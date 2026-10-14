# Paper Drawing Digitizer — Technical Specification

## 1. Overview & Design Goals

This system digitizes black Sharpie strokes drawn on a physical sheet of white paper tracked by 4 ArUco markers (`4x4_1000`, IDs `0, 1, 2, 3`) and renders a perspective-corrected, clean 2D drawing onto the screen in real time.

### Core Simplifications & Assumptions

1. **Known Marker Orientation:** All 4 markers are upright and consistently oriented. For every marker, `corners` are ordered clockwise starting from Top-Left:
   - `corners[0]` = Top-Left (TL)
   - `corners[1]` = Top-Right (TR)
   - `corners[2]` = Bottom-Right (BR)
   - `corners[3]` = Bottom-Left (BL)
2. **Modular Pipeline:** Each stage of the pipeline is an independent module with plain data structures (TypedArrays and simple interfaces) passed between stages.
3. **Simplicity over Edge-Case Heuristics:** Hand, pen, and shadow rejection is handled uniformly by a simple **Temporal Stability Filter** + **Masked Adaptive Threshold**, avoiding complex skin-tone classifiers or multi-scale blob heuristics.

---

## 2. Pipeline Architecture

The system processes each frame through 4 decoupled stages:

```
[Camera ImageData] + [TrackedMarker[4]]
          │
          ▼
┌──────────────────────────────────────────┐
│ Stage 1: PaperGeometry                   │  Computes Homography (Screen ↔ Camera),
│ (homography.ts)                          │  0.5×D paper expansion, & 0.1×D marker boxes
└─────────────────┬────────────────────────┘
                  │ PaperWarpModel { H, invH, markerBoxes }
                  ▼
┌──────────────────────────────────────────┐
│ Stage 2: PerspectiveWarper               │  Warps camera frame to flat grayscale buffer
│ (warper.ts)                              │  and builds binary valid-paper mask
└─────────────────┬────────────────────────┘
                  │ RectifiedFrame { gray, validMask, width, height }
                  ▼
┌──────────────────────────────────────────┐
│ Stage 3: AdaptiveThreshold               │  Dual Integral Image (sum + count) converts
│ (threshold.ts)                           │  rectified luminance to instantaneous ink
└─────────────────┬────────────────────────┘
                  │ InstantInkFrame { ink, validMask }
                  ▼
┌──────────────────────────────────────────┐
│ Stage 4: TemporalInkFilter               │  Rejects moving hand/pen via frame-to-frame
│ (temporal-filter.ts)                     │  motion check & locks stable strokes into RGBA
└─────────────────┬────────────────────────┘
                  │ ImageData (Final Screen Output)
                  ▼
          [Screen Canvas]
```

---

## 3. Coordinate Systems & Marker Layout

### 3.1 Physical Marker Layout & Outer Corners

Because all markers have fixed orientation, the 4 outer corners defining the base quad of the marker layout are deterministic:

- **Top-Left Marker (`id = 0`):** Outer corner is `corners[0]` (TL)
- **Top-Right Marker (`id = 1`):** Outer corner is `corners[1]` (TR)
- **Bottom-Right Marker (`id = 3`):** Outer corner is `corners[2]` (BR)
- **Bottom-Left Marker (`id = 2`):** Outer corner is `corners[3]` (BL)

### 3.2 Paper Boundary ($0.5 \times D$ Outward Expansion) & Marker Mask ($0.1 \times D$ Expansion)

Let $D_u, D_v$ be the normalized marker width and height in rectified space (measured as a fraction of the distance between the outer corners).

- **Full Paper Viewport:** Expands outward by $0.5 \times D$ beyond the 4 outer corners.
- **No-Draw Marker Zones:** Each marker's bounding box in screen space is expanded outward by $0.1 \times D$ on all 4 sides (total box size $= 1.2 \times D$). Pixels inside these 4 boxes are masked out (`validMask[i] = 0`).

---

## 4. Component Specifications & Interfaces

### Shared Types (`src/lib/digitizer/types.ts`)

```ts
import { IPoint } from "../tracker.ts";

export interface IRect {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** Output of Stage 1 (PaperGeometry) */
export interface PaperWarpModel {
  /** 3x3 Homography mapping Screen (u,v) -> Camera (x,y) */
  screenToCameraH: Float64Array;
  /** 3x3 Homography mapping Camera (x,y) -> Screen (u,v) */
  cameraToScreenH: Float64Array;
  /** 4 axis-aligned marker exclusion zones (expanded by 0.1*D) in screen pixels */
  markerExclusionBoxes: IRect[];
  /** 4 outer paper corners in camera space (for debug overlay) */
  cameraPaperQuad: IPoint[];
}

/** Output of Stage 2 (PerspectiveWarper) */
export interface RectifiedFrame {
  width: number;
  height: number;
  /** Rectified grayscale luminance [0..255], length = width * height */
  gray: Uint8Array;
  /** 1 = drawable paper pixel, 0 = inside marker exclusion zone or off-camera */
  validMask: Uint8Array;
}
```

---

### Stage 1: `PaperGeometry` (`src/lib/digitizer/homography.ts`)

**Responsibility:** Given the 4 tracked markers (`id = 0, 1, 2, 3`), compute the perspective transformation between the rectified screen canvas (`width × height`) and the camera image, accounting for the $0.5 \times D$ outward paper margin and the $0.1 \times D$ marker exclusion zones.

#### Algorithm

1. **Validate Markers:** Ensure markers `0, 1, 2, 3` are all present and none are coasting (`isPredicted === false`). If any marker is missing or coasting, return `null` (downstream holds the previous canvas).
2. **Estimate Normalized Marker Size ($D_u, D_v$):**
   - In camera space, compute the average ratio of marker edge length to outer-quad span:
     $$r_u = \frac{1}{2}\left(\frac{\|M_0[1] - M_0[0]\|}{\|M_1[1] - M_0[0]\|} + \frac{\|M_2[2] - M_2[3]\|}{\|M_3[2] - M_2[3]\|}\right)$$
     $$r_v = \frac{1}{2}\left(\frac{\|M_0[3] - M_0[0]\|}{\|M_2[3] - M_0[0]\|} + \frac{\|M_1[2] - M_1[1]\|}{\|M_3[2] - M_1[1]\|}\right)$$
3. **Map Outer Corners directly to Inset Screen Coordinates:**
   - Since the full screen (`[0, W] × [0, H]`) includes a $0.5 \times D$ margin outside the outer corners on all sides, the total span of the screen in units of the outer-corner quad is $1 + 2(0.5 r) = 1 + r$.
   - Therefore, the 4 outer corners (`M0[0], M1[1], M3[2], M2[3]`) map directly to the inset screen rectangle:
     - $u_{min} = W \cdot \frac{0.5 r_u}{1 + r_u}, \quad u_{max} = W - u_{min}$
     - $v_{min} = H \cdot \frac{0.5 r_v}{1 + r_v}, \quad v_{max} = H - v_{min}$
   - Solve the 4-point Direct Linear Transform (DLT) homography `screenToCameraH` mapping:
     - $(u_{min}, v_{min}) \rightarrow M_0.\text{corners}[0]$
     - $(u_{max}, v_{min}) \rightarrow M_1.\text{corners}[1]$
     - $(u_{max}, v_{max}) \rightarrow M_3.\text{corners}[2]$
     - $(u_{min}, v_{max}) \rightarrow M_2.\text{corners}[3]$
4. **Compute Marker Exclusion Boxes ($+0.1 \times D$):**
   - Invert `screenToCameraH` to get `cameraToScreenH`.
   - Project all 4 corners of each marker into screen space, take their bounding box `[minX, minY, maxX, maxY]`, and expand each side outward by $0.1 \times (\text{boxWidth}, \text{boxHeight})$.

#### Public Interface

```ts
export class PaperGeometry {
  constructor(
    private screenWidth: number,
    private screenHeight: number,
    private outwardPaddingRatio = 0.5,
    private markerMarginRatio = 0.1,
  ) {}

  public computeWarpModel(markers: Marker[]): PaperWarpModel | null;
}
```

---

### Stage 2: `PerspectiveWarper` (`src/lib/digitizer/warper.ts`)

**Responsibility:** Inverse-warp the camera `ImageData` into a flat, perspective-corrected grayscale buffer (`gray`) and populate `validMask`.

#### Algorithm

1. Fill `validMask` with `1`.
2. For each of the 4 `markerExclusionBoxes` in `PaperWarpModel`, set `validMask[v * W + u] = 0` for all pixels inside the rectangle.
3. Loop over every screen pixel $(u, v)$:
   - Project $(u, v)$ to camera coordinates $(x, y)$ using `screenToCameraH`.
   - If $(x, y)$ is outside camera bounds, set `validMask[idx] = 0` and `gray[idx] = 255`.
   - Otherwise, sample camera RGB at $(x, y)$ (nearest-neighbor or bilinear) and compute integer luminance:
     $$Y = (77 R + 150 G + 29 B) \gg 8$$

#### Public Interface

```ts
export class PerspectiveWarper {
  constructor(width: number, height: number);
  public warp(cameraImage: ImageData, model: PaperWarpModel): RectifiedFrame;
}
```

---

### Stage 3: `AdaptiveThreshold` (`src/lib/digitizer/threshold.ts`)

**Responsibility:** Convert the rectified grayscale frame (`RectifiedFrame`) into an instantaneous ink intensity buffer (`ink: Uint8Array`, where `0` = white paper, `255` = black ink) while ignoring masked marker regions.

#### Algorithm (Masked Dual Integral Image)

1. Build two Summed Area Tables (`integralSum` and `integralCount`) in a single pass over the frame:
   - If `validMask[idx] === 1`, add `gray[idx]` to `integralSum` and `1` to `integralCount`.
   - If `validMask[idx] === 0`, add `0` to both (markers do not skew local lighting averages).
2. For each pixel $(u, v)$ where `validMask[idx] === 1`:
   - Compute `boxSum` and `boxCount` over a local window of size `windowSize × windowSize` (e.g., `31 × 31`).
   - Compute local paper background brightness: $\mu = \text{boxSum} / \text{boxCount}$.
   - Compute darkness delta: $\Delta = \mu - \text{gray}[idx]$.
   - Apply soft threshold between `thresholdLow` (e.g., `15`) and `thresholdHigh` (e.g., `35`) to produce smooth anti-aliased ink values in `[0..255]`.

#### Public Interface

```ts
export interface ThresholdConfig {
  windowSize?: number; // default: 31
  thresholdLow?: number; // default: 15
  thresholdHigh?: number; // default: 35
}

export class AdaptiveThreshold {
  constructor(width: number, height: number, config?: ThresholdConfig);
  public process(frame: RectifiedFrame): Uint8Array;
}
```

---

### Stage 4: `TemporalInkFilter` (`src/lib/digitizer/temporal-filter.ts`)

**Responsibility:** Reject moving objects (hand, fingers, pen barrel, moving shadows) and maintain a clean, persistent `ImageData` canvas of confirmed ink strokes.

#### Algorithm

1. **Block-Based Motion Detection:**
   - Compare current `frame.gray[i]` with previous frame's `prevGray[i]`.
   - A pixel is flagged as changing if `|gray[i] - prevGray[i]| > motionThreshold` (default: `10`).
   - Update `prevGray[i] = frame.gray[i]`.
   - To prevent the uniform interior of a moving hand or pen from updating, divide the image into coarse blocks (e.g., `16 × 16` px tiles). If any pixels in a tile (or its immediate 3×3 neighboring tiles) are moving, mark the entire tile as **unstable**.
2. **Stability Counter & Persistent Ink Update:**
   - For each pixel $i$:
     - If `validMask[i] === 0` or its tile is **unstable**, set `stabilityCount[i] = 0` (freeze `persistentInk[i]` at its current value).
     - Otherwise, increment `stabilityCount[i]`.
     - When `stabilityCount[i] >= requiredStableFrames` (default: `10` frames $\approx 160\text{ ms}$):
       - Blend `instantInk[i]` into `persistentInk[i]`:
         $$\text{persistentInk}[i] \leftarrow (1 - \alpha) \cdot \text{persistentInk}[i] + \alpha \cdot \text{instantInk}[i]$$
         (default $\alpha = 0.3$).
3. **Render to `ImageData`:**
   - Convert `persistentInk[i]` to screen pixel intensity `out = 255 - persistentInk[i]` (white paper background, black ink strokes).

#### Public Interface

```ts
export interface TemporalFilterConfig {
  motionThreshold?: number; // default: 10 (gray levels)
  tileSize?: number; // default: 16 (px)
  requiredStableFrames?: number; // default: 10 (frames)
  blendAlpha?: number; // default: 0.3
}

export class TemporalInkFilter {
  constructor(width: number, height: number, config?: TemporalFilterConfig);
  public update(frame: RectifiedFrame, instantInk: Uint8Array): ImageData;
  public getOutputImage(): ImageData;
  public reset(): void;
}
```

---

## 5. Top-Level Coordinator & App Integration

### `PaperDigitizer` Facade (`src/lib/digitizer/index.ts`)

Composes the 4 stages behind a single method call for `app.ts`:

```ts
export class PaperDigitizer {
  private geometry: PaperGeometry;
  private warper: PerspectiveWarper;
  private threshold: AdaptiveThreshold;
  private temporalFilter: TemporalInkFilter;

  constructor(width: number, height: number);

  /**
   * Runs the 4-stage pipeline when all 4 markers are stably tracked.
   * Returns the latest digitized ImageData and current warp model (for debug overlays).
   */
  public processFrame(
    cameraImage: ImageData,
    markers: (Marker | TrackedMarker)[],
  ): {
    digitizedImage: ImageData;
    warpModel: PaperWarpModel | null;
  };

  public reset(): void;
}
```

### Integration in `app.ts`

1. Instantiate `const digitizer = new PaperDigitizer(VIDEO_WIDTH, VIDEO_HEIGHT);`
2. In `startDetectionLoop`:
   ```ts
   const { digitizedImage, warpModel } = digitizer.processFrame(imageData, markers);
   ```
3. Render `digitizedImage` onto a dedicated output `<canvas id="paper-canvas">` (or overlay), and optionally draw `warpModel.cameraPaperQuad` on the camera preview canvas to visualize the $0.5 \times D$ tracked boundary.
