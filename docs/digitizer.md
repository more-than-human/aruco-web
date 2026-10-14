# Paper Drawing Digitizer — Technical Specification

## 1. Overview & Design Goals

This system digitizes black Sharpie strokes drawn on a physical sheet of white paper tracked by 4 ArUco markers (`4x4_1000`, IDs `1, 2, 3, 4`) and renders a perspective-corrected, clean 2D drawing onto the screen in real time.

### Core Simplifications & Assumptions

1. **Known Marker Orientation & Paper Layout:** The paper is US Letter in landscape orientation (width = 279.4 mm, height = 215.9 mm). Markers are 20 mm × 20 mm placed with 10 mm margin to the edges.
   Corner marker IDs:
   - Marker 1: Top-Left (TL)
   - Marker 2: Top-Right (TR)
   - Marker 3: Bottom-Right (BR)
   - Marker 4: Bottom-Left (BL)
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

### 3.1 Physical Marker Layout & Centers as Source of Truth

To eliminate jitter caused by individual noisy corner detections, the 4 marker centers (centroids) serve as the source of truth for tracking and perspective calculation:

$$C = \frac{1}{4} \sum_{i=0}^3 \text{corner}_i$$

- **Top-Left Marker (`id = 1`):** Center $C_{TL}$
- **Top-Right Marker (`id = 2`):** Center $C_{TR}$
- **Bottom-Right Marker (`id = 3`):** Center $C_{BR}$
- **Bottom-Left Marker (`id = 4`):** Center $C_{BL}$

### 3.2 Dimensions and Coordinate System

- **Physical Paper Size:** US Letter landscape: $W_{paper} = 279.4\text{ mm}$, $H_{paper} = 215.9\text{ mm}$.
- **Markers:** $20\text{ mm} \times 20\text{ mm}$, inset $10\text{ mm}$ from paper edge to marker outer boundary.
  - Marker center offset from paper edge: $10\text{ mm} + 10\text{ mm} = 20\text{ mm}$.
  - Horizontal center span: $279.4 - 2 \times 20 = 239.4\text{ mm}$.
  - Vertical center span: $215.9 - 2 \times 20 = 175.9\text{ mm}$.
- **Digitizing Zone:** Extends $15\text{ mm}$ beyond marker centers.
  - Digitizing width: $239.4 + 2 \times 15 = 269.4\text{ mm}$.
  - Digitizing height: $175.9 + 2 \times 15 = 205.9\text{ mm}$.
  - Inset ratio: $\frac{15}{269.4}$ horizontally, $\frac{15}{205.9}$ vertically.
- **No-Draw Marker Zones:** Each marker's bounding box in screen space is expanded outward by margin ratio (default $0.2 \times D$) on all 4 sides. Pixels inside these 4 boxes are masked out (`validMask[i] = 0`).

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

**Responsibility:** Given the 4 tracked markers (`id = 0, 1, 2, 3`), compute the perspective transformation between the rectified screen canvas (`width × height`) and the camera image, accounting for the $1.0 \times D$ outward paper margin from marker centers and the $0.1 \times D$ marker exclusion zones.

#### Algorithm

1. **Validate Markers:** Ensure markers `0, 1, 2, 3` are all present and none are coasting (`isPredicted === false`). If any marker is missing or coasting, return `null` (downstream holds the previous canvas).
2. **Compute Marker Centers & Normalized Marker Size ($D_u, D_v$):**
   - For each marker, compute centroid $C = \frac{1}{4} \sum_{i=0}^3 \text{corner}_i$.
   - In camera space, compute the average ratio of marker edge length to center-to-center span:
     $$r_u = \frac{1}{2}\left(\frac{\|M_0[1] - M_0[0]\|}{\|C_1 - C_0\|} + \frac{\|M_2[2] - M_2[3]\|}{\|C_3 - C_2\|}\right)$$
     $$r_v = \frac{1}{2}\left(\frac{\|M_0[3] - M_0[0]\|}{\|C_2 - C_0\|} + \frac{\|M_1[2] - M_1[1]\|}{\|C_3 - C_1\|}\right)$$
3. **Map Marker Centers directly to Inset Screen Coordinates:**
   - Since the full screen (`[0, W] × [0, H]`) includes a $1.0 \times D$ margin outside the marker centers on all sides, the total span of the screen in units of the center span is $1 + 2(1.0 r) = 1 + 2r$.
   - Therefore, the 4 marker centers ($C_0, C_1, C_3, C_2$) map directly to the inset screen rectangle:
     - $u_{min} = W \cdot \frac{r_u}{1 + 2 r_u}, \quad u_{max} = W - u_{min}$
     - $v_{min} = H \cdot \frac{r_v}{1 + 2 r_v}, \quad v_{max} = H - v_{min}$
   - Solve the 4-point Direct Linear Transform (DLT) homography `screenToCameraH` mapping:
     - $(u_{min}, v_{min}) \rightarrow C_0$
     - $(u_{max}, v_{min}) \rightarrow C_1$
     - $(u_{max}, v_{max}) \rightarrow C_3$
     - $(u_{min}, v_{max}) \rightarrow C_2$
4. **Compute Marker Exclusion Boxes ($+0.1 \times D$):**
   - Invert `screenToCameraH` to get `cameraToScreenH`.
   - Project all 4 corners of each marker into screen space, take their bounding box `[minX, minY, maxX, maxY]`, and expand each side outward by $0.1 \times (\text{boxWidth}, \text{boxHeight})$.

#### Public Interface

```ts
export class PaperGeometry {
  constructor(
    private screenWidth: number,
    private screenHeight: number,
    private outwardPaddingRatio = 1.0,
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
