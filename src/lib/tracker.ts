import { Detector, Marker, type IConfig, type IImage, type IPoint } from "./aruco";

export interface IMarkerLike {
  id: number;
  corners: IPoint[];
  hammingDistance: number;
}

export interface IMarkerDetector {
  detect(image: IImage | ImageData | { width: number; height: number; data: Uint8ClampedArray | Uint8Array }): (Marker | IMarkerLike)[];
}

export interface ITrackerConfig extends IConfig {
  /**
   * How many consecutive dropped frames a marker can be predicted
   * before its track is deleted. (Default: 5)
   */
  maxMissedFrames?: number;

  /**
   * Process noise spectral density (q). Controls how quickly the filter
   * adapts to sudden accelerations / direction changes. (Default: 8.0)
   */
  processNoise?: number;

  /**
   * Measurement noise variance (r) in pixels^2. Controls how much the filter
   * trusts the ArUco detector vs. its own motion model. (Default: 2.0)
   */
  measurementNoise?: number;

  /**
   * Velocity damping factor applied during coasting (dropped frames)
   * to prevent markers from flying off-screen indefinitely. (Default: 0.82)
   */
  coastVelocityDamping?: number;

  /**
   * Fixed time step (dt) in seconds/frames. If omitted, standard normalized
   * frame step dt = 1.0 is used unless a timestamp is passed to detect(). (Default: 1.0)
   */
  fixedDt?: number;

  /**
   * Enable KLT Sparse Optical Flow & ArUco Border Verification during missed frames
   * when an image buffer is available. Dramatically improves tracking under motion blur
   * and immediately kills ghost markers when occluded. (Default: true)
   */
  enableOpticalFlow?: boolean;

  /**
   * Half-window radius for KLT optical flow patch tracking in pixels.
   * E.g., radius 5 = 11x11 patch. (Default: 5)
   */
  opticalFlowWinRadius?: number;

  /**
   * Minimum Normalized Cross-Correlation (NCC) score [0..1] required for
   * optical flow corner tracking to be accepted. (Default: 0.60)
   */
  minOpticalFlowConfidence?: number;

  /**
   * Minimum intensity difference (outer white quiet-zone minus inner black border)
   * required to keep a coasting marker alive when visual data is present. (Default: 18)
   */
  minBorderContrast?: number;

  /**
   * Optional pre-instantiated Detector. If not provided, Tracker creates its own.
   */
  detector?: IMarkerDetector;
}

/**
 * Extends the base Marker class from detector.ts so it is a drop-in replacement
 * while exposing optional tracking telemetry.
 */
export class TrackedMarker extends Marker {
  /** True if the marker was not detected by ArUco in the current frame and is coasting/KLT-tracked */
  public isPredicted: boolean;
  /** Number of consecutive frames the ArUco detector missed this marker */
  public missedFrames: number;
  /** Total number of frames this marker track has been alive */
  public age: number;
  /** Estimated corner velocities [{x: vx0, y: vy0}, ...] in pixels per unit dt */
  public velocity: IPoint[];
  /** Tracking confidence in [0, 1] (1.0 when freshly detected, decays or reflects KLT/border score when predicted) */
  public confidence: number;

  constructor(
    id: number,
    corners: IPoint[],
    hammingDistance: number,
    isPredicted: boolean = false,
    missedFrames: number = 0,
    age: number = 1,
    velocity: IPoint[] = [
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { x: 0, y: 0 },
    ],
    confidence: number = 1.0,
  ) {
    super(id, corners, hammingDistance);
    this.isPredicted = isPredicted;
    this.missedFrames = missedFrames;
    this.age = age;
    this.velocity = velocity;
    this.confidence = confidence;
  }
}

/**
 * Fast 2-Level Grayscale Image Pyramid (Level 0: 1x full-res, Level 1: 0.5x half-res).
 * Handles arbitrary motion blur & large frame-to-frame displacements while remaining
 * allocation-free after warmup.
 */
class GrayscalePyramid {
  public width0: number = 0;
  public height0: number = 0;
  public data0: Uint8Array = new Uint8Array(0);

  public width1: number = 0;
  public height1: number = 0;
  public data1: Uint8Array = new Uint8Array(0);

  public updateFromImage(image: { width: number; height: number; data: Uint8ClampedArray | Uint8Array }): void {
    const w0 = image.width | 0;
    const h0 = image.height | 0;
    const src = image.data;
    const numPixels0 = w0 * h0;

    if (this.data0.length !== numPixels0) {
      this.data0 = new Uint8Array(numPixels0);
    }
    this.width0 = w0;
    this.height0 = h0;

    const dst0 = this.data0;

    // Convert RGBA or Grayscale input into contiguous 1-channel luma buffer
    if (src.length === numPixels0 * 4) {
      let j = 0;
      for (let i = 0; i < numPixels0; ++i, j += 4) {
        // Fast integer BT.601 luma: (77*R + 150*G + 29*B) >> 8
        dst0[i] = (77 * src[j] + 150 * src[j + 1] + 29 * src[j + 2]) >> 8;
      }
    } else if (src.length >= numPixels0) {
      dst0.set(src.subarray(0, numPixels0));
    }

    // Build Level 1 (2x2 box-filtered downsample for anti-aliased coarse optical flow)
    const w1 = w0 >> 1;
    const h1 = h0 >> 1;
    const numPixels1 = w1 * h1;

    if (this.data1.length !== numPixels1) {
      this.data1 = new Uint8Array(numPixels1);
    }
    this.width1 = w1;
    this.height1 = h1;

    const dst1 = this.data1;
    for (let y = 0; y < h1; ++y) {
      const row0 = y * 2 * w0;
      const row1 = row0 + w0;
      const outRow = y * w1;
      for (let x = 0; x < w1; ++x) {
        const x2 = x * 2;
        dst1[outRow + x] = (dst0[row0 + x2] + dst0[row0 + x2 + 1] + dst0[row1 + x2] + dst0[row1 + x2 + 1] + 2) >> 2;
      }
    }
  }

  public copyFrom(other: GrayscalePyramid): void {
    if (other.width0 === 0 || other.height0 === 0) return;

    if (this.data0.length !== other.data0.length) {
      this.data0 = new Uint8Array(other.data0.length);
    }
    this.width0 = other.width0;
    this.height0 = other.height0;
    this.data0.set(other.data0);

    if (this.data1.length !== other.data1.length) {
      this.data1 = new Uint8Array(other.data1.length);
    }
    this.width1 = other.width1;
    this.height1 = other.height1;
    this.data1.set(other.data1);
  }
}

/**
 * Bilinear subpixel grayscale sampler.
 */
function sampleBilinear(img: Uint8Array, width: number, height: number, x: number, y: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  if (x0 < 0 || y0 < 0 || x0 + 1 >= width || y0 + 1 >= height) {
    const cx = Math.max(0, Math.min(width - 1, Math.round(x)));
    const cy = Math.max(0, Math.min(height - 1, Math.round(y)));
    return img[cy * width + cx];
  }
  const fx = x - x0;
  const fy = y - y0;
  const ofx = 1.0 - fx;
  const ofy = 1.0 - fy;

  const idx = y0 * width + x0;
  const i00 = img[idx];
  const i10 = img[idx + 1];
  const i01 = img[idx + width];
  const i11 = img[idx + width + 1];

  return ofy * (ofx * i00 + fx * i10) + fy * (ofx * i01 + fx * i11);
}

/**
 * Single-level Inverse Compositional Lucas-Kanade (KLT) patch tracker with
 * Zero-Mean Normalized Cross-Correlation (ZNCC) verification.
 */
function trackPointSingleLevel(
  prevImg: Uint8Array,
  currImg: Uint8Array,
  width: number,
  height: number,
  prevX: number,
  prevY: number,
  initCurrX: number,
  initCurrY: number,
  radius: number,
  maxIters: number,
): { x: number; y: number; ncc: number; valid: boolean } {
  const margin = radius + 2;
  if (
    prevX < margin ||
    prevY < margin ||
    prevX >= width - margin ||
    prevY >= height - margin ||
    initCurrX < margin ||
    initCurrY < margin ||
    initCurrX >= width - margin ||
    initCurrY >= height - margin
  ) {
    return { x: initCurrX, y: initCurrY, ncc: 0, valid: false };
  }

  const winSize = 2 * radius + 1;
  const area = winSize * winSize;
  const T = new Float32Array(area);
  const gxBuf = new Float32Array(area);
  const gyBuf = new Float32Array(area);

  let gxx = 0.0;
  let gxy = 0.0;
  let gyy = 0.0;
  let sumT = 0.0;

  // Precompute template patch T and spatial gradients on prevImg
  let idx = 0;
  for (let wy = -radius; wy <= radius; ++wy) {
    const py = prevY + wy;
    for (let wx = -radius; wx <= radius; ++wx, ++idx) {
      const px = prevX + wx;
      const val = sampleBilinear(prevImg, width, height, px, py);
      T[idx] = val;
      sumT += val;

      const ix = 0.5 * (sampleBilinear(prevImg, width, height, px + 1.0, py) - sampleBilinear(prevImg, width, height, px - 1.0, py));
      const iy = 0.5 * (sampleBilinear(prevImg, width, height, px, py + 1.0) - sampleBilinear(prevImg, width, height, px, py - 1.0));

      gxBuf[idx] = ix;
      gyBuf[idx] = iy;

      gxx += ix * ix;
      gxy += ix * iy;
      gyy += iy * iy;
    }
  }

  // Tikhonov regularization on Hessian so motion-blurred edges remain invertible
  gxx += 1e-3;
  gyy += 1e-3;

  const det = gxx * gyy - gxy * gxy;
  if (det < 1.0) {
    return { x: initCurrX, y: initCurrY, ncc: 0, valid: false };
  }
  const invDet = 1.0 / det;

  let cx = initCurrX;
  let cy = initCurrY;

  // Iterative Gauss-Newton Lucas-Kanade alignment
  for (let iter = 0; iter < maxIters; ++iter) {
    if (cx < margin || cy < margin || cx >= width - margin || cy >= height - margin) {
      return { x: initCurrX, y: initCurrY, ncc: 0, valid: false };
    }

    let bx = 0.0;
    let by = 0.0;
    let k = 0;

    for (let wy = -radius; wy <= radius; ++wy) {
      const qy = cy + wy;
      for (let wx = -radius; wx <= radius; ++wx, ++k) {
        const qx = cx + wx;
        const diff = T[k] - sampleBilinear(currImg, width, height, qx, qy);
        bx += gxBuf[k] * diff;
        by += gyBuf[k] * diff;
      }
    }

    const dx = invDet * (gyy * bx - gxy * by);
    const dy = invDet * (gxx * by - gxy * bx);

    cx += dx;
    cy += dy;

    if (dx * dx + dy * dy < 0.0004) {
      break;
    }
  }

  if (cx < margin || cy < margin || cx >= width - margin || cy >= height - margin) {
    return { x: initCurrX, y: initCurrY, ncc: 0, valid: false };
  }

  // Compute Zero-Mean Normalized Cross-Correlation (ZNCC) to reject occlusions
  const meanT = sumT / area;
  let sumI = 0.0;
  const I = new Float32Array(area);
  let k = 0;
  for (let wy = -radius; wy <= radius; ++wy) {
    const qy = cy + wy;
    for (let wx = -radius; wx <= radius; ++wx, ++k) {
      const val = sampleBilinear(currImg, width, height, cx + wx, qy);
      I[k] = val;
      sumI += val;
    }
  }
  const meanI = sumI / area;

  let num = 0.0;
  let denT = 0.0;
  let denI = 0.0;
  for (let i = 0; i < area; ++i) {
    const dt = T[i] - meanT;
    const di = I[i] - meanI;
    num += dt * di;
    denT += dt * dt;
    denI += di * di;
  }

  const den = Math.sqrt(denT * denI);
  const ncc = den > 1e-4 ? num / den : 0.0;

  return { x: cx, y: cy, ncc, valid: true };
}

/**
 * Coarse-to-Fine 2-Level Pyramidal KLT Tracker for a single corner.
 * Uses Kalman velocity prediction as the initial warm-start at the coarse level.
 */
function trackCornerPyramidalKLT(
  prevPyr: GrayscalePyramid,
  currPyr: GrayscalePyramid,
  prevX: number,
  prevY: number,
  predX: number,
  predY: number,
  winRadius: number,
): { x: number; y: number; ncc: number; valid: boolean } {
  // Level 1 (0.5x resolution): captures large displacements & heavy motion blur
  const coarseRadius = Math.max(3, winRadius - 1);
  const coarseRes = trackPointSingleLevel(
    prevPyr.data1,
    currPyr.data1,
    prevPyr.width1,
    prevPyr.height1,
    prevX * 0.5,
    prevY * 0.5,
    predX * 0.5,
    predY * 0.5,
    coarseRadius,
    6,
  );

  // Use coarse result if valid, otherwise fall back to Kalman prediction at Level 0
  const refineInitX = coarseRes.valid && coarseRes.ncc > 0.35 ? coarseRes.x * 2.0 : predX;
  const refineInitY = coarseRes.valid && coarseRes.ncc > 0.35 ? coarseRes.y * 2.0 : predY;

  // Level 0 (1.0x full resolution): refines to sub-pixel corner accuracy
  const fineRes = trackPointSingleLevel(prevPyr.data0, currPyr.data0, prevPyr.width0, prevPyr.height0, prevX, prevY, refineInitX, refineInitY, winRadius, 8);

  if (fineRes.valid) {
    return fineRes;
  }
  if (coarseRes.valid) {
    return { x: coarseRes.x * 2.0, y: coarseRes.y * 2.0, ncc: coarseRes.ncc, valid: true };
  }
  return { x: predX, y: predY, ncc: 0, valid: false };
}

/**
 * Verifies that a predicted quad in the image actually looks like an ArUco marker:
 * 1. Geometrically convex with reasonable edge ratios and area.
 * 2. Exhibits the characteristic ArUco dark outer border vs. bright quiet-zone contrast,
 *    OR internal binary bit variance.
 * Immediately catches markers that have left the view or been covered by an object.
 */
function verifyArUcoQuadAppearance(
  gray: Uint8Array,
  width: number,
  height: number,
  corners: IPoint[],
  minBorderContrast: number,
): { valid: boolean; score: number } {
  const c0 = corners[0];
  const c1 = corners[1];
  const c2 = corners[2];
  const c3 = corners[3];

  // 1. Check that at least the centroid and 3 corners lie inside the frame
  let insideCount = 0;
  for (let i = 0; i < 4; ++i) {
    if (corners[i].x >= 2 && corners[i].x < width - 2 && corners[i].y >= 2 && corners[i].y < height - 2) {
      insideCount++;
    }
  }
  if (insideCount < 3) {
    return { valid: false, score: 0 };
  }

  // 2. Check polygon convexity & non-self-intersection via cross products
  let sign = 0;
  let minEdgeSq = Infinity;
  let maxEdgeSq = 0;
  for (let i = 0; i < 4; ++i) {
    const a = corners[i];
    const b = corners[(i + 1) & 3];
    const c = corners[(i + 2) & 3];

    const abx = b.x - a.x;
    const aby = b.y - a.y;
    const bcx = c.x - b.x;
    const bcy = c.y - b.y;

    const edgeSq = abx * abx + aby * aby;
    if (edgeSq < minEdgeSq) minEdgeSq = edgeSq;
    if (edgeSq > maxEdgeSq) maxEdgeSq = edgeSq;

    const cross = abx * bcy - aby * bcx;
    if (Math.abs(cross) < 4.0) {
      return { valid: false, score: 0 }; // Collinear / collapsed quad
    }
    const currSign = cross > 0 ? 1 : -1;
    if (sign === 0) {
      sign = currSign;
    } else if (sign !== currSign) {
      return { valid: false, score: 0 }; // Non-convex / bowtie quad
    }
  }

  // Reject implausible aspect ratios or tiny collapsed quads
  if (minEdgeSq < 64.0 || maxEdgeSq / minEdgeSq > 16.0) {
    return { valid: false, score: 0 };
  }

  // 3. Sample ArUco Outer Quiet-Zone (u, v outside [0,1]) vs. Inner Black Border (u, v inside [0,1])
  // Bilinear homography approximation P(u,v) from unit square to quad:
  const evalQuad = (u: number, v: number): { x: number; y: number } => {
    const ou = 1.0 - u;
    const ov = 1.0 - v;
    return {
      x: ov * (ou * c0.x + u * c1.x) + v * (ou * c3.x + u * c2.x),
      y: ov * (ou * c0.y + u * c1.y) + v * (ou * c3.y + u * c2.y),
    };
  };

  // Sample 12 points along the 4 edges (3 per edge) comparing outer quiet-zone vs inner black border
  const sampleTs = [0.25, 0.5, 0.75];
  const dOut = -0.1; // 10% outside marker edge (white quiet zone)
  const dIn = 0.09; // 9% inside marker edge (black border cell)

  let sumOuter = 0.0;
  let sumInner = 0.0;
  let validSamples = 0;

  const pairs: [number, number, number, number][] = [];
  for (let i = 0; i < sampleTs.length; ++i) {
    const t = sampleTs[i];
    pairs.push([t, dOut, t, dIn]); // Top edge
    pairs.push([1.0 - dOut, t, 1.0 - dIn, t]); // Right edge
    pairs.push([t, 1.0 - dOut, t, 1.0 - dIn]); // Bottom edge
    pairs.push([dOut, t, dIn, t]); // Left edge
  }

  for (let i = 0; i < pairs.length; ++i) {
    const [uOut, vOut, uIn, vIn] = pairs[i];
    const pOut = evalQuad(uOut, vOut);
    const pIn = evalQuad(uIn, vIn);

    if (pOut.x >= 1 && pOut.x < width - 1 && pOut.y >= 1 && pOut.y < height - 1 && pIn.x >= 1 && pIn.x < width - 1 && pIn.y >= 1 && pIn.y < height - 1) {
      sumOuter += sampleBilinear(gray, width, height, pOut.x, pOut.y);
      sumInner += sampleBilinear(gray, width, height, pIn.x, pIn.y);
      validSamples++;
    }
  }

  if (validSamples < 6) {
    return { valid: false, score: 0 };
  }

  const meanOuter = sumOuter / validSamples;
  const meanInner = sumInner / validSamples;
  const contrast = meanOuter - meanInner;

  if (contrast < minBorderContrast) {
    return { valid: false, score: Math.max(0, contrast / Math.max(1, minBorderContrast)) };
  }

  const score = Math.min(1.0, contrast / 45.0);
  return { valid: true, score };
}

/**
 * Fits a 2D Similarity Transform (Translation + Isotropic Scale + Rotation) from
 * source points `src` to target points `dst`, and blends `dst` with the rigid
 * transformation of `src` by `rigidityWeight` [0..1].
 *
 * This prevents an individual noisy corner from warping the marker polygon during
 * motion blur or KLT coasting.
 */
function regularizeQuadShape(src: IPoint[], dst: IPoint[], rigidityWeight: number): IPoint[] {
  if (rigidityWeight <= 0) return dst;

  // 1. Centroids
  let scx = 0,
    scy = 0,
    dcx = 0,
    dcy = 0;
  for (let i = 0; i < 4; ++i) {
    scx += src[i].x;
    scy += src[i].y;
    dcx += dst[i].x;
    dcy += dst[i].y;
  }
  scx *= 0.25;
  scy *= 0.25;
  dcx *= 0.25;
  dcy *= 0.25;

  // 2. Procrustes similarity parameters (a = s*cos(theta), b = s*sin(theta))
  let numA = 0.0;
  let numB = 0.0;
  let den = 0.0;
  for (let i = 0; i < 4; ++i) {
    const sx = src[i].x - scx;
    const sy = src[i].y - scy;
    const dx = dst[i].x - dcx;
    const dy = dst[i].y - dcy;

    numA += sx * dx + sy * dy;
    numB += sx * dy - sy * dx;
    den += sx * sx + sy * sy;
  }

  if (den < 1e-3) return dst;

  let a = numA / den;
  let b = numB / den;

  // Clamp instantaneous per-frame scale change to [0.80, 1.25] to prevent collapse/explosion
  const scale = Math.hypot(a, b);
  if (scale > 1e-4) {
    const clampedScale = Math.max(0.8, Math.min(1.25, scale));
    const factor = clampedScale / scale;
    a *= factor;
    b *= factor;
  }

  const wRigid = Math.max(0.0, Math.min(1.0, rigidityWeight));
  const wFree = 1.0 - wRigid;

  const out: IPoint[] = new Array(4);
  for (let i = 0; i < 4; ++i) {
    const sx = src[i].x - scx;
    const sy = src[i].y - scy;
    const rigidX = dcx + (a * sx - b * sy);
    const rigidY = dcy + (b * sx + a * sy);

    out[i] = {
      x: wRigid * rigidX + wFree * dst[i].x,
      y: wRigid * rigidY + wFree * dst[i].y,
    };
  }

  return out;
}

/**
 * 16-State Rigid-Regularized Linear Kalman Filter for a single ArUco Marker.
 *
 * State vector x (Float32Array(16)):
 *   Indices 0..7  : [x0, y0, x1, y1, x2, y2, x3, y3] (Corner positions in pixel space)
 *   Indices 8..15 : [vx0, vy0, vx1, vy1, vx2, vy2, vx3, vy3] (Corner velocities)
 *
 * Combines 2x2 block Kalman filtering with:
 * - Rigid-body velocity decomposition (prevents polygon warping during coasting)
 * - Huber robust innovation gating (prevents motion-blur corner spikes from corrupting velocity)
 * - Lazy 16x16 covariance synchronization
 */
export class MarkerKalmanFilter {
  public readonly id: number;
  public readonly x: Float32Array; // 16-element state vector
  private readonly _P: Float32Array; // 16x16 covariance matrix (lazily synced on access)
  private pDirty: boolean = true;

  // Compact 2x2 block scalars representing the isotropic 16x16 covariance:
  private p00: number;
  private p01: number;
  private p10: number;
  private p11: number;

  // Snapshot of corner coordinates from the previous frame (used as KLT template centers)
  public readonly prevCorners: Float32Array;
  // Last validated canonical template shape (relative to centroid, normalized by radius)
  private readonly canonicalShape: Float32Array;
  private referencePerimeter: number = 0;

  public lastHammingDistance: number;
  public missedFrames: number = 0;
  public age: number = 1;
  public confidence: number = 1.0;
  public lastTimestamp: number | null = null;

  private readonly q: number;
  private readonly r: number;

  constructor(
    id: number,
    initialCorners: IPoint[],
    hammingDistance: number,
    processNoise: number = 8.0,
    measurementNoise: number = 2.0,
    timestamp: number | null = null,
  ) {
    this.id = id;
    this.lastHammingDistance = hammingDistance;
    this.q = processNoise;
    this.r = measurementNoise;
    this.lastTimestamp = timestamp;

    this.x = new Float32Array(16);
    this._P = new Float32Array(256);
    this.prevCorners = new Float32Array(8);
    this.canonicalShape = new Float32Array(8);

    // Initialize positions with the 4 corners, velocities with 0
    for (let i = 0; i < 4; ++i) {
      const cx = initialCorners[i].x;
      const cy = initialCorners[i].y;
      this.x[i * 2] = cx;
      this.x[i * 2 + 1] = cy;
      this.prevCorners[i * 2] = cx;
      this.prevCorners[i * 2 + 1] = cy;
      this.x[8 + i * 2] = 0.0;
      this.x[8 + i * 2 + 1] = 0.0;
    }

    this.updateCanonicalShape(initialCorners, 1.0);

    // Initial covariance: moderate certainty on initial velocity so a single noisy
    // second frame doesn't cause wild velocity overshoot.
    this.p00 = measurementNoise;
    this.p01 = 0.0;
    this.p10 = 0.0;
    this.p11 = processNoise * 2.5;
    this.pDirty = true;
  }

  /**
   * 16x16 row-major covariance matrix P (lazily populated when read).
   */
  public get P(): Float32Array {
    if (this.pDirty) {
      this.syncDenseCovariance();
    }
    return this._P;
  }

  /**
   * Saves current corner positions into `prevCorners` before advancing to a new frame.
   */
  public snapshotPreviousCorners(): void {
    for (let i = 0; i < 8; ++i) {
      this.prevCorners[i] = this.x[i];
    }
  }

  /**
   * Kalman Prediction Step:
   *   x_{k|k-1} = F * x_{k-1}
   *   P_{k|k-1} = F * P_{k-1} * F^T + Q
   *
   * Decomposes corner velocities into rigid-body motion (centroid + angular/scale)
   * and non-rigid deformation, heavily damping non-rigid velocity during coasting
   * so corners never twist or drift apart.
   */
  public predict(dt: number = 1.0, velocityDamping: number = 1.0): void {
    if (dt <= 0) dt = 1.0;

    const x = this.x;

    // 1. Decompose velocity into rigid centroid translation + local relative velocity
    const vcx = 0.25 * (x[8] + x[10] + x[12] + x[14]) * velocityDamping;
    const vcy = 0.25 * (x[9] + x[11] + x[13] + x[15]) * velocityDamping;

    // During coasting (missedFrames > 0), damp individual corner divergence much more aggressively
    const nonRigidDamping = this.missedFrames > 0 ? velocityDamping * 0.35 : velocityDamping * 0.85;

    const prevPts = this.getCorners();
    const rawPredicted: IPoint[] = new Array(4);

    for (let i = 0; i < 4; ++i) {
      const idxX = i * 2;
      const idxY = idxX + 1;

      const dvx = (x[8 + idxX] * velocityDamping - vcx) * nonRigidDamping;
      const dvy = (x[8 + idxY] * velocityDamping - vcy) * nonRigidDamping;

      const vx = vcx + dvx;
      const vy = vcy + dvy;

      x[8 + idxX] = vx;
      x[8 + idxY] = vy;

      rawPredicted[i] = {
        x: x[idxX] + dt * vx,
        y: x[idxY] + dt * vy,
      };
    }

    // 2. Enforce rigid similarity regularization during coasting so quad geometry stays intact
    const rigidity = this.missedFrames > 0 ? 0.85 : 0.25;
    const regPts = regularizeQuadShape(prevPts, rawPredicted, rigidity);

    for (let i = 0; i < 4; ++i) {
      x[i * 2] = regPts[i].x;
      x[i * 2 + 1] = regPts[i].y;
    }

    // 3. Covariance prediction: P_{k|k-1} = F * P_{k-1} * F^T + Q
    const dt2 = dt * dt;
    const dt3 = dt2 * dt;
    const dt4 = dt2 * dt2;

    const q00 = 0.25 * dt4 * this.q;
    const q01 = 0.5 * dt3 * this.q;
    const q11 = dt2 * this.q;

    const prevP00 = this.p00;
    const prevP01 = this.p01;
    const prevP10 = this.p10;
    const prevP11 = this.p11;

    this.p00 = prevP00 + dt * (prevP10 + prevP01) + dt2 * prevP11 + q00;
    this.p01 = prevP01 + dt * prevP11 + q01;
    this.p10 = prevP10 + dt * prevP11 + q01;
    this.p11 = prevP11 + q11;

    this.pDirty = true;
  }

  /**
   * Kalman Measurement Update Step:
   *   K = P_{k|k-1} * H^T * (H * P_{k|k-1} * H^T + R)^{-1}
   *   x_k = x_{k|k-1} + K * (z_k - H * x_{k|k-1})
   *   P_k = (I - K * H) * P_{k|k-1}
   *
   * Includes Huber robust velocity update so motion-blur corner glitches don't
   * inject explosive velocities right before a dropped frame.
   */
  public update(measuredCorners: IPoint[], hammingDistance: number, measurementScale: number = 1.0): void {
    const aligned = this.alignCorners(measuredCorners);

    // Scale effective measurement noise (e.g. slightly higher when updated via KLT vs ArUco)
    const effectiveR = this.r * measurementScale + (hammingDistance > 0 ? hammingDistance * 1.5 : 0);

    // Innovation covariance S = H * P * H^T + R = (p00 + r) * I_8
    const invS = 1.0 / (this.p00 + effectiveR);

    // Kalman gain K = P * H^T * S^{-1} = [ k0 * I_8, k1 * I_8 ]^T
    const k0 = this.p00 * invS;
    const k1 = this.p10 * invS;

    const x = this.x;

    // Compute median innovation (centroid shift) to detect single-corner motion-blur outliers
    let meanDx = 0.0;
    let meanDy = 0.0;
    for (let i = 0; i < 4; ++i) {
      meanDx += aligned[i].x - x[i * 2];
      meanDy += aligned[i].y - x[i * 2 + 1];
    }
    meanDx *= 0.25;
    meanDy *= 0.25;

    // Huber threshold on non-rigid corner residual (relative to marker size)
    const huberDelta = Math.max(4.0, this.referencePerimeter * 0.04);

    for (let i = 0; i < 4; ++i) {
      const idxX = i * 2;
      const idxY = idxX + 1;

      const yX = aligned[i].x - x[idxX];
      const yY = aligned[i].y - x[idxY];

      // Position update trusts the detector measurement
      x[idxX] += k0 * yX;
      x[idxY] += k0 * yY;

      // Velocity update uses Huber weighting on the non-rigid component of the residual
      // so a single blurred corner snapping inward doesn't corrupt that corner's velocity!
      const nonRigidX = yX - meanDx;
      const nonRigidY = yY - meanDy;
      const nonRigidNorm = Math.hypot(nonRigidX, nonRigidY);
      const huberWeight = nonRigidNorm > huberDelta ? huberDelta / nonRigidNorm : 1.0;

      const robustYx = meanDx + huberWeight * nonRigidX;
      const robustYy = meanDy + huberWeight * nonRigidY;

      x[8 + idxX] += k1 * robustYx;
      x[8 + idxY] += k1 * robustYy;
    }

    // Update covariance matrix P_k = (I - K * H) * P_{k|k-1}
    const prevP00 = this.p00;
    const prevP01 = this.p01;
    const prevP10 = this.p10;
    const prevP11 = this.p11;

    const oneMinusK0 = 1.0 - k0;
    this.p00 = oneMinusK0 * prevP00;
    this.p01 = oneMinusK0 * prevP01;
    this.p10 = prevP10 - k1 * prevP00;
    this.p11 = prevP11 - k1 * prevP01;

    this.pDirty = true;

    if (hammingDistance === 0 && measurementScale <= 1.0) {
      this.updateCanonicalShape(this.getCorners(), 0.2);
    }

    this.lastHammingDistance = hammingDistance;
    this.missedFrames = 0;
    this.confidence = 1.0;
    this.age++;
  }

  /**
   * Updates filter state from KLT Optical Flow corner measurements during a missed ArUco frame.
   * Preserves `missedFrames` counter while keeping positions and velocities locked onto the image.
   */
  public applyOpticalFlowUpdate(flowCorners: IPoint[], dt: number, flowConfidence: number): void {
    const prevPts: IPoint[] = [
      { x: this.prevCorners[0], y: this.prevCorners[1] },
      { x: this.prevCorners[2], y: this.prevCorners[3] },
      { x: this.prevCorners[4], y: this.prevCorners[5] },
      { x: this.prevCorners[6], y: this.prevCorners[7] },
    ];

    // Regularize the 4 optical flow corners against the previous frame's quad shape
    // so an individual blurry corner cannot skew the quad
    const regularized = regularizeQuadShape(prevPts, flowCorners, 0.55);

    const invDt = dt > 1e-3 ? 1.0 / dt : 1.0;
    const alphaPos = Math.min(0.92, Math.max(0.5, flowConfidence));
    const alphaVel = 0.65;

    const x = this.x;
    for (let i = 0; i < 4; ++i) {
      const idxX = i * 2;
      const idxY = idxX + 1;

      const targetX = regularized[i].x;
      const targetY = regularized[i].y;

      // Blend KLT measurement with Kalman prediction
      const newX = alphaPos * targetX + (1.0 - alphaPos) * x[idxX];
      const newY = alphaPos * targetY + (1.0 - alphaPos) * x[idxY];

      // Measure actual frame-to-frame corner displacement velocity
      const instVx = (newX - this.prevCorners[idxX]) * invDt;
      const instVy = (newY - this.prevCorners[idxY]) * invDt;

      x[idxX] = newX;
      x[idxY] = newY;
      x[8 + idxX] = alphaVel * instVx + (1.0 - alphaVel) * x[8 + idxX];
      x[8 + idxY] = alphaVel * instVy + (1.0 - alphaVel) * x[8 + idxY];
    }

    this.confidence = flowConfidence;
  }

  /**
   * Updates the stored canonical shape & perimeter of the marker.
   */
  private updateCanonicalShape(corners: IPoint[], blend: number): void {
    let cx = 0,
      cy = 0,
      perim = 0;
    for (let i = 0; i < 4; ++i) {
      cx += corners[i].x;
      cy += corners[i].y;
      const next = corners[(i + 1) & 3];
      perim += Math.hypot(next.x - corners[i].x, next.y - corners[i].y);
    }
    cx *= 0.25;
    cy *= 0.25;

    this.referencePerimeter = this.referencePerimeter === 0 ? perim : (1.0 - blend) * this.referencePerimeter + blend * perim;
    const scale = Math.max(1.0, perim);

    for (let i = 0; i < 4; ++i) {
      const nx = (corners[i].x - cx) / scale;
      const ny = (corners[i].y - cy) / scale;
      this.canonicalShape[i * 2] = (1.0 - blend) * this.canonicalShape[i * 2] + blend * nx;
      this.canonicalShape[i * 2 + 1] = (1.0 - blend) * this.canonicalShape[i * 2 + 1] + blend * ny;
    }
  }

  /**
   * Preserves the ArUco detector's canonical corner orientation (Corner 0 = top-left of binary code).
   * Unlike blind cyclic shifting, ArUco's dictionary decoder already determines which corner is
   * Corner 0. We only permit a cyclic shift if the detector had a high Hamming distance (ambiguous code)
   * AND the 0-shift cost is drastically worse than a shifted alignment.
   */
  private alignCorners(measured: IPoint[]): IPoint[] {
    // When Hamming distance is 0, ArUco's orientation is unambiguous—never corrupt Corner 0!
    if (this.lastHammingDistance === 0) {
      return measured;
    }

    const x = this.x;
    let cost0 = 0.0;
    for (let i = 0; i < 4; ++i) {
      const dx = measured[i].x - x[i * 2];
      const dy = measured[i].y - x[i * 2 + 1];
      cost0 += dx * dx + dy * dy;
    }

    let bestShift = 0;
    let minCost = cost0;

    for (let shift = 1; shift < 4; ++shift) {
      let cost = 0.0;
      for (let i = 0; i < 4; ++i) {
        const pt = measured[(i + shift) & 3];
        const dx = pt.x - x[i * 2];
        const dy = pt.y - x[i * 2 + 1];
        cost += dx * dx + dy * dy;
      }
      // Require shifted cost to be at least 4x smaller than unshifted to override orientation
      if (cost * 4.0 < minCost) {
        minCost = cost;
        bestShift = shift;
      }
    }

    if (bestShift === 0) {
      return measured;
    }

    return [measured[bestShift & 3], measured[(bestShift + 1) & 3], measured[(bestShift + 2) & 3], measured[(bestShift + 3) & 3]];
  }

  /**
   * Synchronizes the 16x16 row-major Float32Array covariance matrix P
   * with the 2x2 block scalars.
   */
  private syncDenseCovariance(): void {
    const P = this._P;
    P.fill(0);
    for (let i = 0; i < 8; ++i) {
      P[i * 16 + i] = this.p00;
      P[i * 16 + (i + 8)] = this.p01;
      P[(i + 8) * 16 + i] = this.p10;
      P[(i + 8) * 16 + (i + 8)] = this.p11;
    }
    this.pDirty = false;
  }

  /**
   * Extracts the 4 filtered corner positions in Detector-compatible IPoint[] format.
   */
  public getCorners(): IPoint[] {
    const x = this.x;
    return [
      { x: x[0], y: x[1] },
      { x: x[2], y: x[3] },
      { x: x[4], y: x[5] },
      { x: x[6], y: x[7] },
    ];
  }

  /**
   * Extracts the 4 corner velocity vectors in IPoint[] format.
   */
  public getVelocities(): IPoint[] {
    const x = this.x;
    return [
      { x: x[8], y: x[9] },
      { x: x[10], y: x[11] },
      { x: x[12], y: x[13] },
      { x: x[14], y: x[15] },
    ];
  }

  /**
   * Exports current state as a TrackedMarker instance (compatible with Marker).
   */
  public toMarker(isPredicted: boolean): TrackedMarker {
    return new TrackedMarker(
      this.id,
      this.getCorners(),
      this.lastHammingDistance,
      isPredicted,
      this.missedFrames,
      this.age,
      this.getVelocities(),
      this.confidence,
    );
  }
}

/**
 * Structure-of-Arrays (SoA) batch buffer for SIMD-friendly multi-marker operations.
 * Packs N markers into contiguous Float32Array registers.
 */
export interface ISoAMarkerBatch {
  count: number;
  ids: Int32Array;
  /** Contiguous [m0_x0, m0_y0, ..., m0_x3, m0_y3, m1_x0, ...] */
  positions: Float32Array;
  /** Contiguous [m0_vx0, m0_vy0, ..., m0_vx3, m0_vy3, m1_vx0, ...] */
  velocities: Float32Array;
}

/**
 * Stateful ArUco Marker Tracker combining:
 * 1. ID-indexed Rigid-Regularized Linear Kalman Filters
 * 2. 2-Level Pyramidal Lucas-Kanade (KLT) Optical Flow during motion-blurred missed frames
 * 3. ArUco Border Contrast & Geometry Verification to immediately prune ghost markers
 *
 * Exposes the exact same detection interface and Marker output structure as Detector.
 */
export class Tracker {
  public readonly config: Required<
    Pick<
      ITrackerConfig,
      | "dictionaryName"
      | "maxMissedFrames"
      | "processNoise"
      | "measurementNoise"
      | "coastVelocityDamping"
      | "fixedDt"
      | "enableOpticalFlow"
      | "opticalFlowWinRadius"
      | "minOpticalFlowConfidence"
      | "minBorderContrast"
    >
  > &
    ITrackerConfig;

  public detector: IMarkerDetector;
  private readonly filters: Map<number, MarkerKalmanFilter>;
  private lastFrameTimestamp: number | null = null;
  private streamConfig: any;
  private mjpeg: any;

  // Ping-pong grayscale image pyramids for KLT optical flow & border verification
  private prevPyramid: GrayscalePyramid = new GrayscalePyramid();
  private currPyramid: GrayscalePyramid = new GrayscalePyramid();
  private hasPrevPyramid: boolean = false;

  constructor(config: ITrackerConfig) {
    this.config = {
      ...config,
      maxMissedFrames: config.maxMissedFrames ?? 5,
      processNoise: config.processNoise ?? 8.0,
      measurementNoise: config.measurementNoise ?? 2.0,
      coastVelocityDamping: config.coastVelocityDamping ?? 0.82,
      fixedDt: config.fixedDt ?? 1.0,
      enableOpticalFlow: config.enableOpticalFlow ?? true,
      opticalFlowWinRadius: config.opticalFlowWinRadius ?? 5,
      minOpticalFlowConfidence: config.minOpticalFlowConfidence ?? 0.6,
      minBorderContrast: config.minBorderContrast ?? 18,
    };

    this.detector = config.detector ?? new Detector(config);
    this.filters = new Map<number, MarkerKalmanFilter>();
  }

  /**
   * Sets or swaps the active underlying detector engine (e.g. WASM SIMD or TypeScript).
   */
  public setDetector(detector: IMarkerDetector): void {
    this.detector = detector;
  }

  /**
   * Returns the count of actively tracked markers (both detected and coasting).
   */
  public get activeTracksCount(): number {
    return this.filters.size;
  }

  /**
   * Detects and tracks markers in an image frame.
   * Returns TrackedMarker[] which is a subclass of Marker[] (100% compatible with Detector.detect).
   *
   * When a tracked marker is missed by the ArUco detector due to motion blur, this method
   * uses KLT sparse optical flow + ArUco border contrast verification on `image` to accurately
   * track the blurred quad or immediately terminate the track if the marker left the scene.
   *
   * @param image IImage or ImageData-like object ({ width, height, data })
   * @param timestampMs Optional monotonic timestamp in milliseconds to compute variable dt
   */
  public detect(image: IImage | ImageData | { width: number; height: number; data: Uint8ClampedArray | Uint8Array }, timestampMs?: number): TrackedMarker[] {
    const rawMarkers = this.detector.detect(image as any);

    const hasImageBuffer =
      this.config.enableOpticalFlow &&
      image &&
      typeof image.width === "number" &&
      typeof image.height === "number" &&
      image.width > 16 &&
      image.height > 16 &&
      image.data &&
      image.data.length >= image.width * image.height;

    if (hasImageBuffer) {
      this.currPyramid.updateFromImage(image as { width: number; height: number; data: Uint8ClampedArray | Uint8Array });
    }

    const result = this.updateWithDetectionsInternal(rawMarkers as Marker[], timestampMs, hasImageBuffer);

    if (hasImageBuffer) {
      // Swap pyramids so currPyramid becomes prevPyramid for the next frame (zero allocation)
      const tmp = this.prevPyramid;
      this.prevPyramid = this.currPyramid;
      this.currPyramid = tmp;
      this.hasPrevPyramid = true;
    }

    return result;
  }

  /**
   * Alias matching Detector.detectImage(width, height, data).
   */
  public detectImage(width: number, height: number, data: Uint8ClampedArray | Uint8Array | number[] | ArrayBuffer, timestampMs?: number): TrackedMarker[] {
    return this.detect(
      {
        width,
        height,
        data: data instanceof Uint8ClampedArray ? data : new Uint8ClampedArray(data as any),
      },
      timestampMs,
    );
  }

  /**
   * Updates internal Kalman Filters given a list of raw Marker detections from a frame.
   * Useful if Detector.detect() is called externally (e.g. in a Web Worker).
   * Optionally accepts `image` to enable KLT optical flow & ghost verification on missed frames.
   */
  public updateWithDetections(
    rawMarkers: (Marker | IMarkerLike)[],
    timestampMs?: number,
    image?: IImage | ImageData | { width: number; height: number; data: Uint8ClampedArray | Uint8Array },
  ): TrackedMarker[] {
    const hasImageBuffer = Boolean(
      this.config.enableOpticalFlow &&
      image &&
      typeof image.width === "number" &&
      typeof image.height === "number" &&
      image.width > 16 &&
      image.height > 16 &&
      image.data &&
      image.data.length >= image.width * image.height,
    );

    if (hasImageBuffer && image) {
      this.currPyramid.updateFromImage(image as { width: number; height: number; data: Uint8ClampedArray | Uint8Array });
    }

    const result = this.updateWithDetectionsInternal(rawMarkers, timestampMs, hasImageBuffer);

    if (hasImageBuffer) {
      const tmp = this.prevPyramid;
      this.prevPyramid = this.currPyramid;
      this.currPyramid = tmp;
      this.hasPrevPyramid = true;
    }

    return result;
  }

  /**
   * Core tracking pipeline combining Kalman prediction, KLT optical flow recovery,
   * ArUco appearance verification, and track lifecycle management.
   */
  private updateWithDetectionsInternal(rawMarkers: (Marker | IMarkerLike)[], timestampMs: number | undefined, hasVisualFrame: boolean): TrackedMarker[] {
    // 1. Compute step time delta (dt)
    let dt = this.config.fixedDt;
    if (timestampMs !== undefined) {
      if (this.lastFrameTimestamp !== null && timestampMs > this.lastFrameTimestamp) {
        const elapsed = timestampMs - this.lastFrameTimestamp;
        // If elapsed time is huge (> 1 sec, e.g. tab backgrounded), clamp dt to fixedDt
        if (elapsed > 1000.0) {
          dt = this.config.fixedDt;
        } else {
          // Normalize milliseconds to ~60fps frame units (16.667ms = 1.0 unit)
          dt = elapsed / (1000.0 / 60.0);
        }
      }
      this.lastFrameTimestamp = timestampMs;
    }
    if (dt <= 0 || !Number.isFinite(dt)) {
      dt = this.config.fixedDt;
    }

    // 2. Group detections by marker ID (if multiple markers share an ID, pick lowest hamming distance)
    const detectedById = new Map<number, Marker | IMarkerLike>();
    for (let i = 0; i < rawMarkers.length; ++i) {
      const m = rawMarkers[i];
      if (!m || !m.corners || m.corners.length < 4) continue;
      const existing = detectedById.get(m.id);
      if (!existing || m.hammingDistance < existing.hammingDistance) {
        detectedById.set(m.id, m);
      }
    }

    const outputMarkers: TrackedMarker[] = [];
    const canRunKLT =
      hasVisualFrame && this.hasPrevPyramid && this.prevPyramid.width0 === this.currPyramid.width0 && this.prevPyramid.height0 === this.currPyramid.height0;

    // 3. Process existing tracked markers
    for (const [id, kf] of this.filters.entries()) {
      const detected = detectedById.get(id);
      kf.snapshotPreviousCorners();

      if (detected) {
        // Case A: Marker was detected by ArUco in this frame
        // If it was previously lost for several frames or jumped a huge distance (e.g. teleported),
        // check if we should hard-reset the filter state instead of creating a huge velocity spike.
        if (this.shouldResetFilterOnReacquire(kf, detected.corners)) {
          const cloned = detected.corners.map((pt) => ({ x: pt.x, y: pt.y }));
          const freshKf = new MarkerKalmanFilter(
            id,
            cloned,
            detected.hammingDistance,
            this.config.processNoise,
            this.config.measurementNoise,
            timestampMs ?? null,
          );
          this.filters.set(id, freshKf);
          outputMarkers.push(freshKf.toMarker(false));
        } else {
          kf.predict(dt, 1.0);
          kf.update(detected.corners, detected.hammingDistance, 1.0);
          kf.lastTimestamp = timestampMs ?? null;
          outputMarkers.push(kf.toMarker(false));
        }
        detectedById.delete(id);
      } else {
        // Case B: Detector missed this marker in this frame
        kf.missedFrames++;

        if (kf.missedFrames > this.config.maxMissedFrames) {
          this.filters.delete(id);
          continue;
        }

        // First run rigid-regularized Kalman prediction to get the warm-start prior
        kf.predict(dt, this.config.coastVelocityDamping);

        if (canRunKLT) {
          // Run 2-level KLT Optical Flow from prevPyramid -> currPyramid for each of the 4 corners
          const flowCorners: IPoint[] = new Array(4);
          let validCornersCount = 0;
          let sumNcc = 0.0;

          for (let c = 0; c < 4; ++c) {
            const prevX = kf.prevCorners[c * 2];
            const prevY = kf.prevCorners[c * 2 + 1];
            const predX = kf.x[c * 2];
            const predY = kf.x[c * 2 + 1];

            const res = trackCornerPyramidalKLT(this.prevPyramid, this.currPyramid, prevX, prevY, predX, predY, this.config.opticalFlowWinRadius);

            if (res.valid && res.ncc >= this.config.minOpticalFlowConfidence) {
              flowCorners[c] = { x: res.x, y: res.y };
              validCornersCount++;
              sumNcc += res.ncc;
            } else {
              flowCorners[c] = { x: predX, y: predY };
            }
          }

          // Require at least 3 of the 4 corners to track coherently via KLT
          if (validCornersCount >= 3) {
            const avgNcc = sumNcc / validCornersCount;
            kf.applyOpticalFlowUpdate(flowCorners, dt, avgNcc);
          } else {
            // Optical flow failed on >= 2 corners (marker likely occluded or left frame)
            this.filters.delete(id);
            continue;
          }

          // Verify that the image region under the tracked quad still has ArUco border contrast
          const currentCorners = kf.getCorners();
          const verification = verifyArUcoQuadAppearance(
            this.currPyramid.data0,
            this.currPyramid.width0,
            this.currPyramid.height0,
            currentCorners,
            this.config.minBorderContrast,
          );

          if (!verification.valid) {
            // Ghost marker rejected! Delete immediately instead of coasting a phantom marker.
            this.filters.delete(id);
            continue;
          }

          kf.confidence = Math.min(kf.confidence, verification.score);
          outputMarkers.push(kf.toMarker(true));
        } else if (hasVisualFrame) {
          // We have a current image (e.g. first frame), verify ArUco border contrast directly
          const verification = verifyArUcoQuadAppearance(
            this.currPyramid.data0,
            this.currPyramid.width0,
            this.currPyramid.height0,
            kf.getCorners(),
            this.config.minBorderContrast,
          );
          if (!verification.valid) {
            this.filters.delete(id);
            continue;
          }
          kf.confidence = verification.score * Math.pow(0.8, kf.missedFrames);
          outputMarkers.push(kf.toMarker(true));
        } else {
          // Pure numeric mode (updateWithDetections called without an image):
          // Only allow coasting if the marker was seen for at least 2 frames (has a valid velocity)
          // and geometric sanity holds.
          if (kf.age < 2 && kf.missedFrames > 1) {
            this.filters.delete(id);
            continue;
          }
          kf.confidence = Math.pow(0.75, kf.missedFrames);
          outputMarkers.push(kf.toMarker(true));
        }
      }
    }

    // 4. Initialize new Kalman Filters for newly discovered marker IDs
    for (const [id, newMarker] of detectedById.entries()) {
      const clonedCorners = newMarker.corners.map((pt) => ({ x: pt.x, y: pt.y }));
      const kf = new MarkerKalmanFilter(
        id,
        clonedCorners,
        newMarker.hammingDistance,
        this.config.processNoise,
        this.config.measurementNoise,
        timestampMs ?? null,
      );
      this.filters.set(id, kf);
      outputMarkers.push(kf.toMarker(false));
    }

    return outputMarkers;
  }

  /**
   * Determines whether a re-detected marker has jumped so far from its coasted prediction
   * that the filter should be cleanly re-initialized rather than slingshotting across the screen.
   */
  private shouldResetFilterOnReacquire(kf: MarkerKalmanFilter, measuredCorners: IPoint[]): boolean {
    let predCx = 0,
      predCy = 0,
      measCx = 0,
      measCy = 0,
      perim = 0;
    for (let i = 0; i < 4; ++i) {
      predCx += kf.x[i * 2];
      predCy += kf.x[i * 2 + 1];
      measCx += measuredCorners[i].x;
      measCy += measuredCorners[i].y;
      const next = measuredCorners[(i + 1) & 3];
      perim += Math.hypot(next.x - measuredCorners[i].x, next.y - measuredCorners[i].y);
    }
    predCx *= 0.25;
    predCy *= 0.25;
    measCx *= 0.25;
    measCy *= 0.25;

    const avgSide = Math.max(16.0, perim * 0.25);
    const centerDist = Math.hypot(measCx - predCx, measCy - predCy);

    // If the marker jumped by more than 2.5x its own width while coasting, reset cleanly
    return kf.missedFrames > 0 && centerDist > avgSide * 2.5;
  }

  /**
   * Initializes continuous RGBA byte stream tracking (mirrors Detector.detectStreamInit).
   */
  public detectStreamInit(width: number, height: number, callback: (image: IImage, markerList: TrackedMarker[]) => void): void {
    this.streamConfig = {
      width,
      height,
      imageSize: width * height * 4,
      index: 0,
      imageData: new Uint8ClampedArray(width * height * 4),
      callback: callback || function (_image: IImage, _markerList: TrackedMarker[]) {},
    };
  }

  /**
   * Feeds arbitrary-sized RGBA chunks into the stream tracker (mirrors Detector.detectStream).
   */
  public detectStream(data: ArrayLike<number>): void {
    for (let i = 0; i < data.length; i++) {
      this.streamConfig.imageData[this.streamConfig.index] = data[i];
      this.streamConfig.index = (this.streamConfig.index + 1) % this.streamConfig.imageSize;
      if (this.streamConfig.index === 0) {
        const image: IImage = {
          width: this.streamConfig.width,
          height: this.streamConfig.height,
          data: this.streamConfig.imageData,
        };
        const markerList = this.detect(image);
        this.streamConfig.callback(image, markerList);
      }
    }
  }

  /**
   * Initializes MJPEG stream tracking (mirrors Detector.detectMJPEGStreamInit).
   */
  public detectMJPEGStreamInit(
    width: number,
    height: number,
    callback: (image: IImage, markerList: TrackedMarker[]) => void,
    decoderFn: (jpegBytes: number[]) => ArrayLike<number>,
  ): void {
    this.mjpeg = {
      decoderFn,
      chunks: [] as number[][],
      SOI: [0xff, 0xd8],
      EOI: [0xff, 0xd9],
    };
    this.detectStreamInit(width, height, callback);
  }

  /**
   * Processes an MJPEG stream chunk (mirrors Detector.detectMJPEGStream).
   */
  public detectMJPEGStream(chunk: number[]): void {
    const SOI = this.mjpeg.SOI;
    const EOI = this.mjpeg.EOI;

    const eoiPos = chunk.findIndex(
      (element: number, index: number, array: number[]) => EOI[0] === element && array.length > index + 1 && EOI[1] === array[index + 1],
    );
    const soiPos = chunk.findIndex(
      (element: number, index: number, array: number[]) => SOI[0] === element && array.length > index + 1 && SOI[1] === array[index + 1],
    );

    if (eoiPos === -1) {
      this.mjpeg.chunks.push(chunk);
    } else {
      const part1 = chunk.slice(0, eoiPos + 2);
      if (part1.length) {
        this.mjpeg.chunks.push(part1);
      }
      if (this.mjpeg.chunks.length) {
        const jpegImage = this.mjpeg.chunks.flat();
        const rgba = this.mjpeg.decoderFn(jpegImage);
        this.detectStream(rgba);
      }
      this.mjpeg.chunks = [];
    }
    if (soiPos > -1) {
      this.mjpeg.chunks = [];
      this.mjpeg.chunks.push(chunk.slice(soiPos));
    }
  }

  /**
   * Returns the current tracked markers without running a new detection step.
   * Optionally predicts forward by `lookaheadDt` without mutating internal filter state.
   */
  public getTrackedMarkers(lookaheadDt: number = 0): TrackedMarker[] {
    const results: TrackedMarker[] = [];

    for (const kf of this.filters.values()) {
      if (lookaheadDt === 0) {
        results.push(kf.toMarker(kf.missedFrames > 0));
      } else {
        const corners = kf.getCorners();
        const velocities = kf.getVelocities();
        const rawExtrapolated: IPoint[] = [
          { x: corners[0].x + lookaheadDt * velocities[0].x, y: corners[0].y + lookaheadDt * velocities[0].y },
          { x: corners[1].x + lookaheadDt * velocities[1].x, y: corners[1].y + lookaheadDt * velocities[1].y },
          { x: corners[2].x + lookaheadDt * velocities[2].x, y: corners[2].y + lookaheadDt * velocities[2].y },
          { x: corners[3].x + lookaheadDt * velocities[3].x, y: corners[3].y + lookaheadDt * velocities[3].y },
        ];
        const extrapolatedCorners = regularizeQuadShape(corners, rawExtrapolated, 0.75);
        results.push(
          new TrackedMarker(kf.id, extrapolatedCorners, kf.lastHammingDistance, kf.missedFrames > 0, kf.missedFrames, kf.age, velocities, kf.confidence),
        );
      }
    }

    return results;
  }

  /**
   * Exports all active tracks in a flat Structure-of-Arrays (SoA) layout
   * for batch SIMD / WebGL / WASM consumption.
   */
  public getSoABatch(): ISoAMarkerBatch {
    const count = this.filters.size;
    const ids = new Int32Array(count);
    const positions = new Float32Array(count * 8);
    const velocities = new Float32Array(count * 8);

    let idx = 0;
    for (const [id, kf] of this.filters.entries()) {
      ids[idx] = id;
      const offset = idx * 8;
      positions.set(kf.x.subarray(0, 8), offset);
      velocities.set(kf.x.subarray(8, 16), offset);
      idx++;
    }

    return { count, ids, positions, velocities };
  }

  /**
   * Retrieves the underlying Kalman Filter for a specific marker ID, if currently tracked.
   */
  public getFilter(markerId: number): MarkerKalmanFilter | undefined {
    return this.filters.get(markerId);
  }

  /**
   * Clears all tracked marker states (or a specific marker ID if provided).
   */
  public reset(markerId?: number): void {
    if (markerId !== undefined) {
      this.filters.delete(markerId);
    } else {
      this.filters.clear();
      this.lastFrameTimestamp = null;
      this.hasPrevPyramid = false;
    }
  }
}
