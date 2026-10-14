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
   * adapts to sudden accelerations / direction changes. (Default: 12.0)
   */
  processNoise?: number;

  /**
   * Measurement noise variance (r) in pixels^2 for detector hits. (Default: 2.0)
   */
  measurementNoise?: number;

  /**
   * Measurement noise variance (r) in pixels^2 when tracked via KLT optical flow. (Default: 4.0)
   */
  kltMeasurementNoise?: number;

  /**
   * Velocity damping factor applied during pure coasting (dropped frames + KLT failure)
   * to prevent markers from flying off-screen. (Default: 0.85)
   */
  coastVelocityDamping?: number;

  /**
   * Fixed time step (dt) in normalized frames. If omitted, standard normalized
   * frame step dt = 1.0 is used unless timestampMs is passed to detect(). (Default: 1.0)
   */
  fixedDt?: number;

  /**
   * Whether to enable Lucas-Kanade optical flow tracking when the ArUco detector
   * drops a frame. (Default: true)
   */
  enableOpticalFlow?: boolean;

  /**
   * Optional pre-instantiated Detector. If not provided, Tracker creates its own.
   */
  detector?: IMarkerDetector;
}

/**
 * Extends the base Marker class from detector.ts so it is a drop-in replacement
 * while exposing tracking telemetry and recovery source.
 */
export class TrackedMarker extends Marker {
  /** True if the marker was not detected by ArUco dictionary decode in the current frame */
  public isPredicted: boolean;
  /** True if recovery was achieved using Lucas-Kanade optical flow tracking on the image */
  public isOpticalFlowTracked: boolean;
  /** Number of consecutive frames the detector missed this marker */
  public missedFrames: number;
  /** Total number of frames this marker track has been alive */
  public age: number;
  /** Estimated corner velocities [{x: vx0, y: vy0}, ...] in pixels per unit dt */
  public velocity: IPoint[];

  constructor(
    id: number,
    corners: IPoint[],
    hammingDistance: number,
    isPredicted: boolean = false,
    isOpticalFlowTracked: boolean = false,
    missedFrames: number = 0,
    age: number = 1,
    velocity: IPoint[] = [
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { x: 0, y: 0 },
    ],
  ) {
    super(id, corners, hammingDistance);
    this.isPredicted = isPredicted;
    this.isOpticalFlowTracked = isOpticalFlowTracked;
    this.missedFrames = missedFrames;
    this.age = age;
    this.velocity = velocity;
  }
}

/**
 * 2-Level Grayscale Pyramid for sub-millisecond Lucas-Kanade feature tracking.
 * Memory is pre-allocated and reused each frame.
 */
class GrayPyramid {
  public width: number = 0;
  public height: number = 0;
  public level0: Uint8Array = new Uint8Array(0);
  public level1: Uint8Array = new Uint8Array(0);
  public w1: number = 0;
  public h1: number = 0;

  public resize(width: number, height: number): void {
    if (this.width === width && this.height === height) return;
    this.width = width;
    this.height = height;
    this.w1 = width >> 1;
    this.h1 = height >> 1;
    this.level0 = new Uint8Array(width * height);
    this.level1 = new Uint8Array(this.w1 * this.h1);
  }

  public copyFrom(other: GrayPyramid): void {
    this.resize(other.width, other.height);
    this.level0.set(other.level0);
    this.level1.set(other.level1);
  }

  public updateFromImage(width: number, height: number, data: Uint8ClampedArray | Uint8Array): void {
    this.resize(width, height);
    const l0 = this.level0;
    const len = width * height;

    if (data.length >= len * 4) {
      // Fast luma conversion: (R * 77 + G * 150 + B * 29) >> 8
      for (let i = 0, j = 0; i < len; ++i, j += 4) {
        l0[i] = (data[j] * 77 + data[j + 1] * 150 + data[j + 2] * 29) >> 8;
      }
    } else {
      l0.set(data.subarray(0, len));
    }

    // Level 1: 2x2 box downsampling for tracking large displacements
    const w0 = this.width;
    const w1 = this.w1;
    const h1 = this.h1;
    const l1 = this.level1;
    for (let y = 0; y < h1; ++y) {
      const row0 = y * 2 * w0;
      const row1 = row0 + w0;
      const dstRow = y * w1;
      for (let x = 0; x < w1; ++x) {
        const x2 = x * 2;
        l1[dstRow + x] = (l0[row0 + x2] + l0[row0 + x2 + 1] + l0[row1 + x2] + l0[row1 + x2 + 1]) >> 2;
      }
    }
  }
}

/**
 * Bilinear sub-pixel sampling helper.
 */
function sampleBilinear(img: Uint8Array, w: number, x: number, y: number): number {
  const ix = x | 0;
  const iy = y | 0;
  const fx = x - ix;
  const fy = y - iy;
  const idx = iy * w + ix;

  const p00 = img[idx];
  const p10 = img[idx + 1];
  const p01 = img[idx + w];
  const p11 = img[idx + w + 1];

  return (1.0 - fy) * ((1.0 - fx) * p00 + fx * p10) + fy * ((1.0 - fx) * p01 + fx * p11);
}

/**
 * Pyramidal Lucas-Kanade corner tracking on high-contrast ArUco corner patches.
 * Rejects tracking on low image gradient structure, occlusion, or geometric inversion.
 */
function trackCornersKLT(prevPyr: GrayPyramid, currPyr: GrayPyramid, prevCorners: IPoint[], initialGuessCorners: IPoint[]): IPoint[] | null {
  const tracked: IPoint[] = [
    { x: initialGuessCorners[0].x, y: initialGuessCorners[0].y },
    { x: initialGuessCorners[1].x, y: initialGuessCorners[1].y },
    { x: initialGuessCorners[2].x, y: initialGuessCorners[2].y },
    { x: initialGuessCorners[3].x, y: initialGuessCorners[3].y },
  ];

  const winRadius = 4; // 9x9 patch window
  const patchSpan = 2 * winRadius + 1;
  const patchSize = patchSpan * patchSpan;
  const maxIterations = 8;

  const I0 = new Float32Array(patchSize);
  const Ix = new Float32Array(patchSize);
  const Iy = new Float32Array(patchSize);

  for (let c = 0; c < 4; ++c) {
    for (let level = 1; level >= 0; --level) {
      const scale = level === 1 ? 0.5 : 1.0;
      const prevImg = level === 1 ? prevPyr.level1 : prevPyr.level0;
      const currImg = level === 1 ? currPyr.level1 : currPyr.level0;
      const w = level === 1 ? prevPyr.w1 : prevPyr.width;
      const h = level === 1 ? prevPyr.h1 : prevPyr.height;

      const px = prevCorners[c].x * scale;
      const py = prevCorners[c].y * scale;
      let qx = tracked[c].x * scale;
      let qy = tracked[c].y * scale;

      if (px < winRadius + 1 || px >= w - winRadius - 2 || py < winRadius + 1 || py >= h - winRadius - 2) {
        return null;
      }

      let Gxx = 0;
      let Gxy = 0;
      let Gyy = 0;
      let idx = 0;

      // Extract template patch and spatial gradients from previous frame
      for (let wy = -winRadius; wy <= winRadius; ++wy) {
        for (let wx = -winRadius; wx <= winRadius; ++wx, ++idx) {
          const sx = px + wx;
          const sy = py + wy;
          I0[idx] = sampleBilinear(prevImg, w, sx, sy);
          const gx = 0.5 * (sampleBilinear(prevImg, w, sx + 1, sy) - sampleBilinear(prevImg, w, sx - 1, sy));
          const gy = 0.5 * (sampleBilinear(prevImg, w, sx, sy + 1) - sampleBilinear(prevImg, w, sx, sy - 1));
          Ix[idx] = gx;
          Iy[idx] = gy;
          Gxx += gx * gx;
          Gxy += gx * gy;
          Gyy += gy * gy;
        }
      }

      const det = Gxx * Gyy - Gxy * Gxy;
      if (det < 1e-3) return null;
      const invDet = 1.0 / det;

      let meanAbsError = 0;
      for (let iter = 0; iter < maxIterations; ++iter) {
        if (qx < winRadius + 1 || qx >= w - winRadius - 2 || qy < winRadius + 1 || qy >= h - winRadius - 2) {
          return null;
        }

        let bx = 0;
        let by = 0;
        let absErrSum = 0;
        idx = 0;

        for (let wy = -winRadius; wy <= winRadius; ++wy) {
          for (let wx = -winRadius; wx <= winRadius; ++wx, ++idx) {
            const it = sampleBilinear(currImg, w, qx + wx, qy + wy) - I0[idx];
            absErrSum += Math.abs(it);
            bx += Ix[idx] * it;
            by += Iy[idx] * it;
          }
        }

        meanAbsError = absErrSum / patchSize;
        const dx = invDet * (Gyy * bx - Gxy * by);
        const dy = invDet * (Gxx * by - Gxy * bx);

        qx -= dx;
        qy -= dy;

        if (dx * dx + dy * dy < 1e-3) break;
      }

      // Reject patch if visual residual indicates occlusion or major lighting shock
      if (level === 0 && meanAbsError > 38.0) {
        return null;
      }

      tracked[c].x = qx / scale;
      tracked[c].y = qy / scale;
    }
  }

  if (!isValidRigidQuadTransition(prevCorners, tracked)) {
    return null;
  }

  return tracked;
}

/**
 * Validates that tracked quad maintains positive convexity and consistent surface area.
 */
function isValidRigidQuadTransition(prev: IPoint[], curr: IPoint[]): boolean {
  let prevArea = 0;
  let currArea = 0;

  for (let i = 0; i < 4; ++i) {
    const j = (i + 1) & 3;
    const k = (i + 2) & 3;

    prevArea += prev[i].x * prev[j].y - prev[j].x * prev[i].y;
    currArea += curr[i].x * curr[j].y - curr[j].x * curr[i].y;

    const crossCurr = (curr[j].x - curr[i].x) * (curr[k].y - curr[j].y) - (curr[j].y - curr[i].y) * (curr[k].x - curr[j].x);
    const crossPrev = (prev[j].x - prev[i].x) * (prev[k].y - prev[j].y) - (prev[j].y - prev[i].y) * (prev[k].x - prev[j].x);

    // Convexity preserved and non-degenerate
    if (crossCurr * crossPrev <= 0) return false;
  }

  const areaRatio = Math.abs(currArea) / Math.max(1.0, Math.abs(prevArea));
  return areaRatio >= 0.55 && areaRatio <= 1.8;
}

/**
 * 16-State Linear Kalman Filter for ArUco Marker tracking with rigid motion preservation.
 *
 * State vector x (Float32Array(16)):
 *   Indices 0..7  : [x0, y0, x1, y1, x2, y2, x3, y3] (Corner positions)
 *   Indices 8..15 : [vx0, vy0, vx1, vy1, vx2, vy2, vx3, vy3] (Corner velocities)
 */
export class MarkerKalmanFilter {
  public readonly id: number;
  public readonly x: Float32Array;
  public readonly P: Float32Array;

  private p00: number;
  private p01: number;
  private p10: number;
  private p11: number;

  public lastHammingDistance: number;
  public missedFrames: number = 0;
  public age: number = 1;
  public lastTimestamp: number | null = null;

  private readonly q: number;
  private readonly defaultR: number;

  constructor(
    id: number,
    initialCorners: IPoint[],
    hammingDistance: number,
    processNoise: number = 12.0,
    measurementNoise: number = 2.0,
    timestamp: number | null = null,
  ) {
    this.id = id;
    this.lastHammingDistance = hammingDistance;
    this.q = processNoise;
    this.defaultR = measurementNoise;
    this.lastTimestamp = timestamp;

    this.x = new Float32Array(16);
    this.P = new Float32Array(256);

    for (let i = 0; i < 4; ++i) {
      this.x[i * 2] = initialCorners[i].x;
      this.x[i * 2 + 1] = initialCorners[i].y;
      this.x[8 + i * 2] = 0.0;
      this.x[8 + i * 2 + 1] = 0.0;
    }

    this.p00 = measurementNoise;
    this.p01 = 0.0;
    this.p10 = 0.0;
    this.p11 = processNoise * 8.0;

    this.syncDenseCovariance();
  }

  /**
   * Kalman Prediction Step.
   * During dropped frames (coasting), applies rigid translation to corners
   * so unobserved jitter never skews or deforms the marker polygon.
   */
  public predict(dt: number = 1.0, velocityDamping: number = 1.0, isCoasting: boolean = false): void {
    if (dt <= 0) dt = 1.0;
    const x = this.x;

    if (isCoasting) {
      // Rigid body translation: enforce shared centroid velocity
      let avgVx = 0;
      let avgVy = 0;
      for (let i = 0; i < 4; ++i) {
        avgVx += x[8 + i * 2];
        avgVy += x[8 + i * 2 + 1];
      }
      avgVx = avgVx * 0.25 * velocityDamping;
      avgVy = avgVy * 0.25 * velocityDamping;

      for (let i = 0; i < 4; ++i) {
        const px = i * 2;
        const py = px + 1;
        x[8 + px] = avgVx;
        x[8 + py] = avgVy;
        x[px] += dt * avgVx;
        x[py] += dt * avgVy;
      }
    } else {
      // Standard linear FMA update per coordinate
      for (let i = 0; i < 8; ++i) {
        const v = x[8 + i] * velocityDamping;
        x[8 + i] = v;
        x[i] += dt * v;
      }
    }

    // Covariance continuous acceleration model
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

    this.syncDenseCovariance();
  }

  /**
   * Kalman Measurement Update Step.
   * Uses measurement noise r appropriate to detector vs optical flow.
   */
  public update(measuredCorners: IPoint[], hammingDistance: number, measurementNoiseOverride?: number): void {
    const r = measurementNoiseOverride ?? this.defaultR;
    const invS = 1.0 / (this.p00 + r);
    const k0 = this.p00 * invS;
    const k1 = this.p10 * invS;

    const x = this.x;
    for (let i = 0; i < 4; ++i) {
      const idxX = i * 2;
      const idxY = idxX + 1;

      const yX = measuredCorners[i].x - x[idxX];
      const yY = measuredCorners[i].y - x[idxY];

      x[idxX] += k0 * yX;
      x[idxY] += k0 * yY;
      x[8 + idxX] += k1 * yX;
      x[8 + idxY] += k1 * yY;
    }

    const prevP00 = this.p00;
    const prevP01 = this.p01;
    const prevP10 = this.p10;
    const prevP11 = this.p11;

    const oneMinusK0 = 1.0 - k0;
    this.p00 = oneMinusK0 * prevP00;
    this.p01 = oneMinusK0 * prevP01;
    this.p10 = prevP10 - k1 * prevP00;
    this.p11 = prevP11 - k1 * prevP01;

    this.syncDenseCovariance();

    this.lastHammingDistance = hammingDistance;
    this.missedFrames = 0;
    this.age++;
  }

  private syncDenseCovariance(): void {
    const P = this.P;
    P.fill(0);
    for (let i = 0; i < 8; ++i) {
      P[i * 16 + i] = this.p00;
      P[i * 16 + (i + 8)] = this.p01;
      P[(i + 8) * 16 + i] = this.p10;
      P[(i + 8) * 16 + (i + 8)] = this.p11;
    }
  }

  public getCorners(): IPoint[] {
    const x = this.x;
    return [
      { x: x[0], y: x[1] },
      { x: x[2], y: x[3] },
      { x: x[4], y: x[5] },
      { x: x[6], y: x[7] },
    ];
  }

  public getVelocities(): IPoint[] {
    const x = this.x;
    return [
      { x: x[8], y: x[9] },
      { x: x[10], y: x[11] },
      { x: x[12], y: x[13] },
      { x: x[14], y: x[15] },
    ];
  }

  public toMarker(isPredicted: boolean, isOpticalFlowTracked: boolean = false): TrackedMarker {
    return new TrackedMarker(
      this.id,
      this.getCorners(),
      this.lastHammingDistance,
      isPredicted,
      isOpticalFlowTracked,
      this.missedFrames,
      this.age,
      this.getVelocities(),
    );
  }
}

export interface ISoAMarkerBatch {
  count: number;
  ids: Int32Array;
  positions: Float32Array;
  velocities: Float32Array;
}

/**
 * Stateful ArUco Marker Tracker combining ID-indexed Kalman Filtering
 * with Pyramidal Lucas-Kanade Optical Flow and rigid geometry preservation.
 */
export class Tracker {
  public readonly config: Required<
    Pick<
      ITrackerConfig,
      | "dictionaryName"
      | "maxMissedFrames"
      | "processNoise"
      | "measurementNoise"
      | "kltMeasurementNoise"
      | "coastVelocityDamping"
      | "fixedDt"
      | "enableOpticalFlow"
    >
  > &
    ITrackerConfig;

  public detector: IMarkerDetector;
  private readonly filters: Map<number, MarkerKalmanFilter>;
  private lastFrameTimestamp: number | null = null;
  private prevPyr: GrayPyramid = new GrayPyramid();
  private currPyr: GrayPyramid = new GrayPyramid();
  private hasPrevPyr: boolean = false;
  private streamConfig: any;
  private mjpeg: any;

  constructor(config: ITrackerConfig) {
    this.config = {
      ...config,
      maxMissedFrames: config.maxMissedFrames ?? 5,
      processNoise: config.processNoise ?? 12.0,
      measurementNoise: config.measurementNoise ?? 2.0,
      kltMeasurementNoise: config.kltMeasurementNoise ?? 4.0,
      coastVelocityDamping: config.coastVelocityDamping ?? 0.85,
      fixedDt: config.fixedDt ?? 1.0,
      enableOpticalFlow: config.enableOpticalFlow ?? true,
    };

    this.detector = config.detector ?? new Detector(config);
    this.filters = new Map<number, MarkerKalmanFilter>();
  }

  public setDetector(detector: IMarkerDetector): void {
    this.detector = detector;
  }

  public get activeTracksCount(): number {
    return this.filters.size;
  }

  public detect(image: IImage | ImageData | { width: number; height: number; data: Uint8ClampedArray | Uint8Array }, timestampMs?: number): TrackedMarker[] {
    const rawMarkers = this.detector.detect(image as any);

    if (this.config.enableOpticalFlow) {
      this.currPyr.updateFromImage(image.width, image.height, image.data);
    }

    const tracked = this.updateWithDetections(rawMarkers as Marker[], timestampMs, this.config.enableOpticalFlow);

    if (this.config.enableOpticalFlow) {
      this.prevPyr.copyFrom(this.currPyr);
      this.hasPrevPyr = true;
    }

    return tracked;
  }

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
   * Updates state with detections. If an existing track is missed,
   * performs Pyramidal KLT tracking on image corners before falling back to rigid coasting.
   */
  public updateWithDetections(rawMarkers: (Marker | IMarkerLike)[], timestampMs?: number, allowOpticalFlowFallback: boolean = false): TrackedMarker[] {
    let dt = this.config.fixedDt;
    if (timestampMs !== undefined) {
      if (this.lastFrameTimestamp !== null && timestampMs > this.lastFrameTimestamp) {
        const elapsed = timestampMs - this.lastFrameTimestamp;
        if (elapsed > 1000.0) {
          dt = this.config.fixedDt;
        } else {
          dt = elapsed / (1000.0 / 60.0);
        }
      }
      this.lastFrameTimestamp = timestampMs;
    }
    if (dt <= 0 || !Number.isFinite(dt)) {
      dt = this.config.fixedDt;
    }

    const detectedById = new Map<number, Marker | IMarkerLike>();
    for (let i = 0; i < rawMarkers.length; ++i) {
      const m = rawMarkers[i];
      const existing = detectedById.get(m.id);
      if (!existing || m.hammingDistance < existing.hammingDistance) {
        detectedById.set(m.id, m);
      }
    }

    const outputMarkers: TrackedMarker[] = [];

    for (const [id, kf] of this.filters.entries()) {
      const detected = detectedById.get(id);

      if (detected) {
        // High confidence direct detector match
        kf.predict(dt, 1.0, false);
        kf.update(detected.corners, detected.hammingDistance, this.config.measurementNoise);
        kf.lastTimestamp = timestampMs ?? null;

        outputMarkers.push(kf.toMarker(false, false));
        detectedById.delete(id);
      } else {
        // Detector dropped this frame: Attempt Pyramidal KLT corner tracking
        let kltSucceeded = false;
        let trackedCorners: IPoint[] | null = null;

        if (allowOpticalFlowFallback && this.hasPrevPyr && kf.missedFrames < this.config.maxMissedFrames) {
          const prevCorners = kf.getCorners();
          // Extrapolate motion guess to seed Lucas-Kanade search
          const velocities = kf.getVelocities();
          const initialGuess: IPoint[] = [
            { x: prevCorners[0].x + dt * velocities[0].x, y: prevCorners[0].y + dt * velocities[0].y },
            { x: prevCorners[1].x + dt * velocities[1].x, y: prevCorners[1].y + dt * velocities[1].y },
            { x: prevCorners[2].x + dt * velocities[2].x, y: prevCorners[2].y + dt * velocities[2].y },
            { x: prevCorners[3].x + dt * velocities[3].x, y: prevCorners[3].y + dt * velocities[3].y },
          ];

          trackedCorners = trackCornersKLT(this.prevPyr, this.currPyr, prevCorners, initialGuess);
          if (trackedCorners !== null) {
            kltSucceeded = true;
          }
        }

        if (kltSucceeded && trackedCorners) {
          // Visual recovery succeeded: update Kalman state with slightly higher noise allowance
          kf.predict(dt, 0.98, false);
          kf.update(trackedCorners, kf.lastHammingDistance, this.config.kltMeasurementNoise);
          kf.missedFrames++;
          kf.lastTimestamp = timestampMs ?? null;

          outputMarkers.push(kf.toMarker(true, true));
        } else {
          // Visual tracking failed: fall back to rigid centroid coasting
          kf.missedFrames++;
          if (kf.missedFrames <= this.config.maxMissedFrames) {
            kf.predict(dt, this.config.coastVelocityDamping, true);
            outputMarkers.push(kf.toMarker(true, false));
          } else {
            this.filters.delete(id);
          }
        }
      }
    }

    // Initialize new filters for novel markers
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
      outputMarkers.push(kf.toMarker(false, false));
    }

    return outputMarkers;
  }

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

  public getTrackedMarkers(lookaheadDt: number = 0): TrackedMarker[] {
    const results: TrackedMarker[] = [];
    for (const kf of this.filters.values()) {
      if (lookaheadDt === 0) {
        results.push(kf.toMarker(kf.missedFrames > 0, false));
      } else {
        const corners = kf.getCorners();
        const velocities = kf.getVelocities();
        const extrapolatedCorners: IPoint[] = [
          { x: corners[0].x + lookaheadDt * velocities[0].x, y: corners[0].y + lookaheadDt * velocities[0].y },
          { x: corners[1].x + lookaheadDt * velocities[1].x, y: corners[1].y + lookaheadDt * velocities[1].y },
          { x: corners[2].x + lookaheadDt * velocities[2].x, y: corners[2].y + lookaheadDt * velocities[2].y },
          { x: corners[3].x + lookaheadDt * velocities[3].x, y: corners[3].y + lookaheadDt * velocities[3].y },
        ];
        results.push(new TrackedMarker(kf.id, extrapolatedCorners, kf.lastHammingDistance, kf.missedFrames > 0, false, kf.missedFrames, kf.age, velocities));
      }
    }
    return results;
  }

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

  public getFilter(markerId: number): MarkerKalmanFilter | undefined {
    return this.filters.get(markerId);
  }

  public reset(markerId?: number): void {
    if (markerId !== undefined) {
      this.filters.delete(markerId);
    } else {
      this.filters.clear();
      this.lastFrameTimestamp = null;
      this.hasPrevPyr = false;
    }
  }
}
