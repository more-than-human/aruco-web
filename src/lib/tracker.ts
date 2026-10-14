import { Detector, Marker, type IConfig, type IImage, type IPoint } from "./aruco";

export type ImageBuffer = IImage | ImageData | { width: number; height: number; data: Uint8ClampedArray | Uint8Array };

export interface IMarkerLike {
  id: number;
  corners: IPoint[];
  hammingDistance: number;
}

export interface IMarkerDetector {
  detect(image: ImageBuffer): (Marker | IMarkerLike)[];
}

export interface ITrackerConfig extends IConfig {
  /** Larger values keep occluded or blurred tracks alive longer, but increase ghost persistence. (Default: 5) */
  maxMissedFrames?: number;

  /** Higher values adapt faster to sudden accelerations, but allow more corner jitter. (Default: 8.0) */
  processNoise?: number;

  /** Higher values produce smoother trajectories by trusting the motion model over raw detections, but increase lag. (Default: 2.0) */
  measurementNoise?: number;

  /** Closer to 1.0 maintains momentum longer during missed frames; lower values brake coasting markers faster. (Default: 0.82) */
  coastVelocityDamping?: number;

  /** Larger values scale predicted velocity displacements further per step when timestamps are omitted. (Default: 1.0) */
  fixedDt?: number;

  /** Enables KLT optical flow and border verification during missed frames. (Default: true) */
  enableOpticalFlow?: boolean;

  /** Larger radii capture broader texture and larger displacements, at higher CPU cost. (Default: 5) */
  opticalFlowWinRadius?: number;

  /** Higher values reject ambiguous optical flow patches more strictly, reducing drift on occlusions. (Default: 0.60) */
  minOpticalFlowConfidence?: number;

  /** Higher values require stronger quiet-zone-to-border contrast to keep coasting markers alive. (Default: 18) */
  minBorderContrast?: number;

  detector?: IMarkerDetector;
}

export interface ISoAMarkerBatch {
  count: number;
  ids: Int32Array;
  positions: Float32Array;
  velocities: Float32Array;
}

const DEFAULT_CONFIG = {
  maxMissedFrames: 5,
  processNoise: 8.0,
  measurementNoise: 2.0,
  coastVelocityDamping: 0.82,
  fixedDt: 1.0,
  enableOpticalFlow: true,
  opticalFlowWinRadius: 5,
  minOpticalFlowConfidence: 0.6,
  minBorderContrast: 18,
} as const;

const FRAME_DURATION_60FPS_MS = 1000.0 / 60.0;
const MAX_VALID_FRAME_GAP_MS = 1000.0;
const REACQUIRE_JUMP_RATIO = 2.5; // Larger values tolerate bigger spatial jumps before resetting velocity state
const EXTRAPOLATION_RIGIDITY = 0.75; // Closer to 1.0 enforces rigid similarity transform over independent corner motion

const BORDER_SAMPLE_PAIRS: ReadonlyArray<readonly [number, number, number, number]> = [0.25, 0.5, 0.75].flatMap((t) => {
  const dOut = -0.1; // More negative samples further outside the marker edge (quiet zone)
  const dIn = 0.09; // Larger positive samples deeper inside the black border cell
  return [
    [t, dOut, t, dIn],
    [1.0 - dOut, t, 1.0 - dIn, t],
    [t, 1.0 - dOut, t, 1.0 - dIn],
    [dOut, t, dIn, t],
  ] as const;
});

export class TrackedMarker extends Marker {
  public isPredicted: boolean;
  public missedFrames: number;
  public age: number;
  public velocity: IPoint[];
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

export class Tracker {
  public readonly config: Required<Pick<ITrackerConfig, keyof typeof DEFAULT_CONFIG | "dictionaryName">> & ITrackerConfig;
  public detector: IMarkerDetector;

  private readonly filters = new Map<number, MarkerKalmanFilter>();
  private lastFrameTimestamp: number | null = null;
  private streamConfig: any;
  private mjpeg: any;

  private prevPyramid = new GrayscalePyramid();
  private currPyramid = new GrayscalePyramid();
  private hasPrevPyramid = false;

  constructor(config: ITrackerConfig) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.detector = config.detector ?? new Detector(config);
  }

  public setDetector(detector: IMarkerDetector): void {
    this.detector = detector;
  }

  public get activeTracksCount(): number {
    return this.filters.size;
  }

  public detect(image: ImageBuffer, timestampMs?: number): TrackedMarker[] {
    const rawMarkers = this.detector.detect(image);
    return this.updateWithDetections(rawMarkers, timestampMs, image);
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

  public updateWithDetections(rawMarkers: (Marker | IMarkerLike)[], timestampMs?: number, image?: ImageBuffer): TrackedMarker[] {
    const hasVisualFrame = Boolean(
      this.config.enableOpticalFlow &&
      image &&
      typeof image.width === "number" &&
      typeof image.height === "number" &&
      image.width > 16 &&
      image.height > 16 &&
      image.data &&
      image.data.length >= image.width * image.height,
    );

    if (hasVisualFrame && image) {
      this.currPyramid.updateFromImage(image);
    }

    const result = this.stepFilters(rawMarkers, this.computeDt(timestampMs), timestampMs ?? null, hasVisualFrame);

    if (hasVisualFrame) {
      [this.prevPyramid, this.currPyramid] = [this.currPyramid, this.prevPyramid];
      this.hasPrevPyramid = true;
    }

    return result;
  }

  public getTrackedMarkers(lookaheadDt: number = 0): TrackedMarker[] {
    const results: TrackedMarker[] = [];

    for (const kf of this.filters.values()) {
      if (lookaheadDt === 0) {
        results.push(kf.toMarker(kf.missedFrames > 0));
        continue;
      }

      const corners = kf.getCorners();
      const velocities = kf.getVelocities();
      const rawExtrapolated = corners.map((c, i) => ({
        x: c.x + lookaheadDt * velocities[i].x,
        y: c.y + lookaheadDt * velocities[i].y,
      }));
      const extrapolatedCorners = regularizeQuadShape(corners, rawExtrapolated, EXTRAPOLATION_RIGIDITY);

      results.push(
        new TrackedMarker(kf.id, extrapolatedCorners, kf.lastHammingDistance, kf.missedFrames > 0, kf.missedFrames, kf.age, velocities, kf.confidence),
      );
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
      this.hasPrevPyramid = false;
    }
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
        this.streamConfig.callback(image, this.detect(image));
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
    const { SOI, EOI } = this.mjpeg;
    const findMarker = (sig: number[]) => chunk.findIndex((el, i, arr) => sig[0] === el && arr.length > i + 1 && sig[1] === arr[i + 1]);

    const eoiPos = findMarker(EOI);
    const soiPos = findMarker(SOI);

    if (eoiPos === -1) {
      this.mjpeg.chunks.push(chunk);
    } else {
      const part1 = chunk.slice(0, eoiPos + 2);
      if (part1.length) this.mjpeg.chunks.push(part1);
      if (this.mjpeg.chunks.length) {
        this.detectStream(this.mjpeg.decoderFn(this.mjpeg.chunks.flat()));
      }
      this.mjpeg.chunks = [];
    }
    if (soiPos > -1) {
      this.mjpeg.chunks = [chunk.slice(soiPos)];
    }
  }

  private computeDt(timestampMs?: number): number {
    let dt = this.config.fixedDt;
    if (timestampMs !== undefined) {
      if (this.lastFrameTimestamp !== null && timestampMs > this.lastFrameTimestamp) {
        const elapsed = timestampMs - this.lastFrameTimestamp;
        dt = elapsed > MAX_VALID_FRAME_GAP_MS ? this.config.fixedDt : elapsed / FRAME_DURATION_60FPS_MS;
      }
      this.lastFrameTimestamp = timestampMs;
    }
    return dt > 0 && Number.isFinite(dt) ? dt : this.config.fixedDt;
  }

  private stepFilters(rawMarkers: (Marker | IMarkerLike)[], dt: number, timestamp: number | null, hasVisualFrame: boolean): TrackedMarker[] {
    const detectedById = new Map<number, Marker | IMarkerLike>();
    for (const m of rawMarkers) {
      if (!m?.corners || m.corners.length < 4) continue;
      const existing = detectedById.get(m.id);
      if (!existing || m.hammingDistance < existing.hammingDistance) {
        detectedById.set(m.id, m);
      }
    }

    const canRunKLT =
      hasVisualFrame && this.hasPrevPyramid && this.prevPyramid.width0 === this.currPyramid.width0 && this.prevPyramid.height0 === this.currPyramid.height0;

    const outputMarkers: TrackedMarker[] = [];

    for (const [id, kf] of this.filters.entries()) {
      const detected = detectedById.get(id);
      kf.snapshotPreviousCorners();

      if (detected) {
        detectedById.delete(id);
        if (this.shouldResetFilterOnReacquire(kf, detected.corners)) {
          outputMarkers.push(this.createFilter(id, detected, timestamp).toMarker(false));
        } else {
          kf.predict(dt, 1.0);
          kf.update(detected.corners, detected.hammingDistance, 1.0);
          kf.lastTimestamp = timestamp;
          outputMarkers.push(kf.toMarker(false));
        }
      } else if (this.coastFilter(kf, dt, canRunKLT, hasVisualFrame)) {
        outputMarkers.push(kf.toMarker(true));
      } else {
        this.filters.delete(id);
      }
    }

    for (const [id, newMarker] of detectedById.entries()) {
      outputMarkers.push(this.createFilter(id, newMarker, timestamp).toMarker(false));
    }

    return outputMarkers;
  }

  private createFilter(id: number, marker: Marker | IMarkerLike, timestamp: number | null): MarkerKalmanFilter {
    const clonedCorners = marker.corners.map((pt) => ({ x: pt.x, y: pt.y }));
    const kf = new MarkerKalmanFilter(id, clonedCorners, marker.hammingDistance, this.config.processNoise, this.config.measurementNoise, timestamp);
    this.filters.set(id, kf);
    return kf;
  }

  private coastFilter(kf: MarkerKalmanFilter, dt: number, canRunKLT: boolean, hasVisualFrame: boolean): boolean {
    if (++kf.missedFrames > this.config.maxMissedFrames) return false;

    kf.predict(dt, this.config.coastVelocityDamping);

    if (canRunKLT) {
      const flowCorners: IPoint[] = new Array(4);
      let validCornersCount = 0;
      let sumNcc = 0.0;

      for (let c = 0; c < 4; ++c) {
        const idxX = c * 2;
        const idxY = idxX + 1;
        const res = trackCornerPyramidalKLT(
          this.prevPyramid,
          this.currPyramid,
          kf.prevCorners[idxX],
          kf.prevCorners[idxY],
          kf.x[idxX],
          kf.x[idxY],
          this.config.opticalFlowWinRadius,
        );

        if (res.valid && res.ncc >= this.config.minOpticalFlowConfidence) {
          flowCorners[c] = { x: res.x, y: res.y };
          validCornersCount++;
          sumNcc += res.ncc;
        } else {
          flowCorners[c] = { x: kf.x[idxX], y: kf.x[idxY] };
        }
      }

      if (validCornersCount < 3) return false;
      kf.applyOpticalFlowUpdate(flowCorners, dt, sumNcc / validCornersCount);
    }

    if (hasVisualFrame) {
      const verification = verifyArUcoQuadAppearance(
        this.currPyramid.data0,
        this.currPyramid.width0,
        this.currPyramid.height0,
        kf.getCorners(),
        this.config.minBorderContrast,
      );
      if (!verification.valid) return false;

      kf.confidence = canRunKLT ? Math.min(kf.confidence, verification.score) : verification.score * Math.pow(0.8, kf.missedFrames);
      return true;
    }

    if (kf.age < 2 && kf.missedFrames > 1) return false;
    kf.confidence = Math.pow(0.75, kf.missedFrames);
    return true;
  }

  private shouldResetFilterOnReacquire(kf: MarkerKalmanFilter, measuredCorners: IPoint[]): boolean {
    if (kf.missedFrames === 0) return false;

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

    const avgSide = Math.max(16.0, perim * 0.25);
    const centerDist = Math.hypot((measCx - predCx) * 0.25, (measCy - predCy) * 0.25);
    return centerDist > avgSide * REACQUIRE_JUMP_RATIO;
  }
}

/**
 * State vector x (Float32Array(16)):
 *   0..7:  [x0, y0, x1, y1, x2, y2, x3, y3]
 *   8..15: [vx0, vy0, vx1, vy1, vx2, vy2, vx3, vy3]
 */
export class MarkerKalmanFilter {
  public readonly id: number;
  public readonly x = new Float32Array(16);
  public readonly prevCorners = new Float32Array(8);

  public lastHammingDistance: number;
  public missedFrames = 0;
  public age = 1;
  public confidence = 1.0;
  public lastTimestamp: number | null;

  private readonly _P = new Float32Array(256);
  private readonly canonicalShape = new Float32Array(8);
  private referencePerimeter = 0;
  private pDirty = true;

  // Isotropic 2x2 block scalars representing the 16x16 state covariance
  private p00: number;
  private p01 = 0.0;
  private p10 = 0.0;
  private p11: number;

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

    for (let i = 0; i < 4; ++i) {
      const idxX = i * 2;
      const idxY = idxX + 1;
      this.x[idxX] = this.prevCorners[idxX] = initialCorners[i].x;
      this.x[idxY] = this.prevCorners[idxY] = initialCorners[i].y;
    }

    this.updateCanonicalShape(initialCorners, 1.0);
    this.p00 = measurementNoise;
    this.p11 = processNoise * 2.5;
  }

  public get P(): Float32Array {
    if (this.pDirty) {
      this._P.fill(0);
      for (let i = 0; i < 8; ++i) {
        this._P[i * 16 + i] = this.p00;
        this._P[i * 16 + (i + 8)] = this.p01;
        this._P[(i + 8) * 16 + i] = this.p10;
        this._P[(i + 8) * 16 + (i + 8)] = this.p11;
      }
      this.pDirty = false;
    }
    return this._P;
  }

  public snapshotPreviousCorners(): void {
    this.prevCorners.set(this.x.subarray(0, 8));
  }

  public predict(dt: number = 1.0, velocityDamping: number = 1.0): void {
    if (dt <= 0) dt = 1.0;

    const x = this.x;
    const vcx = 0.25 * (x[8] + x[10] + x[12] + x[14]) * velocityDamping;
    const vcy = 0.25 * (x[9] + x[11] + x[13] + x[15]) * velocityDamping;

    // Lower multiplier damps individual corner divergence more aggressively relative to centroid motion
    const nonRigidDamping = this.missedFrames > 0 ? velocityDamping * 0.35 : velocityDamping * 0.85;
    const prevPts = this.getCorners();
    const rawPredicted: IPoint[] = new Array(4);

    for (let i = 0; i < 4; ++i) {
      const idxX = i * 2;
      const idxY = idxX + 1;
      const vx = vcx + (x[8 + idxX] * velocityDamping - vcx) * nonRigidDamping;
      const vy = vcy + (x[8 + idxY] * velocityDamping - vcy) * nonRigidDamping;

      x[8 + idxX] = vx;
      x[8 + idxY] = vy;
      rawPredicted[i] = { x: x[idxX] + dt * vx, y: x[idxY] + dt * vy };
    }

    const regPts = regularizeQuadShape(prevPts, rawPredicted, this.missedFrames > 0 ? 0.85 : 0.25);
    for (let i = 0; i < 4; ++i) {
      x[i * 2] = regPts[i].x;
      x[i * 2 + 1] = regPts[i].y;
    }

    const dt2 = dt * dt;
    const q00 = 0.25 * dt2 * dt2 * this.q;
    const q01 = 0.5 * dt2 * dt * this.q;
    const q11 = dt2 * this.q;

    const { p00, p01, p10, p11 } = this;
    this.p00 = p00 + dt * (p10 + p01) + dt2 * p11 + q00;
    this.p01 = p01 + dt * p11 + q01;
    this.p10 = p10 + dt * p11 + q01;
    this.p11 = p11 + q11;
    this.pDirty = true;
  }

  public update(measuredCorners: IPoint[], hammingDistance: number, measurementScale: number = 1.0): void {
    const aligned = this.alignCorners(measuredCorners);
    const effectiveR = this.r * measurementScale + (hammingDistance > 0 ? hammingDistance * 1.5 : 0);
    const invS = 1.0 / (this.p00 + effectiveR);
    const k0 = this.p00 * invS;
    const k1 = this.p10 * invS;

    const x = this.x;
    let meanDx = 0.0;
    let meanDy = 0.0;
    for (let i = 0; i < 4; ++i) {
      meanDx += aligned[i].x - x[i * 2];
      meanDy += aligned[i].y - x[i * 2 + 1];
    }
    meanDx *= 0.25;
    meanDy *= 0.25;

    // Smaller threshold clips single-corner non-rigid velocity spikes more aggressively
    const huberDelta = Math.max(4.0, this.referencePerimeter * 0.04);

    for (let i = 0; i < 4; ++i) {
      const idxX = i * 2;
      const idxY = idxX + 1;
      const yX = aligned[i].x - x[idxX];
      const yY = aligned[i].y - x[idxY];

      x[idxX] += k0 * yX;
      x[idxY] += k0 * yY;

      const nonRigidX = yX - meanDx;
      const nonRigidY = yY - meanDy;
      const nonRigidNorm = Math.hypot(nonRigidX, nonRigidY);
      const huberWeight = nonRigidNorm > huberDelta ? huberDelta / nonRigidNorm : 1.0;

      x[8 + idxX] += k1 * (meanDx + huberWeight * nonRigidX);
      x[8 + idxY] += k1 * (meanDy + huberWeight * nonRigidY);
    }

    const { p00, p01, p10, p11 } = this;
    const oneMinusK0 = 1.0 - k0;
    this.p00 = oneMinusK0 * p00;
    this.p01 = oneMinusK0 * p01;
    this.p10 = p10 - k1 * p00;
    this.p11 = p11 - k1 * p01;
    this.pDirty = true;

    if (hammingDistance === 0 && measurementScale <= 1.0) {
      this.updateCanonicalShape(this.getCorners(), 0.2);
    }

    this.lastHammingDistance = hammingDistance;
    this.missedFrames = 0;
    this.confidence = 1.0;
    this.age++;
  }

  public applyOpticalFlowUpdate(flowCorners: IPoint[], dt: number, flowConfidence: number): void {
    const prevPts: IPoint[] = [
      { x: this.prevCorners[0], y: this.prevCorners[1] },
      { x: this.prevCorners[2], y: this.prevCorners[3] },
      { x: this.prevCorners[4], y: this.prevCorners[5] },
      { x: this.prevCorners[6], y: this.prevCorners[7] },
    ];

    const regularized = regularizeQuadShape(prevPts, flowCorners, 0.55);
    const invDt = dt > 1e-3 ? 1.0 / dt : 1.0;
    const alphaPos = Math.min(0.92, Math.max(0.5, flowConfidence));
    const alphaVel = 0.65;

    const x = this.x;
    for (let i = 0; i < 4; ++i) {
      const idxX = i * 2;
      const idxY = idxX + 1;
      const newX = alphaPos * regularized[i].x + (1.0 - alphaPos) * x[idxX];
      const newY = alphaPos * regularized[i].y + (1.0 - alphaPos) * x[idxY];
      const instVx = (newX - this.prevCorners[idxX]) * invDt;
      const instVy = (newY - this.prevCorners[idxY]) * invDt;

      x[idxX] = newX;
      x[idxY] = newY;
      x[8 + idxX] = alphaVel * instVx + (1.0 - alphaVel) * x[8 + idxX];
      x[8 + idxY] = alphaVel * instVy + (1.0 - alphaVel) * x[8 + idxY];
    }

    this.confidence = flowConfidence;
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
      const idxX = i * 2;
      const idxY = idxX + 1;
      this.canonicalShape[idxX] = (1.0 - blend) * this.canonicalShape[idxX] + blend * ((corners[i].x - cx) / scale);
      this.canonicalShape[idxY] = (1.0 - blend) * this.canonicalShape[idxY] + blend * ((corners[i].y - cy) / scale);
    }
  }

  private alignCorners(measured: IPoint[]): IPoint[] {
    if (this.lastHammingDistance === 0) return measured;

    const shiftCost = (shift: number) => {
      let cost = 0.0;
      for (let i = 0; i < 4; ++i) {
        const pt = measured[(i + shift) & 3];
        const dx = pt.x - this.x[i * 2];
        const dy = pt.y - this.x[i * 2 + 1];
        cost += dx * dx + dy * dy;
      }
      return cost;
    };

    let bestShift = 0;
    let minCost = shiftCost(0);

    for (let shift = 1; shift < 4; ++shift) {
      const cost = shiftCost(shift);
      if (cost * 4.0 < minCost) {
        minCost = cost;
        bestShift = shift;
      }
    }

    return bestShift === 0 ? measured : [measured[bestShift & 3], measured[(bestShift + 1) & 3], measured[(bestShift + 2) & 3], measured[(bestShift + 3) & 3]];
  }
}

class GrayscalePyramid {
  public width0 = 0;
  public height0 = 0;
  public data0 = new Uint8Array(0);

  public width1 = 0;
  public height1 = 0;
  public data1 = new Uint8Array(0);

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
    if (src.length === numPixels0 * 4) {
      for (let i = 0, j = 0; i < numPixels0; ++i, j += 4) {
        dst0[i] = (77 * src[j] + 150 * src[j + 1] + 29 * src[j + 2]) >> 8;
      }
    } else if (src.length >= numPixels0) {
      dst0.set(src.subarray(0, numPixels0));
    }

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
}

function trackCornerPyramidalKLT(
  prevPyr: GrayscalePyramid,
  currPyr: GrayscalePyramid,
  prevX: number,
  prevY: number,
  predX: number,
  predY: number,
  winRadius: number,
): { x: number; y: number; ncc: number; valid: boolean } {
  const coarseRes = trackPointSingleLevel(
    prevPyr.data1,
    currPyr.data1,
    prevPyr.width1,
    prevPyr.height1,
    prevX * 0.5,
    prevY * 0.5,
    predX * 0.5,
    predY * 0.5,
    Math.max(3, winRadius - 1),
    6,
  );

  const useCoarse = coarseRes.valid && coarseRes.ncc > 0.35;
  const refineInitX = useCoarse ? coarseRes.x * 2.0 : predX;
  const refineInitY = useCoarse ? coarseRes.y * 2.0 : predY;

  const fineRes = trackPointSingleLevel(prevPyr.data0, currPyr.data0, prevPyr.width0, prevPyr.height0, prevX, prevY, refineInitX, refineInitY, winRadius, 8);

  if (fineRes.valid) return fineRes;
  if (coarseRes.valid) return { x: coarseRes.x * 2.0, y: coarseRes.y * 2.0, ncc: coarseRes.ncc, valid: true };
  return { x: predX, y: predY, ncc: 0, valid: false };
}

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
  const inBounds = (x: number, y: number) => x >= margin && y >= margin && x < width - margin && y < height - margin;

  if (!inBounds(prevX, prevY) || !inBounds(initCurrX, initCurrY)) {
    return { x: initCurrX, y: initCurrY, ncc: 0, valid: false };
  }

  const winSize = 2 * radius + 1;
  const area = winSize * winSize;
  const T = new Float32Array(area);
  const gxBuf = new Float32Array(area);
  const gyBuf = new Float32Array(area);

  let gxx = 1e-3;
  let gxy = 0.0;
  let gyy = 1e-3;
  let sumT = 0.0;

  for (let wy = -radius, idx = 0; wy <= radius; ++wy) {
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

  const det = gxx * gyy - gxy * gxy;
  if (det < 1.0) return { x: initCurrX, y: initCurrY, ncc: 0, valid: false };
  const invDet = 1.0 / det;

  let cx = initCurrX;
  let cy = initCurrY;

  for (let iter = 0; iter < maxIters; ++iter) {
    if (!inBounds(cx, cy)) return { x: initCurrX, y: initCurrY, ncc: 0, valid: false };

    let bx = 0.0;
    let by = 0.0;
    for (let wy = -radius, k = 0; wy <= radius; ++wy) {
      const qy = cy + wy;
      for (let wx = -radius; wx <= radius; ++wx, ++k) {
        const diff = T[k] - sampleBilinear(currImg, width, height, cx + wx, qy);
        bx += gxBuf[k] * diff;
        by += gyBuf[k] * diff;
      }
    }

    const dx = invDet * (gyy * bx - gxy * by);
    const dy = invDet * (gxx * by - gxy * bx);
    cx += dx;
    cy += dy;

    if (dx * dx + dy * dy < 0.0004) break;
  }

  if (!inBounds(cx, cy)) return { x: initCurrX, y: initCurrY, ncc: 0, valid: false };

  const I = new Float32Array(area);
  let sumI = 0.0;
  for (let wy = -radius, k = 0; wy <= radius; ++wy) {
    const qy = cy + wy;
    for (let wx = -radius; wx <= radius; ++wx, ++k) {
      const val = sampleBilinear(currImg, width, height, cx + wx, qy);
      I[k] = val;
      sumI += val;
    }
  }

  const meanT = sumT / area;
  const meanI = sumI / area;
  let num = 0.0,
    denT = 0.0,
    denI = 0.0;

  for (let i = 0; i < area; ++i) {
    const dt = T[i] - meanT;
    const di = I[i] - meanI;
    num += dt * di;
    denT += dt * dt;
    denI += di * di;
  }

  const den = Math.sqrt(denT * denI);
  return { x: cx, y: cy, ncc: den > 1e-4 ? num / den : 0.0, valid: true };
}

function verifyArUcoQuadAppearance(
  gray: Uint8Array,
  width: number,
  height: number,
  corners: IPoint[],
  minBorderContrast: number,
): { valid: boolean; score: number } {
  let insideCount = 0;
  for (let i = 0; i < 4; ++i) {
    if (corners[i].x >= 2 && corners[i].x < width - 2 && corners[i].y >= 2 && corners[i].y < height - 2) {
      insideCount++;
    }
  }
  if (insideCount < 3) return { valid: false, score: 0 };

  let sign = 0;
  let minEdgeSq = Infinity;
  let maxEdgeSq = 0;

  for (let i = 0; i < 4; ++i) {
    const a = corners[i];
    const b = corners[(i + 1) & 3];
    const c = corners[(i + 2) & 3];

    const abx = b.x - a.x;
    const aby = b.y - a.y;
    const edgeSq = abx * abx + aby * aby;
    if (edgeSq < minEdgeSq) minEdgeSq = edgeSq;
    if (edgeSq > maxEdgeSq) maxEdgeSq = edgeSq;

    const cross = abx * (c.y - b.y) - aby * (c.x - b.x);
    if (Math.abs(cross) < 4.0) return { valid: false, score: 0 };

    const currSign = cross > 0 ? 1 : -1;
    if (sign === 0) sign = currSign;
    else if (sign !== currSign) return { valid: false, score: 0 };
  }

  if (minEdgeSq < 64.0 || maxEdgeSq / minEdgeSq > 16.0) {
    return { valid: false, score: 0 };
  }

  const [c0, c1, c2, c3] = corners;
  const evalQuad = (u: number, v: number) => {
    const ou = 1.0 - u;
    const ov = 1.0 - v;
    return {
      x: ov * (ou * c0.x + u * c1.x) + v * (ou * c3.x + u * c2.x),
      y: ov * (ou * c0.y + u * c1.y) + v * (ou * c3.y + u * c2.y),
    };
  };

  let sumOuter = 0.0;
  let sumInner = 0.0;
  let validSamples = 0;

  for (const [uOut, vOut, uIn, vIn] of BORDER_SAMPLE_PAIRS) {
    const pOut = evalQuad(uOut, vOut);
    const pIn = evalQuad(uIn, vIn);

    if (pOut.x >= 1 && pOut.x < width - 1 && pOut.y >= 1 && pOut.y < height - 1 && pIn.x >= 1 && pIn.x < width - 1 && pIn.y >= 1 && pIn.y < height - 1) {
      sumOuter += sampleBilinear(gray, width, height, pOut.x, pOut.y);
      sumInner += sampleBilinear(gray, width, height, pIn.x, pIn.y);
      validSamples++;
    }
  }

  if (validSamples < 6) return { valid: false, score: 0 };

  const contrast = (sumOuter - sumInner) / validSamples;
  if (contrast < minBorderContrast) {
    return { valid: false, score: Math.max(0, contrast / Math.max(1, minBorderContrast)) };
  }

  return { valid: true, score: Math.min(1.0, contrast / 45.0) };
}

/**
 * Blends `dst` with a Procrustes similarity transform of `src` onto `dst`.
 * Higher `rigidityWeight` [0..1] preserves `src` polygon angles/proportions over independent corner positions.
 */
function regularizeQuadShape(src: IPoint[], dst: IPoint[], rigidityWeight: number): IPoint[] {
  if (rigidityWeight <= 0) return dst;

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

  let numA = 0.0,
    numB = 0.0,
    den = 0.0;
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

  const scale = Math.hypot(a, b);
  if (scale > 1e-4) {
    const factor = Math.max(0.8, Math.min(1.25, scale)) / scale;
    a *= factor;
    b *= factor;
  }

  const wRigid = Math.max(0.0, Math.min(1.0, rigidityWeight));
  const wFree = 1.0 - wRigid;

  return dst.map((d, i) => {
    const sx = src[i].x - scx;
    const sy = src[i].y - scy;
    return {
      x: wRigid * (dcx + a * sx - b * sy) + wFree * d.x,
      y: wRigid * (dcy + b * sx + a * sy) + wFree * d.y,
    };
  });
}

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
  const idx = y0 * width + x0;

  return (1.0 - fy) * (ofx * img[idx] + fx * img[idx + 1]) + fy * (ofx * img[idx + width] + fx * img[idx + width + 1]);
}
