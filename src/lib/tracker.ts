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
   * adapts to sudden accelerations / direction changes. (Default: 15.0)
   */
  processNoise?: number;

  /**
   * Measurement noise variance (r) in pixels^2. Controls how much the filter
   * trusts the ArUco detector vs. its own motion model. (Default: 2.5)
   */
  measurementNoise?: number;

  /**
   * Velocity damping factor applied during coasting (dropped frames)
   * to prevent markers from flying off-screen indefinitely. (Default: 0.85)
   */
  coastVelocityDamping?: number;

  /**
   * Fixed time step (dt) in seconds/frames. If omitted, standard normalized
   * frame step dt = 1.0 is used unless a timestamp is passed to detect(). (Default: 1.0)
   */
  fixedDt?: number;

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
  /** True if the marker was not detected in the current frame and is coasting on Kalman prediction */
  public isPredicted: boolean;
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
    this.missedFrames = missedFrames;
    this.age = age;
    this.velocity = velocity;
  }
}

/**
 * 16-State Linear Kalman Filter for a single ArUco Marker.
 *
 * State vector x (Float32Array(16)):
 *   Indices 0..7  : [x0, y0, x1, y1, x2, y2, x3, y3] (Corner positions in pixel space)
 *   Indices 8..15 : [vx0, vy0, vx1, vy1, vx2, vy2, vx3, vy3] (Corner velocities)
 *
 * Measurement vector z (Float32Array(8)):
 *   Indices 0..7  : [x0, y0, x1, y1, x2, y2, x3, y3]
 *
 * Covariance Matrix P (Float32Array(256)):
 *   16x16 dense row-major representation synchronized with 8 decoupled 2x2 blocks.
 */
export class MarkerKalmanFilter {
  public readonly id: number;
  public readonly x: Float32Array; // 16-element state vector
  public readonly P: Float32Array; // 16x16 covariance matrix (row-major)

  // Compact 2x2 block scalars representing the isotropic 16x16 covariance:
  // P = [ p00 * I_8   p01 * I_8 ]
  //     [ p10 * I_8   p11 * I_8 ]
  private p00: number;
  private p01: number;
  private p10: number;
  private p11: number;

  public lastHammingDistance: number;
  public missedFrames: number = 0;
  public age: number = 1;
  public lastTimestamp: number | null = null;

  private readonly q: number;
  private readonly r: number;

  constructor(
    id: number,
    initialCorners: IPoint[],
    hammingDistance: number,
    processNoise: number = 15.0,
    measurementNoise: number = 2.5,
    timestamp: number | null = null,
  ) {
    this.id = id;
    this.lastHammingDistance = hammingDistance;
    this.q = processNoise;
    this.r = measurementNoise;
    this.lastTimestamp = timestamp;

    this.x = new Float32Array(16);
    this.P = new Float32Array(256);

    // Initialize positions with the 4 corners, velocities with 0
    for (let i = 0; i < 4; ++i) {
      this.x[i * 2] = initialCorners[i].x;
      this.x[i * 2 + 1] = initialCorners[i].y;
      this.x[8 + i * 2] = 0.0;
      this.x[8 + i * 2 + 1] = 0.0;
    }

    // Initial covariance: high certainty on position, high uncertainty on initial velocity
    this.p00 = measurementNoise;
    this.p01 = 0.0;
    this.p10 = 0.0;
    this.p11 = processNoise * 10.0;

    this.syncDenseCovariance();
  }

  /**
   * Kalman Prediction Step:
   *   x_{k|k-1} = F * x_{k-1}
   *   P_{k|k-1} = F * P_{k-1} * F^T + Q
   */
  public predict(dt: number = 1.0, velocityDamping: number = 1.0): void {
    if (dt <= 0) dt = 1.0;

    // 1. State prediction (vectorized loop over the 8 coordinates: FMA pattern x_i += dt * vx_i)
    const x = this.x;
    for (let i = 0; i < 8; ++i) {
      const v = x[8 + i] * velocityDamping;
      x[8 + i] = v;
      x[i] = x[i] + dt * v;
    }

    // 2. Covariance prediction: P_{k|k-1} = F * P_{k-1} * F^T + Q
    // Continuous white noise acceleration model discretized over dt:
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
   * Kalman Measurement Update Step:
   *   K = P_{k|k-1} * H^T * (H * P_{k|k-1} * H^T + R)^{-1}
   *   x_k = x_{k|k-1} + K * (z_k - H * x_{k|k-1})
   *   P_k = (I - K * H) * P_{k|k-1}
   */
  public update(measuredCorners: IPoint[], hammingDistance: number): void {
    const aligned = this.alignCorners(measuredCorners);

    // Innovation covariance S = H * P * H^T + R = (p00 + r) * I_8
    const invS = 1.0 / (this.p00 + this.r);

    // Kalman gain K = P * H^T * S^{-1} = [ k0 * I_8, k1 * I_8 ]^T
    const k0 = this.p00 * invS;
    const k1 = this.p10 * invS;

    // Update state vector x_k = x_{k|k-1} + K * y
    const x = this.x;
    for (let i = 0; i < 4; ++i) {
      const idxX = i * 2;
      const idxY = idxX + 1;

      const yX = aligned[i].x - x[idxX];
      const yY = aligned[i].y - x[idxY];

      // Position update
      x[idxX] += k0 * yX;
      x[idxY] += k0 * yY;

      // Velocity update
      x[8 + idxX] += k1 * yX;
      x[8 + idxY] += k1 * yY;
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

    this.syncDenseCovariance();

    this.lastHammingDistance = hammingDistance;
    this.missedFrames = 0;
    this.age++;
  }

  /**
   * Ensures cyclic corner ordering consistency between predicted state and new measurement
   * to prevent 90/180/270-degree rotational jumps from corrupting velocity estimates.
   */
  private alignCorners(measured: IPoint[]): IPoint[] {
    let bestShift = 0;
    let minCost = Infinity;
    const x = this.x;

    for (let shift = 0; shift < 4; ++shift) {
      let cost = 0.0;
      for (let i = 0; i < 4; ++i) {
        const pt = measured[(i + shift) & 3];
        const dx = pt.x - x[i * 2];
        const dy = pt.y - x[i * 2 + 1];
        cost += dx * dx + dy * dy;
      }
      if (cost < minCost) {
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
    const P = this.P;
    P.fill(0);
    for (let i = 0; i < 8; ++i) {
      P[i * 16 + i] = this.p00;
      P[i * 16 + (i + 8)] = this.p01;
      P[(i + 8) * 16 + i] = this.p10;
      P[(i + 8) * 16 + (i + 8)] = this.p11;
    }
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
    return new TrackedMarker(this.id, this.getCorners(), this.lastHammingDistance, isPredicted, this.missedFrames, this.age, this.getVelocities());
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
 * Stateful ArUco Marker Tracker using ID-indexed Linear Kalman Filters.
 * Exposes the exact same detection interface and Marker output structure as Detector.
 */
export class Tracker {
  public readonly config: Required<
    Pick<ITrackerConfig, "dictionaryName" | "maxMissedFrames" | "processNoise" | "measurementNoise" | "coastVelocityDamping" | "fixedDt">
  > &
    ITrackerConfig;

  public detector: IMarkerDetector;
  private readonly filters: Map<number, MarkerKalmanFilter>;
  private lastFrameTimestamp: number | null = null;
  private streamConfig: any;
  private mjpeg: any;

  constructor(config: ITrackerConfig) {
    this.config = {
      ...config,
      maxMissedFrames: config.maxMissedFrames ?? 5,
      processNoise: config.processNoise ?? 15.0,
      measurementNoise: config.measurementNoise ?? 2.5,
      coastVelocityDamping: config.coastVelocityDamping ?? 0.85,
      fixedDt: config.fixedDt ?? 1.0,
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
   * @param image IImage or ImageData-like object ({ width, height, data })
   * @param timestampMs Optional monotonic timestamp in milliseconds to compute variable dt
   */
  public detect(image: IImage | ImageData | { width: number; height: number; data: Uint8ClampedArray | Uint8Array }, timestampMs?: number): TrackedMarker[] {
    const rawMarkers = this.detector.detect(image as any);
    return this.updateWithDetections(rawMarkers as Marker[], timestampMs);
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
   */
  public updateWithDetections(rawMarkers: (Marker | IMarkerLike)[], timestampMs?: number): TrackedMarker[] {
    // Compute step time delta (dt)
    let dt = this.config.fixedDt;
    if (timestampMs !== undefined) {
      if (this.lastFrameTimestamp !== null && timestampMs > this.lastFrameTimestamp) {
        const elapsed = timestampMs - this.lastFrameTimestamp;
        // If elapsed time is huge (> 1 sec, e.g. tab backgrounded), clamp dt to fixedDt
        if (elapsed > 1000.0) {
          dt = this.config.fixedDt;
        } else {
          // Normalize milliseconds to ~60fps frame units (16.667ms = 1.0 unit) so noise parameters stay scale-invariant
          dt = elapsed / (1000.0 / 60.0);
        }
      }
      this.lastFrameTimestamp = timestampMs;
    }
    if (dt <= 0 || !Number.isFinite(dt)) {
      dt = this.config.fixedDt;
    }

    // Group detections by marker ID (if multiple markers share an ID, pick lowest hamming distance)
    const detectedById = new Map<number, Marker | IMarkerLike>();
    for (let i = 0; i < rawMarkers.length; ++i) {
      const m = rawMarkers[i];
      const existing = detectedById.get(m.id);
      if (!existing || m.hammingDistance < existing.hammingDistance) {
        detectedById.set(m.id, m);
      }
    }

    const outputMarkers: TrackedMarker[] = [];

    // 1. Process existing tracked markers
    for (const [id, kf] of this.filters.entries()) {
      const detected = detectedById.get(id);

      if (detected) {
        // Marker was detected in this frame: Predict + Measurement Update
        kf.predict(dt, 1.0);
        kf.update(detected.corners, detected.hammingDistance);
        kf.lastTimestamp = timestampMs ?? null;

        outputMarkers.push(kf.toMarker(false));
        detectedById.delete(id);
      } else {
        // Detector dropped this marker in this frame: Coast using Prediction Step
        kf.missedFrames++;

        if (kf.missedFrames <= this.config.maxMissedFrames) {
          kf.predict(dt, this.config.coastVelocityDamping);
          outputMarkers.push(kf.toMarker(true));
        } else {
          // Exceeded maximum coasting frames: prune track
          this.filters.delete(id);
        }
      }
    }

    // 2. Initialize new Kalman Filters for newly discovered marker IDs
    for (const [id, newMarker] of detectedById.entries()) {
      // Clone corners so external mutations don't affect internal state
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
        const extrapolatedCorners: IPoint[] = [
          { x: corners[0].x + lookaheadDt * velocities[0].x, y: corners[0].y + lookaheadDt * velocities[0].y },
          { x: corners[1].x + lookaheadDt * velocities[1].x, y: corners[1].y + lookaheadDt * velocities[1].y },
          { x: corners[2].x + lookaheadDt * velocities[2].x, y: corners[2].y + lookaheadDt * velocities[2].y },
          { x: corners[3].x + lookaheadDt * velocities[3].x, y: corners[3].y + lookaheadDt * velocities[3].y },
        ];
        results.push(new TrackedMarker(kf.id, extrapolatedCorners, kf.lastHammingDistance, kf.missedFrames > 0, kf.missedFrames, kf.age, velocities));
      }
    }

    return results;
  }

  /**
   * Exports all active tracks in a flat Structure-of-Arrays (SoA) layout
   * as described in theory.md for batch SIMD / WebGL / WASM consumption.
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
    }
  }
}
