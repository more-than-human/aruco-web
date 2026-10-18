import { CvScope } from "./cv-scope.ts";
import { extractInkMask, type InkExtractorOptions } from "./ink-extractor.ts";
import { isOpenCVReady } from "./opencv.ts";
import type { PaperTrackingResult } from "./types.ts";
import { traceContoursChaikin, type VectorizerOptions, type VectorShape } from "./vectorizer.ts";

export interface InkDigitizerConfig {
  extractor?: InkExtractorOptions;
  vectorizer?: VectorizerOptions;
  /** Max update rate in Hz (default: 8 Hz) */
  targetFps?: number;
}

export class InkDigitizer {
  private config: InkDigitizerConfig;
  private isProcessing = false;
  private lastProcessTime = 0;
  private minIntervalMs: number;
  private latestVectorShape: VectorShape | null = null;
  private latestPath2D: Path2D | null = null;
  private lastDurationMs = 0;

  constructor(config?: InkDigitizerConfig) {
    this.config = config ?? {};
    const targetFps = this.config.targetFps ?? 8;
    this.minIntervalMs = 1000 / targetFps;
  }

  public reset(): void {
    this.latestVectorShape = null;
    this.latestPath2D = null;
    this.lastProcessTime = 0;
    this.isProcessing = false;
    this.lastDurationMs = 0;
  }

  public getLatestVectorShape(): VectorShape | null {
    return this.latestVectorShape;
  }

  public getLatestPath2D(): Path2D | null {
    return this.latestPath2D;
  }

  public getLastDurationMs(): number {
    return this.lastDurationMs;
  }

  /**
   * Evaluates camera frame for ink extraction and vectorization.
   * Runs asynchronously without blocking the main 30/60 fps camera loop.
   */
  public processFrame(cameraImageData: ImageData, tracking: PaperTrackingResult, now: number = performance.now()): void {
    if (!isOpenCVReady()) return;
    if (!tracking.isTracking || !tracking.boundary?.isLocked) return;
    if (this.isProcessing) return;
    if (now - this.lastProcessTime < this.minIntervalMs) return;

    this.isProcessing = true;
    this.lastProcessTime = now;

    // Run asynchronously to allow UI to breathe
    queueMicrotask(() => {
      const t0 = performance.now();
      try {
        CvScope.use((scope) => {
          const inkMat = extractInkMask(scope, cameraImageData, tracking, this.config.extractor);
          if (!inkMat) return;

          const pxPerMm = tracking.pxPerMm[0];
          const shape = traceContoursChaikin(inkMat, pxPerMm, this.config.vectorizer);

          this.latestVectorShape = shape;
          this.latestPath2D = shape.svgPath ? new Path2D(shape.svgPath) : null;
          this.lastDurationMs = performance.now() - t0;
        });
      } catch (err) {
        console.error("Error during ink vectorization:", err);
      } finally {
        this.isProcessing = false;
      }
    });
  }
}
