import { CvScope } from "./cv-scope.ts";
import { extractInkMask, type InkExtractorOptions } from "./ink-extractor.ts";
import { isOpenCVReady } from "./opencv.ts";
import { SegmentTracker } from "./segment-tracker.ts";
import { segmentDrawings, type SegmenterOptions } from "./segmenter.ts";
import type { DrawingSegment, PaperTrackingResult, Point2D } from "./types.ts";
import type { VectorizerOptions } from "./vectorizer.ts";

export interface InkDigitizerConfig {
  extractor?: InkExtractorOptions;
  segmenter?: SegmenterOptions;
  vectorizer?: VectorizerOptions;
  /** Max update rate in Hz (default: 8 Hz) */
  targetFps?: number;
}

export class InkDigitizer {
  private config: InkDigitizerConfig;
  private tracker = new SegmentTracker();
  private isProcessing = false;
  private lastProcessTime = 0;
  private minIntervalMs: number;
  private latestSegments: DrawingSegment[] = [];
  private lastDurationMs = 0;
  private activeSegmentId: string | null = null;

  constructor(config?: InkDigitizerConfig) {
    this.config = config ?? {};
    const targetFps = this.config.targetFps ?? 8;
    this.minIntervalMs = 1000 / targetFps;
  }

  public reset(): void {
    this.latestSegments = [];
    this.tracker.reset();
    this.lastProcessTime = 0;
    this.isProcessing = false;
    this.lastDurationMs = 0;
    this.activeSegmentId = null;
  }

  public getSegments(): DrawingSegment[] {
    return this.latestSegments;
  }

  public getActiveSegment(): DrawingSegment | null {
    if (!this.activeSegmentId) return null;
    return this.latestSegments.find((s) => s.id === this.activeSegmentId) ?? null;
  }

  public getLastDurationMs(): number {
    return this.lastDurationMs;
  }

  /**
   * Tests whether the finger pointer coordinates intersect any segment
   * and marks the intersected segment as active.
   */
  public updatePointer(pointerCanvasPos: Point2D | null, ctx?: CanvasRenderingContext2D): DrawingSegment | null {
    this.activeSegmentId = null;

    if (!pointerCanvasPos) {
      for (const seg of this.latestSegments) {
        seg.isActive = false;
      }
      return null;
    }

    let active: DrawingSegment | null = null;

    for (const seg of this.latestSegments) {
      const isIntersecting = this.isPointInSegment(seg, pointerCanvasPos, ctx);
      seg.isActive = isIntersecting;
      if (isIntersecting && !active) {
        active = seg;
        this.activeSegmentId = seg.id;
      }
    }

    return active;
  }

  private isPointInSegment(seg: DrawingSegment, pt: Point2D, ctx?: CanvasRenderingContext2D): boolean {
    const [bx, by, bw, bh] = seg.bboxPx;
    const pad = 12;
    if (pt.x < bx - pad || pt.x > bx + bw + pad || pt.y < by - pad || pt.y > by + bh + pad) {
      return false;
    }

    if (!ctx) {
      return true;
    }

    ctx.save();
    ctx.lineWidth = 16;
    const inPath = ctx.isPointInPath(seg.path2d, pt.x, pt.y, "evenodd") || ctx.isPointInStroke(seg.path2d, pt.x, pt.y);
    ctx.restore();

    // If small icon or letter, touching anywhere inside the padded bbox is accepted
    const isSmall = bw <= 40 && bh <= 40;
    return inPath || isSmall;
  }

  /**
   * Evaluates camera frame for ink extraction, stroke clustering, and vectorization.
   * Runs asynchronously without blocking the main 30/60 fps camera loop.
   */
  public processFrame(cameraImageData: ImageData, tracking: PaperTrackingResult, now: number = performance.now()): void {
    if (!isOpenCVReady()) return;
    if (!tracking.isTracking || !tracking.boundary?.isLocked) return;
    if (this.isProcessing) return;
    if (now - this.lastProcessTime < this.minIntervalMs) return;

    this.isProcessing = true;
    this.lastProcessTime = now;

    queueMicrotask(() => {
      const t0 = performance.now();
      try {
        CvScope.use((scope) => {
          const inkMat = extractInkMask(scope, cameraImageData, tracking, this.config.extractor);
          if (!inkMat) return;

          const pxPerMm = tracking.pxPerMm[0];
          const rawSegments = segmentDrawings(scope, inkMat, pxPerMm, {
            ...this.config.segmenter,
            vectorizerOptions: this.config.vectorizer,
          });

          const trackedSegments = this.tracker.update(rawSegments);

          // Preserve active state if active segment continues to exist
          if (this.activeSegmentId) {
            for (const s of trackedSegments) {
              if (s.id === this.activeSegmentId) {
                s.isActive = true;
              }
            }
          }

          this.latestSegments = trackedSegments;
          this.lastDurationMs = performance.now() - t0;
        });
      } catch (err) {
        console.error("Error during ink segmentation & vectorization:", err);
      } finally {
        this.isProcessing = false;
      }
    });
  }
}
