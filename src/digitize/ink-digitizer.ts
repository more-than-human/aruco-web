import { CvScope } from "./cv-scope.ts";
import { extractInkMask, type InkExtractorOptions } from "./ink-extractor.ts";
import { isOpenCVReady } from "./opencv.ts";
import { isPointInHull, pointToBoxDistance, SegmentTracker } from "./segment-tracker.ts";
import { segmentDrawings, type SegmenterOptions } from "./segmenter.ts";
import type { DrawingSegment, PaperTrackingResult, Point2D, Rect } from "./types.ts";
import type { VectorizerOptions } from "./vectorizer.ts";

/** Proximity distance in canvas pixels (~25mm) within which a segment is frozen */
const PROXIMITY_DISTANCE_PX = 100;

/** Duration in ms to hold frozen state after finger leaves proximity to prevent occlusion dropouts */
const FREEZE_HOLD_MS = 500;

export interface InkDigitizerConfig {
  extractor?: InkExtractorOptions;
  segmenter?: SegmenterOptions;
  vectorizer?: VectorizerOptions;
  /** Max update rate in Hz (default: 8 Hz) */
  targetFps?: number;
  /** Proximity distance in canvas pixels for freezing segments (default: 100) */
  proximityDistancePx?: number;
  /** Grace hold time in ms for frozen segments (default: 500) */
  freezeHoldMs?: number;
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
  private proximityDistancePx: number;
  private freezeHoldMs: number;

  constructor(config?: InkDigitizerConfig) {
    this.config = config ?? {};
    const targetFps = this.config.targetFps ?? 8;
    this.minIntervalMs = 1000 / targetFps;
    this.proximityDistancePx = this.config.proximityDistancePx ?? PROXIMITY_DISTANCE_PX;
    this.freezeHoldMs = this.config.freezeHoldMs ?? FREEZE_HOLD_MS;
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
   * Tests whether the finger pointer coordinates or mouse pointer coordinates
   * intersect any segment, and manages the frozen state when pointer is near a segment.
   */
  public updatePointer(
    fingerCanvasPos: Point2D | null,
    mouseCanvasPos: Point2D | null = null,
    ctx?: CanvasRenderingContext2D,
    now: number = performance.now(),
  ): DrawingSegment | null {
    this.activeSegmentId = null;

    let active: DrawingSegment | null = null;

    for (const seg of this.latestSegments) {
      let fingerIntersecting = false;

      // 1. Finger proximity & intersection
      if (fingerCanvasPos) {
        const dist = pointToBoxDistance(fingerCanvasPos, seg.bboxPx);
        const isNear = dist <= this.proximityDistancePx;
        fingerIntersecting = this.isPointInSegment(seg, fingerCanvasPos, ctx);

        if (isNear || fingerIntersecting) {
          seg.isFrozen = true;
          seg.lastNearMs = now;
        } else if (now - (seg.lastNearMs ?? 0) > this.freezeHoldMs) {
          seg.isFrozen = false;
        }
      } else {
        if (now - (seg.lastNearMs ?? 0) > this.freezeHoldMs) {
          seg.isFrozen = false;
        }
      }

      // 2. Mouse hover & intersection
      const mouseIntersecting = mouseCanvasPos !== null && this.isPointInSegment(seg, mouseCanvasPos, ctx);

      // Freeze when mouse hovers so camera exposure or hand shadow doesn't distort segment
      if (mouseIntersecting) {
        seg.isFrozen = true;
        seg.lastNearMs = now;
      }

      const isIntersecting = fingerIntersecting || mouseIntersecting;
      seg.isActive = isIntersecting;

      if (isIntersecting && !active) {
        active = seg;
        this.activeSegmentId = seg.id;
      }
    }

    return active;
  }

  public findSegmentAtPoint(pt: Point2D, ctx?: CanvasRenderingContext2D): DrawingSegment | null {
    for (const seg of this.latestSegments) {
      if (this.isPointInSegment(seg, pt, ctx)) {
        return seg;
      }
    }
    return null;
  }

  public isPointInSegment(seg: DrawingSegment, pt: Point2D, _ctx?: CanvasRenderingContext2D): boolean {
    const [bx, by, bw, bh] = seg.bboxPx;
    const pad = 12;
    if (pt.x < bx - pad || pt.x > bx + bw + pad || pt.y < by - pad || pt.y > by + bh + pad) {
      return false;
    }

    if (seg.hull && seg.hull.length >= 3) {
      return isPointInHull(seg.hull, pt, pad);
    }

    // Fallback if hull not available (e.g. small icon or letter)
    const isSmall = bw <= 40 && bh <= 40;
    return isSmall;
  }

  /**
   * Evaluates camera frame for ink extraction, stroke clustering, and vectorization.
   * Freezes segments in proximity to the finger, ignoring new detection in their regions
   * and using the last detected segment as the ground truth.
   */
  public processFrame(cameraImageData: ImageData, tracking: PaperTrackingResult, now: number = performance.now()): void {
    if (!isOpenCVReady()) return;
    if (!tracking.isTracking || !tracking.boundary?.isLocked) return;
    if (this.isProcessing) return;
    if (now - this.lastProcessTime < this.minIntervalMs) return;

    this.isProcessing = true;
    this.lastProcessTime = now;

    // Snapshot currently frozen segments before async processing
    const frozenSegments = this.latestSegments.filter((s) => s.isFrozen);
    const frozenRegions: Rect[] = frozenSegments.map((s) => ({
      x: s.bboxPx[0],
      y: s.bboxPx[1],
      width: s.bboxPx[2],
      height: s.bboxPx[3],
    }));

    queueMicrotask(() => {
      const t0 = performance.now();
      try {
        CvScope.use((scope) => {
          const inkMat = extractInkMask(scope, cameraImageData, tracking, this.config.extractor, frozenRegions);
          if (!inkMat) return;

          const pxPerMm = tracking.pxPerMm[0];
          const rawSegments = segmentDrawings(scope, inkMat, pxPerMm, {
            ...this.config.segmenter,
            vectorizerOptions: this.config.vectorizer,
          });

          const trackedSegments = this.tracker.update(rawSegments, frozenSegments);

          // Preserve active and frozen states
          for (const s of trackedSegments) {
            if (s.id === this.activeSegmentId) {
              s.isActive = true;
            }
            if (frozenSegments.some((f) => f.id === s.id)) {
              s.isFrozen = true;
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
