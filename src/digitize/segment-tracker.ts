import type { DrawingSegment } from "./types.ts";

/**
 * Computes Intersection-over-Union (IoU) for two axis-aligned bounding boxes [x, y, w, h].
 */
export function boxIoU(a: [number, number, number, number], b: [number, number, number, number]): number {
  const [ax0, ay0, aw, ah] = a;
  const [bx0, by0, bw, bh] = b;
  const iw = Math.max(0.0, Math.min(ax0 + aw, bx0 + bw) - Math.max(ax0, bx0));
  const ih = Math.max(0.0, Math.min(ay0 + ah, by0 + bh) - Math.max(ay0, by0));
  const inter = iw * ih;
  const union = aw * ah + bw * bh - inter;
  return union > 0 ? inter / union : 0.0;
}

/**
 * Tracks drawing segment identities stably over time via greedy bipartite IoU matching.
 * Follows digitizer-v2.md Section 2.5 DrawingTracker.
 */
export class SegmentTracker {
  private iouThreshold: number;
  private previousSegments: DrawingSegment[] = [];
  private nextId = 1;

  constructor(iouThreshold: number = 0.3) {
    this.iouThreshold = iouThreshold;
  }

  public reset(): void {
    this.previousSegments = [];
    this.nextId = 1;
  }

  /**
   * Matches incoming segments with previous frame segments to maintain persistent IDs (e.g. d1, d2).
   */
  public update(currentSegments: DrawingSegment[]): DrawingSegment[] {
    // Generate all candidate match pairs sorted by descending IoU
    const pairs: Array<{ iou: number; currIdx: number; prevIdx: number }> = [];

    for (let i = 0; i < currentSegments.length; i++) {
      for (let j = 0; j < this.previousSegments.length; j++) {
        const iou = boxIoU(currentSegments[i].bboxMm, this.previousSegments[j].bboxMm);
        if (iou >= this.iouThreshold) {
          pairs.push({ iou, currIdx: i, prevIdx: j });
        }
      }
    }

    pairs.sort((a, b) => b.iou - a.iou);

    const assignedIds = new Map<number, string>();
    const usedPrev = new Set<number>();

    for (const { currIdx, prevIdx } of pairs) {
      if (assignedIds.has(currIdx) || usedPrev.has(prevIdx)) {
        continue;
      }
      assignedIds.set(currIdx, this.previousSegments[prevIdx].id);
      usedPrev.add(prevIdx);
    }

    for (let i = 0; i < currentSegments.length; i++) {
      const seg = currentSegments[i];
      if (assignedIds.has(i)) {
        seg.id = assignedIds.get(i)!;
      } else {
        seg.id = `d${this.nextId++}`;
      }
    }

    this.previousSegments = currentSegments;
    return currentSegments;
  }
}
