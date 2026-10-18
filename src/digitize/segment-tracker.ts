import type { DrawingSegment, Point2D } from "./types.ts";

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
 * Calculates Euclidean distance from a 2D point to an axis-aligned bounding box [x, y, w, h].
 * Returns 0 if the point is inside the box.
 */
export function pointToBoxDistance(pt: Point2D, box: [number, number, number, number]): number {
  const [bx, by, bw, bh] = box;
  const dx = Math.max(bx - pt.x, 0, pt.x - (bx + bw));
  const dy = Math.max(by - pt.y, 0, pt.y - (by + bh));
  return Math.hypot(dx, dy);
}

/**
 * Tracks drawing segment identities stably over time via greedy bipartite IoU matching.
 * Preserves frozen segments intact when finger is nearby to prevent occlusion dropouts.
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
   * Automatically retains any frozen segments from the previous frame as the ground truth.
   */
  public update(currentSegments: DrawingSegment[], frozenSegments: DrawingSegment[] = []): DrawingSegment[] {
    const frozenIds = new Set(frozenSegments.map((s) => s.id));
    const nonFrozenPrev = this.previousSegments.filter((s) => !frozenIds.has(s.id));

    // Generate candidate match pairs between newly detected segments and previous non-frozen segments
    const pairs: Array<{ iou: number; currIdx: number; prevIdx: number }> = [];

    for (let i = 0; i < currentSegments.length; i++) {
      for (let j = 0; j < nonFrozenPrev.length; j++) {
        const iou = boxIoU(currentSegments[i].bboxMm, nonFrozenPrev[j].bboxMm);
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
      assignedIds.set(currIdx, nonFrozenPrev[prevIdx].id);
      usedPrev.add(prevIdx);
    }

    // Allocate IDs for current segments
    for (let i = 0; i < currentSegments.length; i++) {
      const seg = currentSegments[i];
      if (assignedIds.has(i)) {
        seg.id = assignedIds.get(i)!;
      } else {
        while (frozenIds.has(`d${this.nextId}`)) {
          this.nextId++;
        }
        seg.id = `d${this.nextId++}`;
      }
      seg.isFrozen = false;
    }

    // Combine frozen segments (truth) and newly detected segments
    const allSegments = [...frozenSegments, ...currentSegments];

    // Sort into reading order: rows of 30 mm, then left-to-right
    allSegments.sort((a, b) => {
      const rowA = Math.round(a.bboxMm[1] / 30);
      const rowB = Math.round(b.bboxMm[1] / 30);
      return rowA !== rowB ? rowA - rowB : a.bboxMm[0] - b.bboxMm[0];
    });

    this.previousSegments = allSegments;
    return allSegments;
  }
}
