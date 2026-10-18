import { isPointInPolygon } from "./homography.ts";
import type { DrawingSegment, Point2D } from "./types.ts";

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

/**
 * Tests whether a point is inside or within tolerance of a polygon hull.
 */
export function isPointInHull(hull: Point2D[], pt: Point2D, tolerancePx: number = 12): boolean {
  if (hull.length < 3) return false;

  if (isPointInPolygon(pt, hull)) {
    return true;
  }

  if (tolerancePx > 0) {
    const tolSq = tolerancePx * tolerancePx;
    const n = hull.length;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      if (pointToSegmentDistSq(pt.x, pt.y, hull[j].x, hull[j].y, hull[i].x, hull[i].y) <= tolSq) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Computes 2D convex hull of a set of points using Andrew's monotone chain algorithm.
 * Returns vertices in counter-clockwise order.
 */
export function computeConvexHull(points: Point2D[]): Point2D[] {
  const n = points.length;
  if (n <= 2) return [...points];

  const sorted = points.slice().sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));

  const unique: Point2D[] = [];
  for (let i = 0; i < sorted.length; i++) {
    if (i === 0 || sorted[i].x !== sorted[i - 1].x || sorted[i].y !== sorted[i - 1].y) {
      unique.push(sorted[i]);
    }
  }
  if (unique.length <= 2) return unique;

  const cross = (o: Point2D, a: Point2D, b: Point2D) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: Point2D[] = [];
  for (const p of unique) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: Point2D[] = [];
  for (let i = unique.length - 1; i >= 0; i--) {
    const p = unique[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  lower.pop();
  upper.pop();

  return lower.concat(upper);
}

/**
 * Creates a closed Path2D representing a polygon hull.
 */
export function createHullPath2D(hull: Point2D[]): Path2D {
  const path = new Path2D();
  if (hull.length === 0) return path;
  path.moveTo(hull[0].x, hull[0].y);
  for (let i = 1; i < hull.length; i++) {
    path.lineTo(hull[i].x, hull[i].y);
  }
  path.closePath();
  return path;
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
 * Squared distance from a point (px, py) to a line segment (x1, y1)-(x2, y2).
 */
export function pointToSegmentDistSq(px: number, py: number, x1: number, y1: number, x2: number, y2: number): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) {
    const ex = px - x1;
    const ey = py - y1;
    return ex * ex + ey * ey;
  }
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lenSq));
  const qx = x1 + t * dx;
  const qy = y1 + t * dy;
  const rx = px - qx;
  const ry = py - qy;
  return rx * rx + ry * ry;
}

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
