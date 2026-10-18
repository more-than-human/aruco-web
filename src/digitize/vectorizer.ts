import type { Mat } from "@techstark/opencv-js";
import { CvScope } from "./cv-scope.ts";
import type { Point2D } from "./types.ts";

export interface VectorShape {
  /** SVG path "d" string ready for <path d="..." fill-rule="evenodd" /> or Path2D */
  svgPath: string;
  /** Individual SVG subpaths */
  subpaths: string[];
  /** Total number of vector outlines */
  outlineCount: number;
  /** Width in millimeters */
  widthMm: number;
  /** Height in millimeters */
  heightMm: number;
}

export interface VectorizerOptions {
  /** Filter out contours with pixel area smaller than this (default: 10) */
  turdsizePx?: number;
  /** Polygon approximation epsilon in millimeters (default: 0.2) */
  approxEpsilonMm?: number;
  /** Number of Chaikin smoothing iterations (default: 2) */
  chaikinIterations?: number;
  /** Morphological smoothing kernel diameter in millimeters (default: 0.75) */
  smoothKernelMm?: number;
  /** Gaussian pre-smoothing blur in millimeters (default: 0.5) */
  smoothBlurMm?: number;
  /** Outward normal expansion distance in pixels (default: 0.5) */
  outwardOffsetPx?: number;
}

/**
 * Subdivides polygon edges longer than `spacing` to ensure uniform vertex density
 * prior to Chaikin curve subdivision.
 */
export function densifyPolygon(pts: Point2D[], spacing: number): Point2D[] {
  const n = pts.length;
  if (n < 3) return pts;

  const out: Point2D[] = [];
  const safeSpacing = Math.max(1, spacing);

  for (let i = 0; i < n; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % n];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.hypot(dx, dy);
    const steps = Math.max(1, Math.ceil(dist / safeSpacing));

    for (let s = 0; s < steps; s++) {
      const t = s / steps;
      out.push({ x: a.x + dx * t, y: a.y + dy * t });
    }
  }

  return out;
}

/**
 * Applies Chaikin's corner-cutting algorithm for quadratic B-spline smoothing.
 * Replaces each edge with two points at 1/4 and 3/4 positions:
 *   Q = 0.75 * P0 + 0.25 * P1
 *   R = 0.25 * P0 + 0.75 * P1
 */
export function applyChaikin(pts: Point2D[], iterations: number = 2): Point2D[] {
  let curr = pts;
  for (let iter = 0; iter < iterations; iter++) {
    const n = curr.length;
    if (n < 3) break;

    const next: Point2D[] = [];
    for (let i = 0; i < n; i++) {
      const p0 = curr[i];
      const p1 = curr[(i + 1) % n];
      next.push({
        x: 0.75 * p0.x + 0.25 * p1.x,
        y: 0.75 * p0.y + 0.25 * p1.y,
      });
      next.push({
        x: 0.25 * p0.x + 0.75 * p1.x,
        y: 0.25 * p0.y + 0.75 * p1.y,
      });
    }
    curr = next;
  }
  return curr;
}

/**
 * Offsets polygon vertices along their outward normal vectors to align with pixel edges.
 * Samples the binary mask to ensure normals face outward (toward the paper background).
 */
export function offsetOutward(pts: Point2D[], maskData: Uint8Array, w: number, h: number, dist: number = 0.5): Point2D[] {
  const n = pts.length;
  if (n < 3) return pts;

  const normals: Point2D[] = [];
  let inwardCount = 0;
  let probeCount = 0;

  for (let i = 0; i < n; i++) {
    const prev = pts[(i - 1 + n) % n];
    const next = pts[(i + 1) % n];
    const tx = next.x - prev.x;
    const ty = next.y - prev.y;
    const len = Math.hypot(tx, ty) || 1e-9;
    const nx = ty / len;
    const ny = -tx / len;
    normals.push({ x: nx, y: ny });

    // Probe 1.5px along normal
    const px = Math.round(pts[i].x + nx * 1.5);
    const py = Math.round(pts[i].y + ny * 1.5);
    if (px >= 0 && px < w && py >= 0 && py < h) {
      probeCount++;
      if (maskData[py * w + px] > 0) {
        inwardCount++;
      }
    }
  }

  // If majority of probe points were inside ink, flip normal sign
  const sign = probeCount > 0 && inwardCount / probeCount > 0.5 ? -1.0 : 1.0;

  return pts.map((p, i) => ({
    x: p.x + normals[i].x * dist * sign,
    y: p.y + normals[i].y * dist * sign,
  }));
}

/**
 * Pure OpenCV.js v5 Contour + Chaikin Vectorizer.
 * Converts a binary ink Mat (255=ink, 0=paper) into smooth SVG vector paths
 * formatted for rendering with `fill-rule="evenodd"`.
 */
export function traceContoursChaikin(inkMat: Mat, pxPerMm: number, options?: VectorizerOptions, offset?: Point2D): VectorShape {
  const turdsizePx = options?.turdsizePx ?? 10;
  const approxEpsilonMm = options?.approxEpsilonMm ?? 0.2;
  const chaikinIterations = options?.chaikinIterations ?? 2;
  const smoothKernelMm = options?.smoothKernelMm ?? 0.75;
  const smoothBlurMm = options?.smoothBlurMm ?? 0.5;
  const outwardOffsetPx = options?.outwardOffsetPx ?? 0.5;
  const offsetX = offset?.x ?? 0;
  const offsetY = offset?.y ?? 0;

  return CvScope.use((scope) => {
    const cv = scope.cv;

    // 1. Morphological close + open to smooth single-pixel edge roughness
    const kSize = Math.max(3, Math.round(smoothKernelMm * pxPerMm) | 1);
    const kernel = scope.manage(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kSize, kSize)));
    const morph = scope.mat();
    cv.morphologyEx(inkMat, morph, cv.MORPH_CLOSE, kernel);
    cv.morphologyEx(morph, morph, cv.MORPH_OPEN, kernel);

    // 2. Light Gaussian blur and binary threshold
    const bSize = Math.max(3, Math.round(smoothBlurMm * pxPerMm * 2) | 1);
    cv.GaussianBlur(morph, morph, new cv.Size(bSize, bSize), 0, 0, cv.BORDER_DEFAULT);
    const smooth = scope.mat();
    cv.threshold(morph, smooth, 127, 255, cv.THRESH_BINARY);

    // 3. Extract 2-level contour tree (outer contours + holes)
    const contours = scope.matVector();
    const hierarchy = scope.mat();
    cv.findContours(smooth, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE);

    const smoothData = smooth.data;
    const smoothWidth = smooth.cols;
    const smoothHeight = smooth.rows;

    const subpaths: string[] = [];

    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const area = cv.contourArea(c);
      if (area < turdsizePx) {
        c.delete();
        continue;
      }

      // Polygon approximation
      const approx = scope.mat();
      cv.approxPolyDP(c, approx, approxEpsilonMm * pxPerMm, true);
      const numPts = approx.rows;

      if (numPts < 3) {
        c.delete();
        approx.delete();
        continue;
      }

      // Read vertices from cv.Mat (CV_32SC2)
      const rawPoints: Point2D[] = [];
      const ptsData = approx.data32S;
      for (let j = 0; j < numPts; j++) {
        rawPoints.push({ x: ptsData[j * 2], y: ptsData[j * 2 + 1] });
      }

      c.delete();
      approx.delete();

      // Densify polygon edges longer than spacing
      const densified = densifyPolygon(rawPoints, pxPerMm);

      // Chaikin corner rounding
      const smoothed = applyChaikin(densified, chaikinIterations);

      // Outward normal offset (+0.5px)
      const offsetPts = offsetOutward(smoothed, smoothData, smoothWidth, smoothHeight, outwardOffsetPx);

      if (offsetPts.length >= 3) {
        let d = `M${(offsetPts[0].x + offsetX).toFixed(1)},${(offsetPts[0].y + offsetY).toFixed(1)}`;
        for (let k = 1; k < offsetPts.length; k++) {
          d += `L${(offsetPts[k].x + offsetX).toFixed(1)},${(offsetPts[k].y + offsetY).toFixed(1)}`;
        }
        d += "Z";
        subpaths.push(d);
      }
    }

    return {
      svgPath: subpaths.join(" "),
      subpaths,
      outlineCount: subpaths.length,
      widthMm: inkMat.cols / pxPerMm,
      heightMm: inkMat.rows / pxPerMm,
    };
  });
}
