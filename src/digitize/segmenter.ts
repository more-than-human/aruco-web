import type { Mat } from "@techstark/opencv-js";
import { CvScope } from "./cv-scope.ts";
import { createHullPath2D } from "./segment-tracker.ts";
import type { DrawingSegment } from "./types.ts";
import { traceContoursChaikin, type VectorizerOptions } from "./vectorizer.ts";

export interface SegmenterOptions {
  /** Maximum gap in millimeters to fuse adjacent strokes into a drawing (default: 6.0) */
  mergeDilateMm?: number;
  /** Discard candidate drawings with total ink area smaller than this in mm² (default: 15.0) */
  minDrawingAreaMm2?: number;
  /** Downsampling scale factor for fast stroke clustering (default: 2) */
  downsampleScale?: number;
  /** Vectorizer configuration options */
  vectorizerOptions?: VectorizerOptions;
}

/**
 * Segments a binary ink image into distinct drawing regions via morphological dilation
 * and connected components, then vectorizes each cluster into an individual DrawingSegment.
 * Follows digitizer-v2.md Section 2.5 and Section 3.8.
 */
export function segmentDrawings(scope: CvScope, inkMat: Mat, pxPerMm: number, options?: SegmenterOptions): DrawingSegment[] {
  const cv = scope.cv;
  const mergeDilateMm = options?.mergeDilateMm ?? 6.0;
  const minDrawingAreaMm2 = options?.minDrawingAreaMm2 ?? 15.0;
  const scale = options?.downsampleScale ?? 2;

  const w = inkMat.cols;
  const h = inkMat.rows;
  const smallW = Math.floor(w / scale);
  const smallH = Math.floor(h / scale);

  // 1. Downscaled stroke clustering for fast morphological fusion
  const smallMat = scope.mat();
  cv.resize(inkMat, smallMat, new cv.Size(smallW, smallH), 0, 0, cv.INTER_NEAREST);

  const kDilate = Math.max(3, Math.round(mergeDilateMm * (pxPerMm / scale)) | 1);
  const dilateKernel = scope.manage(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kDilate, kDilate)));
  const grouped = scope.mat();
  cv.dilate(smallMat, grouped, dilateKernel, new cv.Point(-1, -1), 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());

  // 2. Connected components on grouped clusters
  const gLabels = scope.mat();
  const gStats = scope.mat();
  const gCentroids = scope.mat();
  const nGroups = cv.connectedComponentsWithStats(grouped, gLabels, gStats, gCentroids, 8, cv.CV_32S);

  const gStatsData = gStats.data32S;
  const inkData = inkMat.data;
  const minInkPixels = minDrawingAreaMm2 * pxPerMm * pxPerMm;

  const segments: DrawingSegment[] = [];

  // Iterate over foreground groups (1 to nGroups - 1)
  for (let l = 1; l < nGroups; l++) {
    const rawGx = gStatsData[l * 5 + cv.CC_STAT_LEFT] * scale;
    const rawGy = gStatsData[l * 5 + cv.CC_STAT_TOP] * scale;
    const rawGw = gStatsData[l * 5 + cv.CC_STAT_WIDTH] * scale;
    const rawGh = gStatsData[l * 5 + cv.CC_STAT_HEIGHT] * scale;

    // Expand search box by padding to capture full anti-aliased stroke edges
    const pad = 8;
    const rx = Math.max(0, rawGx - pad);
    const ry = Math.max(0, rawGy - pad);
    const rw = Math.min(w - rx, rawGw + pad * 2);
    const rh = Math.min(h - ry, rawGh + pad * 2);

    // Tighten bounds to actual non-zero ink pixels within this bounding box
    let minX = rw;
    let minY = rh;
    let maxX = -1;
    let maxY = -1;
    let inkCount = 0;

    for (let y = ry; y < ry + rh; y++) {
      const rowOffset = y * w;
      for (let x = rx; x < rx + rw; x++) {
        if (inkData[rowOffset + x] > 0) {
          inkCount++;
          const lx = x - rx;
          const ly = y - ry;
          if (lx < minX) minX = lx;
          if (lx > maxX) maxX = lx;
          if (ly < minY) minY = ly;
          if (ly > maxY) maxY = ly;
        }
      }
    }

    if (inkCount < minInkPixels || maxX < minX || maxY < minY) {
      continue;
    }

    const tightX = rx + minX;
    const tightY = ry + minY;
    const tightW = maxX - minX + 1;
    const tightH = maxY - minY + 1;

    // Extract isolated tight crop
    const tightMat = scope.mat(tightH, tightW, cv.CV_8UC1);
    const tightData = tightMat.data;
    for (let y = 0; y < tightH; y++) {
      const srcOffset = (tightY + y) * w + tightX;
      const dstOffset = y * tightW;
      tightData.set(inkData.subarray(srcOffset, srcOffset + tightW), dstOffset);
    }

    // Vectorize tight crop with canvas coordinate offset
    const shape = traceContoursChaikin(tightMat, pxPerMm, options?.vectorizerOptions, { x: tightX, y: tightY });

    if (shape.outlineCount === 0 || !shape.svgPath) {
      continue;
    }

    const areaMm2 = inkCount / (pxPerMm * pxPerMm);

    segments.push({
      id: "",
      bboxPx: [tightX, tightY, tightW, tightH],
      bboxMm: [tightX / pxPerMm, tightY / pxPerMm, tightW / pxPerMm, tightH / pxPerMm],
      inkAreaMm2: areaMm2,
      svgPath: shape.svgPath,
      path2d: new Path2D(shape.svgPath),
      hull: shape.hull,
      hullPath2d: createHullPath2D(shape.hull),
      isActive: false,
      isFrozen: false,
    });
  }

  // Sort into reading order: rows of 30 mm, then left-to-right
  segments.sort((a, b) => {
    const rowA = Math.round(a.bboxMm[1] / 30);
    const rowB = Math.round(b.bboxMm[1] / 30);
    return rowA !== rowB ? rowA - rowB : a.bboxMm[0] - b.bboxMm[0];
  });

  return segments;
}
