import type { Mat } from "@techstark/opencv-js";
import { CvScope } from "./cv-scope.ts";
import { computeConvexHull, createHullPath2D } from "./segment-tracker.ts";
import { getMemoryViews, wasmExtractRasterCropToCanvas, wasmScanGroupBbox, withScratch } from "./segment-wasm.ts";
import type { DrawingSegment, PaperTrackingResult, Rect } from "./types.ts";

const DEFAULT_ADAPTIVE_BLOCK_MM = 7.0;
const DEFAULT_ADAPTIVE_C = 14;
const DEFAULT_EDGE_MARGIN_MM = 5.0;
const DEFAULT_MARKER_PAD_MM = 4.0;
const DEFAULT_FROZEN_PAD_MM = 15.0;
const DEFAULT_MERGE_DILATE_MM = 6.0;
const DEFAULT_MIN_DRAWING_AREA_MM2 = 15.0;
const DEFAULT_DOWNSAMPLE_SCALE = 3;

// Endian-aware 32-bit pixel values for fast RGBA array writes
const IS_LITTLE_ENDIAN = new Uint8Array(new Uint32Array([0x12345678]).buffer)[0] === 0x78;
const INK_PIXEL_32 = IS_LITTLE_ENDIAN ? 0xff2a170f : 0x0f172aff; // #0f172a

export interface InkProcessorOptions {
  /** Adaptive threshold neighborhood block size in millimeters (default: 7.0) */
  adaptiveBlockMm?: number;
  /** Adaptive threshold constant subtraction (default: 14) */
  adaptiveC?: number;
  /** Width of outer sheet border to zero out in millimeters (default: 5.0) */
  edgeMarginMm?: number;
  /** Padding around marker exclusion boxes in millimeters (default: 4.0) */
  markerPadMm?: number;
  /** Maximum gap in millimeters to fuse adjacent strokes into a drawing (default: 6.0) */
  mergeDilateMm?: number;
  /** Discard candidate drawings with total ink area smaller than this in mm² (default: 15.0) */
  minDrawingAreaMm2?: number;
  /** Downsampling scale factor for fast stroke clustering (default: 3) */
  downsampleScale?: number;
}

/**
 * End-to-end extraction and raster segmentation pipeline.
 */
export function processInk(
  scope: CvScope,
  cameraImageData: ImageData,
  tracking: PaperTrackingResult,
  options?: InkProcessorOptions,
  frozenRegions?: Rect[],
): DrawingSegment[] {
  const inkMat = extractInkMask(scope, cameraImageData, tracking, options, frozenRegions);
  if (!inkMat) return [];

  const pxPerMm = tracking.pxPerMm[0];
  return segmentDrawings(scope, inkMat, pxPerMm, options);
}

/**
 * Extracts a clean binary ink Mat from camera frame via single-channel perspective warp,
 * box-mean adaptive thresholding, and morphological opening.
 */
export function extractInkMask(
  scope: CvScope,
  cameraImageData: ImageData,
  tracking: PaperTrackingResult,
  options?: InkProcessorOptions,
  frozenRegions?: Rect[],
): Mat | null {
  if (!tracking.isTracking || !tracking.cameraToCanvasH) {
    return null;
  }

  const cv = scope.cv;
  const [canvasWidth, canvasHeight] = tracking.canvasSize;
  const pxPerMm = tracking.pxPerMm[0];

  const adaptiveBlockMm = options?.adaptiveBlockMm ?? DEFAULT_ADAPTIVE_BLOCK_MM;
  const adaptiveC = options?.adaptiveC ?? DEFAULT_ADAPTIVE_C;
  const edgeMarginMm = options?.edgeMarginMm ?? DEFAULT_EDGE_MARGIN_MM;
  const markerPadMm = options?.markerPadMm ?? DEFAULT_MARKER_PAD_MM;

  // Convert camera frame to 1-channel grayscale in camera space to reduce warp operations by 75%
  const cameraRgba = scope.manage(cv.matFromImageData(cameraImageData));
  const cameraGray = scope.mat();
  cv.cvtColor(cameraRgba, cameraGray, cv.COLOR_RGBA2GRAY);

  // Warp single-channel grayscale perspective
  const rectifiedGray = scope.mat();
  const H_mat = scope.manage(cv.matFromArray(3, 3, cv.CV_64FC1, Array.from(tracking.cameraToCanvasH)));
  const canvasDsize = new cv.Size(canvasWidth, canvasHeight);
  const borderWhite = new cv.Scalar(255);
  cv.warpPerspective(cameraGray, rectifiedGray, H_mat, canvasDsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, borderWhite);

  // Fast box-mean adaptive thresholding (O(1) via integral image)
  const inkMask = scope.mat();
  const blockSize = Math.max(3, Math.round(adaptiveBlockMm * pxPerMm) | 1);
  cv.adaptiveThreshold(rectifiedGray, inkMask, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, blockSize, adaptiveC);

  // Direct rectangle fills to eliminate submatrix wrapper allocations
  const zeroScalar = new cv.Scalar(0);
  const marginPx = Math.round(edgeMarginMm * pxPerMm);
  if (marginPx > 0) {
    cv.rectangle(inkMask, new cv.Point(0, 0), new cv.Point(canvasWidth, marginPx), zeroScalar, -1);
    cv.rectangle(inkMask, new cv.Point(0, canvasHeight - marginPx), new cv.Point(canvasWidth, canvasHeight), zeroScalar, -1);
    cv.rectangle(inkMask, new cv.Point(0, 0), new cv.Point(marginPx, canvasHeight), zeroScalar, -1);
    cv.rectangle(inkMask, new cv.Point(canvasWidth - marginPx, 0), new cv.Point(canvasWidth, canvasHeight), zeroScalar, -1);
  }

  const padPx = Math.round(markerPadMm * pxPerMm);
  for (const box of tracking.markerExclusionBoxes) {
    const x0 = Math.max(0, Math.floor(box.x - padPx));
    const y0 = Math.max(0, Math.floor(box.y - padPx));
    const x1 = Math.min(canvasWidth, Math.ceil(box.x + box.width + padPx));
    const y1 = Math.min(canvasHeight, Math.ceil(box.y + box.height + padPx));
    if (x1 > x0 && y1 > y0) {
      cv.rectangle(inkMask, new cv.Point(x0, y0), new cv.Point(x1, y1), zeroScalar, -1);
    }
  }

  if (frozenRegions && frozenRegions.length > 0) {
    const frozenPadPx = Math.round(DEFAULT_FROZEN_PAD_MM * pxPerMm);
    for (const region of frozenRegions) {
      const x0 = Math.max(0, Math.floor(region.x - frozenPadPx));
      const y0 = Math.max(0, Math.floor(region.y - frozenPadPx));
      const x1 = Math.min(canvasWidth, Math.ceil(region.x + region.width + frozenPadPx));
      const y1 = Math.min(canvasHeight, Math.ceil(region.y + region.height + frozenPadPx));
      if (x1 > x0 && y1 > y0) {
        cv.rectangle(inkMask, new cv.Point(x0, y0), new cv.Point(x1, y1), zeroScalar, -1);
      }
    }
  }

  // Fast 3x3 morphological open to strip camera sensor specks and isolated noise
  const k3 = scope.manage(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3)));
  cv.morphologyEx(inkMask, inkMask, cv.MORPH_OPEN, k3);

  return inkMask;
}

/**
 * Segments a binary ink image into distinct raster drawing regions using downscaled
 * morphological dilation and a single connected components pass.
 */
export function segmentDrawings(scope: CvScope, inkMat: Mat, pxPerMm: number, options?: InkProcessorOptions): DrawingSegment[] {
  const cv = scope.cv;
  const mergeDilateMm = options?.mergeDilateMm ?? DEFAULT_MERGE_DILATE_MM;
  const minDrawingAreaMm2 = options?.minDrawingAreaMm2 ?? DEFAULT_MIN_DRAWING_AREA_MM2;
  const scale = options?.downsampleScale ?? DEFAULT_DOWNSAMPLE_SCALE;

  const w = inkMat.cols;
  const h = inkMat.rows;
  const smallW = Math.floor(w / scale);
  const smallH = Math.floor(h / scale);

  // Downscaled stroke clustering
  const smallMat = scope.mat();
  cv.resize(inkMat, smallMat, new cv.Size(smallW, smallH), 0, 0, cv.INTER_NEAREST);

  const kDilate = Math.max(3, Math.round(mergeDilateMm * (pxPerMm / scale)) | 1);
  const dilateKernel = scope.manage(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kDilate, kDilate)));
  const grouped = scope.mat();
  cv.dilate(smallMat, grouped, dilateKernel);

  // Single connected components on downscaled grouped image
  const gLabels = scope.mat();
  const gStats = scope.mat();
  const gCentroids = scope.mat();
  const nGroups = cv.connectedComponentsWithStats(grouped, gLabels, gStats, gCentroids, 8, cv.CV_32S);

  const gStatsData = gStats.data32S;
  const inkData = inkMat.data;
  const minInkPixels = minDrawingAreaMm2 * pxPerMm * pxPerMm;

  const segments: DrawingSegment[] = [];

  withScratch((alloc) => {
    const numPixels = w * h;
    const inkPtr = alloc(numPixels);
    const { u8View } = getMemoryViews();
    u8View.set(inkData, inkPtr);

    for (let l = 1; l < nGroups; l++) {
      const rawGx = gStatsData[l * 5 + cv.CC_STAT_LEFT] * scale;
      const rawGy = gStatsData[l * 5 + cv.CC_STAT_TOP] * scale;
      const rawGw = gStatsData[l * 5 + cv.CC_STAT_WIDTH] * scale;
      const rawGh = gStatsData[l * 5 + cv.CC_STAT_HEIGHT] * scale;

      const pad = 4;
      const rx = Math.max(0, rawGx - pad);
      const ry = Math.max(0, rawGy - pad);
      const rw = Math.min(w - rx, rawGw + pad * 2);
      const rh = Math.min(h - ry, rawGh + pad * 2);

      const scan = wasmScanGroupBbox(inkPtr, w, rx, ry, rw, rh);
      if (scan.inkCount < minInkPixels || scan.maxX < scan.minX || scan.maxY < scan.minY) {
        continue;
      }

      const tightX = rx + scan.minX;
      const tightY = ry + scan.minY;
      const tightW = scan.maxX - scan.minX + 1;
      const tightH = scan.maxY - scan.minY + 1;

      const rasterCanvas = createRasterCanvas(tightW, tightH);
      const rasterCtx = rasterCanvas.getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;

      const alphaMask = wasmExtractRasterCropToCanvas(inkPtr, w, tightX, tightY, tightW, tightH, INK_PIXEL_32, rasterCtx);

      const hull = computeConvexHull([scan.pMinX, scan.pMaxX, scan.pMinY, scan.pMaxY, scan.pMinSum, scan.pMaxSum, scan.pMinDiff, scan.pMaxDiff]);
      const hullPath2d = createHullPath2D(hull);
      const areaMm2 = scan.inkCount / (pxPerMm * pxPerMm);

      let cachedActiveCanvas: CanvasImageSource | undefined;

      segments.push({
        id: "",
        bboxPx: [tightX, tightY, tightW, tightH],
        bboxMm: [tightX / pxPerMm, tightY / pxPerMm, tightW / pxPerMm, tightH / pxPerMm],
        inkAreaMm2: areaMm2,
        rasterCanvas,
        get activeCanvas() {
          if (!cachedActiveCanvas) {
            cachedActiveCanvas = createActiveCanvas(rasterCanvas, tightW, tightH);
          }
          return cachedActiveCanvas;
        },
        set activeCanvas(canvas: CanvasImageSource | undefined) {
          cachedActiveCanvas = canvas;
        },
        alphaMask,
        svgPath: "",
        path2d: new Path2D(),
        hull,
        hullPath2d,
        isActive: false,
        isFrozen: false,
      });
    }
  });

  // Sort into reading order: rows of 30 mm, then left-to-right
  segments.sort((a, b) => {
    const rowA = Math.round(a.bboxMm[1] / 30);
    const rowB = Math.round(b.bboxMm[1] / 30);
    return rowA !== rowB ? rowA - rowB : a.bboxMm[0] - b.bboxMm[0];
  });

  return segments;
}

export function createActiveCanvas(rasterCanvas: CanvasImageSource, width: number, height: number): HTMLCanvasElement | OffscreenCanvas {
  const canvas = createRasterCanvas(width, height);
  const ctx = canvas.getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (ctx) {
    ctx.drawImage(rasterCanvas, 0, 0);
    ctx.globalCompositeOperation = "source-in";
    ctx.fillStyle = "#2563eb";
    ctx.fillRect(0, 0, width, height);
  }
  return canvas;
}

function createRasterCanvas(width: number, height: number): HTMLCanvasElement | OffscreenCanvas {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(width, height);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}
