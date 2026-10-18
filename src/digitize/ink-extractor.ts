import type { Mat } from "@techstark/opencv-js";
import { CvScope } from "./cv-scope.ts";
import type { PaperTrackingResult, Rect } from "./types.ts";

export interface InkExtractorOptions {
  /** Size of Gaussian blur kernel (default: 5) */
  blurKsize?: number;
  /** Adaptive threshold neighborhood block size in millimeters (default: 15.0) */
  adaptiveBlockMm?: number;
  /** Adaptive threshold constant subtraction (default: 15) */
  adaptiveC?: number;
  /** Width of outer sheet border to zero out in millimeters (default: 5.0) */
  edgeMarginMm?: number;
  /** Additional padding around marker exclusion boxes in millimeters (default: 4.0) */
  markerPadMm?: number;
  /** Discard isolated connected components smaller than this area in mm² (default: 3.0) */
  minSpeckAreaMm2?: number;
}

/**
 * Extracts a clean binary ink Mat (255=ink, 0=paper) from the camera image
 * using perspective rectification, adaptive Gaussian thresholding, and speck suppression.
 */
export function extractInkMask(
  scope: CvScope,
  cameraImageData: ImageData,
  tracking: PaperTrackingResult,
  options?: InkExtractorOptions,
  frozenRegions?: Rect[],
): Mat | null {
  if (!tracking.isTracking || !tracking.cameraToCanvasH) {
    return null;
  }

  const cv = scope.cv;
  const [canvasWidth, canvasHeight] = tracking.canvasSize;
  const pxPerMm = tracking.pxPerMm[0];

  const blurKsize = options?.blurKsize ?? 5;
  const adaptiveBlockMm = options?.adaptiveBlockMm ?? 15.0;
  const adaptiveC = options?.adaptiveC ?? 15;
  const edgeMarginMm = options?.edgeMarginMm ?? 5.0;
  const markerPadMm = options?.markerPadMm ?? 4.0;
  const minSpeckAreaMm2 = options?.minSpeckAreaMm2 ?? 3.0;

  // 1. Wrap camera ImageData into OpenCV Mat
  const cameraMat = scope.manage(cv.matFromImageData(cameraImageData));

  // 2. Warp camera perspective to rectified canvas resolution
  const rectifiedRgba = scope.mat();
  const H_mat = scope.manage(cv.matFromArray(3, 3, cv.CV_64FC1, Array.from(tracking.cameraToCanvasH)));
  const canvasDsize = new cv.Size(canvasWidth, canvasHeight);
  const borderWhite = new cv.Scalar(255, 255, 255, 255);

  cv.warpPerspective(cameraMat, rectifiedRgba, H_mat, canvasDsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, borderWhite);

  // 3. Grayscale conversion & Gaussian noise filtering
  const gray = scope.mat();
  cv.cvtColor(rectifiedRgba, gray, cv.COLOR_RGBA2GRAY);

  const blur = scope.mat();
  const kSize = Math.max(3, blurKsize | 1);
  cv.GaussianBlur(gray, blur, new cv.Size(kSize, kSize), 0, 0, cv.BORDER_DEFAULT);

  // 4. Adaptive Gaussian Thresholding (inverse: ink becomes 255, paper becomes 0)
  const inkMask = scope.mat();
  const blockSize = Math.max(3, Math.round(adaptiveBlockMm * pxPerMm) | 1);
  cv.adaptiveThreshold(blur, inkMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blockSize, adaptiveC);

  const zeroScalar = new cv.Scalar(0);

  // 5. Zero out outer paper margin (e.g. 5mm)
  const marginPx = Math.round(edgeMarginMm * pxPerMm);
  if (marginPx > 0) {
    const topRect = new cv.Rect(0, 0, canvasWidth, marginPx);
    const btmRect = new cv.Rect(0, canvasHeight - marginPx, canvasWidth, marginPx);
    const lftRect = new cv.Rect(0, 0, marginPx, canvasHeight);
    const rgtRect = new cv.Rect(canvasWidth - marginPx, 0, marginPx, canvasHeight);

    inkMask.roi(topRect).setTo(zeroScalar);
    inkMask.roi(btmRect).setTo(zeroScalar);
    inkMask.roi(lftRect).setTo(zeroScalar);
    inkMask.roi(rgtRect).setTo(zeroScalar);
  }

  // 6. Zero out marker exclusion zones (plus padding)
  const padPx = Math.round(markerPadMm * pxPerMm);
  for (const box of tracking.markerExclusionBoxes) {
    const x0 = Math.max(0, Math.floor(box.x - padPx));
    const y0 = Math.max(0, Math.floor(box.y - padPx));
    const x1 = Math.min(canvasWidth, Math.ceil(box.x + box.width + padPx));
    const y1 = Math.min(canvasHeight, Math.ceil(box.y + box.height + padPx));

    if (x1 > x0 && y1 > y0) {
      const markerRect = new cv.Rect(x0, y0, x1 - x0, y1 - y0);
      inkMask.roi(markerRect).setTo(zeroScalar);
    }
  }

  // 7. Zero out regions occupied by frozen segments (ignoring new detections near finger)
  if (frozenRegions && frozenRegions.length > 0) {
    const frozenPadPx = Math.round(15.0 * pxPerMm);
    for (const region of frozenRegions) {
      const x0 = Math.max(0, Math.floor(region.x - frozenPadPx));
      const y0 = Math.max(0, Math.floor(region.y - frozenPadPx));
      const x1 = Math.min(canvasWidth, Math.ceil(region.x + region.width + frozenPadPx));
      const y1 = Math.min(canvasHeight, Math.ceil(region.y + region.height + frozenPadPx));

      if (x1 > x0 && y1 > y0) {
        const frozenRect = new cv.Rect(x0, y0, x1 - x0, y1 - y0);
        inkMask.roi(frozenRect).setTo(zeroScalar);
      }
    }
  }

  // 8. Speck removal via connected components
  const labels = scope.mat();
  const stats = scope.mat();
  const centroids = scope.mat();
  const numComponents = cv.connectedComponentsWithStats(inkMask, labels, stats, centroids, 8, cv.CV_32S);

  const minSpeckPx = minSpeckAreaMm2 * pxPerMm * pxPerMm;
  const statsData = stats.data32S;
  const labelsData = labels.data32S;
  const inkData = inkMask.data;

  // Identify components below minSpeckPx
  const isSpeck = new Uint8Array(numComponents);
  for (let i = 1; i < numComponents; i++) {
    const area = statsData[i * 5 + cv.CC_STAT_AREA];
    if (area < minSpeckPx) {
      isSpeck[i] = 1;
    }
  }

  for (let idx = 0; idx < labelsData.length; idx++) {
    if (isSpeck[labelsData[idx]] === 1) {
      inkData[idx] = 0;
    }
  }

  return inkMask;
}
