import type { IPoint, Marker } from "../aruco-tracking/aruco.ts";
import type { TrackedMarker } from "../aruco-tracking/tracker.ts";
import { findHomography4Pts, invert3x3, multiply3x3, projectPoint } from "./homography.ts";
import { DEFAULT_SHEET_CONFIG, type PaperBoundary, type PaperSheetConfig, type PaperTrackingResult, type Point2D, type Rect } from "./types.ts";

interface EstablishedPaperPosition {
  boundary: PaperBoundary;
  cameraToMmH: Float64Array;
  mmToCameraH: Float64Array;
  cameraToCanvasH: Float64Array;
  canvasToCameraH: Float64Array;
  markerExclusionBoxes: Rect[];
}

export class PaperTracker {
  private config: PaperSheetConfig;
  private lastPosition: EstablishedPaperPosition | null = null;
  private lastTrackingResult: PaperTrackingResult;

  constructor(config?: Partial<PaperSheetConfig>) {
    this.config = { ...DEFAULT_SHEET_CONFIG, ...config };
    this.lastTrackingResult = this.createEmptyResult();
  }

  public reset(): void {
    this.lastPosition = null;
    this.lastTrackingResult = this.createEmptyResult();
  }

  /**
   * Updates tracking state using detected markers from the current camera frame.
   * Up to 3 markers can be occluded without losing track; the last paper position is used.
   * When all 4 markers are occluded, the paper is treated as absent.
   */
  public update(markers: (Marker | TrackedMarker)[], _timestampMs: number = performance.now()): PaperTrackingResult {
    const visibleMarkers = new Map<number, Point2D[]>();

    for (const m of markers) {
      if (!this.config.markerIds.includes(m.id)) continue;
      if (!m.corners || m.corners.length < 4) continue;
      if ("isPredicted" in m && (m as TrackedMarker).isPredicted) continue;

      visibleMarkers.set(
        m.id,
        m.corners.map((c: IPoint) => ({ x: c.x, y: c.y })),
      );
    }

    const visibleCount = visibleMarkers.size;

    // When all 4 markers are occluded, treat the paper as absent
    if (visibleCount === 0) {
      this.lastPosition = null;
      this.lastTrackingResult = this.createEmptyResult();
      return this.lastTrackingResult;
    }

    // When all 4 markers are visible, calculate new paper position
    if (visibleCount === 4) {
      const [idTL, idTR, idBR, idBL] = this.config.markerIds;
      const cornersTL = visibleMarkers.get(idTL)!;
      const cornersTR = visibleMarkers.get(idTR)!;
      const cornersBR = visibleMarkers.get(idBR)!;
      const cornersBL = visibleMarkers.get(idBL)!;

      const getCenter = (corners: Point2D[]): Point2D => ({
        x: (corners[0].x + corners[1].x + corners[2].x + corners[3].x) * 0.25,
        y: (corners[0].y + corners[1].y + corners[2].y + corners[3].y) * 0.25,
      });

      const cTL_cam = getCenter(cornersTL);
      const cTR_cam = getCenter(cornersTR);
      const cBR_cam = getCenter(cornersBR);
      const cBL_cam = getCenter(cornersBL);

      const cOffset = this.config.paperEdgePaddingMm + this.config.markerSizeMm * 0.5;
      const cTL_mm: Point2D = { x: cOffset, y: cOffset };
      const cTR_mm: Point2D = { x: this.config.sheetWidthMm - cOffset, y: cOffset };
      const cBR_mm: Point2D = {
        x: this.config.sheetWidthMm - cOffset,
        y: this.config.sheetHeightMm - cOffset,
      };
      const cBL_mm: Point2D = { x: cOffset, y: this.config.sheetHeightMm - cOffset };

      const cameraPts: Point2D[] = [cTL_cam, cTR_cam, cBR_cam, cBL_cam];
      const mmPts: Point2D[] = [cTL_mm, cTR_mm, cBR_mm, cBL_mm];

      const cameraToMmH = findHomography4Pts(cameraPts, mmPts);
      const mmToCameraH = cameraToMmH ? invert3x3(cameraToMmH) : null;

      if (!cameraToMmH || !mmToCameraH) {
        if (this.lastPosition) {
          this.lastTrackingResult = this.createHeldResult(visibleCount);
          return this.lastTrackingResult;
        }
        this.lastTrackingResult = this.createEmptyResult();
        return this.lastTrackingResult;
      }

      const sx = this.config.canvasWidth / this.config.sheetWidthMm;
      const sy = this.config.canvasHeight / this.config.sheetHeightMm;

      const S = new Float64Array([sx, 0, 0, 0, sy, 0, 0, 0, 1]);
      const S_inv = new Float64Array([1 / sx, 0, 0, 0, 1 / sy, 0, 0, 0, 1]);

      const cameraToCanvasH = multiply3x3(S, cameraToMmH);
      const canvasToCameraH = multiply3x3(mmToCameraH, S_inv);

      const outerPaperMm: Point2D[] = [
        { x: 0, y: 0 },
        { x: this.config.sheetWidthMm, y: 0 },
        { x: this.config.sheetWidthMm, y: this.config.sheetHeightMm },
        { x: 0, y: this.config.sheetHeightMm },
      ];

      const quadPx = outerPaperMm.map((p) => projectPoint(mmToCameraH, p.x, p.y));

      let minX = Infinity,
        minY = Infinity,
        maxX = -Infinity,
        maxY = -Infinity;
      for (const p of quadPx) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      }
      const bboxPx: [number, number, number, number] = [Math.floor(minX), Math.floor(minY), Math.ceil(maxX - minX), Math.ceil(maxY - minY)];

      let reprojErrSum = 0;
      for (let i = 0; i < 4; i++) {
        const proj = projectPoint(mmToCameraH, mmPts[i].x, mmPts[i].y);
        const dx = proj.x - cameraPts[i].x;
        const dy = proj.y - cameraPts[i].y;
        reprojErrSum += Math.hypot(dx, dy);
      }
      const reprojErrPx = reprojErrSum / 4.0;

      const markerExclusionBoxes: Rect[] = [];
      const markerCornersList = [cornersTL, cornersTR, cornersBR, cornersBL];
      for (const corners of markerCornersList) {
        let bMinX = Infinity,
          bMinY = Infinity,
          bMaxX = -Infinity,
          bMaxY = -Infinity;

        for (const c of corners) {
          const cp = projectPoint(cameraToCanvasH, c.x, c.y);
          if (cp.x < bMinX) bMinX = cp.x;
          if (cp.x > bMaxX) bMaxX = cp.x;
          if (cp.y < bMinY) bMinY = cp.y;
          if (cp.y > bMaxY) bMaxY = cp.y;
        }

        const bw = bMaxX - bMinX;
        const bh = bMaxY - bMinY;
        const padX = this.config.markerMarginRatio * bw;
        const padY = this.config.markerMarginRatio * bh;

        markerExclusionBoxes.push({
          x: Math.max(0, bMinX - padX),
          y: Math.max(0, bMinY - padY),
          width: Math.min(this.config.canvasWidth, bMaxX + padX) - Math.max(0, bMinX - padX),
          height: Math.min(this.config.canvasHeight, bMaxY + padY) - Math.max(0, bMinY - padY),
        });
      }

      const boundary: PaperBoundary = {
        quadPx,
        bboxPx,
        isLocked: reprojErrPx < 4.0,
        reprojErrPx,
      };

      this.lastPosition = {
        boundary,
        cameraToMmH,
        mmToCameraH,
        cameraToCanvasH,
        canvasToCameraH,
        markerExclusionBoxes,
      };

      this.lastTrackingResult = {
        isTracking: true,
        isHeld: false,
        visibleCount: 4,
        boundary,
        cameraToMmH,
        mmToCameraH,
        cameraToCanvasH,
        canvasToCameraH,
        markerExclusionBoxes,
        sheetMm: [this.config.sheetWidthMm, this.config.sheetHeightMm],
        canvasSize: [this.config.canvasWidth, this.config.canvasHeight],
        pxPerMm: [sx, sy],
      };

      return this.lastTrackingResult;
    }

    // Up to 3 markers are occluded (visibleCount is 1, 2, or 3)
    // If tracking was already established, use the last paper position
    if (this.lastPosition) {
      this.lastTrackingResult = this.createHeldResult(visibleCount);
      return this.lastTrackingResult;
    }

    // No prior paper position established yet
    this.lastTrackingResult = this.createEmptyResult();
    this.lastTrackingResult.visibleCount = visibleCount;
    return this.lastTrackingResult;
  }

  private createHeldResult(visibleCount: number): PaperTrackingResult {
    const sx = this.config.canvasWidth / this.config.sheetWidthMm;
    const sy = this.config.canvasHeight / this.config.sheetHeightMm;
    const pos = this.lastPosition!;

    return {
      isTracking: true,
      isHeld: true,
      visibleCount,
      boundary: pos.boundary,
      cameraToMmH: pos.cameraToMmH,
      mmToCameraH: pos.mmToCameraH,
      cameraToCanvasH: pos.cameraToCanvasH,
      canvasToCameraH: pos.canvasToCameraH,
      markerExclusionBoxes: pos.markerExclusionBoxes,
      sheetMm: [this.config.sheetWidthMm, this.config.sheetHeightMm],
      canvasSize: [this.config.canvasWidth, this.config.canvasHeight],
      pxPerMm: [sx, sy],
    };
  }

  /**
   * Tests whether a camera pixel coordinate is physically within the paper sheet.
   */
  public isPointInsidePaper(cameraPt: Point2D): boolean {
    if (!this.lastTrackingResult.isTracking || !this.lastTrackingResult.cameraToMmH) {
      return false;
    }
    const mm = projectPoint(this.lastTrackingResult.cameraToMmH, cameraPt.x, cameraPt.y);
    return mm.x >= 0 && mm.x <= this.config.sheetWidthMm && mm.y >= 0 && mm.y <= this.config.sheetHeightMm;
  }

  /**
   * Maps a camera pixel coordinate to paper millimeters.
   */
  public cameraToMm(cameraPt: Point2D): Point2D | null {
    if (!this.lastTrackingResult.isTracking || !this.lastTrackingResult.cameraToMmH) {
      return null;
    }
    return projectPoint(this.lastTrackingResult.cameraToMmH, cameraPt.x, cameraPt.y);
  }

  /**
   * Maps paper millimeters to camera pixel coordinate.
   */
  public mmToCamera(mmPt: Point2D): Point2D | null {
    if (!this.lastTrackingResult.isTracking || !this.lastTrackingResult.mmToCameraH) {
      return null;
    }
    return projectPoint(this.lastTrackingResult.mmToCameraH, mmPt.x, mmPt.y);
  }

  /**
   * Maps a camera pixel coordinate to canvas pixel coordinate.
   */
  public cameraToCanvas(cameraPt: Point2D): Point2D | null {
    if (!this.lastTrackingResult.isTracking || !this.lastTrackingResult.cameraToCanvasH) {
      return null;
    }
    return projectPoint(this.lastTrackingResult.cameraToCanvasH, cameraPt.x, cameraPt.y);
  }

  /**
   * Maps a canvas pixel coordinate to camera pixel coordinate.
   */
  public canvasToCamera(canvasPt: Point2D): Point2D | null {
    if (!this.lastTrackingResult.isTracking || !this.lastTrackingResult.canvasToCameraH) {
      return null;
    }
    return projectPoint(this.lastTrackingResult.canvasToCameraH, canvasPt.x, canvasPt.y);
  }

  private createEmptyResult(): PaperTrackingResult {
    const sx = this.config.canvasWidth / this.config.sheetWidthMm;
    const sy = this.config.canvasHeight / this.config.sheetHeightMm;
    return {
      isTracking: false,
      isHeld: false,
      visibleCount: 0,
      boundary: null,
      cameraToMmH: null,
      mmToCameraH: null,
      cameraToCanvasH: null,
      canvasToCameraH: null,
      markerExclusionBoxes: [],
      sheetMm: [this.config.sheetWidthMm, this.config.sheetHeightMm],
      canvasSize: [this.config.canvasWidth, this.config.canvasHeight],
      pxPerMm: [sx, sy],
    };
  }
}
