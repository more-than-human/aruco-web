import type { IPoint, Marker } from "../aruco-tracking/aruco.ts";
import type { TrackedMarker } from "../aruco-tracking/tracker.ts";
import { findHomography4Pts, invert3x3, multiply3x3, projectPoint } from "./homography.ts";
import {
  DEFAULT_SHEET_CONFIG,
  type PaperBoundary,
  type PaperSheetConfig,
  type PaperTrackingResult,
  type Point2D,
  type Rect,
} from "./types.ts";

interface MarkerRecord {
  id: number;
  corners: Point2D[];
  center: Point2D;
  velocity: Point2D;
  lastSeenMs: number;
}

export class PaperTracker {
  private config: PaperSheetConfig;
  private markersMap = new Map<number, MarkerRecord>();
  private lastTrackingResult: PaperTrackingResult;
  private lastUpdateTimeMs = 0;

  constructor(config?: Partial<PaperSheetConfig>) {
    this.config = { ...DEFAULT_SHEET_CONFIG, ...config };
    this.lastTrackingResult = this.createEmptyResult();
  }

  public reset(): void {
    this.markersMap.clear();
    this.lastTrackingResult = this.createEmptyResult();
    this.lastUpdateTimeMs = 0;
  }

  /**
   * Updates tracking state using detected markers from the current camera frame.
   */
  public update(
    markers: (Marker | TrackedMarker)[],
    timestampMs: number = performance.now(),
  ): PaperTrackingResult {
    const dt = this.lastUpdateTimeMs > 0 ? (timestampMs - this.lastUpdateTimeMs) / 1000 : 0.033;
    this.lastUpdateTimeMs = timestampMs;

    let visibleInFrame = 0;

    // Process all currently detected markers matching our 4 target IDs
    for (const m of markers) {
      if (!this.config.markerIds.includes(m.id)) continue;
      if (!m.corners || m.corners.length < 4) continue;

      visibleInFrame++;
      const corners: Point2D[] = m.corners.map((c: IPoint) => ({ x: c.x, y: c.y }));
      const center: Point2D = {
        x: (corners[0].x + corners[1].x + corners[2].x + corners[3].x) * 0.25,
        y: (corners[0].y + corners[1].y + corners[2].y + corners[3].y) * 0.25,
      };

      const prev = this.markersMap.get(m.id);
      let velocity: Point2D = { x: 0, y: 0 };
      if (prev && dt > 0 && dt < 0.5) {
        velocity = {
          x: (center.x - prev.center.x) / dt,
          y: (center.y - prev.center.y) / dt,
        };
      }

      this.markersMap.set(m.id, {
        id: m.id,
        corners,
        center,
        velocity,
        lastSeenMs: timestampMs,
      });
    }

    // If zero markers are visible and hold time has elapsed, paper is lost
    if (visibleInFrame === 0) {
      let anyValid = false;
      for (const rec of this.markersMap.values()) {
        if (timestampMs - rec.lastSeenMs <= this.config.holdDurationMs) {
          anyValid = true;
          break;
        }
      }
      if (!anyValid) {
        this.lastTrackingResult = this.createEmptyResult();
        return this.lastTrackingResult;
      }
    }

    // Verify all 4 required corner markers are available (either live or within hold window)
    const [idTL, idTR, idBR, idBL] = this.config.markerIds;
    const recTL = this.markersMap.get(idTL);
    const recTR = this.markersMap.get(idTR);
    const recBR = this.markersMap.get(idBR);
    const recBL = this.markersMap.get(idBL);

    if (!recTL || !recTR || !recBR || !recBL) {
      this.lastTrackingResult = this.createEmptyResult();
      return this.lastTrackingResult;
    }

    // Check if any marker has exceeded hold time
    const maxAge = Math.max(
      timestampMs - recTL.lastSeenMs,
      timestampMs - recTR.lastSeenMs,
      timestampMs - recBR.lastSeenMs,
      timestampMs - recBL.lastSeenMs,
    );

    if (maxAge > this.config.holdDurationMs) {
      this.lastTrackingResult = this.createEmptyResult();
      return this.lastTrackingResult;
    }

    const isHeld = visibleInFrame < 4;

    // Estimate centers (applying linear velocity extrapolation if held)
    const getEffectiveCenter = (rec: MarkerRecord): Point2D => {
      const ageS = (timestampMs - rec.lastSeenMs) / 1000;
      if (ageS > 0.01) {
        return {
          x: rec.center.x + rec.velocity.x * ageS,
          y: rec.center.y + rec.velocity.y * ageS,
        };
      }
      return rec.center;
    };

    const cTL_cam = getEffectiveCenter(recTL);
    const cTR_cam = getEffectiveCenter(recTR);
    const cBR_cam = getEffectiveCenter(recBR);
    const cBL_cam = getEffectiveCenter(recBL);

    // Marker centers in paper physical millimeters
    const cOffset = this.config.paperEdgePaddingMm + this.config.markerSizeMm * 0.5;
    const cTL_mm: Point2D = { x: cOffset, y: cOffset };
    const cTR_mm: Point2D = { x: this.config.sheetWidthMm - cOffset, y: cOffset };
    const cBR_mm: Point2D = {
      x: this.config.sheetWidthMm - cOffset,
      y: this.config.sheetHeightMm - cOffset,
    };
    const cBL_mm: Point2D = { x: cOffset, y: this.config.sheetHeightMm - cOffset };

    // Compute homography Camera Pixels -> Paper Millimeters
    const cameraPts: Point2D[] = [cTL_cam, cTR_cam, cBR_cam, cBL_cam];
    const mmPts: Point2D[] = [cTL_mm, cTR_mm, cBR_mm, cBL_mm];

    const cameraToMmH = findHomography4Pts(cameraPts, mmPts);
    if (!cameraToMmH) {
      this.lastTrackingResult = this.createEmptyResult();
      return this.lastTrackingResult;
    }

    const mmToCameraH = invert3x3(cameraToMmH);
    if (!mmToCameraH) {
      this.lastTrackingResult = this.createEmptyResult();
      return this.lastTrackingResult;
    }

    // Scale matrix between Paper mm and Canvas pixels
    const sx = this.config.canvasWidth / this.config.sheetWidthMm;
    const sy = this.config.canvasHeight / this.config.sheetHeightMm;

    // S = diag(sx, sy, 1)
    const S = new Float64Array([sx, 0, 0, 0, sy, 0, 0, 0, 1]);
    const S_inv = new Float64Array([1 / sx, 0, 0, 0, 1 / sy, 0, 0, 0, 1]);

    const cameraToCanvasH = multiply3x3(S, cameraToMmH);
    const canvasToCameraH = multiply3x3(mmToCameraH, S_inv);

    // Project physical sheet outer corners (0,0), (W,0), (W,H), (0,H) to camera coordinates
    const outerPaperMm: Point2D[] = [
      { x: 0, y: 0 },
      { x: this.config.sheetWidthMm, y: 0 },
      { x: this.config.sheetWidthMm, y: this.config.sheetHeightMm },
      { x: 0, y: this.config.sheetHeightMm },
    ];

    const quadPx = outerPaperMm.map((p) => projectPoint(mmToCameraH, p.x, p.y));

    // Calculate axis-aligned bounding box of projected paper in camera pixels
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
    const bboxPx: [number, number, number, number] = [
      Math.floor(minX),
      Math.floor(minY),
      Math.ceil(maxX - minX),
      Math.ceil(maxY - minY),
    ];

    // Compute reprojection error of the 4 marker centers
    let reprojErrSum = 0;
    for (let i = 0; i < 4; i++) {
      const proj = projectPoint(mmToCameraH, mmPts[i].x, mmPts[i].y);
      const dx = proj.x - cameraPts[i].x;
      const dy = proj.y - cameraPts[i].y;
      reprojErrSum += Math.hypot(dx, dy);
    }
    const reprojErrPx = reprojErrSum / 4.0;

    // Compute marker exclusion boxes in canvas coordinates
    const markerExclusionBoxes: Rect[] = [];
    const markerRecords = [recTL, recTR, recBR, recBL];
    for (const rec of markerRecords) {
      let bMinX = Infinity,
        bMinY = Infinity,
        bMaxX = -Infinity,
        bMaxY = -Infinity;

      for (const c of rec.corners) {
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
      isLocked: !isHeld && reprojErrPx < 4.0,
      reprojErrPx,
    };

    this.lastTrackingResult = {
      isTracking: true,
      isHeld,
      visibleCount: visibleInFrame,
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

  /**
   * Tests whether a camera pixel coordinate is physically within the paper sheet.
   */
  public isPointInsidePaper(cameraPt: Point2D): boolean {
    if (!this.lastTrackingResult.isTracking || !this.lastTrackingResult.cameraToMmH) {
      return false;
    }
    const mm = projectPoint(this.lastTrackingResult.cameraToMmH, cameraPt.x, cameraPt.y);
    return (
      mm.x >= 0 &&
      mm.x <= this.config.sheetWidthMm &&
      mm.y >= 0 &&
      mm.y <= this.config.sheetHeightMm
    );
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
