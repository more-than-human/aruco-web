import type { IPoint, Marker } from "../aruco-tracking/aruco.ts";
import { TrackedMarker } from "../aruco-tracking/tracker.ts";
import {
  DIGITIZING_ZONE_EXTEND_MM,
  LETTER_LANDSCAPE_HEIGHT_MM,
  LETTER_LANDSCAPE_WIDTH_MM,
  MARKER_SIZE_MM,
  PAPER_EDGE_PADDING_MM,
  type IRect,
  type PaperWarpModel,
} from "./types.ts";

/**
 * Computes perspective transformations and digitizing geometry
 * for a US Letter sheet tracked by 4 corner markers (TL: 1, TR: 2, BR: 3, BL: 4).
 */
export class PaperGeometry {
  private screenWidth: number;
  private screenHeight: number;
  private markerMarginRatio: number;
  private lastKnownMarkers = new Map<number, Marker | TrackedMarker>();

  public static readonly PAPER_WIDTH_MM = LETTER_LANDSCAPE_WIDTH_MM;
  public static readonly PAPER_HEIGHT_MM = LETTER_LANDSCAPE_HEIGHT_MM;
  public static readonly MARKER_SIZE_MM = MARKER_SIZE_MM;
  public static readonly PAPER_EDGE_PADDING_MM = PAPER_EDGE_PADDING_MM;
  public static readonly DIGITIZING_ZONE_EXTEND_MM = DIGITIZING_ZONE_EXTEND_MM;

  public static readonly MARKER_CENTER_OFFSET_MM = PAPER_EDGE_PADDING_MM + MARKER_SIZE_MM * 0.5;

  public static readonly CENTER_SPAN_U_MM = LETTER_LANDSCAPE_WIDTH_MM - 2 * PaperGeometry.MARKER_CENTER_OFFSET_MM;
  public static readonly CENTER_SPAN_V_MM = LETTER_LANDSCAPE_HEIGHT_MM - 2 * PaperGeometry.MARKER_CENTER_OFFSET_MM;

  public static readonly DIGITIZING_EXTEND_MM = DIGITIZING_ZONE_EXTEND_MM;
  public static readonly DIGITIZING_WIDTH_MM = PaperGeometry.CENTER_SPAN_U_MM + 2 * PaperGeometry.DIGITIZING_EXTEND_MM;
  public static readonly DIGITIZING_HEIGHT_MM = PaperGeometry.CENTER_SPAN_V_MM + 2 * PaperGeometry.DIGITIZING_EXTEND_MM;

  constructor(screenWidth: number, screenHeight: number, _outwardPaddingRatio?: number, markerMarginRatio = 0.2) {
    this.screenWidth = screenWidth;
    this.screenHeight = screenHeight;
    this.markerMarginRatio = markerMarginRatio;
  }

  public computeWarpModel(markers: (Marker | TrackedMarker)[]): PaperWarpModel | null {
    // Update last known locations with currently detected/available markers (IDs 0..3)
    let visibleMarkerCount = 0;
    for (const m of markers) {
      if (m.id >= 0 && m.id <= 3 && m.corners && m.corners.length === 4) {
        this.lastKnownMarkers.set(m.id, m);
        visibleMarkerCount++;
      }
    }

    // Paper is only tracked while at least one corner marker is currently visible.
    // With 0 markers present, treat the paper as absent (do not coast on stale data).
    if (visibleMarkerCount === 0) {
      this.lastKnownMarkers.clear();
      return null;
    }

    const m1 = this.lastKnownMarkers.get(0);
    const m2 = this.lastKnownMarkers.get(1);
    const m3 = this.lastKnownMarkers.get(2);
    const m4 = this.lastKnownMarkers.get(3);

    if (!m1 || !m2 || !m3 || !m4) {
      return null;
    }

    const markerList = [m1, m2, m3, m4];

    const getCenter = (m: Marker | TrackedMarker): IPoint => {
      const c = m.corners;
      return {
        x: (c[0].x + c[1].x + c[2].x + c[3].x) * 0.25,
        y: (c[0].y + c[1].y + c[2].y + c[3].y) * 0.25,
      };
    };

    const cTL = getCenter(m1);
    const cTR = getCenter(m2);
    const cBR = getCenter(m3);
    const cBL = getCenter(m4);

    const u_min = this.screenWidth * (PaperGeometry.DIGITIZING_EXTEND_MM / PaperGeometry.DIGITIZING_WIDTH_MM);
    const u_max = this.screenWidth - u_min;
    const v_min = this.screenHeight * (PaperGeometry.DIGITIZING_EXTEND_MM / PaperGeometry.DIGITIZING_HEIGHT_MM);
    const v_max = this.screenHeight - v_min;

    const srcPts: [number, number][] = [
      [u_min, v_min],
      [u_max, v_min],
      [u_max, v_max],
      [u_min, v_max],
    ];
    const dstPts: [number, number][] = [
      [cTL.x, cTL.y],
      [cTR.x, cTR.y],
      [cBR.x, cBR.y],
      [cBL.x, cBL.y],
    ];

    const screenToCameraH = PaperGeometry.findHomography(srcPts, dstPts);
    if (!screenToCameraH) return null;

    const cameraToScreenH = PaperGeometry.invert3x3(screenToCameraH);
    if (!cameraToScreenH) return null;

    const markerExclusionBoxes: IRect[] = [];
    for (const m of markerList) {
      let bMinX = Infinity;
      let bMinY = Infinity;
      let bMaxX = -Infinity;
      let bMaxY = -Infinity;

      for (const corner of m.corners) {
        const [su, sv] = PaperGeometry.projectPoint(cameraToScreenH, corner.x, corner.y);
        if (su < bMinX) bMinX = su;
        if (su > bMaxX) bMaxX = su;
        if (sv < bMinY) bMinY = sv;
        if (sv > bMaxY) bMaxY = sv;
      }

      const boxW = bMaxX - bMinX;
      const boxH = bMaxY - bMinY;
      const padX = this.markerMarginRatio * boxW;
      const padY = this.markerMarginRatio * boxH;

      markerExclusionBoxes.push({
        minX: Math.max(0, bMinX - padX),
        minY: Math.max(0, bMinY - padY),
        maxX: Math.min(this.screenWidth, bMaxX + padX),
        maxY: Math.min(this.screenHeight, bMaxY + padY),
      });
    }

    const paperMarginScreenX =
      this.screenWidth * ((PaperGeometry.MARKER_CENTER_OFFSET_MM - PaperGeometry.DIGITIZING_EXTEND_MM) / PaperGeometry.DIGITIZING_WIDTH_MM);
    const paperMarginScreenY =
      this.screenHeight * ((PaperGeometry.MARKER_CENTER_OFFSET_MM - PaperGeometry.DIGITIZING_EXTEND_MM) / PaperGeometry.DIGITIZING_HEIGHT_MM);

    const paperLeft = -paperMarginScreenX;
    const paperRight = this.screenWidth + paperMarginScreenX;
    const paperTop = -paperMarginScreenY;
    const paperBottom = this.screenHeight + paperMarginScreenY;

    const [cTL_x, cTL_y] = PaperGeometry.projectPoint(screenToCameraH, paperLeft, paperTop);
    const [cTR_x, cTR_y] = PaperGeometry.projectPoint(screenToCameraH, paperRight, paperTop);
    const [cBR_x, cBR_y] = PaperGeometry.projectPoint(screenToCameraH, paperRight, paperBottom);
    const [cBL_x, cBL_y] = PaperGeometry.projectPoint(screenToCameraH, paperLeft, paperBottom);

    const cameraPaperQuad: IPoint[] = [
      { x: cTL_x, y: cTL_y },
      { x: cTR_x, y: cTR_y },
      { x: cBR_x, y: cBR_y },
      { x: cBL_x, y: cBL_y },
    ];

    return {
      screenToCameraH,
      cameraToScreenH,
      markerExclusionBoxes,
      cameraPaperQuad,
    };
  }

  /**
   * Solves for 3x3 homography H such that dst = H * src for 4 point correspondences.
   */
  public static findHomography(src: [number, number][], dst: [number, number][]): Float64Array | null {
    const A: number[][] = [];
    const b: number[] = [];

    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i];
      const [u, v] = dst[i];
      A.push([x, y, 1, 0, 0, 0, -x * u, -y * u]);
      b.push(u);
      A.push([0, 0, 0, x, y, 1, -x * v, -y * v]);
      b.push(v);
    }

    const h = PaperGeometry.solveLinearSystem(A, b);
    if (!h) return null;

    return new Float64Array([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]);
  }

  public static solveLinearSystem(A: number[][], b: number[]): number[] | null {
    const n = 8;
    const M: number[][] = Array.from({ length: n }, (_, i) => [...A[i], b[i]]);

    for (let col = 0; col < n; col++) {
      let maxRow = col;
      let maxVal = Math.abs(M[col][col]);
      for (let row = col + 1; row < n; row++) {
        const val = Math.abs(M[row][col]);
        if (val > maxVal) {
          maxVal = val;
          maxRow = row;
        }
      }

      if (maxVal < 1e-10) {
        return null;
      }

      if (maxRow !== col) {
        const tmp = M[col];
        M[col] = M[maxRow];
        M[maxRow] = tmp;
      }

      const pivot = M[col][col];
      for (let c = col; c <= n; c++) {
        M[col][c] /= pivot;
      }

      for (let row = 0; row < n; row++) {
        if (row !== col) {
          const factor = M[row][col];
          if (Math.abs(factor) > 1e-12) {
            for (let c = col; c <= n; c++) {
              M[row][c] -= factor * M[col][c];
            }
          }
        }
      }
    }

    return M.map((row) => row[n]);
  }

  public static invert3x3(m: Float64Array): Float64Array | null {
    const a00 = m[0],
      a01 = m[1],
      a02 = m[2];
    const a10 = m[3],
      a11 = m[4],
      a12 = m[5];
    const a20 = m[6],
      a21 = m[7],
      a22 = m[8];

    const c00 = a11 * a22 - a12 * a21;
    const c01 = -(a10 * a22 - a12 * a20);
    const c02 = a10 * a21 - a11 * a20;

    const det = a00 * c00 + a01 * c01 + a02 * c02;
    if (Math.abs(det) < 1e-12) return null;

    const invDet = 1.0 / det;

    const c10 = -(a01 * a22 - a02 * a21);
    const c11 = a00 * a22 - a02 * a20;
    const c12 = -(a00 * a21 - a01 * a20);

    const c20 = a01 * a12 - a02 * a11;
    const c21 = -(a00 * a12 - a02 * a10);
    const c22 = a00 * a11 - a01 * a10;

    return new Float64Array([c00 * invDet, c10 * invDet, c20 * invDet, c01 * invDet, c11 * invDet, c21 * invDet, c02 * invDet, c12 * invDet, c22 * invDet]);
  }

  public static projectPoint(H: Float64Array, x: number, y: number): [number, number] {
    const w = H[6] * x + H[7] * y + H[8];
    const invW = w !== 0 ? 1.0 / w : 1.0;
    return [(H[0] * x + H[1] * y + H[2]) * invW, (H[3] * x + H[4] * y + H[5]) * invW];
  }

  public reset(): void {
    this.lastKnownMarkers.clear();
  }
}
