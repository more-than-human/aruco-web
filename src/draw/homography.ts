import type { IPoint, Marker } from "../lib/aruco.ts";
import { TrackedMarker } from "../lib/tracker.ts";
import type { IRect, PaperWarpModel } from "./types.ts";

/**
 * Stage 1: PaperGeometry
 * Given the 4 tracked markers (id = 0, 1, 2, 3), computes the perspective
 * transformation between screen canvas (width x height, assumed portrait A4)
 * and camera image, accounting for 1.0xD outward paper margin from marker centers
 * and 0.1xD marker exclusion zones.
 *
 * Markers are assumed placed on the paper with portrait orientation:
 * - Marker 0: Top-Left
 * - Marker 1: Top-Right
 * - Marker 2: Bottom-Left
 * - Marker 3: Bottom-Right
 */
export class PaperGeometry {
  private screenWidth: number;
  private screenHeight: number;
  private outwardPaddingRatio: number;
  private markerMarginRatio: number;

  constructor(screenWidth: number, screenHeight: number, outwardPaddingRatio = 1.0, markerMarginRatio = 0.1) {
    this.screenWidth = screenWidth;
    this.screenHeight = screenHeight;
    this.outwardPaddingRatio = outwardPaddingRatio;
    this.markerMarginRatio = markerMarginRatio;
  }

  public computeWarpModel(markers: (Marker | TrackedMarker)[]): PaperWarpModel | null {
    // 1. Validate Markers: Ensure markers 0, 1, 2, 3 are all present and none are coasting
    const m0 = markers.find((m) => m.id === 0);
    const m1 = markers.find((m) => m.id === 1);
    const m2 = markers.find((m) => m.id === 2);
    const m3 = markers.find((m) => m.id === 3);

    if (!m0 || !m1 || !m2 || !m3) {
      return null;
    }

    const markerList = [m0, m1, m2, m3];
    for (const m of markerList) {
      if (m instanceof TrackedMarker && m.isPredicted) {
        return null;
      }
    }

    // Marker Centers:
    // Averaging 4 corners gives a much more stable position than individual jittery corners
    const getCenter = (m: Marker | TrackedMarker): IPoint => {
      const c = m.corners;
      return {
        x: (c[0].x + c[1].x + c[2].x + c[3].x) * 0.25,
        y: (c[0].y + c[1].y + c[2].y + c[3].y) * 0.25,
      };
    };

    const c0 = getCenter(m0); // M0 Center (Top-Left)
    const c1 = getCenter(m1); // M1 Center (Top-Right)
    const c2 = getCenter(m2); // M2 Center (Bottom-Left)
    const c3 = getCenter(m3); // M3 Center (Bottom-Right)

    const dist = (a: IPoint, b: IPoint) => Math.hypot(a.x - b.x, a.y - b.y);

    // 2. Estimate Normalized Marker Size (D_u, D_v) relative to center-to-center spans
    const spanU0 = dist(c1, c0);
    const spanU1 = dist(c3, c2);
    const spanV0 = dist(c2, c0);
    const spanV1 = dist(c3, c1);

    if (spanU0 === 0 || spanU1 === 0 || spanV0 === 0 || spanV1 === 0) {
      return null;
    }

    const r_u = 0.5 * (dist(m0.corners[1], m0.corners[0]) / spanU0 + dist(m2.corners[2], m2.corners[3]) / spanU1);
    const r_v = 0.5 * (dist(m0.corners[3], m0.corners[0]) / spanV0 + dist(m1.corners[2], m1.corners[1]) / spanV1);

    // 3. Map Marker Centers directly to Inset Screen Coordinates
    // The screen represents [0, screenWidth] x [0, screenHeight].
    // Each side expands outward from the marker center by outwardPaddingRatio * D.
    // Total screen span = centerSpan + 2 * outwardPaddingRatio * D = centerSpan * (1 + 2 * outwardPaddingRatio * r).
    const totalExpansionU = 2 * this.outwardPaddingRatio * r_u;
    const totalExpansionV = 2 * this.outwardPaddingRatio * r_v;

    const u_min = this.screenWidth * ((this.outwardPaddingRatio * r_u) / (1 + totalExpansionU));
    const u_max = this.screenWidth - u_min;
    const v_min = this.screenHeight * ((this.outwardPaddingRatio * r_v) / (1 + totalExpansionV));
    const v_max = this.screenHeight - v_min;

    // DLT Homography screenToCameraH:
    // (u_min, v_min) -> c0 (M0 Center)
    // (u_max, v_min) -> c1 (M1 Center)
    // (u_max, v_max) -> c3 (M3 Center)
    // (u_min, v_max) -> c2 (M2 Center)
    const srcPts: [number, number][] = [
      [u_min, v_min],
      [u_max, v_min],
      [u_max, v_max],
      [u_min, v_max],
    ];
    const dstPts: [number, number][] = [
      [c0.x, c0.y],
      [c1.x, c1.y],
      [c3.x, c3.y],
      [c2.x, c2.y],
    ];

    const screenToCameraH = PaperGeometry.findHomography(srcPts, dstPts);
    if (!screenToCameraH) return null;

    const cameraToScreenH = PaperGeometry.invert3x3(screenToCameraH);
    if (!cameraToScreenH) return null;

    // 4. Compute Marker Exclusion Boxes (+0.1 * D):
    // Project all 4 corners of each marker into screen space, take bounding box [minX, minY, maxX, maxY],
    // and expand each side outward by markerMarginRatio * (boxWidth, boxHeight).
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

    // 4 outer paper corners in camera space (for debug overlay)
    // Project the full screen corners (0,0), (W,0), (W,H), (0,H) back to camera space
    const [cTL_x, cTL_y] = PaperGeometry.projectPoint(screenToCameraH, 0, 0);
    const [cTR_x, cTR_y] = PaperGeometry.projectPoint(screenToCameraH, this.screenWidth, 0);
    const [cBR_x, cBR_y] = PaperGeometry.projectPoint(screenToCameraH, this.screenWidth, this.screenHeight);
    const [cBL_x, cBL_y] = PaperGeometry.projectPoint(screenToCameraH, 0, this.screenHeight);

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
   * Standard DLT via 8x8 linear system with H[8] = 1.
   */
  public static findHomography(src: [number, number][], dst: [number, number][]): Float64Array | null {
    // 8 equations for 8 unknowns (h00, h01, h02, h10, h11, h12, h20, h21) with h22 = 1.
    // For each point i:
    // xi' = (h00*xi + h01*yi + h02) / (h20*xi + h21*yi + 1)
    // yi' = (h10*xi + h11*yi + h12) / (h20*xi + h21*yi + 1)
    // =>
    // xi*h00 + yi*h01 + h02 + 0 + 0 + 0 - xi*xi'*h20 - yi*xi'*h21 = xi'
    // 0 + 0 + 0 + xi*h10 + yi*h11 + h12 - xi*yi'*h20 - yi*yi'*h21 = yi'

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

    // Solve 8x8 linear system A * h = b via Gaussian elimination with partial pivoting
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
}
