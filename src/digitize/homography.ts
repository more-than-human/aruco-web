import type { Point2D } from "./types.ts";

/**
 * Solves for 3x3 homography H such that dst = H * src for 4 point correspondences.
 * Points are given in [x, y] format.
 */
export function findHomography4Pts(src: Point2D[], dst: Point2D[]): Float64Array | null {
  if (src.length < 4 || dst.length < 4) return null;

  const A: number[][] = [];
  const b: number[] = [];

  for (let i = 0; i < 4; i++) {
    const { x, y } = src[i];
    const { x: u, y: v } = dst[i];
    A.push([x, y, 1, 0, 0, 0, -x * u, -y * u]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -x * v, -y * v]);
    b.push(v);
  }

  const h = solveLinearSystem8x8(A, b);
  if (!h) return null;

  return new Float64Array([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]);
}

/**
 * Solves Ax = b for an 8x8 linear system using Gaussian elimination with partial pivoting.
 */
function solveLinearSystem8x8(A: number[][], b: number[]): number[] | null {
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

/**
 * Robust 3x3 matrix inversion.
 */
export function invert3x3(m: Float64Array): Float64Array | null {
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

/**
 * Multiplies two 3x3 matrices (A @ B).
 */
export function multiply3x3(A: Float64Array, B: Float64Array): Float64Array {
  const out = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] = A[r * 3 + 0] * B[0 * 3 + c] + A[r * 3 + 1] * B[1 * 3 + c] + A[r * 3 + 2] * B[2 * 3 + c];
    }
  }
  return out;
}

/**
 * Projects a 2D point using a 3x3 homography matrix.
 */
export function projectPoint(H: Float64Array, x: number, y: number): Point2D {
  const w = H[6] * x + H[7] * y + H[8];
  const invW = Math.abs(w) > 1e-12 ? 1.0 / w : 1.0;
  return {
    x: (H[0] * x + H[1] * y + H[2]) * invW,
    y: (H[3] * x + H[4] * y + H[5]) * invW,
  };
}

/**
 * Projects multiple 2D points using a 3x3 homography matrix.
 */
export function projectPoints(H: Float64Array, pts: Point2D[]): Point2D[] {
  return pts.map((p) => projectPoint(H, p.x, p.y));
}

/**
 * Point-in-polygon containment test using ray-casting algorithm.
 */
export function isPointInPolygon(point: Point2D, polygon: Point2D[]): boolean {
  let inside = false;
  const n = polygon.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i].x,
      yi = polygon[i].y;
    const xj = polygon[j].x,
      yj = polygon[j].y;
    const intersect = yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}
