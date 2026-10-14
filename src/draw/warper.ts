import type { PaperWarpModel, RectifiedFrame } from "./types.ts";

/**
 * Stage 2: PerspectiveWarper
 * Inverse-warps camera ImageData into a flat, perspective-corrected
 * grayscale buffer (gray) and populates validMask.
 */
export class PerspectiveWarper {
  private width: number;
  private height: number;
  private gray: Uint8Array;
  private validMask: Uint8Array;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.gray = new Uint8Array(width * height);
    this.validMask = new Uint8Array(width * height);
  }

  public warp(cameraImage: ImageData, model: PaperWarpModel): RectifiedFrame {
    const W = this.width;
    const H = this.height;
    const camW = cameraImage.width;
    const camH = cameraImage.height;
    const camData = cameraImage.data;
    const Hmat = model.screenToCameraH;

    // 1. Fill validMask with 1
    this.validMask.fill(1);

    // 2. For each of the markerExclusionBoxes, set validMask = 0 inside
    for (const box of model.markerExclusionBoxes) {
      const startX = Math.max(0, Math.floor(box.minX));
      const endX = Math.min(W - 1, Math.ceil(box.maxX));
      const startY = Math.max(0, Math.floor(box.minY));
      const endY = Math.min(H - 1, Math.ceil(box.maxY));

      for (let v = startY; v <= endY; v++) {
        const rowOffset = v * W;
        for (let u = startX; u <= endX; u++) {
          this.validMask[rowOffset + u] = 0;
        }
      }
    }

    // 3. Loop over every screen pixel (u, v):
    // Project (u, v) to camera coordinates (x, y) using screenToCameraH.
    // If (x, y) outside camera bounds -> validMask = 0, gray = 255.
    // Otherwise sample camera RGB and compute Y = (77*R + 150*G + 29*B) >> 8.
    const h00 = Hmat[0],
      h01 = Hmat[1],
      h02 = Hmat[2];
    const h10 = Hmat[3],
      h11 = Hmat[4],
      h12 = Hmat[5];
    const h20 = Hmat[6],
      h21 = Hmat[7],
      h22 = Hmat[8];

    let idx = 0;
    for (let v = 0; v < H; v++) {
      for (let u = 0; u < W; u++, idx++) {
        const w = h20 * u + h21 * v + h22;
        if (Math.abs(w) < 1e-12) {
          this.validMask[idx] = 0;
          this.gray[idx] = 255;
          continue;
        }

        const invW = 1.0 / w;
        const cx = (h00 * u + h01 * v + h02) * invW;
        const cy = (h10 * u + h11 * v + h12) * invW;

        const ix = Math.round(cx);
        const iy = Math.round(cy);

        if (ix < 0 || ix >= camW || iy < 0 || iy >= camH) {
          this.validMask[idx] = 0;
          this.gray[idx] = 255;
        } else {
          const camIdx = (iy * camW + ix) << 2;
          const r = camData[camIdx];
          const g = camData[camIdx + 1];
          const b = camData[camIdx + 2];
          this.gray[idx] = (77 * r + 150 * g + 29 * b) >> 8;
        }
      }
    }

    return {
      width: W,
      height: H,
      gray: this.gray,
      validMask: this.validMask,
    };
  }
}
