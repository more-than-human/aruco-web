import type { RectifiedFrame } from "./types.ts";

export interface TemporalFilterConfig {
  motionThreshold?: number; // default: 10 (gray levels)
  tileSize?: number; // default: 16 (px)
  requiredStableFrames?: number; // default: 10 (frames)
  blendAlpha?: number; // default: 0.3
}

/**
 * Stage 4: TemporalInkFilter
 * Rejects moving objects (hand, fingers, pen barrel, moving shadows) and
 * maintains a clean, persistent ImageData canvas of confirmed ink strokes.
 */
export class TemporalInkFilter {
  private width: number;
  private height: number;
  private motionThreshold: number;
  private tileSize: number;
  private requiredStableFrames: number;
  private blendAlpha: number;

  private prevGray: Uint8Array | null = null;
  private persistentInk: Float32Array;
  private stabilityCount: Uint16Array;
  private outputImageData: ImageData;

  private tilesX: number;
  private tilesY: number;
  private tileHasMotion: Uint8Array;
  private tileUnstable: Uint8Array;

  constructor(width: number, height: number, config?: TemporalFilterConfig) {
    this.width = width;
    this.height = height;
    this.motionThreshold = config?.motionThreshold ?? 10;
    this.tileSize = config?.tileSize ?? 16;
    this.requiredStableFrames = config?.requiredStableFrames ?? 10;
    this.blendAlpha = config?.blendAlpha ?? 0.3;

    const totalPixels = width * height;
    this.persistentInk = new Float32Array(totalPixels);
    this.stabilityCount = new Uint16Array(totalPixels);
    this.outputImageData = new ImageData(width, height);

    this.tilesX = Math.ceil(width / this.tileSize);
    this.tilesY = Math.ceil(height / this.tileSize);
    const numTiles = this.tilesX * this.tilesY;
    this.tileHasMotion = new Uint8Array(numTiles);
    this.tileUnstable = new Uint8Array(numTiles);

    this.initOutputImage();
  }

  private initOutputImage(): void {
    const data = this.outputImageData.data;
    data.fill(255); // White background, full opacity
  }

  public reset(): void {
    this.prevGray = null;
    this.persistentInk.fill(0);
    this.stabilityCount.fill(0);
    this.initOutputImage();
  }

  public getOutputImage(): ImageData {
    return this.outputImageData;
  }

  public update(frame: RectifiedFrame, instantInk: Uint8Array): ImageData {
    const W = this.width;
    const H = this.height;
    const gray = frame.gray;
    const validMask = frame.validMask;
    const tileSize = this.tileSize;
    const tilesX = this.tilesX;
    const tilesY = this.tilesY;

    // Reset tile flags
    this.tileHasMotion.fill(0);
    this.tileUnstable.fill(0);

    // 1. Block-Based Motion Detection
    if (!this.prevGray) {
      this.prevGray = new Uint8Array(W * H);
      this.prevGray.set(gray);
      // First frame: mark all tiles as having motion so we don't instantly blend before stability
      this.tileHasMotion.fill(1);
    } else {
      let idx = 0;
      for (let y = 0; y < H; y++) {
        const ty = Math.floor(y / tileSize);
        const tyOffset = ty * tilesX;
        for (let x = 0; x < W; x++, idx++) {
          const diff = Math.abs(gray[idx] - this.prevGray[idx]);
          if (diff > this.motionThreshold) {
            const tx = Math.floor(x / tileSize);
            this.tileHasMotion[tyOffset + tx] = 1;
          }
        }
      }
      this.prevGray.set(gray);
    }

    // Dilate motion across immediate 3x3 neighboring tiles:
    // If any pixels in a tile (or its immediate 3x3 neighboring tiles) are moving, mark the entire tile as unstable.
    for (let ty = 0; ty < tilesY; ty++) {
      const minTy = Math.max(0, ty - 1);
      const maxTy = Math.min(tilesY - 1, ty + 1);

      for (let tx = 0; tx < tilesX; tx++) {
        const minTx = Math.max(0, tx - 1);
        const maxTx = Math.min(tilesX - 1, tx + 1);

        let unstable = 0;
        outer: for (let ny = minTy; ny <= maxTy; ny++) {
          const nRowOffset = ny * tilesX;
          for (let nx = minTx; nx <= maxTx; nx++) {
            if (this.tileHasMotion[nRowOffset + nx] === 1) {
              unstable = 1;
              break outer;
            }
          }
        }
        this.tileUnstable[ty * tilesX + tx] = unstable;
      }
    }

    // 2. Stability Counter & Persistent Ink Update & 3. Render to ImageData
    const alpha = this.blendAlpha;
    const reqFrames = this.requiredStableFrames;
    const outData = this.outputImageData.data;

    let idx = 0;
    for (let y = 0; y < H; y++) {
      const ty = Math.floor(y / tileSize);
      const tyOffset = ty * tilesX;

      for (let x = 0; x < W; x++, idx++) {
        const tx = Math.floor(x / tileSize);
        const isTileUnstable = this.tileUnstable[tyOffset + tx] === 1;

        if (validMask[idx] === 0) {
          // Explicitly clear ink and reset stability for excluded areas (e.g. markers, out of bounds)
          this.persistentInk[idx] = 0;
          this.stabilityCount[idx] = 0;
        } else if (isTileUnstable) {
          // Freeze persistentInk at current value during transient motion
          this.stabilityCount[idx] = 0;
        } else {
          this.stabilityCount[idx]++;
          if (this.stabilityCount[idx] >= reqFrames) {
            this.persistentInk[idx] = (1 - alpha) * this.persistentInk[idx] + alpha * instantInk[idx];
          }
        }

        // Render to ImageData:
        // out = 255 - persistentInk[i] (white paper background, black ink strokes)
        const inkVal = validMask[idx] === 0 ? 0 : this.persistentInk[idx];
        const pixelVal = Math.max(0, Math.min(255, Math.round(255 - inkVal)));

        const outIdx = idx << 2;
        outData[outIdx] = pixelVal;
        outData[outIdx + 1] = pixelVal;
        outData[outIdx + 2] = pixelVal;
        outData[outIdx + 3] = 255;
      }
    }

    return this.outputImageData;
  }
}
