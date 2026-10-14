import type { RectifiedFrame } from "./types.ts";

export interface ThresholdConfig {
  windowSize?: number; // default: 31
  thresholdLow?: number; // default: 15
  thresholdHigh?: number; // default: 35
}

/**
 * Stage 3: AdaptiveThreshold
 * Converts rectified grayscale frame (RectifiedFrame) into an instantaneous ink intensity buffer
 * (ink: Uint8Array, where 0 = white paper, 255 = black ink) using masked dual integral images.
 */
export class AdaptiveThreshold {
  private width: number;
  private height: number;
  private windowSize: number;
  private thresholdLow: number;
  private thresholdHigh: number;

  private integralSum: Int32Array;
  private integralCount: Int32Array;
  private ink: Uint8Array;

  constructor(width: number, height: number, config?: ThresholdConfig) {
    this.width = width;
    this.height = height;
    this.windowSize = config?.windowSize ?? 31;
    this.thresholdLow = config?.thresholdLow ?? 15;
    this.thresholdHigh = config?.thresholdHigh ?? 35;

    // Integral images of size (width + 1) * (height + 1)
    const intSize = (width + 1) * (height + 1);
    this.integralSum = new Int32Array(intSize);
    this.integralCount = new Int32Array(intSize);
    this.ink = new Uint8Array(width * height);
  }

  public process(frame: RectifiedFrame): Uint8Array {
    const W = this.width;
    const H = this.height;
    const gray = frame.gray;
    const validMask = frame.validMask;
    const intW = W + 1;

    const integralSum = this.integralSum;
    const integralCount = this.integralCount;
    const ink = this.ink;

    // 1. Build two Summed Area Tables (integralSum and integralCount) in a single pass
    // First row and first column remain 0
    let srcIdx = 0;
    for (let y = 0; y < H; y++) {
      let rowSum = 0;
      let rowCount = 0;
      const intRowOffset = (y + 1) * intW;
      const prevIntRowOffset = y * intW;

      for (let x = 0; x < W; x++, srcIdx++) {
        if (validMask[srcIdx] === 1) {
          rowSum += gray[srcIdx];
          rowCount += 1;
        }

        const intIdx = intRowOffset + (x + 1);
        const aboveIntIdx = prevIntRowOffset + (x + 1);

        integralSum[intIdx] = integralSum[aboveIntIdx] + rowSum;
        integralCount[intIdx] = integralCount[aboveIntIdx] + rowCount;
      }
    }

    // 2. Compute local threshold and instantaneous ink
    const halfWin = Math.floor(this.windowSize / 2);
    const low = this.thresholdLow;
    const high = this.thresholdHigh;
    const range = high - low;

    srcIdx = 0;
    for (let y = 0; y < H; y++) {
      const y0 = Math.max(0, y - halfWin);
      const y1 = Math.min(H, y + halfWin + 1);

      const r0Offset = y0 * intW;
      const r1Offset = y1 * intW;

      for (let x = 0; x < W; x++, srcIdx++) {
        if (validMask[srcIdx] === 0) {
          ink[srcIdx] = 0;
          continue;
        }

        const x0 = Math.max(0, x - halfWin);
        const x1 = Math.min(W, x + halfWin + 1);

        // Sub-rectangle sum and count
        const boxCount = integralCount[r1Offset + x1] - integralCount[r0Offset + x1] - integralCount[r1Offset + x0] + integralCount[r0Offset + x0];

        if (boxCount === 0) {
          ink[srcIdx] = 0;
          continue;
        }

        const boxSum = integralSum[r1Offset + x1] - integralSum[r0Offset + x1] - integralSum[r1Offset + x0] + integralSum[r0Offset + x0];

        const mu = boxSum / boxCount;
        const delta = mu - gray[srcIdx];

        if (delta <= low) {
          ink[srcIdx] = 0;
        } else if (delta >= high) {
          ink[srcIdx] = 255;
        } else {
          // Soft threshold for smooth anti-aliased ink
          ink[srcIdx] = Math.round(((delta - low) / range) * 255);
        }
      }
    }

    return ink;
  }
}
