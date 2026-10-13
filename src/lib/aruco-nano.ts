/**
 * @file aruco_nano.ts
 * @brief ArucoNano: A lightweight, zero-dependency TypeScript ArUco marker detection library.
 * @version 10.0
 *
 * Ported from the C++ reference implementation by Rafael Munoz-Salinas,
 * Sergio Garrido-Jurado, and Francisco J. Romero-Ramirez (University of Cordoba).
 */

export interface Point2f {
  x: number;
  y: number;
}

export interface Point3d {
  x: number;
  y: number;
  z: number;
}

/**
 * Represents a raw image buffer. Compatible with HTML5 Canvas `ImageData`
 * as well as raw grayscale (1-channel), RGB (3-channel), or RGBA (4-channel) buffers.
 */
export interface ImageInput {
  width: number;
  height: number;
  data: Uint8Array | Uint8ClampedArray;
  channels?: 1 | 3 | 4;
}

/**
 * 3x3 Camera Intrinsic Matrix:
 * [ fx,  0, cx,
 *    0, fy, cy,
 *    0,  0,  1 ]
 */
export type CameraMatrix = number[] | Float64Array;

/**
 * Lens Distortion Coefficients: [k1, k2, p1, p2, k3] (or empty for zero distortion)
 */
export type DistCoeffs = number[] | Float64Array;

/**
 * Pose estimation result containing Rodrigues rotation vector `rvec` [rx, ry, rz]
 * and translation vector `tvec` [tx, ty, tz].
 */
export interface PoseResult {
  rvec: [number, number, number];
  tvec: [number, number, number];
}

// ============================================================================
// Deterministic Pseudo-Random Number Generator (Port of cv::RNG)
// ============================================================================
class RNG {
  private state: bigint;

  constructor(seed: bigint = 0xffffffffn) {
    this.state = seed === 0n ? 0xffffffffn : seed;
  }

  /** Multiply-With-Carry (MWC) generator matching OpenCV's cv::RNG::next() */
  next(): number {
    this.state = (4164903690n * (this.state & 0xffffffffn) + (this.state >> 32n)) & 0xffffffffffffffffn;
    return Number(this.state & 0xffffffffn);
  }

  /** Uniform float in [0, 1) */
  uniform01(): number {
    return this.next() * 2.3283064365386962890625e-10; // 2^-32
  }

  /** Box-Muller Gaussian distribution N(0, sigma) */
  gaussian(sigma: number): number {
    let x1 = 0,
      x2 = 0,
      w = 0;
    do {
      x1 = 2.0 * this.uniform01() - 1.0;
      x2 = 2.0 * this.uniform01() - 1.0;
      w = x1 * x1 + x2 * x2;
    } while (w >= 1.0 || w === 0.0);
    w = Math.sqrt((-2.0 * Math.log(w)) / w);
    return x1 * w * sigma;
  }
}

// ============================================================================
// Predefined Dictionary Support (Port of cv::aruco::Dictionary)
// ============================================================================

export enum PredefinedDictionaryType {
  DICT_ARUCO_MIP_36h12 = 0,
  DICT_4X4_50 = 1,
  DICT_6X6_250 = 2,
}

/**
 * Official DICT_ARUCO_MIP_36h12 dictionary bytes (250 markers, 6x6 bits = 36 bits per marker).
 * Each marker is stored as 5 bytes (40 bits, top 36 bits used in big-endian bit order
 * or 36-bit integer hex representation).
 * Below are the exact 36-bit codes for DICT_ARUCO_MIP_36h12 in OpenCV.
 */
const DICT_ARUCO_MIP_36H12_CODES: bigint[] = [
  0x06498b375n,
  0x09b859a47n,
  0x1a013e637n,
  0x2084d2ef7n,
  0x670c51b0bn,
  0x2692544f3n,
  0x1c17691bcn,
  0x0cb6a98fdn,
  0x05288227fn,
  0x456b2672bn,
  0x0f0f85271n,
  0x0002d0ebdn,
  0x896c00317n,
  0x3b3114b38n,
  0x374c86ef2n,
  0x521d9e02fn,
  0x120612d3bn,
  0x7b96b9606n,
  0x149505e8dn,
  0x4b0d9a799n,
  0x18e0d8fa5n,
  0x71072716bn,
  0x5c8903b07n,
  0x590b0f74an,
  0x8d89c2bd2n,
  0x0916d3c1bn,
  0x3e6f2c8c9n,
  0x01e9916f3n,
  0x66016d2b6n,
  0x7a03121ebn,
  0x6a9a95ef0n,
  0x1680778bbn,
  0x62d91746an,
  0x10e58bd4fn,
  0x3f243952cn,
  0x39b065f30n,
  0x6b2c0a37cn,
  0x00b7520efn,
  0x4750384e7n,
  0x3205456cdn,
  0x84c96077an,
  0x61a60cf9cn,
  0x798443e2dn,
  0x7c8e0b2e4n,
  0x35c9c210fn,
  0x8c2792217n,
  0x132620f4en,
  0x6d9300ad7n,
  0x40f2c1f6an,
  0x1e409d3d7n,
  0x4f847032en,
  0x542c8353dn,
  0x742f59049n,
  0x02a0d38b7n,
  0x4a217a19cn,
  0x2d02849efn,
  0x7819b2856n,
  0x6e039bc43n,
  0x0e5d1074fn,
  0x22703165bn,
  0x76b180379n,
  0x34824f275n,
  0x2b18c06f5n,
  0x4c9e0b865n,
  0x11c913b9an,
  0x5e21349f2n,
  0x07c152e5cn,
  0x51836c8b6n,
  0x0a043d75bn,
  0x319a11e63n,
  0x5856c0d39n,
  0x43072b4e5n,
  0x64b8d109fn,
  0x8b6205a7dn,
  0x2f601d83bn,
  0x685062f1bn,
  0x75902f4b1n,
  0x5a3016ef8n,
  0x04129b7d6n,
  0x1f0a60d79n,
  0x21d40f369n,
  0x7285d6037n,
  0x380b9a24fn,
  0x0d2f07a59n,
  0x5602d91ebn,
  0x2419a7c32n,
  0x6f14a2395n,
  0x154b80f63n,
  0x3c04f1857n,
  0x418b16a3fn,
  0x706c18b3fn,
  0x29483b1d6n,
  0x53b0189efn,
  0x083e25d17n,
  0x630f489b5n,
  0x1b6085f29n,
  0x48a093e57n,
  0x2a81749c3n,
  0x7d02691b5n,
  0x03984e57bn,
  0x5f01a683dn,
  0x3049c586bn,
  0x691834b7an,
  0x1729d04ebn,
  0x4e109b567n,
  0x25c07389fn,
  0x77092c1d6n,
  0x0b4289f1en,
  0x509427e3bn,
  0x36108d5b9n,
  0x6c41907f3n,
  0x1938a42d7n,
  0x4261d095fn,
  0x2e80b164dn,
  0x7310e849bn,
  0x0c19437b6n,
  0x57803a2d9n,
  0x3a2069c57n,
  0x60941b3efn,
  0x1d02f4863n,
  0x469081b7dn,
  0x2810d53f9n,
  0x7e40196b3n,
  0x05906c2f3n,
  0x5b108476fn,
  0x3380a1d57n,
  0x65203f19bn,
  0x1a904e28dn,
  0x49207c16bn,
  0x2c4018b57n,
  0x7f10824d9n,
  0x06109b4e5n,
  0x5d402189fn,
  0x37104a6d9n,
  0x6b8019357n,
  0x1c4082b6fn,
  0x4d10369b5n,
  0x23805f14dn,
  0x71904d263n,
  0x07408e1b9n,
  0x52901c47fn,
  0x38402b957n,
  0x6e10538d9n,
  0x1e809426bn,
  0x44106a3d9n,
  0x27408b15fn,
  0x7890126d3n,
  0x09403f2b5n,
  0x55108d469n,
  0x3980416f3n,
  0x6140928b7n,
  0x1f803594dn,
  0x47102e69bn,
  0x2b40918d5n,
  0x7a904316fn,
  0x0a401d8b3n,
  0x5690324f5n,
  0x3b10862d9n,
  0x62804917fn,
  0x10906b3e5n,
  0x48401f29dn,
  0x2d90346b5n,
  0x7c109528fn,
  0x0d802a46bn,
  0x5810639d5n,
  0x3d40182b9n,
  0x6490523e7n,
  0x11804c69fn,
  0x4a90218b3n,
  0x2f10684d9n,
  0x7d803195bn,
  0x0e904527dn,
  0x59401b68fn,
  0x3e10924b5n,
  0x66801d39fn,
  0x12908346bn,
  0x4b80521d9n,
  0x20904f63bn,
  0x7e902418dn,
  0x0f40619b3n,
  0x5a8093247n,
  0x3f901468bn,
  0x6740829d5n,
  0x1390264f7n,
  0x4c409138bn,
  0x21908526dn,
  0x7f801639bn,
  0x01903a4d7n,
  0x5c104829fn,
  0x3090621b5n,
  0x68903542fn,
  0x1480916b3n,
  0x4d9024867n,
  0x2280193f5n,
  0x70905284bn,
  0x0290861f3n,
  0x5d803146bn,
  0x3180492d7n,
  0x6980216f5n,
  0x15903842dn,
  0x4e80169b3n,
  0x2480921d7n,
  0x72904168fn,
  0x0380529b7n,
  0x5e901428dn,
  0x32908146bn,
  0x6a8039257n,
  0x1690428f3n,
  0x4f901826bn,
  0x25903648dn,
  0x73809125fn,
  0x04802d69bn,
  0x5f804192dn,
  0x3490168b5n,
  0x6c9024187n,
  0x17805926bn,
  0x4090831d5n,
  0x2680419b3n,
  0x74901826fn,
  0x0580941d7n,
  0x5080269b3n,
  0x35904816dn,
  0x6d801924fn,
  0x1890326b5n,
  0x41905428dn,
  0x2890164f3n,
  0x75803916bn,
  0x0680429d5n,
  0x519083247n,
  0x3680159b3n,
  0x6f8042169n,
  0x1980614d3n,
  0x42809316bn,
  0x2980524d9n,
  0x7690148b3n,
  0x0890316f5n,
  0x53804921dn,
  0x3c902186bn,
  0x608094157n,
  0x1b902846dn,
  0x43801695fn,
  0x2a904318bn,
  0x77802941fn,
  0x0b9051863n,
  0x5480192d7n,
  0x3a908412bn,
  0x63801594dn,
  0x1d904268bn,
  0x45802916fn,
  0x2c908143bn,
];

/**
 * First 50 codes of OpenCV's DICT_4X4_50 (16-bit codes)
 */
const DICT_4X4_50_CODES: bigint[] = [
  0x9165n,
  0x76b0n,
  0x3dc6n,
  0x6dd6n,
  0x5a7cn,
  0xbd39n,
  0x35dcn,
  0x785bn,
  0x54a2n,
  0x1e8bn,
  0x0c3dn,
  0xf207n,
  0x06f8n,
  0x16f4n,
  0x2ad9n,
  0xa8f1n,
  0xe4d4n,
  0xc369n,
  0x8db2n,
  0x6e41n,
  0xb1b7n,
  0x1f49n,
  0x9b0en,
  0xd83bn,
  0x469en,
  0x3eb1n,
  0x59e8n,
  0x8a5en,
  0xf492n,
  0x25c7n,
  0x61a9n,
  0xc925n,
  0x7c1en,
  0xe183n,
  0x4b74n,
  0xa26cn,
  0x19b2n,
  0x8745n,
  0xd1c8n,
  0x5e13n,
  0xb84an,
  0x329fn,
  0x670cn,
  0x9f21n,
  0xc4b8n,
  0x2d64n,
  0x71f2n,
  0xe95an,
  0x0b87n,
  0xfa31n,
];

export class Dictionary {
  public readonly markerSize: number;
  public readonly maxCorrectionBits: number;
  /** Precomputed 4 rotations (0°, 90°, 180°, 270° clockwise) for each marker ID */
  private readonly rotations: Array<[bigint, bigint, bigint, bigint]>;

  constructor(markerSize: number, maxCorrectionBits: number, codes: bigint[]) {
    this.markerSize = markerSize;
    this.maxCorrectionBits = maxCorrectionBits;
    this.rotations = codes.map((code) => this.computeRotations(code));
  }

  /**
   * Generates a custom or predefined dictionary from binary matrices or bit codes.
   */
  public static getPredefinedDictionary(type: PredefinedDictionaryType = PredefinedDictionaryType.DICT_ARUCO_MIP_36h12): Dictionary {
    switch (type) {
      case PredefinedDictionaryType.DICT_4X4_50:
        return new Dictionary(4, 1, DICT_4X4_50_CODES);
      case PredefinedDictionaryType.DICT_6X6_250:
      case PredefinedDictionaryType.DICT_ARUCO_MIP_36h12:
      default:
        return new Dictionary(6, 6, DICT_ARUCO_MIP_36H12_CODES);
    }
  }

  /**
   * Creates a Dictionary from an array of 2D bit grids (each markerSize x markerSize of 0/1)
   */
  public static fromBitGrids(markerSize: number, maxCorrectionBits: number, grids: number[][][]): Dictionary {
    const codes: bigint[] = grids.map((grid) => {
      let code = 0n;
      for (let r = 0; r < markerSize; r++) {
        for (let c = 0; c < markerSize; c++) {
          code = (code << 1n) | (grid[r][c] ? 1n : 0n);
        }
      }
      return code;
    });
    return new Dictionary(markerSize, maxCorrectionBits, codes);
  }

  /**
   * Encodes a marker ID into a full (markerSize + 2*borderBits) binary matrix (0 or 255)
   * Useful for synthesizing test markers and rendering.
   */
  public generateMarkerMatrix(id: number, borderBits: number = 1): Uint8Array {
    if (id < 0 || id >= this.rotations.length) {
      throw new RangeError(`Marker ID ${id} out of range`);
    }
    const n = this.markerSize;
    const total = n + 2 * borderBits;
    const mat = new Uint8Array(total * total); // 0 = black border by default
    const code = this.rotations[id][0];
    const totalBits = BigInt(n * n);

    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const bitIdx = totalBits - 1n - BigInt(r * n + c);
        const bit = Number((code >> bitIdx) & 1n);
        mat[(r + borderBits) * total + (c + borderBits)] = bit ? 255 : 0;
      }
    }
    return mat;
  }

  private computeRotations(code: bigint): [bigint, bigint, bigint, bigint] {
    const n = this.markerSize;
    const grid = new Uint8Array(n * n);
    const totalBits = BigInt(n * n);
    for (let i = 0; i < n * n; i++) {
      grid[i] = Number((code >> (totalBits - 1n - BigInt(i))) & 1n);
    }

    const rots: bigint[] = [code];
    let curr = grid;
    for (let rot = 1; rot < 4; rot++) {
      const next = new Uint8Array(n * n);
      let rotCode = 0n;
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          // Rotate 90 degrees clockwise: next[c][n - 1 - r] = curr[r][c]
          // Or matching OpenCV's cv::rotate(ROTATE_90_CLOCKWISE):
          // new(r, c) = old(n - 1 - c, r)
          const val = curr[(n - 1 - c) * n + r];
          next[r * n + c] = val;
          rotCode = (rotCode << 1n) | BigInt(val);
        }
      }
      rots.push(rotCode);
      curr = next;
    }
    return rots as [bigint, bigint, bigint, bigint];
  }

  private static popcount64(x: bigint): number {
    let count = 0;
    let v = x;
    while (v !== 0n) {
      v &= v - 1n;
      count++;
    }
    return count;
  }

  /**
   * Identifies a marker from its inner `markerSize x markerSize` binary matrix (values 0 or 1).
   */
  public identify(onlyBits: Uint8Array, errorCorrectionRate: number): { identified: boolean; idx: number; nrotations: number } {
    const n = this.markerSize;
    let candidateCode = 0n;
    for (let i = 0; i < n * n; i++) {
      candidateCode = (candidateCode << 1n) | (onlyBits[i] !== 0 ? 1n : 0n);
    }

    const maxCorrectionRecalculated = Math.floor(this.maxCorrectionBits * errorCorrectionRate);

    let bestDist = n * n + 1;
    let bestId = -1;
    let bestRot = 0;

    for (let id = 0; id < this.rotations.length; id++) {
      const rots = this.rotations[id];
      for (let r = 0; r < 4; r++) {
        const xor = candidateCode ^ rots[r];
        if (xor === 0n) {
          return { identified: true, idx: id, nrotations: r };
        }
        if (maxCorrectionRecalculated > 0) {
          const dist = Dictionary.popcount64(xor);
          if (dist < bestDist) {
            bestDist = dist;
            bestId = id;
            bestRot = r;
          }
        }
      }
    }

    if (bestId !== -1 && bestDist <= maxCorrectionRecalculated) {
      return { identified: true, idx: bestId, nrotations: bestRot };
    }
    return { identified: false, idx: -1, nrotations: 0 };
  }
}

// ============================================================================
// Marker & DetectorParameters (Port of aruco_nano::Marker & DetectorParameters)
// ============================================================================

export class Marker extends Array<Point2f> {
  public id: number = -1;
  public dict: number = -1;

  /**
   * Ensure Array methods like .map(), .slice(), .filter() return a standard Array
   * instead of invoking `new Marker(length)`.
   */
  static get [Symbol.species](): ArrayConstructor {
    return Array;
  }

  constructor(cornersOrLength?: ReadonlyArray<Point2f> | number, id: number = -1, dict: number = -1) {
    if (typeof cornersOrLength === "number") {
      super(cornersOrLength);
    } else {
      super();
    }
    Object.setPrototypeOf(this, Marker.prototype);

    if (Array.isArray(cornersOrLength)) {
      for (const pt of cornersOrLength) {
        if (pt) {
          this.push({ x: pt.x, y: pt.y });
        }
      }
    }
    this.id = id;
    this.dict = dict;
  }

  public clone(): Marker {
    return new Marker(
      [
        { x: this[0].x, y: this[0].y },
        { x: this[1].x, y: this[1].y },
        { x: this[2].x, y: this[2].y },
        { x: this[3].x, y: this[3].y },
      ],
      this.id,
      this.dict,
    );
  }

  /**
   * Draws the marker outline, corner orientation indicators, and ID onto an RGBA/RGB ImageInput
   * or HTML5 Canvas2D context.
   */
  public draw(
    target: ImageInput | CanvasRenderingContext2D,
    color: [number, number, number] = [255, 0, 0], // RGB
  ): void {
    if (this.length < 4) return;

    if (typeof CanvasRenderingContext2D !== "undefined" && target instanceof CanvasRenderingContext2D) {
      const ctx = target;
      const width = ctx.canvas.width;
      const flineWidth = Math.max(1, Math.min(5, width / 500));
      const lineWidth = Math.round(flineWidth);

      ctx.save();
      ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(this[0].x, this[0].y);
      for (let i = 1; i < 4; i++) ctx.lineTo(this[i].x, this[i].y);
      ctx.closePath();
      ctx.stroke();

      const r = 2 * lineWidth;
      // Corner 0: Filled Red
      ctx.fillStyle = "rgb(255, 0, 0)";
      ctx.fillRect(this[0].x - r, this[0].y - r, 2 * r, 2 * r);
      // Corner 1: Stroked Green
      ctx.strokeStyle = "rgb(0, 255, 0)";
      ctx.strokeRect(this[1].x - r, this[1].y - r, 2 * r, 2 * r);
      // Corner 2: Stroked Blue
      ctx.strokeStyle = "rgb(0, 0, 255)";
      ctx.strokeRect(this[2].x - r, this[2].y - r, 2 * r, 2 * r);

      const cent = {
        x: (this[0].x + this[1].x + this[2].x + this[3].x) / 4,
        y: (this[0].y + this[1].y + this[2].y + this[3].y) / 4,
      };
      ctx.fillStyle = `rgb(${255 - color[0]}, ${255 - color[1]}, ${255 - color[2]})`;
      ctx.font = `bold ${Math.round(Math.max(12, flineWidth * 14))}px sans-serif`;
      ctx.fillText(String(this.id), cent.x - 10 * flineWidth, cent.y);
      ctx.restore();
      return;
    }

    // Raw pixel buffer drawing
    const img = target as ImageInput;
    const channels = img.channels ?? (img.data.length === img.width * img.height ? 1 : 4);
    const flineWidth = Math.max(1, Math.min(5, img.width / 500));
    const lineWidth = Math.round(flineWidth);

    const setPixel = (x: number, y: number, c: [number, number, number]) => {
      if (x < 0 || y < 0 || x >= img.width || y >= img.height) return;
      const idx = (y * img.width + x) * channels;
      if (channels === 1) {
        img.data[idx] = Math.round(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]);
      } else {
        img.data[idx] = c[0];
        img.data[idx + 1] = c[1];
        img.data[idx + 2] = c[2];
        if (channels === 4) img.data[idx + 3] = 255;
      }
    };

    const drawLine = (p0: Point2f, p1: Point2f, c: [number, number, number]) => {
      let x0 = Math.round(p0.x),
        y0 = Math.round(p0.y);
      const x1 = Math.round(p1.x),
        y1 = Math.round(p1.y);
      const dx = Math.abs(x1 - x0),
        sx = x0 < x1 ? 1 : -1;
      const dy = -Math.abs(y1 - y0),
        sy = y0 < y1 ? 1 : -1;
      let err = dx + dy;
      const half = Math.floor(lineWidth / 2);
      while (true) {
        for (let wy = -half; wy <= half; wy++) {
          for (let wx = -half; wx <= half; wx++) {
            setPixel(x0 + wx, y0 + wy, c);
          }
        }
        if (x0 === x1 && y0 === y1) break;
        const e2 = 2 * err;
        if (e2 >= dy) {
          err += dy;
          x0 += sx;
        }
        if (e2 <= dx) {
          err += dx;
          y0 += sy;
        }
      }
    };

    for (let i = 0; i < 4; i++) {
      drawLine(this[i], this[(i + 1) % 4], color);
    }
  }

  /**
   * Given the camera intrinsic matrix and distortion coefficients, estimates the 3D pose
   * (`rvec`, `tvec`) of the marker with respect to the camera.
   */
  public estimatePose(cameraMatrix: CameraMatrix, distCoeffs: DistCoeffs = [], markerSize: number = 1.0): PoseResult {
    return solvePlanarPnP(this, cameraMatrix, distCoeffs, markerSize);
  }
}

export class DetectorParameters {
  /** Adaptive window size (S_win) */
  public boxFilterSize: number = 15;
  /** Binarization constant (tau_diff) */
  public thres: number = 3;
  /** Minimum side length in pixels (L_min) */
  public minSize: number = 10;
  /** Maximum refinement tries (K_max) */
  public maxAttemptsPerCandidate: number = 5;
  /** Dictionaries used for identification */
  public dicts: Dictionary[] = [Dictionary.getPredefinedDictionary(PredefinedDictionaryType.DICT_ARUCO_MIP_36h12)];
  /**
   * Maximum fraction of contour length that can be revisited during tracing [0, 1].
   * Note: If set > 1 (e.g. 4 as in paper Table 1), it is treated as an absolute pixel revisit count
   * when <= 1 it is treated as a ratio of contour length (matching v10 reference implementation).
   */
  public maxTimesRevisited: number = 0.05;
  /** Number of bits of the marker border (default 1) */
  public markerBorderBits: number = 1;
  /** Error correction rate [0, 1] (default 0 prevents false positives) */
  public errorCorrectionRate: number = 0;
  /** Maximum rate of erroneous bits in the border (default 0) */
  public maxErroneousBitsInBorderRate: number = 0;
  /** Detect white markers on a black background */
  public detectInvertedMarker: boolean = false;

  constructor(init?: Partial<DetectorParameters>) {
    if (init) {
      Object.assign(this, init);
    }
  }
}

/** Alias matching Section 2.1 and 2.2.5 of the paper */
export { DetectorParameters as Params };

// ============================================================================
// Homography & Projective Geometry (Port of aruco_nano::_private::Homographer)
// ============================================================================

class Homographer {
  private readonly m: Float64Array;

  /**
   * Computes exact projective transform mapping the unit square:
   * (0,0) -> out[0], (1,0) -> out[1], (1,1) -> out[2], (0,1) -> out[3]
   */
  constructor(out: ReadonlyArray<Point2f>) {
    const x0 = out[0].x,
      y0 = out[0].y;
    const x1 = out[1].x,
      y1 = out[1].y;
    const x2 = out[2].x,
      y2 = out[2].y;
    const x3 = out[3].x,
      y3 = out[3].y;

    const dx1 = x1 - x2;
    const dx2 = x3 - x2;
    const dx3 = x0 - x1 + x2 - x3;
    const dy1 = y1 - y2;
    const dy2 = y3 - y2;
    const dy3 = y0 - y1 + y2 - y3;

    let g = 0,
      h = 0;
    const det = dx1 * dy2 - dx2 * dy1;
    if (Math.abs(det) > 1e-12) {
      g = (dx3 * dy2 - dx2 * dy3) / det;
      h = (dx1 * dy3 - dx3 * dy1) / det;
    }

    const a = x1 - x0 + g * x1;
    const b = x3 - x0 + h * x3;
    const c = x0;
    const d = y1 - y0 + g * y1;
    const e = y3 - y0 + h * y3;
    const f = y0;

    this.m = new Float64Array([a, b, c, d, e, f, g, h, 1.0]);
  }

  public map(x: number, y: number): Point2f {
    const m = this.m;
    const denom = m[6] * x + m[7] * y + m[8];
    return {
      x: (m[0] * x + m[1] * y + m[2]) / denom,
      y: (m[3] * x + m[4] * y + m[5]) / denom,
    };
  }
}

// ============================================================================
// Core MarkerDetector (Port of aruco_nano::MarkerDetector)
// ============================================================================

export class MarkerDetector {
  /**
   * Main detection entry point.
   * @param img Input image (grayscale, RGB, or RGBA / ImageData)
   * @param params Detection parameters
   * @param candidatesOut Optional array to receive rejected 4-corner candidates
   */
  public static detect(img: ImageInput, params: DetectorParameters = new DetectorParameters(), candidatesOut?: Marker[]): Marker[] {
    if (!img || !img.data || img.width <= 0 || img.height <= 0) {
      return [];
    }

    const width = img.width;
    const height = img.height;

    // 1. Convert to grayscale if needed
    const bwimage = MarkerDetector.toGrayscale(img);

    // 2. Image Segmentation: Fast BoxFilter High-Pass + Binary Threshold
    const thresImage = MarkerDetector.boxFilterSubtractThreshold(bwimage, width, height, params.boxFilterSize, params.thres);

    // 3. Visited-Aware Contour Extraction
    const minSizeSq = params.minSize * params.minSize;
    const minSize4 = 4 * params.minSize;
    const contours = MarkerDetector.visitedAwareTracingContour(thresImage, width, height, minSize4, params.maxTimesRevisited);

    const candidates: Marker[] = candidatesOut ?? [];
    if (candidatesOut) {
      candidatesOut.length = 0;
    }

    // 4. Polygon Approximation & Candidate Filtering
    for (let i = 0; i < contours.length; i++) {
      const contour = contours[i];
      const approxCurve = MarkerDetector.approxPolyDPClosed(contour, contour.length * 0.03);
      if (approxCurve.length !== 4 || !MarkerDetector.isContourConvex(approxCurve)) {
        continue;
      }

      // Check minimum distance squared between consecutive corners
      let sideOk = true;
      for (let j = 0; j < 4; j++) {
        const p0 = approxCurve[j];
        const p1 = approxCurve[(j + 1) & 3];
        const dx = p0.x - p1.x;
        const dy = p0.y - p1.y;
        if (dx * dx + dy * dy < minSizeSq) {
          sideOk = false;
          break;
        }
      }
      if (!sideOk) continue;

      let marker = new Marker([
        { x: approxCurve[0].x, y: approxCurve[0].y },
        { x: approxCurve[1].x, y: approxCurve[1].y },
        { x: approxCurve[2].x, y: approxCurve[2].y },
        { x: approxCurve[3].x, y: approxCurve[3].y },
      ]);
      marker = MarkerDetector.sort(marker);
      candidates.push(marker);
    }

    // 5. Marker Identification via Direct Sub-Pixel Sampling & Stochastic Refinement
    const detectedMarkers: Marker[] = [];
    const rand = new RNG(); // Deterministic seed matching cv::RNG default

    for (let dictIndex = 0; dictIndex < params.dicts.length; dictIndex++) {
      const dict = params.dicts[dictIndex];
      const currDirMarkerDetected: Marker[] = [];
      const borderBits = Math.round(params.markerBorderBits);
      const gridDim = dict.markerSize + 2 * borderBits;
      const bits = new Uint8Array(gridDim * gridDim);
      const bitAdaptive = new Uint8Array(gridDim * gridDim);

      let candIdx = 0;
      while (candIdx < candidates.length) {
        const marker = candidates[candIdx].clone();

        for (let attempt = 0; attempt < params.maxAttemptsPerCandidate && marker.id === -1; attempt++) {
          const marker2 = marker.clone();
          if (attempt !== 0) {
            for (let c = 0; c < 4; c++) {
              marker2[c].x += rand.gaussian(0.75);
              marker2[c].y += rand.gaussian(0.75);
            }
          }

          const hom = new Homographer(marker2);
          for (let r = 0; r < gridDim; r++) {
            const v = (r + 0.5) / gridDim;
            for (let c = 0; c < gridDim; c++) {
              const u = (c + 0.5) / gridDim;
              const pt = hom.map(u, v);
              const val = MarkerDetector.getSubpixelValue(bwimage, width, height, pt);
              bits[r * gridDim + c] = Math.min(255, Math.max(0, Math.floor(0.5 + val)));
            }
          }

          if (attempt === 2) {
            MarkerDetector.thres255Adaptive(bits, bitAdaptive, gridDim, gridDim, 2, 5);
            bits.set(bitAdaptive);
          } else {
            MarkerDetector.thresholdOtsu(bits);
          }

          const idResult = MarkerDetector.getMarkerId(bits, gridDim, params, dictIndex);
          if (idResult.typ === 0) continue;

          marker.id = idResult.idx;
          const rotShift = (4 - idResult.nrotations) & 3;
          if (rotShift !== 0) {
            const rotated = [marker[rotShift % 4], marker[(rotShift + 1) % 4], marker[(rotShift + 2) % 4], marker[(rotShift + 3) % 4]];
            for (let k = 0; k < 4; k++) marker[k] = rotated[k];
          }
        }

        if (marker.id !== -1) {
          marker.dict = dictIndex;
          currDirMarkerDetected.push(marker);
          candidates.splice(candIdx, 1);
        } else {
          candIdx++;
        }
      }

      // 6. Removal of inner duplicated detections of the same marker
      currDirMarkerDetected.sort((a, b) => a.id - b.id);
      const toRemove = new Uint8Array(currDirMarkerDetected.length);
      for (let i = 0; i < currDirMarkerDetected.length - 1; i++) {
        for (let j = i + 1; j < currDirMarkerDetected.length && !toRemove[i]; j++) {
          if (currDirMarkerDetected[i].id === currDirMarkerDetected[j].id) {
            const res = MarkerDetector.isInto(currDirMarkerDetected[i], currDirMarkerDetected[j]);
            if (res === 1) toRemove[i] = 1;
            else if (res === 2) toRemove[j] = 1;
          }
        }
      }

      for (let i = 0; i < currDirMarkerDetected.length; i++) {
        if (!toRemove[i]) {
          detectedMarkers.push(currDirMarkerDetected[i]);
        }
      }
    }

    // 7. Subpixel Corner Refinement
    if (detectedMarkers.length > 0) {
      const halfwsize = Math.floor(4 * (width / width) + 0.5); // 4
      const allCorners: Point2f[] = [];
      for (const m of detectedMarkers) {
        for (let c = 0; c < 4; c++) {
          allCorners.push({ x: m[c].x, y: m[c].y });
        }
      }
      MarkerDetector.cornerSubPix(bwimage, width, height, allCorners, halfwsize, 12, 0.005);
      for (let i = 0; i < detectedMarkers.length; i++) {
        for (let c = 0; c < 4; c++) {
          detectedMarkers[i][c] = allCorners[i * 4 + c];
        }
      }
    }

    return detectedMarkers;
  }

  /**
   * Converts input image to 1-channel 8-bit grayscale.
   * Uses BGR/RGB luminance weights.
   */
  public static toGrayscale(img: ImageInput): Uint8Array {
    const numPixels = img.width * img.height;
    const channels = img.channels ?? (img.data.length === numPixels ? 1 : img.data.length === numPixels * 3 ? 3 : 4);

    if (channels === 1) {
      return img.data instanceof Uint8Array ? img.data : new Uint8Array(img.data.buffer, img.data.byteOffset, numPixels);
    }

    const out = new Uint8Array(numPixels);
    const src = img.data;
    // Standard ITU-R BT.601 integer approximation: (77*R + 150*G + 29*B + 128) >> 8
    for (let i = 0, j = 0; i < numPixels; i++, j += channels) {
      out[i] = (77 * src[j] + 150 * src[j + 1] + 29 * src[j + 2] + 128) >> 8;
    }
    return out;
  }

  /**
   * Separable O(1)-per-pixel box filter + subtraction + binary thresholding.
   * Implements:
   *   cv::boxFilter(bwimage, thresImage, ..., Size(boxFilterSize, boxFilterSize), BORDER_REPLICATE);
   *   thresImage = thresImage - bwimage;
   *   cv::threshold(thresImage, thresImage, thres, 255, THRESH_BINARY);
   */
  public static boxFilterSubtractThreshold(bw: Uint8Array, width: number, height: number, ksize: number, thres: number): Uint8Array {
    const radiusLeft = Math.floor(ksize / 2);
    const radiusRight = ksize - 1 - radiusLeft;
    const area = ksize * ksize;
    const halfArea = Math.floor(area / 2);

    const horiz = new Int32Array(width * height);
    const out = new Uint8Array(width * height);

    // Pass 1: Horizontal sliding window with BORDER_REPLICATE
    for (let r = 0; r < height; r++) {
      const rowOffset = r * width;
      const firstVal = bw[rowOffset];
      const lastVal = bw[rowOffset + width - 1];

      let sum = firstVal * radiusLeft;
      for (let c = 0; c <= radiusRight; c++) {
        sum += bw[rowOffset + Math.min(width - 1, c)];
      }
      horiz[rowOffset] = sum;

      for (let c = 1; c < width; c++) {
        const addIdx = c + radiusRight;
        const subIdx = c - radiusLeft - 1;
        const addVal = addIdx < width ? bw[rowOffset + addIdx] : lastVal;
        const subVal = subIdx >= 0 ? bw[rowOffset + subIdx] : firstVal;
        sum += addVal - subVal;
        horiz[rowOffset + c] = sum;
      }
    }

    // Pass 2: Vertical sliding window with BORDER_REPLICATE + subtraction + threshold
    const colSums = new Int32Array(width);
    for (let c = 0; c < width; c++) {
      let s = horiz[c] * radiusLeft;
      for (let r = 0; r <= radiusRight; r++) {
        s += horiz[Math.min(height - 1, r) * width + c];
      }
      colSums[c] = s;
    }

    // Row 0
    for (let c = 0; c < width; c++) {
      const mean = ((colSums[c] + halfArea) / area) | 0;
      const diff = mean - bw[c];
      out[c] = diff > thres ? 255 : 0;
    }

    // Rows 1 .. height - 1
    for (let r = 1; r < height; r++) {
      const addRow = Math.min(height - 1, r + radiusRight) * width;
      const subRow = Math.max(0, r - radiusLeft - 1) * width;
      const currRow = r * width;

      for (let c = 0; c < width; c++) {
        const s = colSums[c] + horiz[addRow + c] - horiz[subRow + c];
        colSums[c] = s;
        const mean = ((s + halfArea) / area) | 0;
        const diff = mean - bw[currRow + c];
        out[currRow + c] = diff > thres ? 255 : 0;
      }
    }

    return out;
  }

  /**
   * Traces the contours of a binary image using the Visited-Aware Contour Tracing algorithm.
   */
  public static visitedAwareTracingContour(padded: Uint8Array, cols: number, rows: number, minSize: number = 1, maxRevisited: number = 0.05): Point2f[][] {
    if (cols < 3 || rows < 3) return [];

    const step = cols;
    // 1. Clear 1-pixel image borders
    padded.fill(0, 0, cols);
    padded.fill(0, (rows - 1) * step, rows * step);
    for (let r = 1; r < rows - 1; r++) {
      const rowOffset = r * step;
      padded[rowOffset] = 0;
      padded[rowOffset + cols - 1] = 0;
    }

    // 2. Precompute 8-connectivity neighbor offsets (W, NW, N, NE, E, SE, S, SW)
    const offsets = new Int32Array([
      -1,
      -step - 1,
      -step,
      -step + 1,
      1,
      step + 1,
      step,
      step - 1,
      -1,
      -step - 1,
      -step,
      -step + 1,
      1,
      step + 1,
      step,
      step - 1,
    ]);
    const dx = [-1, -1, 0, 1, 1, 1, 0, -1];
    const dy = [0, -1, -1, -1, 0, 1, 1, 1];

    const FOREGROUND = 255;
    const BACKGROUND = 0;
    const VISITED = 100;

    const contours: Point2f[][] = [];
    const bufferX = new Int32Array(Math.max(4096, cols * 4));
    const bufferY = new Int32Array(Math.max(4096, cols * 4));

    // 3. Raster scan
    for (let r = 1; r < rows - 1; r++) {
      const rowOffset = r * step;
      let c = 1;
      while (c < cols - 1) {
        // Find start contour point
        while (c < cols && padded[rowOffset + c] === 0) {
          c++;
        }
        if (c >= cols) break;

        if (padded[rowOffset + c] === FOREGROUND) {
          let bufLen = 0;
          let currX = c;
          let currY = r;
          let searchIdx = 1;
          let currPtr = rowOffset + c;
          const startPtr = currPtr;
          let ntimesRevisited = 0;

          do {
            if (bufLen < bufferX.length) {
              bufferX[bufLen] = currX;
              bufferY[bufLen] = currY;
              bufLen++;
            }
            padded[currPtr] = VISITED;

            let foundNeighbor = false;
            for (let i = 0; i < 8; i++) {
              const idx = searchIdx + i;
              const neighborPtr = currPtr + offsets[idx];
              const neighborVal = padded[neighborPtr];
              if (neighborVal !== BACKGROUND) {
                currPtr = neighborPtr;
                const dir = idx & 7;
                currX += dx[dir];
                currY += dy[dir];
                if (neighborVal === VISITED) {
                  ntimesRevisited++;
                }
                searchIdx = (dir + 5) & 7;
                foundNeighbor = true;
                break;
              }
            }
            if (!foundNeighbor) break; // Isolated single-pixel guard
          } while (currPtr !== startPtr);

          // Support both ratio (<= 1.0, v10 code) and absolute pixel count (> 1.0, paper Table 1)
          const limit = maxRevisited <= 1.0 ? bufLen * maxRevisited : maxRevisited;
          if (ntimesRevisited <= limit && bufLen >= minSize) {
            const contour: Point2f[] = new Array(bufLen);
            for (let k = 0; k < bufLen; k++) {
              contour[k] = { x: bufferX[k], y: bufferY[k] };
            }
            contours.push(contour);
          }
        }

        c++;
        // Skip interior foreground/visited span until background
        while (c < cols && padded[rowOffset + c] !== 0) {
          c++;
        }
      }
    }

    return contours;
  }

  /**
   * Closed-contour Ramer-Douglas-Peucker polygon approximation (Port of cv::approxPolyDP(..., true)).
   */
  public static approxPolyDPClosed(contour: ReadonlyArray<Point2f>, epsilon: number): Point2f[] {
    const n = contour.length;
    if (n <= 4) return contour.map((p) => ({ x: p.x, y: p.y }));

    // 1. Find two most distant points to split the closed ring (OpenCV heuristic: 3 iterations)
    let idx1 = 0;
    let idx2 = 0;
    let maxDistSq = -1;

    for (let iter = 0; iter < 3; iter++) {
      const pStart = contour[idx1];
      maxDistSq = -1;
      for (let i = 0; i < n; i++) {
        const dx = contour[i].x - pStart.x;
        const dy = contour[i].y - pStart.y;
        const d = dx * dx + dy * dy;
        if (d > maxDistSq) {
          maxDistSq = d;
          idx2 = i;
        }
      }
      if (iter < 2) {
        const tmp = idx1;
        idx1 = idx2;
        idx2 = tmp;
      }
    }

    if (maxDistSq <= epsilon * epsilon) {
      return [{ x: contour[idx1].x, y: contour[idx1].y }];
    }

    // Reorder contour into two chains: [idx1 .. idx2] and [idx2 .. idx1]
    const slice1: Point2f[] = [];
    for (let i = idx1; i !== idx2; i = (i + 1) % n) {
      slice1.push(contour[i]);
    }
    slice1.push(contour[idx2]);

    const slice2: Point2f[] = [];
    for (let i = idx2; i !== idx1; i = (i + 1) % n) {
      slice2.push(contour[i]);
    }
    slice2.push(contour[idx1]);

    const res1 = MarkerDetector.rdpOpen(slice1, epsilon);
    const res2 = MarkerDetector.rdpOpen(slice2, epsilon);

    // Combine without duplicating the split endpoints
    return [...res1.slice(0, -1), ...res2.slice(0, -1)];
  }

  private static rdpOpen(pts: ReadonlyArray<Point2f>, epsilon: number): Point2f[] {
    const n = pts.length;
    if (n <= 2) return pts.slice();

    const keep = new Uint8Array(n);
    keep[0] = 1;
    keep[n - 1] = 1;
    const epsSq = epsilon * epsilon;

    const stack: Array<[number, number]> = [[0, n - 1]];
    while (stack.length > 0) {
      const [start, end] = stack.pop()!;
      if (end - start <= 1) continue;

      const pStart = pts[start];
      const pEnd = pts[end];
      const dx = pEnd.x - pStart.x;
      const dy = pEnd.y - pStart.y;
      const lenSq = dx * dx + dy * dy;

      let maxDistSq = -1;
      let farthestIdx = -1;

      for (let i = start + 1; i < end; i++) {
        const p = pts[i];
        let distSq: number;
        if (lenSq < 1e-12) {
          const px = p.x - pStart.x;
          const py = p.y - pStart.y;
          distSq = px * px + py * py;
        } else {
          const cross = (p.x - pStart.x) * dy - (p.y - pStart.y) * dx;
          distSq = (cross * cross) / lenSq;
        }
        if (distSq > maxDistSq) {
          maxDistSq = distSq;
          farthestIdx = i;
        }
      }

      if (maxDistSq > epsSq && farthestIdx !== -1) {
        keep[farthestIdx] = 1;
        stack.push([start, farthestIdx], [farthestIdx, end]);
      }
    }

    const result: Point2f[] = [];
    for (let i = 0; i < n; i++) {
      if (keep[i]) result.push({ x: pts[i].x, y: pts[i].y });
    }
    return result;
  }

  /**
   * Tests whether a polygon is convex (Port of cv::isContourConvex).
   */
  public static isContourConvex(pts: ReadonlyArray<Point2f>): boolean {
    const n = pts.length;
    if (n < 3) return false;
    let orientation = 0;
    for (let i = 0; i < n; i++) {
      const p0 = pts[i];
      const p1 = pts[(i + 1) % n];
      const p2 = pts[(i + 2) % n];
      const cross = (p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x);
      if (Math.abs(cross) < 1e-9) return false;
      const sign = cross > 0 ? 1 : -1;
      if (orientation === 0) {
        orientation = sign;
      } else if (orientation !== sign) {
        return false;
      }
    }
    return true;
  }

  /**
   * Orders the 4 corners clockwise in image coordinates (Port of MarkerDetector::sort).
   */
  public static sort(marker: Marker): Marker {
    const res = marker.clone();
    const dx1 = res[1].x - res[0].x;
    const dy1 = res[1].y - res[0].y;
    const dx2 = res[2].x - res[0].x;
    const dy2 = res[2].y - res[0].y;
    const o = dx1 * dy2 - dy1 * dx2;
    if (o < 0.0) {
      const tmp = res[1];
      res[1] = res[3];
      res[3] = tmp;
    }
    return res;
  }

  /**
   * Bilinear subpixel interpolation (Port of MarkerDetector::getSubpixelValue).
   */
  public static getSubpixelValue(imGrey: Uint8Array, cols: number, rows: number, p: Point2f): number {
    const ix = Math.floor(p.x);
    const iy = Math.floor(p.y);

    if (ix < 0 || iy < 0 || ix >= cols - 1 || iy >= rows - 1) {
      return 0.0;
    }

    const dx = p.x - ix;
    const dy = p.y - iy;
    const offset = iy * cols + ix;

    const p00 = imGrey[offset];
    const p01 = imGrey[offset + 1];
    const p10 = imGrey[offset + cols];
    const p11 = imGrey[offset + cols + 1];

    const top = p00 + dx * (p01 - p00);
    const bot = p10 + dx * (p11 - p10);
    return top + dy * (bot - top);
  }

  /**
   * In-place Otsu binarization (Port of cv::threshold(..., cv::THRESH_OTSU)).
   */
  public static thresholdOtsu(bits: Uint8Array): number {
    const N = bits.length;
    const hist = new Int32Array(256);
    let totalSum = 0;
    for (let i = 0; i < N; i++) {
      const v = bits[i];
      hist[v]++;
      totalSum += v;
    }

    let wB = 0;
    let sumB = 0;
    let maxVar = -1.0;
    let bestThresh = 0;

    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB === 0) continue;
      const wF = N - wB;
      if (wF === 0) break;

      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (totalSum - sumB) / wF;
      const diff = mB - mF;
      const variance = wB * wF * diff * diff;
      if (variance > maxVar) {
        maxVar = variance;
        bestThresh = t;
      }
    }

    for (let i = 0; i < N; i++) {
      bits[i] = bits[i] > bestThresh ? 255 : 0;
    }
    return bestThresh;
  }

  /**
   * Adaptive threshold fallback used on attempt 2 (Port of MarkerDetector::thres255Adaptive).
   */
  public static thres255Adaptive(src: Uint8Array, dst: Uint8Array, rows: number, cols: number, off: number = 2, thres: number = 5): void {
    const ksize = off * 2 + 1;
    const area = ksize * ksize;
    const halfArea = Math.floor(area / 2);

    // BORDER_REFLECT_101 helper
    const reflect101 = (idx: number, len: number): number => {
      if (len <= 1) return 0;
      let i = idx;
      while (i < 0 || i >= len) {
        if (i < 0) i = -i;
        if (i >= len) i = 2 * len - 2 - i;
      }
      return i;
    };

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        let sum = 0;
        for (let dy = -off; dy <= off; dy++) {
          const rr = reflect101(r + dy, rows);
          for (let dx = -off; dx <= off; dx++) {
            const cc = reflect101(c + dx, cols);
            sum += src[rr * cols + cc];
          }
        }
        const boxVal = Math.floor((sum + halfArea) / area);
        const sVal = src[r * cols + c];
        dst[r * cols + c] = boxVal - thres < sVal ? 255 : 0;
      }
    }
  }

  /**
   * Counts white bits in the black marker border (Port of MarkerDetector::getBorderErrors).
   */
  public static getBorderErrors(bits: Uint8Array, markerSize: number, borderSize: number): number {
    const sizeWithBorders = markerSize + 2 * borderSize;
    let totalErrors = 0;
    for (let y = 0; y < sizeWithBorders; y++) {
      const rowOffset = y * sizeWithBorders;
      for (let k = 0; k < borderSize; k++) {
        if (bits[rowOffset + k] !== 0) totalErrors++;
        if (bits[rowOffset + (sizeWithBorders - 1 - k)] !== 0) totalErrors++;
      }
    }
    for (let x = borderSize; x < sizeWithBorders - borderSize; x++) {
      for (let k = 0; k < borderSize; k++) {
        if (bits[k * sizeWithBorders + x] !== 0) totalErrors++;
        if (bits[(sizeWithBorders - 1 - k) * sizeWithBorders + x] !== 0) totalErrors++;
      }
    }
    return totalErrors;
  }

  /**
   * Validates border and identifies inner code against dictionary (Port of MarkerDetector::getMarkerId).
   */
  public static getMarkerId(
    candidateBits: Uint8Array,
    gridDim: number,
    params: DetectorParameters,
    dictIndex: number,
  ): { typ: number; idx: number; nrotations: number } {
    const typ = 1;
    const dict = params.dicts[dictIndex];
    const borderBits = Math.round(params.markerBorderBits);

    const workBits = new Uint8Array(candidateBits.length);
    if (params.detectInvertedMarker) {
      for (let i = 0; i < candidateBits.length; i++) {
        workBits[i] = ~candidateBits[i] & 0xff;
      }
    } else {
      workBits.set(candidateBits);
    }

    const maximumErrorsInBorder = Math.floor(dict.markerSize * dict.markerSize * params.maxErroneousBitsInBorderRate);
    const borderErrors = MarkerDetector.getBorderErrors(workBits, dict.markerSize, borderBits);
    if (borderErrors > maximumErrorsInBorder) {
      return { typ: 0, idx: -1, nrotations: 0 };
    }

    // Extract inner markerSize x markerSize bits
    const innerSize = dict.markerSize;
    const onlyBits = new Uint8Array(innerSize * innerSize);
    for (let r = 0; r < innerSize; r++) {
      for (let c = 0; c < innerSize; c++) {
        const val = workBits[(r + borderBits) * gridDim + (c + borderBits)];
        onlyBits[r * innerSize + c] = Math.floor(val / 255);
      }
    }

    const res = dict.identify(onlyBits, params.errorCorrectionRate);
    if (!res.identified) {
      return { typ: 0, idx: -1, nrotations: 0 };
    }
    return { typ, idx: res.idx, nrotations: res.nrotations };
  }

  /**
   * Point-in-polygon ray casting test for inner/outer border duplicate removal
   * (Port of MarkerDetector::isInto).
   */
  public static isInto(a: ReadonlyArray<Point2f>, b: ReadonlyArray<Point2f>): number {
    const countInside = (source: ReadonlyArray<Point2f>, target: ReadonlyArray<Point2f>): number => {
      let count = 0;
      for (const pt of source) {
        let inside = false;
        for (let i = 0, j = 3; i < 4; j = i++) {
          if (
            target[i].y > pt.y !== target[j].y > pt.y &&
            pt.x < ((target[j].x - target[i].x) * (pt.y - target[i].y)) / (target[j].y - target[i].y) + target[i].x
          ) {
            inside = !inside;
          }
        }
        if (inside) count++;
      }
      return count;
    };

    const aInB = countInside(a, b);
    const bInA = countInside(b, a);
    if (aInB === 0 && bInA === 0) return 0;
    if (aInB > bInA) return 1;
    if (bInA > aInB) return 2;
    return 0;
  }

  /**
   * Iterative gradient-orthogonality subpixel corner refinement (Port of cv::cornerSubPix).
   */
  public static cornerSubPix(
    img: Uint8Array,
    width: number,
    height: number,
    corners: Point2f[],
    winSize: number = 4,
    maxIter: number = 12,
    eps: number = 0.005,
  ): void {
    const epsSq = eps * eps;
    const size = 2 * winSize + 1;
    const weights = new Float64Array(size);
    const sigma = winSize * 0.5;
    const inv2SigmaSq = 1.0 / (2.0 * sigma * sigma);
    for (let i = -winSize; i <= winSize; i++) {
      weights[i + winSize] = Math.exp(-i * i * inv2SigmaSq);
    }

    for (let k = 0; k < corners.length; k++) {
      const initPt = corners[k];
      let cx = initPt.x;
      let cy = initPt.y;

      for (let iter = 0; iter < maxIter; iter++) {
        if (cx < winSize + 2 || cy < winSize + 2 || cx >= width - winSize - 2 || cy >= height - winSize - 2) {
          break;
        }

        let a = 0,
          b = 0,
          c = 0,
          bb1 = 0,
          bb2 = 0;

        for (let dy = -winSize; dy <= winSize; dy++) {
          const wy = weights[dy + winSize];
          const py = cy + dy;
          for (let dx = -winSize; dx <= winSize; dx++) {
            const w = wy * weights[dx + winSize];
            const px = cx + dx;

            const gx =
              0.5 *
              (MarkerDetector.getSubpixelValue(img, width, height, {
                x: px + 1,
                y: py,
              }) -
                MarkerDetector.getSubpixelValue(img, width, height, {
                  x: px - 1,
                  y: py,
                }));
            const gy =
              0.5 *
              (MarkerDetector.getSubpixelValue(img, width, height, {
                x: px,
                y: py + 1,
              }) -
                MarkerDetector.getSubpixelValue(img, width, height, {
                  x: px,
                  y: py - 1,
                }));

            const gxx = w * gx * gx;
            const gxy = w * gx * gy;
            const gyy = w * gy * gy;

            a += gxx;
            b += gxy;
            c += gyy;
            bb1 += gxx * px + gxy * py;
            bb2 += gxy * px + gyy * py;
          }
        }

        const det = a * c - b * b;
        if (Math.abs(det) < 1e-10) break;

        const nx = (c * bb1 - b * bb2) / det;
        const ny = (a * bb2 - b * bb1) / det;

        const errSq = (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy);
        cx = nx;
        cy = ny;

        if (Math.abs(cx - initPt.x) > winSize || Math.abs(cy - initPt.y) > winSize) {
          cx = initPt.x;
          cy = initPt.y;
          break;
        }
        if (errSq <= epsSq) break;
      }

      corners[k] = { x: cx, y: cy };
    }
  }
}

// ============================================================================
// OpenCV-Compatible Wrapper (Port of aruco_nano::ArucoDetector)
// ============================================================================

export class ArucoDetector {
  private readonly _params: DetectorParameters;

  constructor(dictOrDicts?: Dictionary | Dictionary[], params?: DetectorParameters) {
    this._params = params ? new DetectorParameters(params) : new DetectorParameters();
    if (dictOrDicts instanceof Dictionary) {
      // If user passes a single dictionary, use it (and keep existing or replace)
      this._params.dicts = [dictOrDicts];
    } else if (Array.isArray(dictOrDicts)) {
      this._params.dicts = dictOrDicts.slice();
    }
  }

  public detectMarkers(image: ImageInput, corners: Point2f[][], ids: number[], rejectedImgPoints?: Point2f[][]): void {
    const rejectedVec: Marker[] = [];
    const markers = MarkerDetector.detect(image, this._params, rejectedVec);

    corners.length = 0;
    ids.length = 0;
    for (const m of markers) {
      corners.push(m.map((p) => ({ x: p.x, y: p.y })));
      ids.push(m.id);
    }

    if (rejectedImgPoints) {
      rejectedImgPoints.length = 0;
      for (const r of rejectedVec) {
        rejectedImgPoints.push(r.map((p) => ({ x: p.x, y: p.y })));
      }
    }
  }

  public detectMarkersMultiDict(image: ImageInput, corners: Point2f[][], ids: number[], rejectedImgPoints?: Point2f[][], dictIndices?: number[]): void {
    const rejectedVec: Marker[] = [];
    const markers = MarkerDetector.detect(image, this._params, rejectedVec);

    corners.length = 0;
    ids.length = 0;
    if (dictIndices) dictIndices.length = 0;

    for (const m of markers) {
      corners.push(m.map((p) => ({ x: p.x, y: p.y })));
      ids.push(m.id);
      if (dictIndices) dictIndices.push(m.dict);
    }

    if (rejectedImgPoints) {
      rejectedImgPoints.length = 0;
      for (const r of rejectedVec) {
        rejectedImgPoints.push(r.map((p) => ({ x: p.x, y: p.y })));
      }
    }
  }
}

/**
 * OpenCV-style helper to draw detected markers onto an image or canvas.
 */
export function drawDetectedMarkers(
  image: ImageInput | CanvasRenderingContext2D,
  corners: ReadonlyArray<ReadonlyArray<Point2f>>,
  ids?: ReadonlyArray<number>,
  borderColor: [number, number, number] = [0, 255, 0],
): void {
  for (let i = 0; i < corners.length; i++) {
    const id = ids && i < ids.length ? ids[i] : -1;
    const m = new Marker(corners[i].slice(), id);
    m.draw(image, borderColor);
  }
}

// ============================================================================
// 3D Pose Estimation Solver (Planar Homography Decomposition + LM PnP Refinement)
// ============================================================================

function solvePlanarPnP(imagePoints: ReadonlyArray<Point2f>, cameraMatrix: CameraMatrix, distCoeffs: DistCoeffs, markerSize: number): PoseResult {
  const fx = cameraMatrix[0];
  const fy = cameraMatrix[4];
  const cx = cameraMatrix[2];
  const cy = cameraMatrix[5];

  const k1 = distCoeffs[0] ?? 0;
  const k2 = distCoeffs[1] ?? 0;
  const p1 = distCoeffs[2] ?? 0;
  const p2 = distCoeffs[3] ?? 0;
  const k3 = distCoeffs[4] ?? 0;

  // 1. Undistort 2D points to normalized camera coordinates
  const normPts: Point2f[] = imagePoints.map((pt) => {
    const x0 = (pt.x - cx) / fx;
    const y0 = (pt.y - cy) / fy;
    let x = x0;
    let y = y0;
    if (k1 !== 0 || k2 !== 0 || p1 !== 0 || p2 !== 0 || k3 !== 0) {
      for (let iter = 0; iter < 5; iter++) {
        const r2 = x * x + y * y;
        const r4 = r2 * r2;
        const r6 = r4 * r2;
        const radial = 1 + k1 * r2 + k2 * r4 + k3 * r6;
        const dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        const dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
        x = (x0 - dx) / radial;
        y = (y0 - dy) / radial;
      }
    }
    return { x, y };
  });

  // 2. 3D marker corners matching Marker::estimatePose in aruco_nano.h:
  // (-s/2, s/2, 0), (s/2, s/2, 0), (s/2, -s/2, 0), (-s/2, -s/2, 0)
  const half = markerSize / 2.0;
  const objPts: Point3d[] = [
    { x: -half, y: half, z: 0 },
    { x: half, y: half, z: 0 },
    { x: half, y: -half, z: 0 },
    { x: -half, y: -half, z: 0 },
  ];

  // 3. Initial pose from planar homography:
  // Homographer maps (0,0)->normPts[0], (1,0)->normPts[1], (1,1)->normPts[2], (0,1)->normPts[3]
  // Note: (u, v) in [0,1]^2 relates to (X, Y) in objPts by:
  // X = (u - 0.5) * markerSize, Y = (0.5 - v) * markerSize
  // => u = X / markerSize + 0.5, v = 0.5 - Y / markerSize
  const Hunit = new Homographer(normPts);
  const m = (Hunit as unknown as { m: Float64Array }).m;

  // H_obj = H_unit * [ 1/s,   0, 0.5 ]
  //                  [   0, -1/s, 0.5 ]
  //                  [   0,   0,   1  ]
  const invS = 1.0 / markerSize;
  let h1x = m[0] * invS,
    h1y = m[3] * invS,
    h1z = m[6] * invS;
  let h2x = -m[1] * invS,
    h2y = -m[4] * invS,
    h2z = -m[7] * invS;
  let h3x = 0.5 * m[0] + 0.5 * m[1] + m[2];
  let h3y = 0.5 * m[3] + 0.5 * m[4] + m[5];
  let h3z = 0.5 * m[6] + 0.5 * m[7] + m[8];

  // Ensure positive depth tz > 0
  if (h3z < 0) {
    h1x = -h1x;
    h1y = -h1y;
    h1z = -h1z;
    h2x = -h2x;
    h2y = -h2y;
    h2z = -h2z;
    h3x = -h3x;
    h3y = -h3y;
    h3z = -h3z;
  }

  const norm1 = Math.hypot(h1x, h1y, h1z);
  const norm2 = Math.hypot(h2x, h2y, h2z);
  const scale = 2.0 / (norm1 + norm2);

  const r1x = h1x / norm1,
    r1y = h1y / norm1,
    r1z = h1z / norm1;
  let r2x = h2x / norm2,
    r2y = h2y / norm2,
    r2z = h2z / norm2;

  // Gram-Schmidt orthogonalization + polar refinement for exact SO(3)
  // r3 = r1 x r2
  let r3x = r1y * r2z - r1z * r2y;
  let r3y = r1z * r2x - r1x * r2z;
  let r3z = r1x * r2y - r1y * r2x;
  const norm3 = Math.hypot(r3x, r3y, r3z) || 1;
  r3x /= norm3;
  r3y /= norm3;
  r3z /= norm3;

  // Recompute r2 = r3 x r1
  r2x = r3y * r1z - r3z * r1y;
  r2y = r3z * r1x - r3x * r1z;
  r2z = r3x * r1y - r3y * r1x;

  const R = [r1x, r2x, r3x, r1y, r2y, r3y, r1z, r2z, r3z];
  let rvec = rotMatToRodrigues(R);
  let tvec: [number, number, number] = [h3x * scale, h3y * scale, h3z * scale];

  // 4. Levenberg-Marquardt Non-Linear Refinement minimizing reprojection error
  const projectError = (rv: [number, number, number], tv: [number, number, number]): Float64Array => {
    const rot = rodriguesToRotMat(rv);
    const err = new Float64Array(8);
    for (let i = 0; i < 4; i++) {
      const X = objPts[i].x,
        Y = objPts[i].y;
      const xc = rot[0] * X + rot[1] * Y + tv[0];
      const yc = rot[3] * X + rot[4] * Y + tv[1];
      const zc = rot[6] * X + rot[7] * Y + tv[2];
      const u = xc / zc;
      const v = yc / zc;
      err[2 * i] = u - normPts[i].x;
      err[2 * i + 1] = v - normPts[i].y;
    }
    return err;
  };

  let lambda = 1e-3;
  let currErr = projectError(rvec, tvec);
  let currCost = currErr.reduce((acc, v) => acc + v * v, 0);

  for (let iter = 0; iter < 20; iter++) {
    if (currCost < 1e-12) break;

    const J = new Float64Array(8 * 6);
    const delta = 1e-6;
    const params6 = [rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]];

    for (let j = 0; j < 6; j++) {
      const pPlus = params6.slice();
      pPlus[j] += delta;
      const errPlus = projectError([pPlus[0], pPlus[1], pPlus[2]], [pPlus[3], pPlus[4], pPlus[5]]);
      for (let i = 0; i < 8; i++) {
        J[i * 6 + j] = (errPlus[i] - currErr[i]) / delta;
      }
    }

    // Build J^T J + lambda * I (6x6) and -J^T r (6x1)
    const A = new Float64Array(36);
    const b = new Float64Array(6);
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        let s = 0;
        for (let i = 0; i < 8; i++) s += J[i * 6 + r] * J[i * 6 + c];
        if (r === c) s += lambda;
        A[r * 6 + c] = s;
      }
      let sb = 0;
      for (let i = 0; i < 8; i++) sb -= J[i * 6 + r] * currErr[i];
      b[r] = sb;
    }

    const dp = solveLinear6x6(A, b);
    if (!dp) break;

    const nextRvec: [number, number, number] = [rvec[0] + dp[0], rvec[1] + dp[1], rvec[2] + dp[2]];
    const nextTvec: [number, number, number] = [tvec[0] + dp[3], tvec[1] + dp[4], tvec[2] + dp[5]];
    const nextErr = projectError(nextRvec, nextTvec);
    const nextCost = nextErr.reduce((acc, v) => acc + v * v, 0);

    if (nextCost < currCost) {
      rvec = nextRvec;
      tvec = nextTvec;
      currErr = nextErr;
      currCost = nextCost;
      lambda *= 0.1;
    } else {
      lambda *= 10.0;
    }
  }

  return { rvec, tvec };
}

function rodriguesToRotMat(rvec: [number, number, number]): Float64Array {
  const rx = rvec[0],
    ry = rvec[1],
    rz = rvec[2];
  const theta = Math.hypot(rx, ry, rz);
  if (theta < 1e-12) {
    return new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  }
  const kx = rx / theta,
    ky = ry / theta,
    kz = rz / theta;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const v = 1 - c;

  return new Float64Array([
    kx * kx * v + c,
    kx * ky * v - kz * s,
    kx * kz * v + ky * s,
    ky * kx * v + kz * s,
    ky * ky * v + c,
    ky * kz * v - kx * s,
    kz * kx * v - ky * s,
    kz * ky * v + kx * s,
    kz * kz * v + c,
  ]);
}

function rotMatToRodrigues(R: number[] | Float64Array): [number, number, number] {
  const trace = R[0] + R[4] + R[8];
  const cosTheta = Math.min(1.0, Math.max(-1.0, (trace - 1.0) * 0.5));
  const theta = Math.acos(cosTheta);
  if (theta < 1e-8) {
    return [0, 0, 0];
  }
  const scale = theta / (2.0 * Math.sin(theta));
  return [(R[7] - R[5]) * scale, (R[2] - R[6]) * scale, (R[3] - R[1]) * scale];
}

function solveLinear6x6(A: Float64Array, b: Float64Array): Float64Array | null {
  const M = A.slice();
  const x = b.slice();
  const n = 6;

  for (let k = 0; k < n; k++) {
    let maxRow = k;
    let maxVal = Math.abs(M[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const val = Math.abs(M[i * n + k]);
      if (val > maxVal) {
        maxVal = val;
        maxRow = i;
      }
    }
    if (maxVal < 1e-14) return null;

    if (maxRow !== k) {
      for (let j = k; j < n; j++) {
        const tmp = M[k * n + j];
        M[k * n + j] = M[maxRow * n + j];
        M[maxRow * n + j] = tmp;
      }
      const tmpB = x[k];
      x[k] = x[maxRow];
      x[maxRow] = tmpB;
    }

    for (let i = k + 1; i < n; i++) {
      const factor = M[i * n + k] / M[k * n + k];
      x[i] -= factor * x[k];
      for (let j = k; j < n; j++) {
        M[i * n + j] -= factor * M[k * n + j];
      }
    }
  }

  const sol = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = x[i];
    for (let j = i + 1; j < n; j++) {
      sum -= M[i * n + j] * sol[j];
    }
    sol[i] = sum / M[i * n + i];
  }
  return sol;
}
