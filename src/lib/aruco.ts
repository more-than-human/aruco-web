import { DICTIONARY_PACKED } from "./dictionary-packed";

export interface IPoint {
  x: number;
  y: number;
}

export interface IImage {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export class Image implements IImage {
  public width: number;
  public height: number;
  public data: Uint8ClampedArray;

  constructor(width: number = 0, height: number = 0, data?: Uint8ClampedArray | Uint8Array | number[] | ArrayBuffer) {
    this.width = width;
    this.height = height;

    if (data instanceof Uint8ClampedArray) {
      // Optimal: zero-copy
      this.data = data;
    } else if (data instanceof Uint8Array) {
      // Convert from Uint8Array
      this.data = new Uint8ClampedArray(data);
    } else if (Array.isArray(data)) {
      // Backwards compatibility: convert from Array
      this.data = new Uint8ClampedArray(data);
    } else if (data instanceof ArrayBuffer) {
      // From raw buffer
      this.data = new Uint8ClampedArray(data);
    } else {
      // Create empty
      this.data = new Uint8ClampedArray(width * height);
    }
  }

  // Canvas integration (zero-copy)
  static fromImageData(imageData: ImageData): Image {
    return new Image(imageData.width, imageData.height, imageData.data);
  }

  toImageData(): ImageData {
    // Create ImageData using the proper constructor
    // TypeScript may have issues with types, so we use a workaround
    const imageData = new ImageData(this.width, this.height);
    imageData.data.set(this.data);
    return imageData;
  }

  // Helper methods
  clone(): Image {
    const img = new Image(this.width, this.height);
    img.data.set(this.data);
    return img;
  }

  fill(value: number): this {
    this.data.fill(value);
    return this;
  }

  getPixel(x: number, y: number): number {
    return this.data[y * this.width + x];
  }

  setPixel(x: number, y: number, value: number): void {
    this.data[y * this.width + x] = value;
  }

  copy(other: Image): this {
    this.width = other.width;
    this.height = other.height;
    this.data = new Uint8ClampedArray(other.data);
    return this;
  }
}

class BlurStack {
  public color: number;
  public next: any;
  constructor() {
    this.color = 0;
    this.next = null;
  }
}

export class CV {
  static grayscale(imageSrc: IImage): Image {
    const src = imageSrc.data;
    const dst = new Image(imageSrc.width, imageSrc.height);
    const dstData = dst.data;

    let i = 0;
    let j = 0;
    const len = src.length;

    // Process RGBA to grayscale
    while (i < len) {
      // Weighted average: 0.299R + 0.587G + 0.114B
      dstData[j++] = (src[i] * 0.299 + src[i + 1] * 0.587 + src[i + 2] * 0.114 + 0.5) & 0xff;
      i += 4;
    }

    return dst;
  }

  static threshold(imageSrc: IImage, imageDst: IImage, threshold: number): IImage {
    const src = imageSrc.data;
    const dst = imageDst.data;
    const len = src.length;
    const tab = new Uint8Array(256);

    // Build lookup table (ottimizzato con TypedArray)
    for (let i = 0; i < 256; ++i) {
      tab[i] = i <= threshold ? 0 : 255;
    }

    // Apply threshold using lookup
    for (let i = 0; i < len; ++i) {
      dst[i] = tab[src[i]];
    }

    imageDst.width = imageSrc.width;
    imageDst.height = imageSrc.height;

    return imageDst;
  }

  static adaptiveThreshold(imageSrc: IImage, imageDst: IImage, kernelSize: number, threshold: number): IImage {
    const src = imageSrc.data;
    const dst = imageDst.data;
    const len = src.length;
    const tab = new Uint8Array(768);

    // First: blur the image
    CV.stackBoxBlur(imageSrc, imageDst, kernelSize);

    // Build lookup table (768 = 3 * 256 per gestire range -255 to 510)
    for (let i = 0; i < 768; ++i) {
      tab[i] = i - 255 <= -threshold ? 255 : 0;
    }

    // Apply adaptive threshold WITH CRITICAL +255 OFFSET
    for (let i = 0; i < len; ++i) {
      // CRITICAL: +255 offset to handle negative values
      dst[i] = tab[src[i] - dst[i] + 255];
    }

    imageDst.width = imageSrc.width;
    imageDst.height = imageSrc.height;

    return imageDst;
  }

  static otsu(imageSrc: IImage): number {
    const src = imageSrc.data;
    const len = src.length;
    const hist = new Uint32Array(256);
    let threshold = 0;
    let sum = 0;
    let sumB = 0;
    let wB = 0;
    let wF = 0;
    let max = 0;

    for (let i = 0; i < len; ++i) {
      hist[src[i]]++;
    }

    for (let i = 0; i < 256; ++i) {
      sum += hist[i] * i;
    }

    for (let i = 0; i < 256; ++i) {
      wB += hist[i];
      if (0 !== wB) {
        wF = len - wB;
        if (0 === wF) {
          break;
        }

        sumB += hist[i] * i;

        const mu = sumB / wB - (sum - sumB) / wF;
        const between = wB * wF * mu * mu;

        if (between > max) {
          max = between;
          threshold = i;
        }
      }
    }

    return threshold;
  }

  static stackBoxBlurMult = [1, 171, 205, 293, 57, 373, 79, 137, 241, 27, 391, 357, 41, 19, 283, 265];

  static stackBoxBlurShift = [0, 9, 10, 11, 9, 12, 10, 11, 12, 9, 13, 13, 10, 9, 13, 13];

  static stackBoxBlur(imageSrc: IImage, imageDst: IImage, kernelSize: number): IImage {
    const src = imageSrc.data;
    const dst = imageDst.data;
    const height = imageSrc.height;
    const width = imageSrc.width;
    const heightMinus1 = height - 1;
    const widthMinus1 = width - 1;
    const size = kernelSize + kernelSize + 1;
    const radius = kernelSize + 1;
    const mult = CV.stackBoxBlurMult[kernelSize];
    const shift = CV.stackBoxBlurShift[kernelSize];

    let stack = new BlurStack();
    const stackStart = stack;
    for (let i = 1; i < size; ++i) {
      stack = stack.next = new BlurStack();
    }
    stack.next = stackStart;

    let pos = 0;

    for (let y = 0; y < height; ++y) {
      const start = pos;

      let color = src[pos];
      let sum = radius * color;

      stack = stackStart;
      for (let i = 0; i < radius; ++i) {
        stack.color = color;
        stack = stack.next;
      }
      for (let i = 1; i < radius; ++i) {
        stack.color = src[pos + i];
        sum += stack.color;
        stack = stack.next;
      }

      stack = stackStart;
      for (let x = 0; x < width; ++x) {
        dst[pos++] = (sum * mult) >>> shift;

        let p = x + radius;
        p = start + (p < widthMinus1 ? p : widthMinus1);
        sum -= stack.color - src[p];

        stack.color = src[p];
        stack = stack.next;
      }
    }

    for (let x = 0; x < width; ++x) {
      pos = x;
      let start = pos + width;

      let color = dst[pos];
      let sum = radius * color;

      stack = stackStart;
      for (let i = 0; i < radius; ++i) {
        stack.color = color;
        stack = stack.next;
      }
      for (let i = 1; i < radius; ++i) {
        stack.color = dst[start];
        sum += stack.color;
        stack = stack.next;

        start += width;
      }

      stack = stackStart;
      for (let y = 0; y < height; ++y) {
        dst[pos] = (sum * mult) >>> shift;

        let p = y + radius;
        p = x + (p < heightMinus1 ? p : heightMinus1) * width;
        sum -= stack.color - dst[p];

        stack.color = dst[p];
        stack = stack.next;

        pos += width;
      }
    }

    return imageDst;
  }

  static gaussianBlur(imageSrc: IImage, imageDst: IImage, imageMean: IImage, kernelSize: number): IImage {
    var kernel = CV.gaussianKernel(kernelSize);

    imageDst.width = imageSrc.width;
    imageDst.height = imageSrc.height;

    imageMean.width = imageSrc.width;
    imageMean.height = imageSrc.height;

    CV.gaussianBlurFilter(imageSrc, imageMean, kernel, true);
    CV.gaussianBlurFilter(imageMean, imageDst, kernel, false);

    return imageDst;
  }

  static gaussianBlurFilter(imageSrc: IImage, imageDst: IImage, kernel: any[], horizontal: boolean): IImage {
    const src = imageSrc.data;
    const dst = imageDst.data;
    const height = imageSrc.height;
    const width = imageSrc.width;
    let pos = 0;
    const limit = kernel.length >> 1;

    for (let i = 0; i < height; ++i) {
      for (let j = 0; j < width; ++j) {
        let value = 0.0;

        for (let k = -limit; k <= limit; ++k) {
          let cur;
          if (horizontal) {
            cur = pos + k;
            if (j + k < 0) {
              cur = pos;
            } else if (j + k >= width) {
              cur = pos;
            }
          } else {
            cur = pos + k * width;
            if (i + k < 0) {
              cur = pos;
            } else if (i + k >= height) {
              cur = pos;
            }
          }

          value += kernel[limit + k] * src[cur];
        }

        dst[pos++] = horizontal ? value : (value + 0.5) | 0;
      }
    }

    return imageDst;
  }

  static gaussianKernel(kernelSize: number): number[] {
    let tab = [[1], [0.25, 0.5, 0.25], [0.0625, 0.25, 0.375, 0.25, 0.0625], [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]],
      kernel = [],
      center,
      sigma,
      scale2X,
      sum,
      x,
      i;

    if (kernelSize <= 7 && kernelSize % 2 === 1) {
      kernel = tab[kernelSize >> 1];
    } else {
      center = (kernelSize - 1.0) * 0.5;
      sigma = 0.8 + 0.3 * (center - 1.0);
      scale2X = -0.5 / (sigma * sigma);
      sum = 0.0;
      for (i = 0; i < kernelSize; ++i) {
        x = i - center;
        sum += kernel[i] = Math.exp(scale2X * x * x);
      }
      sum = 1 / sum;
      for (i = 0; i < kernelSize; ++i) {
        kernel[i] *= sum;
      }
    }

    return kernel;
  }

  static findContours(imageSrc: IImage, binary: any): number[] {
    let width = imageSrc.width,
      height = imageSrc.height,
      contours = [],
      src,
      deltas,
      pos,
      pix,
      nbd,
      outer,
      hole,
      i,
      j;

    src = CV.binaryBorder(imageSrc, binary);

    deltas = CV.neighborhoodDeltas(width + 2);

    pos = width + 3;
    nbd = 1;

    for (i = 0; i < height; ++i, pos += 2) {
      for (j = 0; j < width; ++j, ++pos) {
        pix = src[pos];

        if (0 !== pix) {
          outer = hole = false;

          if (1 === pix && 0 === src[pos - 1]) {
            outer = true;
          } else if (pix >= 1 && 0 === src[pos + 1]) {
            hole = true;
          }

          if (outer || hole) {
            ++nbd;

            contours.push(CV.borderFollowing(src, pos, nbd, { x: j, y: i }, hole, deltas));
          }
        }
      }
    }

    return contours;
  }

  static borderFollowing(
    src: { [x: string]: any },
    pos: number,
    nbd: number,
    point: {
      x: any;
      y: any;
    },
    hole: boolean,
    deltas: any[],
  ) {
    let contour: any = [],
      pos1,
      pos3,
      pos4,
      s,
      s_end;

    contour.hole = hole;

    s = s_end = hole ? 0 : 4;
    do {
      s = (s - 1) & 7;
      pos1 = pos + deltas[s];
      if (src[pos1] !== 0) {
        break;
      }
    } while (s !== s_end);

    if (s === s_end) {
      src[pos] = -nbd;
      contour.push({ x: point.x, y: point.y });
    } else {
      pos3 = pos;

      while (true) {
        s_end = s;

        do {
          pos4 = pos3 + deltas[++s];
        } while (src[pos4] === 0);

        s &= 7;

        if ((s - 1) >>> 0 < s_end >>> 0) {
          src[pos3] = -nbd;
        } else if (src[pos3] === 1) {
          src[pos3] = nbd;
        }

        contour.push({ x: point.x, y: point.y });

        point.x += CV.neighborhood[s][0];
        point.y += CV.neighborhood[s][1];

        if (pos4 === pos && pos3 === pos1) {
          break;
        }

        pos3 = pos4;
        s = (s + 4) & 7;
      }
    }

    return contour;
  }

  static neighborhood: number[][] = [
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
  ];

  static neighborhoodDeltas(width: number): number[] {
    let deltas: number[] = [],
      len = CV.neighborhood.length,
      i = 0;

    for (; i < len; ++i) {
      deltas[i] = CV.neighborhood[i][0] + CV.neighborhood[i][1] * width;
    }

    return deltas.concat(deltas);
  }

  static approxPolyDP(contour: any, epsilon: number): IPoint[] {
    let slice = { start_index: 0, end_index: 0 },
      right_slice = { start_index: 0, end_index: 0 },
      poly = [],
      stack: { start_index: number; end_index: number }[] = [],
      len = contour.length,
      pt: IPoint,
      start_pt,
      end_pt,
      dist,
      max_dist = 0,
      le_eps,
      dx,
      dy,
      i,
      j,
      k;

    epsilon *= epsilon;

    k = 0;

    for (i = 0; i < 3; ++i) {
      max_dist = 0;

      k = (k + right_slice.start_index) % len;
      start_pt = contour[k];
      if (++k === len) {
        k = 0;
      }

      for (j = 1; j < len; ++j) {
        pt = contour[k];
        if (++k === len) {
          k = 0;
        }

        dx = pt.x - start_pt.x;
        dy = pt.y - start_pt.y;
        dist = dx * dx + dy * dy;

        if (dist > max_dist) {
          max_dist = dist;
          right_slice.start_index = j;
        }
      }
    }

    if (max_dist <= epsilon) {
      poly.push({ x: start_pt.x, y: start_pt.y });
    } else {
      slice.start_index = k;
      slice.end_index = right_slice.start_index += slice.start_index;

      right_slice.start_index -= right_slice.start_index >= len ? len : 0;
      right_slice.end_index = slice.start_index;
      if (right_slice.end_index < right_slice.start_index) {
        right_slice.end_index += len;
      }

      stack.push({
        start_index: right_slice.start_index,
        end_index: right_slice.end_index,
      });
      stack.push({
        start_index: slice.start_index,
        end_index: slice.end_index,
      });
    }

    while (stack.length !== 0) {
      slice = stack.pop()!;

      end_pt = contour[slice.end_index % len];
      start_pt = contour[(k = slice.start_index % len)];
      if (++k === len) {
        k = 0;
      }

      if (slice.end_index <= slice.start_index + 1) {
        le_eps = true;
      } else {
        max_dist = 0;

        dx = end_pt.x - start_pt.x;
        dy = end_pt.y - start_pt.y;

        for (i = slice.start_index + 1; i < slice.end_index; ++i) {
          pt = contour[k];
          if (++k === len) {
            k = 0;
          }

          dist = Math.abs((pt.y - start_pt.y) * dx - (pt.x - start_pt.x) * dy);

          if (dist > max_dist) {
            max_dist = dist;
            right_slice.start_index = i;
          }
        }

        le_eps = max_dist * max_dist <= epsilon * (dx * dx + dy * dy);
      }

      if (le_eps) {
        poly.push({ x: start_pt.x, y: start_pt.y });
      } else {
        right_slice.end_index = slice.end_index;
        slice.end_index = right_slice.start_index;

        stack.push({
          start_index: right_slice.start_index,
          end_index: right_slice.end_index,
        });
        stack.push({
          start_index: slice.start_index,
          end_index: slice.end_index,
        });
      }
    }

    return poly;
  }

  static warp(imageSrc: IImage, imageDst: IImage, contour: IPoint[], warpSize: number): IImage {
    const src = imageSrc.data;
    const width = imageSrc.width;
    const height = imageSrc.height;

    // Ensure dst has enough space
    if (imageDst.data.length < warpSize * warpSize || imageDst.width !== warpSize || imageDst.height !== warpSize) {
      imageDst.width = warpSize;
      imageDst.height = warpSize;
      imageDst.data = new Uint8ClampedArray(warpSize * warpSize);
    }

    const dst = imageDst.data;
    let pos = 0;

    const m = CV.getPerspectiveTransform(contour, warpSize - 1);

    let r = m[8];
    let s = m[2];
    let t = m[5];

    for (let i = 0; i < warpSize; ++i) {
      r += m[7];
      s += m[1];
      t += m[4];

      let u = r;
      let v = s;
      let w = t;

      for (let j = 0; j < warpSize; ++j) {
        u += m[6];
        v += m[0];
        w += m[3];

        const x = v / u;
        const y = w / u;

        const sx1 = x >>> 0;
        const sx2 = sx1 === width - 1 ? sx1 : sx1 + 1;
        const dx1 = x - sx1;
        const dx2 = 1.0 - dx1;

        const sy1 = y >>> 0;
        const sy2 = sy1 === height - 1 ? sy1 : sy1 + 1;
        const dy1 = y - sy1;
        const dy2 = 1.0 - dy1;

        const p1 = sy1 * width;
        const p2 = p1;
        const p3 = sy2 * width;
        const p4 = p3;

        dst[pos++] = (dy2 * (dx2 * src[p1 + sx1] + dx1 * src[p2 + sx2]) + dy1 * (dx2 * src[p3 + sx1] + dx1 * src[p4 + sx2])) & 0xff;
      }
    }

    return imageDst;
  }

  static getPerspectiveTransform(src: IPoint[], size: number): number[] {
    let rq = CV.square2quad(src);

    rq[0] /= size;
    rq[1] /= size;
    rq[3] /= size;
    rq[4] /= size;
    rq[6] /= size;
    rq[7] /= size;

    return rq;
  }

  static square2quad(src: IPoint[]): number[] {
    let sq = [],
      px,
      py,
      dx1,
      dx2,
      dy1,
      dy2,
      den;

    px = src[0].x - src[1].x + src[2].x - src[3].x;
    py = src[0].y - src[1].y + src[2].y - src[3].y;

    if (0 === px && 0 === py) {
      sq[0] = src[1].x - src[0].x;
      sq[1] = src[2].x - src[1].x;
      sq[2] = src[0].x;
      sq[3] = src[1].y - src[0].y;
      sq[4] = src[2].y - src[1].y;
      sq[5] = src[0].y;
      sq[6] = 0;
      sq[7] = 0;
      sq[8] = 1;
    } else {
      dx1 = src[1].x - src[2].x;
      dx2 = src[3].x - src[2].x;
      dy1 = src[1].y - src[2].y;
      dy2 = src[3].y - src[2].y;
      den = dx1 * dy2 - dx2 * dy1;

      sq[6] = (px * dy2 - dx2 * py) / den;
      sq[7] = (dx1 * py - px * dy1) / den;
      sq[8] = 1;
      sq[0] = src[1].x - src[0].x + sq[6] * src[1].x;
      sq[1] = src[3].x - src[0].x + sq[7] * src[3].x;
      sq[2] = src[0].x;
      sq[3] = src[1].y - src[0].y + sq[6] * src[1].y;
      sq[4] = src[3].y - src[0].y + sq[7] * src[3].y;
      sq[5] = src[0].y;
    }

    return sq;
  }

  static isContourConvex(contour: IPoint[]): boolean {
    let orientation = 0,
      convex = true,
      len = contour.length,
      i = 0,
      j = 0,
      cur_pt,
      prev_pt,
      dxdy0,
      dydx0,
      dx0,
      dy0,
      dx,
      dy;

    prev_pt = contour[len - 1];
    cur_pt = contour[0];

    dx0 = cur_pt.x - prev_pt.x;
    dy0 = cur_pt.y - prev_pt.y;

    for (; i < len; ++i) {
      if (++j === len) {
        j = 0;
      }

      prev_pt = cur_pt;
      cur_pt = contour[j];

      dx = cur_pt.x - prev_pt.x;
      dy = cur_pt.y - prev_pt.y;
      dxdy0 = dx * dy0;
      dydx0 = dy * dx0;

      orientation |= dydx0 > dxdy0 ? 1 : dydx0 < dxdy0 ? 2 : 3;

      if (3 === orientation) {
        convex = false;
        break;
      }

      dx0 = dx;
      dy0 = dy;
    }

    return convex;
  }

  static perimeter(poly: IPoint[]): number {
    let len = poly.length,
      i = 0,
      j = len - 1,
      p = 0.0,
      dx,
      dy;

    for (; i < len; j = i++) {
      dx = poly[i].x - poly[j].x;
      dy = poly[i].y - poly[j].y;

      p += Math.sqrt(dx * dx + dy * dy);
    }

    return p;
  }

  static minEdgeLength(poly: IPoint[]): number {
    let len = poly.length,
      i = 0,
      j = len - 1,
      min = Infinity,
      d,
      dx,
      dy;

    for (; i < len; j = i++) {
      dx = poly[i].x - poly[j].x;
      dy = poly[i].y - poly[j].y;

      d = dx * dx + dy * dy;

      if (d < min) {
        min = d;
      }
    }

    return Math.sqrt(min);
  }

  static countNonZero(imageSrc: IImage, square: any): number {
    const src = imageSrc.data;
    const height = square.height;
    const width = square.width;
    let pos = square.x + square.y * imageSrc.width;
    const span = imageSrc.width - width;
    let nz = 0;

    for (let i = 0; i < height; ++i) {
      for (let j = 0; j < width; ++j) {
        if (0 !== src[pos++]) {
          ++nz;
        }
      }

      pos += span;
    }

    return nz;
  }

  static binaryBorder(imageSrc: IImage, dst: number[]): number[] {
    const src = imageSrc.data;
    const height = imageSrc.height;
    const width = imageSrc.width;
    let posSrc = 0;
    let posDst = 0;

    for (let j = -2; j < width; ++j) {
      dst[posDst++] = 0;
    }

    for (let i = 0; i < height; ++i) {
      dst[posDst++] = 0;

      for (let j = 0; j < width; ++j) {
        dst[posDst++] = src[posSrc++] === 0 ? 0 : 1;
      }

      dst[posDst++] = 0;
    }

    for (let j = -2; j < width; ++j) {
      dst[posDst++] = 0;
    }

    return dst;
  }
}

interface DictionaryConfig {
  nBits: number;
  tau: number;
  codeList: (number | string | number[])[];
}

interface DictionaryEntry {
  id: number;
}

export class DICTIONARIES {
  static getDictionary(name: keyof typeof DICTIONARIES): DictionaryConfig {
    const dictionary = DICTIONARY_PACKED[name.toLocaleLowerCase()];
    if (dictionary && typeof dictionary === "object" && "nBits" in dictionary && "tau" in dictionary && "codeList" in dictionary) {
      return dictionary as DictionaryConfig;
    }
    throw new Error(`The dictionary "${name}" is not recognized.`);
  }
}

export interface IConfig {
  dictionaryName: "aruco" | "4x4_1000" | "5x5_1000" | "6x6_1000" | "mip_36h12";
  maxHammingDistance?: number;
  limit?: number;
}

export class Dictionary {
  private codes: Record<string, DictionaryEntry> = {};
  private codeList: string[] = [];
  public tau: number = 0;
  public nBits: number = 0;
  public markSize: number = 0;
  public limit?: number;
  private dicName: string;

  constructor(dicName: string, limit?: number) {
    this.codes = {};
    this.codeList = [];
    this.tau = 0;
    this.dicName = dicName;
    this.limit = limit;
    this.initialize(dicName, limit);
  }

  private initialize(dicName: string, limit?: number): void {
    this.codes = {};
    this.codeList = [];
    this.tau = 0;
    this.nBits = 0;
    this.markSize = 0;
    this.dicName = dicName;
    this.limit = limit;
    const dictionary = DICTIONARIES.getDictionary(dicName as keyof typeof DICTIONARIES);
    // console.log('dicName', dicName);

    // console.log(dictionary);

    if (!dictionary) throw new Error('The dictionary "' + dicName + '" is not recognized.');

    this.nBits = dictionary.nBits;
    this.markSize = Math.sqrt(dictionary.nBits) + 2;
    const count = limit !== undefined && limit >= 0 ? Math.min(limit, dictionary.codeList.length) : dictionary.codeList.length;
    for (let i = 0; i < count; i++) {
      const code = this.parseCode(dictionary.codeList[i], dictionary.nBits);
      this.codeList.push(code);
      this.codes[code] = { id: i };
    }

    this.tau = dictionary.tau || this.calculateTau();
  }

  private parseCode(rawCode: number | string | number[], nBits: number): string {
    if (typeof rawCode === "number") {
      return rawCode.toString(2).padStart(nBits, "0");
    } else if (typeof rawCode === "string") {
      return parseInt(rawCode, 16).toString(2).padStart(nBits, "0");
    } else if (Array.isArray(rawCode)) {
      return rawCode
        .map((byte) => byte.toString(2).padStart(8, "0"))
        .join("")
        .slice(0, nBits);
    } else {
      throw new Error(`Invalid code: ${rawCode}`);
    }
  }

  find(bits: string[][]): { id: number; distance: number } | null {
    const binaryString = bits.flat().join("");
    let closestMatch: { id: number; distance: number } | null = null;

    for (const code of this.codeList) {
      const distance = this.hammingDistance(binaryString, code);
      if (distance < this.tau) {
        if (!closestMatch || distance < closestMatch.distance) {
          closestMatch = { id: this.codes[code].id, distance };
        }
      }
    }

    return closestMatch;
  }
  protected _hex2bin(hex: number, nBits: number): string {
    return hex.toString(2).padStart(nBits, "0");
  }

  protected _bytes2bin(byteList: number[], nBits: number): string {
    let bits = "";
    for (const byte of byteList) {
      bits += byte.toString(2).padStart(bits.length + 8 > nBits ? nBits - bits.length : 8, "0");
    }
    return bits;
  }

  private hammingDistance(str1: string, str2: string): number {
    if (str1.length !== str2.length) {
      throw new Error("Hamming distance requires inputs of the same length.");
    }
    return Array.from(str1).reduce((distance, char, index) => {
      return distance + (char !== str2[index] ? 1 : 0);
    }, 0);
  }

  private calculateTau(): number {
    let tau = Number.MAX_VALUE;
    for (let i = 0; i < this.codeList.length; i++)
      for (let j = i + 1; j < this.codeList.length; j++) {
        const distance = this.hammingDistance(this.codeList[i], this.codeList[j]);
        tau = distance < tau ? distance : tau;
      }
    return tau;
  }

  generateSVG(id: number): string {
    const code = this.codeList[id];
    if (code == null)
      throw new Error(
        'The id "' + id + '" is not valid for the dictionary "' + this.dicName + '". ID must be between 0 and ' + (this.codeList.length - 1) + " included.",
      );
    const size = this.markSize - 2;
    let svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + (size + 4) + " " + (size + 4) + '">';
    svg += '<rect x="0" y="0" width="' + (size + 4) + '" height="' + (size + 4) + '" fill="white"/>';
    svg += '<rect x="1" y="1" width="' + (size + 2) + '" height="' + (size + 2) + '" fill="black"/>';
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (code[y * size + x] == "1") svg += '<rect x="' + (x + 2) + '" y="' + (y + 2) + '" width="1" height="1" fill="white"/>';
      }
    }
    svg += "</svg>";
    return svg;
  }
}

export class Marker {
  public id: number;
  public corners: IPoint[];
  public hammingDistance: number;

  constructor(id: number, corners: any, hammingDistance: number) {
    this.id = id;
    this.corners = corners;
    this.hammingDistance = hammingDistance;
  }
}

export class Detector {
  public config: IConfig;
  private dictionary: Dictionary;
  private grey!: Image;
  private thres!: Image;
  private homography!: Image;
  private binary: any[];
  private contours: any[];
  private polys: any[];
  private candidates: any[];
  private streamConfig: any;
  private mjpeg: any;
  static readonly mjpeg: any;

  constructor(config: IConfig) {
    this.config = config;
    this.binary = [];
    this.contours = [];
    this.polys = [];
    this.candidates = [];
    this.dictionary = new Dictionary(config.dictionaryName, config.limit);
    this.dictionary.tau = config.maxHammingDistance != null ? config.maxHammingDistance : this.dictionary.tau;
  }

  detectImage(width: number, height: number, data: any) {
    return this.detect({
      width: width,
      height: height,
      data: data,
    });
  }

  detectStreamInit(width: number, height: number, callback: any) {
    this.streamConfig = {};
    this.streamConfig.width = width;
    this.streamConfig.height = height;
    this.streamConfig.imageSize = width * height * 4; //provided image must be a sequence of rgba bytes (4 bytes represent a pixel)
    this.streamConfig.index = 0;
    this.streamConfig.imageData = new Uint8ClampedArray(this.streamConfig.imageSize);
    this.streamConfig.callback = callback || function (_image: any, _markerList: any) {};
  }

  //accept data chunks of different sizes
  detectStream(data: any) {
    for (let i = 0; i < data.length; i++) {
      this.streamConfig.imageData[this.streamConfig.index] = data[i];
      this.streamConfig.index = (this.streamConfig.index + 1) % this.streamConfig.imageSize;
      if (this.streamConfig.index == 0) {
        const image = {
          width: this.streamConfig.width,
          height: this.streamConfig.height,
          data: this.streamConfig.imageData,
        };
        const markerList = this.detect(image);
        this.streamConfig.callback(image, markerList);
      }
    }
  }

  detectMJPEGStreamInit(width: number, height: number, callback: any, decoderFn: any) {
    this.mjpeg = {
      decoderFn: decoderFn,
      chunks: [],
      SOI: [0xff, 0xd8],
      EOI: [0xff, 0xd9],
    };
    this.detectStreamInit(width, height, callback);
  }

  detectMJPEGStream(chunk: any) {
    const eoiPos = chunk.findIndex(function (element: any, index: any, array: any) {
      return Detector.mjpeg.EOI[0] == element && array.length > index + 1 && Detector.mjpeg.EOI[1] == array[index + 1];
    });
    const soiPos = chunk.findIndex(function (element: any, index: any, array: any) {
      return Detector.mjpeg.SOI[0] == element && array.length > index + 1 && Detector.mjpeg.SOI[1] == array[index + 1];
    });

    if (eoiPos === -1) {
      this.mjpeg.chunks.push(chunk);
    } else {
      const part1 = chunk.slice(0, eoiPos + 2);
      if (part1.length) {
        this.mjpeg.chunks.push(part1);
      }
      if (this.mjpeg.chunks.length) {
        const jpegImage = this.mjpeg.chunks.flat();
        const rgba = this.mjpeg.decoderFn(jpegImage);
        this.detectStream(rgba);
      }
      this.mjpeg.chunks = [];
    }
    if (soiPos > -1) {
      this.mjpeg.chunks = [];
      this.mjpeg.chunks.push(chunk.slice(soiPos));
    }
  }

  detect(image: any): Marker[] {
    this.grey = CV.grayscale(image);

    if (!this.thres || this.thres.width !== this.grey.width || this.thres.height !== this.grey.height) {
      this.thres = new Image(this.grey.width, this.grey.height);
    }

    CV.adaptiveThreshold(this.grey, this.thres, 2, 7);

    this.contours = CV.findContours(this.thres, this.binary);

    this.candidates = this.findCandidates(this.contours, image.width * 0.01, 0.05, 10);

    this.candidates = this.clockwiseCorners(this.candidates);
    const minDist = Math.max(30, image.width * 0.05);
    this.candidates = this.notTooNear(this.candidates, minDist);

    return this.findMarkers(this.grey, this.candidates, 49);
  }

  findCandidates(contours: any, minSize: number, epsilon: number, minLength: number): IPoint[][] {
    let candidates: IPoint[][] = [],
      len = contours.length,
      contour,
      poly,
      i;

    this.polys = [];

    for (i = 0; i < len; ++i) {
      contour = contours[i];

      if (contour.length >= minSize) {
        poly = CV.approxPolyDP(contour, contour.length * epsilon);

        this.polys.push(poly);

        if (4 === poly.length && CV.isContourConvex(poly)) {
          if (CV.minEdgeLength(poly) >= minLength) {
            candidates.push(poly);
          }
        }
      }
    }

    return candidates;
  }

  clockwiseCorners(candidates: any) {
    let len = candidates.length,
      dx1,
      dx2,
      dy1,
      dy2,
      swap;

    for (let i = 0; i < len; ++i) {
      dx1 = candidates[i][1].x - candidates[i][0].x;
      dy1 = candidates[i][1].y - candidates[i][0].y;
      dx2 = candidates[i][2].x - candidates[i][0].x;
      dy2 = candidates[i][2].y - candidates[i][0].y;

      if (dx1 * dy2 - dy1 * dx2 < 0) {
        swap = candidates[i][1];
        candidates[i][1] = candidates[i][3];
        candidates[i][3] = swap;
      }
    }

    return candidates;
  }

  notTooNear(candidates: IPoint[][], minDist: number): IPoint[][] {
    const notTooNear: IPoint[][] = [];
    const len = candidates.length;
    const tooNearFlags = new Array(len).fill(false); // ✅ Array separato per i flag

    let dist, dx, dy;

    for (let i = 0; i < len; ++i) {
      for (let j = i + 1; j < len; ++j) {
        dist = 0;

        for (let k = 0; k < 4; ++k) {
          dx = candidates[i][k].x - candidates[j][k].x;
          dy = candidates[i][k].y - candidates[j][k].y;

          dist += dx * dx + dy * dy;
        }

        if (dist / 4 < minDist * minDist) {
          // ✅ Confronta perimetri usando CV.perimeter
          const perimI = CV.perimeter(candidates[i]);
          const perimJ = CV.perimeter(candidates[j]);

          if (perimI < perimJ) {
            tooNearFlags[i] = true; // ✅ Marca nel flag array
          } else {
            tooNearFlags[j] = true; // ✅ Marca nel flag array
          }
        }
      }
    }

    // ✅ Filtra usando il flag array
    for (let i = 0; i < len; ++i) {
      if (!tooNearFlags[i]) {
        notTooNear.push(candidates[i]);
      }
    }

    return notTooNear;
  }

  findMarkers(imageSrc: IImage, candidates: any, warpSize: number) {
    let markers = [],
      len = candidates.length,
      candidate,
      marker;

    for (let i = 0; i < len; ++i) {
      candidate = candidates[i];

      if (!this.homography || this.homography.width !== warpSize || this.homography.height !== warpSize) {
        this.homography = new Image(warpSize, warpSize);
      }

      CV.warp(imageSrc, this.homography, candidate, warpSize);

      CV.threshold(this.homography, this.homography, CV.otsu(this.homography));

      marker = this.getMarker(this.homography, candidate);
      if (marker) {
        markers.push(marker);
      }
    }

    return markers;
  }

  getMarker(imageSrc: IImage, candidate: any) {
    let markSize = this.dictionary.markSize;
    let width = (imageSrc.width / markSize) >>> 0,
      minZero = (width * width) >> 1,
      bits: number[][] = [],
      rotations = [],
      square,
      inc,
      i,
      j;

    for (i = 0; i < markSize; ++i) {
      inc = 0 === i || markSize - 1 === i ? 1 : markSize - 1;

      for (j = 0; j < markSize; j += inc) {
        square = {
          x: j * width,
          y: i * width,
          width: width,
          height: width,
        };
        if (CV.countNonZero(imageSrc, square) > minZero) {
          return null;
        }
      }
    }

    for (i = 0; i < markSize - 2; ++i) {
      bits[i] = [];

      for (j = 0; j < markSize - 2; ++j) {
        square = {
          x: (j + 1) * width,
          y: (i + 1) * width,
          width: width,
          height: width,
        };

        bits[i][j] = CV.countNonZero(imageSrc, square) > minZero ? 1 : 0;
      }
    }

    rotations[0] = bits;

    let foundMin = null;
    let rot = 0;
    for (i = 0; i < 4; i++) {
      const found = this.dictionary.find(rotations[i]);
      if (found && (foundMin === null || found.distance < foundMin.distance)) {
        foundMin = found;
        rot = i;
        if (foundMin.distance === 0) break;
      }
      rotations[i + 1] = this.rotate(rotations[i]);
    }

    if (foundMin) return new Marker(foundMin.id, this.rotate2(candidate, 4 - rot), foundMin.distance);

    return null;
  }

  hammingDistance(bits: any) {
    let ids = [
        [1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 1, 0],
      ],
      dist = 0,
      sum,
      minSum,
      i,
      j,
      k;

    for (i = 0; i < 5; ++i) {
      minSum = Infinity;

      for (j = 0; j < 4; ++j) {
        sum = 0;

        for (k = 0; k < 5; ++k) {
          sum += bits[i][k] === ids[j][k] ? 0 : 1;
        }

        if (sum < minSum) {
          minSum = sum;
        }
      }

      dist += minSum;
    }

    return dist;
  }

  mat2id(bits: any) {
    let id = 0,
      i;

    for (i = 0; i < 5; ++i) {
      id <<= 1;
      id |= bits[i][1];
      id <<= 1;
      id |= bits[i][3];
    }

    return id;
  }

  rotate(src: any): any {
    let dst: any[][] = [],
      len = src.length;

    for (let i = 0; i < len; ++i) {
      dst[i] = [];
      for (let j = 0; j < src[i].length; ++j) {
        dst[i][j] = src[src[i].length - j - 1][i];
      }
    }

    return dst;
  }

  rotate2 = function (src: any, rotation: any) {
    let dst = [],
      len = src.length;

    for (let i = 0; i < len; ++i) {
      dst[i] = src[(rotation + i) % len];
    }

    return dst;
  };
}
