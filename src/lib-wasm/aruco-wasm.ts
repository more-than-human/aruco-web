import { DICTIONARY_PACKED } from "../shared/dictionary-packed";

export interface IPoint {
  x: number;
  y: number;
}

export interface IImage {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export interface IConfig {
  dictionaryName: "aruco" | "4x4_1000" | "5x5_1000" | "6x6_1000" | "mip_36h12";
  maxHammingDistance?: number;
  limit?: number;
}

interface DictionaryConfig {
  nBits: number;
  tau: number;
  codeList: (number | string | number[])[];
}

export interface ArUcoWasmExports {
  memory: WebAssembly.Memory;
  get_heap_ptr(): number;
  set_heap_ptr(ptr: number): void;
  alloc(size: number): number;

  // Vec3 & Mat3 (SIMD Float32)
  vec3_add(a: number, b: number, out: number): void;
  vec3_sub(a: number, b: number, out: number): void;
  vec3_mult(a: number, b: number, out: number): void;
  vec3_add_scalar(a: number, s: number, out: number): void;
  vec3_mult_scalar(a: number, s: number, out: number): void;
  vec3_dot(a: number, b: number): number;
  vec3_cross(a: number, b: number, out: number): void;
  vec3_square(a: number): number;
  vec3_normalize(a: number): number;
  vec3_inverse(a: number, out: number): void;
  vec3_min_index(a: number): number;

  mat3_transpose(a: number, out: number): void;
  mat3_mult(a: number, b: number, out: number): void;
  mat3_mult_vector(m: number, v: number, out: number): void;
  mat3_from_rows(a: number, b: number, c: number, out: number): void;
  mat3_from_diagonal(a: number, out: number): void;

  // CV (SIMD Accelerated)
  cv_grayscale(src: number, dst: number, width: number, height: number): void;
  cv_threshold(src: number, dst: number, len: number, thres: number): void;
  cv_otsu(src: number, len: number): number;
  cv_stack_box_blur(src: number, dst: number, width: number, height: number, kernelSize: number): void;
  cv_adaptive_threshold(src: number, dst: number, width: number, height: number, kernelSize: number, threshold: number): void;
  cv_gaussian_kernel(kernelSize: number, kernelPtr: number): void;
  cv_gaussian_blur_filter(src: number, dst: number, width: number, height: number, kernelPtr: number, kernelLen: number, horizontal: number): void;
  cv_gaussian_blur(src: number, dst: number, mean: number, width: number, height: number, kernelSize: number): void;
  cv_binary_border(src: number, dst: number, width: number, height: number): void;
  cv_neighborhood_deltas(width: number, deltasPtr: number): void;
  cv_border_following(src: number, pos: number, nbd: number, ptX: number, ptY: number, hole: number, deltas: number, outPts: number): number;
  cv_find_contours(src: number, width: number, height: number, binaryBuf: number, contoursHeaders: number, contoursPts: number): number;
  cv_approx_poly_dp(contourPts: number, len: number, epsilon: number, outPoly: number, stackBuf: number): number;
  cv_square2quad(srcPts: number, outSq: number): void;
  cv_get_perspective_transform(srcPts: number, size: number, outRq: number): void;
  cv_warp(src: number, dst: number, width: number, height: number, contourPts: number, warpSize: number): void;
  cv_is_contour_convex(polyPts: number, len: number): number;
  cv_perimeter(polyPts: number, len: number): number;
  cv_min_edge_length(polyPts: number, len: number): number;
  cv_count_non_zero(src: number, imgWidth: number, sqX: number, sqY: number, sqW: number, sqH: number): number;

  // SVD
  svd_pythag(a: number, b: number): number;
  svd_sign(a: number, b: number): number;
  svd_svdcmp(a: number, m: number, n: number, w: number, v: number): number;

  // Posit
  posit_angle(ax: number, ay: number, bx: number, by: number, cx: number, cy: number): number;
  posit_init(positPtr: number, modelSize: number, focalLength: number): void;
  posit_pos(positPtr: number, ptsPtr: number, epsPtr: number, r1Ptr: number, r2Ptr: number, t1Ptr: number, t2Ptr: number): void;
  posit_get_error(positPtr: number, ptsPtr: number, rPtr: number, tPtr: number): number;
  posit_iterate(positPtr: number, ptsPtr: number, rPtr: number, tPtr: number): number;
  posit_pose(positPtr: number, ptsPtr: number, poseOut: number): void;

  // Dictionary & Detector
  dict_calculate_tau(dictPtr: number): number;
  dict_find_bits(dictPtr: number, bitsMask: bigint): bigint;
  detector_rotate(src: number, dst: number, size: number): void;
  detector_rotate2(src: number, rotation: number, dst: number): void;
  detector_hamming_distance(bitsPtr: number): number;
  detector_mat2id(bitsPtr: number): number;
  detector_find_candidates(
    contoursHeaders: number,
    numContours: number,
    minSize: number,
    epsilon: number,
    minLength: number,
    candidatesOut: number,
    polyScratch: number,
    stackScratch: number,
  ): number;
  detector_clockwise_corners(candidates: number, len: number): void;
  detector_not_too_near(candidates: number, len: number, minDist: number, outCandidates: number, flagsScratch: number): number;
  detector_get_marker(homography: number, warpSize: number, candidatePtr: number, dictPtr: number, markerOut: number): number;
  detector_find_markers(
    greySrc: number,
    width: number,
    height: number,
    candidates: number,
    numCandidates: number,
    warpSize: number,
    homographyBuf: number,
    dictPtr: number,
    markersOut: number,
  ): number;
  detector_detect(
    rgbaSrc: number,
    width: number,
    height: number,
    dictPtr: number,
    greyBuf: number,
    thresBuf: number,
    binaryBuf: number,
    contoursHeaders: number,
    contoursPts: number,
    candidatesBuf1: number,
    candidatesBuf2: number,
    polyScratch: number,
    stackScratch: number,
    homographyBuf: number,
    markersOut: number,
  ): number;
}

let wasm: ArUcoWasmExports;
let u8View: Uint8Array;
let i32View: Int32Array;
let f32View: Float32Array;
let f64View: Float64Array;
let u64View: BigUint64Array;

function refreshViews(): void {
  const buf = wasm.memory.buffer;
  if (!u8View || u8View.buffer !== buf) {
    u8View = new Uint8Array(buf);
    i32View = new Int32Array(buf);
    f32View = new Float32Array(buf);
    f64View = new Float64Array(buf);
    u64View = new BigUint64Array(buf);
  }
}

/**
 * Initializes the ArUco WebAssembly module from a compiled binary or URL/BufferSource.
 */
export async function initArUcoWasm(source: BufferSource | WebAssembly.Module | Response | Promise<Response>): Promise<ArUcoWasmExports> {
  const imports = {
    env: {
      sin: Math.sin,
      cos: Math.cos,
      acos: Math.acos,
      atan: Math.atan,
      exp: Math.exp,
    },
  };

  let instance: WebAssembly.Instance;
  if (source instanceof WebAssembly.Module) {
    instance = await WebAssembly.instantiate(source, imports);
  } else if (source instanceof Response || source instanceof Promise) {
    const res = await WebAssembly.instantiateStreaming(source, imports);
    instance = res.instance;
  } else {
    const res = await WebAssembly.instantiate(source, imports);
    instance = res.instance;
  }

  wasm = instance.exports as unknown as ArUcoWasmExports;
  refreshViews();
  return wasm;
}

/**
 *Scratch Arena helper for temporary Wasm calls (restores heap pointer automatically)
 */
function withScratch<T>(fn: (alloc: (bytes: number) => number) => T): T {
  const saved = wasm.get_heap_ptr();
  try {
    return fn((bytes: number) => wasm.alloc(bytes));
  } finally {
    wasm.set_heap_ptr(saved);
  }
}

// ============================================================================
// Vec3 & Mat3 (Backed by Wasm SIMD Float32 operations)
// ============================================================================
const MATH_SCRATCH_A = 0x1800; // 6144
const MATH_SCRATCH_B = 0x1840; // 6208
const MATH_SCRATCH_C = 0x1880; // 6272
const MATH_SCRATCH_OUT = 0x18c0; // 6336

export class Vec3 {
  public data: Float32Array;

  constructor(x = 0, y = 0, z = 0) {
    this.data = new Float32Array([x, y, z]);
  }

  get v(): number[] {
    return [this.data[0], this.data[1], this.data[2]];
  }

  set v(arr: number[]) {
    this.data[0] = arr[0] || 0;
    this.data[1] = arr[1] || 0;
    this.data[2] = arr[2] || 0;
  }

  get x(): number {
    return this.data[0];
  }
  set x(val: number) {
    this.data[0] = val;
  }
  get y(): number {
    return this.data[1];
  }
  set y(val: number) {
    this.data[1] = val;
  }
  get z(): number {
    return this.data[2];
  }
  set z(val: number) {
    this.data[2] = val;
  }

  copy(other: Vec3): this {
    this.data.set(other.data);
    return this;
  }

  clone(): Vec3 {
    return new Vec3(this.data[0], this.data[1], this.data[2]);
  }

  addInPlace(other: Vec3): this {
    Vec3.add(this, other, this);
    return this;
  }

  subInPlace(other: Vec3): this {
    Vec3.sub(this, other, this);
    return this;
  }

  multScalarInPlace(scalar: number): this {
    Vec3.multScalar(this, scalar, this);
    return this;
  }

  normalizeInPlace(): number {
    return this.normalize();
  }

  normalize(): number {
    refreshViews();
    f32View.set(this.data, MATH_SCRATCH_A >> 2);
    const len = wasm.vec3_normalize(MATH_SCRATCH_A);
    this.data.set(f32View.subarray(MATH_SCRATCH_A >> 2, (MATH_SCRATCH_A >> 2) + 3));
    return len;
  }

  square(): number {
    refreshViews();
    f32View.set(this.data, MATH_SCRATCH_A >> 2);
    return wasm.vec3_square(MATH_SCRATCH_A);
  }

  minIndex(): number {
    refreshViews();
    f32View.set(this.data, MATH_SCRATCH_A >> 2);
    return wasm.vec3_min_index(MATH_SCRATCH_A);
  }

  static add(a: Vec3, b: Vec3, out: Vec3 = new Vec3()): Vec3 {
    refreshViews();
    f32View.set(a.data, MATH_SCRATCH_A >> 2);
    f32View.set(b.data, MATH_SCRATCH_B >> 2);
    wasm.vec3_add(MATH_SCRATCH_A, MATH_SCRATCH_B, MATH_SCRATCH_OUT);
    out.data.set(f32View.subarray(MATH_SCRATCH_OUT >> 2, (MATH_SCRATCH_OUT >> 2) + 3));
    return out;
  }

  static sub(a: Vec3, b: Vec3, out: Vec3 = new Vec3()): Vec3 {
    refreshViews();
    f32View.set(a.data, MATH_SCRATCH_A >> 2);
    f32View.set(b.data, MATH_SCRATCH_B >> 2);
    wasm.vec3_sub(MATH_SCRATCH_A, MATH_SCRATCH_B, MATH_SCRATCH_OUT);
    out.data.set(f32View.subarray(MATH_SCRATCH_OUT >> 2, (MATH_SCRATCH_OUT >> 2) + 3));
    return out;
  }

  static mult(a: Vec3, b: Vec3, out: Vec3 = new Vec3()): Vec3 {
    refreshViews();
    f32View.set(a.data, MATH_SCRATCH_A >> 2);
    f32View.set(b.data, MATH_SCRATCH_B >> 2);
    wasm.vec3_mult(MATH_SCRATCH_A, MATH_SCRATCH_B, MATH_SCRATCH_OUT);
    out.data.set(f32View.subarray(MATH_SCRATCH_OUT >> 2, (MATH_SCRATCH_OUT >> 2) + 3));
    return out;
  }

  static addScalar(a: Vec3, scalar: number, out: Vec3 = new Vec3()): Vec3 {
    refreshViews();
    f32View.set(a.data, MATH_SCRATCH_A >> 2);
    wasm.vec3_add_scalar(MATH_SCRATCH_A, scalar, MATH_SCRATCH_OUT);
    out.data.set(f32View.subarray(MATH_SCRATCH_OUT >> 2, (MATH_SCRATCH_OUT >> 2) + 3));
    return out;
  }

  static multScalar(a: Vec3, scalar: number, out: Vec3 = new Vec3()): Vec3 {
    refreshViews();
    f32View.set(a.data, MATH_SCRATCH_A >> 2);
    wasm.vec3_mult_scalar(MATH_SCRATCH_A, scalar, MATH_SCRATCH_OUT);
    out.data.set(f32View.subarray(MATH_SCRATCH_OUT >> 2, (MATH_SCRATCH_OUT >> 2) + 3));
    return out;
  }

  static cross(a: Vec3, b: Vec3, out: Vec3 = new Vec3()): Vec3 {
    refreshViews();
    f32View.set(a.data, MATH_SCRATCH_A >> 2);
    f32View.set(b.data, MATH_SCRATCH_B >> 2);
    wasm.vec3_cross(MATH_SCRATCH_A, MATH_SCRATCH_B, MATH_SCRATCH_OUT);
    out.data.set(f32View.subarray(MATH_SCRATCH_OUT >> 2, (MATH_SCRATCH_OUT >> 2) + 3));
    return out;
  }

  static dot(a: Vec3, b: Vec3): number {
    refreshViews();
    f32View.set(a.data, MATH_SCRATCH_A >> 2);
    f32View.set(b.data, MATH_SCRATCH_B >> 2);
    return wasm.vec3_dot(MATH_SCRATCH_A, MATH_SCRATCH_B);
  }

  static inverse(a: Vec3): Vec3 {
    refreshViews();
    const out = new Vec3();
    f32View.set(a.data, MATH_SCRATCH_A >> 2);
    wasm.vec3_inverse(MATH_SCRATCH_A, MATH_SCRATCH_OUT);
    out.data.set(f32View.subarray(MATH_SCRATCH_OUT >> 2, (MATH_SCRATCH_OUT >> 2) + 3));
    return out;
  }

  static addNew(a: Vec3, b: Vec3): Vec3 {
    return Vec3.add(a, b, new Vec3());
  }
  static subNew(a: Vec3, b: Vec3): Vec3 {
    return Vec3.sub(a, b, new Vec3());
  }
  static multNew(a: Vec3, b: Vec3): Vec3 {
    return Vec3.mult(a, b, new Vec3());
  }
  static crossNew(a: Vec3, b: Vec3): Vec3 {
    return Vec3.cross(a, b, new Vec3());
  }
  static multScalarNew(a: Vec3, scalar: number): Vec3 {
    return Vec3.multScalar(a, scalar, new Vec3());
  }
  static addScalarNew(a: Vec3, scalar: number): Vec3 {
    return Vec3.addScalar(a, scalar, new Vec3());
  }
}

export class Mat3 {
  public data: Float32Array;

  constructor() {
    this.data = new Float32Array(9);
  }

  get m(): number[][] {
    const d = this.data;
    return [
      [d[0], d[1], d[2]],
      [d[3], d[4], d[5]],
      [d[6], d[7], d[8]],
    ];
  }

  set m(matrix: number[][]) {
    this.data.set([matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][0], matrix[1][1], matrix[1][2], matrix[2][0], matrix[2][1], matrix[2][2]]);
  }

  get(row: number, col: number): number {
    return this.data[row * 3 + col];
  }

  set(row: number, col: number, value: number): void {
    this.data[row * 3 + col] = value;
  }

  copy(a: Mat3): this {
    this.data.set(a.data);
    return this;
  }

  static clone(a: Mat3): Mat3 {
    const matrix = new Mat3();
    matrix.data.set(a.data);
    return matrix;
  }

  transposeInPlace(): this {
    Mat3.transpose(this, this);
    return this;
  }

  static transpose(a: Mat3, out: Mat3 = new Mat3()): Mat3 {
    refreshViews();
    f32View.set(a.data, MATH_SCRATCH_A >> 2);
    wasm.mat3_transpose(MATH_SCRATCH_A, MATH_SCRATCH_OUT);
    out.data.set(f32View.subarray(MATH_SCRATCH_OUT >> 2, (MATH_SCRATCH_OUT >> 2) + 9));
    return out;
  }

  static mult(a: Mat3, b: Mat3, out: Mat3 = new Mat3()): Mat3 {
    refreshViews();
    f32View.set(a.data, MATH_SCRATCH_A >> 2);
    f32View.set(b.data, MATH_SCRATCH_B >> 2);
    wasm.mat3_mult(MATH_SCRATCH_A, MATH_SCRATCH_B, MATH_SCRATCH_OUT);
    out.data.set(f32View.subarray(MATH_SCRATCH_OUT >> 2, (MATH_SCRATCH_OUT >> 2) + 9));
    return out;
  }

  static multVector(m: Mat3, v: Vec3, out: Vec3 = new Vec3()): Vec3 {
    refreshViews();
    f32View.set(m.data, MATH_SCRATCH_A >> 2);
    f32View.set(v.data, MATH_SCRATCH_B >> 2);
    wasm.mat3_mult_vector(MATH_SCRATCH_A, MATH_SCRATCH_B, MATH_SCRATCH_OUT);
    out.data.set(f32View.subarray(MATH_SCRATCH_OUT >> 2, (MATH_SCRATCH_OUT >> 2) + 3));
    return out;
  }

  static fromRows(a: Vec3, b: Vec3, c: Vec3): Mat3 {
    refreshViews();
    const out = new Mat3();
    f32View.set(a.data, MATH_SCRATCH_A >> 2);
    f32View.set(b.data, MATH_SCRATCH_B >> 2);
    f32View.set(c.data, MATH_SCRATCH_C >> 2);
    wasm.mat3_from_rows(MATH_SCRATCH_A, MATH_SCRATCH_B, MATH_SCRATCH_C, MATH_SCRATCH_OUT);
    out.data.set(f32View.subarray(MATH_SCRATCH_OUT >> 2, (MATH_SCRATCH_OUT >> 2) + 9));
    return out;
  }

  static fromDiagonal(a: Vec3): Mat3 {
    refreshViews();
    const out = new Mat3();
    f32View.set(a.data, MATH_SCRATCH_A >> 2);
    wasm.mat3_from_diagonal(MATH_SCRATCH_A, MATH_SCRATCH_OUT);
    out.data.set(f32View.subarray(MATH_SCRATCH_OUT >> 2, (MATH_SCRATCH_OUT >> 2) + 9));
    return out;
  }

  static identity(): Mat3 {
    const matrix = new Mat3();
    matrix.data[0] = 1;
    matrix.data[4] = 1;
    matrix.data[8] = 1;
    return matrix;
  }

  static multNew(a: Mat3, b: Mat3): Mat3 {
    return Mat3.mult(a, b, new Mat3());
  }
  static transposeNew(a: Mat3): Mat3 {
    return Mat3.transpose(a, new Mat3());
  }
  static multVectorNew(m: Mat3, v: Vec3): Vec3 {
    return Mat3.multVector(m, v, new Vec3());
  }

  column(index: number): Vec3 {
    const d = this.data;
    return new Vec3(d[index], d[3 + index], d[6 + index]);
  }

  row(index: number): Vec3 {
    const d = this.data;
    const offset = index * 3;
    return new Vec3(d[offset], d[offset + 1], d[offset + 2]);
  }
}

// ============================================================================
// Image & Computer Vision (CV) Wrapper
// ============================================================================
export class Image implements IImage {
  public width: number;
  public height: number;
  public data: Uint8ClampedArray;

  constructor(width: number = 0, height: number = 0, data?: Uint8ClampedArray | Uint8Array | number[] | ArrayBuffer) {
    this.width = width;
    this.height = height;
    if (data instanceof Uint8ClampedArray) {
      this.data = data;
    } else if (data instanceof Uint8Array || Array.isArray(data) || data instanceof ArrayBuffer) {
      this.data = new Uint8ClampedArray(data);
    } else {
      this.data = new Uint8ClampedArray(width * height);
    }
  }

  static fromImageData(imageData: ImageData): Image {
    return new Image(imageData.width, imageData.height, imageData.data);
  }

  toImageData(): ImageData {
    const imageData = new ImageData(this.width, this.height);
    imageData.data.set(this.data);
    return imageData;
  }

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

export class CV {
  static grayscale(imageSrc: IImage): Image {
    return withScratch((alloc) => {
      refreshViews();
      const srcPtr = alloc(imageSrc.data.length);
      const dstLen = imageSrc.width * imageSrc.height;
      const dstPtr = alloc(dstLen);
      u8View.set(imageSrc.data, srcPtr);
      wasm.cv_grayscale(srcPtr, dstPtr, imageSrc.width, imageSrc.height);
      const dst = new Image(imageSrc.width, imageSrc.height);
      dst.data.set(u8View.subarray(dstPtr, dstPtr + dstLen));
      return dst;
    });
  }

  static threshold(imageSrc: IImage, imageDst: IImage, threshold: number): IImage {
    return withScratch((alloc) => {
      refreshViews();
      const len = imageSrc.data.length;
      const srcPtr = alloc(len);
      const dstPtr = alloc(len);
      u8View.set(imageSrc.data, srcPtr);
      wasm.cv_threshold(srcPtr, dstPtr, len, threshold);
      imageDst.width = imageSrc.width;
      imageDst.height = imageSrc.height;
      imageDst.data.set(u8View.subarray(dstPtr, dstPtr + len));
      return imageDst;
    });
  }

  static adaptiveThreshold(imageSrc: IImage, imageDst: IImage, kernelSize: number, threshold: number): IImage {
    return withScratch((alloc) => {
      refreshViews();
      const len = imageSrc.data.length;
      const srcPtr = alloc(len);
      const dstPtr = alloc(len);
      u8View.set(imageSrc.data, srcPtr);
      wasm.cv_adaptive_threshold(srcPtr, dstPtr, imageSrc.width, imageSrc.height, kernelSize, threshold);
      imageDst.width = imageSrc.width;
      imageDst.height = imageSrc.height;
      imageDst.data.set(u8View.subarray(dstPtr, dstPtr + len));
      return imageDst;
    });
  }

  static otsu(imageSrc: IImage): number {
    return withScratch((alloc) => {
      refreshViews();
      const len = imageSrc.data.length;
      const srcPtr = alloc(len);
      u8View.set(imageSrc.data, srcPtr);
      return wasm.cv_otsu(srcPtr, len);
    });
  }

  static stackBoxBlur(imageSrc: IImage, imageDst: IImage, kernelSize: number): IImage {
    return withScratch((alloc) => {
      refreshViews();
      const len = imageSrc.data.length;
      const srcPtr = alloc(len);
      const dstPtr = alloc(len);
      u8View.set(imageSrc.data, srcPtr);
      wasm.cv_stack_box_blur(srcPtr, dstPtr, imageSrc.width, imageSrc.height, kernelSize);
      imageDst.width = imageSrc.width;
      imageDst.height = imageSrc.height;
      imageDst.data.set(u8View.subarray(dstPtr, dstPtr + len));
      return imageDst;
    });
  }

  static gaussianBlur(imageSrc: IImage, imageDst: IImage, imageMean: IImage, kernelSize: number): IImage {
    return withScratch((alloc) => {
      refreshViews();
      const len = imageSrc.data.length;
      const srcPtr = alloc(len);
      const dstPtr = alloc(len);
      const meanPtr = alloc(len);
      u8View.set(imageSrc.data, srcPtr);
      wasm.cv_gaussian_blur(srcPtr, dstPtr, meanPtr, imageSrc.width, imageSrc.height, kernelSize);
      imageDst.width = imageMean.width = imageSrc.width;
      imageDst.height = imageMean.height = imageSrc.height;
      imageMean.data.set(u8View.subarray(meanPtr, meanPtr + len));
      imageDst.data.set(u8View.subarray(dstPtr, dstPtr + len));
      return imageDst;
    });
  }

  static gaussianKernel(kernelSize: number): number[] {
    return withScratch((alloc) => {
      refreshViews();
      const kPtr = alloc(kernelSize * 8);
      wasm.cv_gaussian_kernel(kernelSize, kPtr);
      return Array.from(f64View.subarray(kPtr >> 3, (kPtr >> 3) + kernelSize));
    });
  }

  static findContours(imageSrc: IImage, _binary?: number[]): any[] {
    return withScratch((alloc) => {
      refreshViews();
      const w = imageSrc.width;
      const h = imageSrc.height;
      const srcPtr = alloc(w * h);
      const binLen = (w + 2) * (h + 2);
      const binPtr = alloc(binLen * 4);
      const hdrsPtr = alloc(w * h * 12);
      const ptsPtr = alloc(w * h * 8);

      u8View.set(imageSrc.data, srcPtr);
      const numContours = wasm.cv_find_contours(srcPtr, w, h, binPtr, hdrsPtr, ptsPtr);

      const contours: any[] = [];
      for (let i = 0; i < numContours; i++) {
        const hdrIdx = (hdrsPtr >> 2) + i * 3;
        const cPtsPtr = i32View[hdrIdx];
        const cLen = i32View[hdrIdx + 1];
        const hole = i32View[hdrIdx + 2] !== 0;
        const contour: any = [];
        contour.hole = hole;
        for (let j = 0; j < cLen; j++) {
          const pIdx = (cPtsPtr >> 2) + j * 2;
          contour.push({ x: i32View[pIdx], y: i32View[pIdx + 1] });
        }
        contours.push(contour);
      }
      return contours;
    });
  }

  static approxPolyDP(contour: IPoint[], epsilon: number): IPoint[] {
    return withScratch((alloc) => {
      refreshViews();
      const len = contour.length;
      const inPtr = alloc(len * 8);
      const outPtr = alloc(len * 8);
      const stackPtr = alloc(len * 8);

      for (let i = 0; i < len; i++) {
        i32View[(inPtr >> 2) + i * 2] = contour[i].x;
        i32View[(inPtr >> 2) + i * 2 + 1] = contour[i].y;
      }

      const outLen = wasm.cv_approx_poly_dp(inPtr, len, epsilon, outPtr, stackPtr);
      const poly: IPoint[] = [];
      for (let i = 0; i < outLen; i++) {
        poly.push({
          x: i32View[(outPtr >> 2) + i * 2],
          y: i32View[(outPtr >> 2) + i * 2 + 1],
        });
      }
      return poly;
    });
  }

  static warp(imageSrc: IImage, imageDst: IImage, contour: IPoint[], warpSize: number): IImage {
    return withScratch((alloc) => {
      refreshViews();
      const srcPtr = alloc(imageSrc.data.length);
      const dstLen = warpSize * warpSize;
      const dstPtr = alloc(dstLen);
      const quadPtr = alloc(64);

      u8View.set(imageSrc.data, srcPtr);
      for (let i = 0; i < 4; i++) {
        f64View[(quadPtr >> 3) + i * 2] = contour[i].x;
        f64View[(quadPtr >> 3) + i * 2 + 1] = contour[i].y;
      }

      wasm.cv_warp(srcPtr, dstPtr, imageSrc.width, imageSrc.height, quadPtr, warpSize);

      if (imageDst.data.length < dstLen || imageDst.width !== warpSize || imageDst.height !== warpSize) {
        imageDst.width = warpSize;
        imageDst.height = warpSize;
        imageDst.data = new Uint8ClampedArray(dstLen);
      }
      imageDst.data.set(u8View.subarray(dstPtr, dstPtr + dstLen));
      return imageDst;
    });
  }

  static getPerspectiveTransform(src: IPoint[], size: number): number[] {
    return withScratch((alloc) => {
      refreshViews();
      const srcPtr = alloc(64);
      const outPtr = alloc(72);
      for (let i = 0; i < 4; i++) {
        f64View[(srcPtr >> 3) + i * 2] = src[i].x;
        f64View[(srcPtr >> 3) + i * 2 + 1] = src[i].y;
      }
      wasm.cv_get_perspective_transform(srcPtr, size, outPtr);
      return Array.from(f64View.subarray(outPtr >> 3, (outPtr >> 3) + 9));
    });
  }

  static square2quad(src: IPoint[]): number[] {
    return withScratch((alloc) => {
      refreshViews();
      const srcPtr = alloc(64);
      const outPtr = alloc(72);
      for (let i = 0; i < 4; i++) {
        f64View[(srcPtr >> 3) + i * 2] = src[i].x;
        f64View[(srcPtr >> 3) + i * 2 + 1] = src[i].y;
      }
      wasm.cv_square2quad(srcPtr, outPtr);
      return Array.from(f64View.subarray(outPtr >> 3, (outPtr >> 3) + 9));
    });
  }

  static isContourConvex(contour: IPoint[]): boolean {
    return withScratch((alloc) => {
      refreshViews();
      const len = contour.length;
      const ptr = alloc(len * 8);
      for (let i = 0; i < len; i++) {
        i32View[(ptr >> 2) + i * 2] = contour[i].x;
        i32View[(ptr >> 2) + i * 2 + 1] = contour[i].y;
      }
      return wasm.cv_is_contour_convex(ptr, len) !== 0;
    });
  }

  static perimeter(poly: IPoint[]): number {
    return withScratch((alloc) => {
      refreshViews();
      const len = poly.length;
      const ptr = alloc(len * 8);
      for (let i = 0; i < len; i++) {
        i32View[(ptr >> 2) + i * 2] = poly[i].x;
        i32View[(ptr >> 2) + i * 2 + 1] = poly[i].y;
      }
      return wasm.cv_perimeter(ptr, len);
    });
  }

  static minEdgeLength(poly: IPoint[]): number {
    return withScratch((alloc) => {
      refreshViews();
      const len = poly.length;
      const ptr = alloc(len * 8);
      for (let i = 0; i < len; i++) {
        i32View[(ptr >> 2) + i * 2] = poly[i].x;
        i32View[(ptr >> 2) + i * 2 + 1] = poly[i].y;
      }
      return wasm.cv_min_edge_length(ptr, len);
    });
  }

  static countNonZero(imageSrc: IImage, square: { x: number; y: number; width: number; height: number }): number {
    return withScratch((alloc) => {
      refreshViews();
      const srcPtr = alloc(imageSrc.data.length);
      u8View.set(imageSrc.data, srcPtr);
      return wasm.cv_count_non_zero(srcPtr, imageSrc.width, square.x, square.y, square.width, square.height);
    });
  }

  static binaryBorder(imageSrc: IImage, dst: number[]): number[] {
    return withScratch((alloc) => {
      refreshViews();
      const w = imageSrc.width;
      const h = imageSrc.height;
      const srcPtr = alloc(w * h);
      const dstLen = (w + 2) * (h + 2);
      const dstPtr = alloc(dstLen * 4);
      u8View.set(imageSrc.data, srcPtr);
      wasm.cv_binary_border(srcPtr, dstPtr, w, h);
      for (let i = 0; i < dstLen; i++) {
        dst[i] = i32View[(dstPtr >> 2) + i];
      }
      return dst;
    });
  }
}

// ============================================================================
// SVD, Posit, and Pose Wrapper
// ============================================================================
export class SVD {
  static svdcmp(a: number[][], m: number, n: number, w: number[], v: number[][]): boolean {
    return withScratch((alloc) => {
      refreshViews();
      const aPtr = alloc(m * n * 8);
      const wPtr = alloc(n * 8);
      const vPtr = alloc(n * n * 8);

      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          f64View[(aPtr >> 3) + i * n + j] = a[i][j];
        }
      }

      const ok = wasm.svd_svdcmp(aPtr, m, n, wPtr, vPtr) !== 0;

      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          a[i][j] = f64View[(aPtr >> 3) + i * n + j];
        }
      }
      for (let i = 0; i < n; i++) {
        w[i] = f64View[(wPtr >> 3) + i];
        for (let j = 0; j < n; j++) {
          v[i][j] = f64View[(vPtr >> 3) + i * n + j];
        }
      }
      return ok;
    });
  }

  static pythag(a: number, b: number): number {
    return wasm.svd_pythag(a, b);
  }

  static sign(a: number, b: number): number {
    return wasm.svd_sign(a, b);
  }
}

export class Pose {
  public bestError: number;
  public bestRotation: number[][];
  public bestTranslation: number[];
  public alternativeError: number;
  public alternativeRotation: number[][];
  public alternativeTranslation: number[];

  constructor(error1: number, rotation1: number[][], translation1: number[], error2: number, rotation2: number[][], translation2: number[]) {
    this.bestError = error1;
    this.bestRotation = rotation1;
    this.bestTranslation = translation1;
    this.alternativeError = error2;
    this.alternativeRotation = rotation2;
    this.alternativeTranslation = translation2;
  }
}

export class Posit {
  public modelSize: number;
  public focalLength: number;
  private positPtr: number;

  constructor(modelSize: number, focalLength: number) {
    this.modelSize = modelSize;
    this.focalLength = focalLength;
    this.positPtr = wasm.alloc(272);
    wasm.posit_init(this.positPtr, modelSize, focalLength);
  }

  pose(points: IPoint[]): Pose {
    return withScratch((alloc) => {
      refreshViews();
      const ptsPtr = alloc(64);
      const poseOut = alloc(208);

      for (let i = 0; i < 4; i++) {
        f64View[(ptsPtr >> 3) + i * 2] = points[i].x;
        f64View[(ptsPtr >> 3) + i * 2 + 1] = points[i].y;
      }

      wasm.posit_pose(this.positPtr, ptsPtr, poseOut);

      const base = poseOut >> 3;
      const bestError = f64View[base];
      const bestRotation = [
        [f64View[base + 1], f64View[base + 2], f64View[base + 3]],
        [f64View[base + 4], f64View[base + 5], f64View[base + 6]],
        [f64View[base + 7], f64View[base + 8], f64View[base + 9]],
      ];
      const bestTranslation = [f64View[base + 10], f64View[base + 11], f64View[base + 12]];

      const altError = f64View[base + 13];
      const altRotation = [
        [f64View[base + 14], f64View[base + 15], f64View[base + 16]],
        [f64View[base + 17], f64View[base + 18], f64View[base + 19]],
        [f64View[base + 20], f64View[base + 21], f64View[base + 22]],
      ];
      const altTranslation = [f64View[base + 23], f64View[base + 24], f64View[base + 25]];

      return new Pose(bestError, bestRotation, bestTranslation, altError, altRotation, altTranslation);
    });
  }
}

// ============================================================================
// Dictionary, Marker, and Detector Wrapper
// ============================================================================
export class DICTIONARIES {
  static getDictionary(name: string): DictionaryConfig {
    const dictionary = DICTIONARY_PACKED[name.toLocaleLowerCase()];
    if (dictionary && typeof dictionary === "object" && "nBits" in dictionary && "tau" in dictionary && "codeList" in dictionary) {
      return dictionary as DictionaryConfig;
    }
    throw new Error(`The dictionary "${name}" is not recognized.`);
  }
}

export class Dictionary {
  private codeList: string[] = [];
  public tau: number = 0;
  public nBits: number = 0;
  public markSize: number = 0;
  public limit?: number;
  private dicName: string;
  public dictPtr: number = 0;

  constructor(dicName: string, limit?: number) {
    this.dicName = dicName;
    this.limit = limit;
    this.initialize(dicName, limit);
  }

  private initialize(dicName: string, limit?: number): void {
    const dictionary = DICTIONARIES.getDictionary(dicName);
    this.nBits = dictionary.nBits;
    this.markSize = Math.sqrt(dictionary.nBits) + 2;
    const count = limit !== undefined && limit >= 0 ? Math.min(limit, dictionary.codeList.length) : dictionary.codeList.length;

    refreshViews();
    this.dictPtr = wasm.alloc(16 + count * 8);
    i32View[this.dictPtr >> 2] = this.nBits;
    i32View[(this.dictPtr >> 2) + 1] = this.markSize;
    i32View[(this.dictPtr >> 2) + 3] = count;

    const codesBase = (this.dictPtr + 16) >> 3;
    for (let i = 0; i < count; i++) {
      const binStr = this.parseCode(dictionary.codeList[i], dictionary.nBits);
      this.codeList.push(binStr);
      u64View[codesBase + i] = BigInt("0b" + binStr);
    }

    if (dictionary.tau) {
      this.tau = dictionary.tau;
      i32View[(this.dictPtr >> 2) + 2] = this.tau;
    } else {
      this.tau = wasm.dict_calculate_tau(this.dictPtr);
    }
  }

  public setTau(tau: number): void {
    this.tau = tau;
    refreshViews();
    i32View[(this.dictPtr >> 2) + 2] = tau;
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
    }
    throw new Error(`Invalid code: ${rawCode}`);
  }

  find(bits: (string | number)[][]): { id: number; distance: number } | null {
    const binaryString = bits.flat().join("");
    const mask = BigInt("0b" + binaryString);
    const res = wasm.dict_find_bits(this.dictPtr, mask);
    if (res < 0n) return null;
    return {
      id: Number(res & 0xffffffffn),
      distance: Number(res >> 32n),
    };
  }

  generateSVG(id: number): string {
    const code = this.codeList[id];
    if (code == null) {
      throw new Error(`The id "${id}" is not valid for the dictionary "${this.dicName}". ID must be between 0 and ${this.codeList.length - 1} included.`);
    }
    const size = this.markSize - 2;
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size + 4} ${size + 4}">`;
    svg += `<rect x="0" y="0" width="${size + 4}" height="${size + 4}" fill="white"/>`;
    svg += `<rect x="1" y="1" width="${size + 2}" height="${size + 2}" fill="black"/>`;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (code[y * size + x] === "1") {
          svg += `<rect x="${x + 2}" y="${y + 2}" width="1" height="1" fill="white"/>`;
        }
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

  constructor(id: number, corners: IPoint[], hammingDistance: number) {
    this.id = id;
    this.corners = corners;
    this.hammingDistance = hammingDistance;
  }
}

export class Detector {
  public config: IConfig;
  private dictionary: Dictionary;
  private streamConfig: any;
  private mjpeg: any;

  constructor(config: IConfig) {
    this.config = config;
    this.dictionary = new Dictionary(config.dictionaryName, config.limit);
    if (config.maxHammingDistance != null) {
      this.dictionary.setTau(config.maxHammingDistance);
    }
  }

  detectImage(width: number, height: number, data: Uint8ClampedArray | Uint8Array): Marker[] {
    return this.detect({ width, height, data });
  }

  detect(image: { width: number; height: number; data: Uint8ClampedArray | Uint8Array }): Marker[] {
    return withScratch((alloc) => {
      refreshViews();
      const { width, height, data } = image;
      const numPixels = width * height;

      const rgbaPtr = alloc(data.length);
      const greyPtr = alloc(numPixels);
      const thresPtr = alloc(numPixels);
      const binaryPtr = alloc((width + 2) * (height + 2) * 4);
      const contoursHeaders = alloc(Math.max(4096, numPixels) * 12);
      const contoursPts = alloc(Math.max(16384, numPixels * 2) * 8);
      const candidatesBuf1 = alloc(4096 * 32);
      const candidatesBuf2 = alloc(4096 * 32);
      const polyScratch = alloc(16384 * 8);
      const stackScratch = alloc(16384 * 8);
      const homographyBuf = alloc(49 * 49);
      const markersOut = alloc(1024 * 40);

      refreshViews();
      u8View.set(data, rgbaPtr);

      const numMarkers = wasm.detector_detect(
        rgbaPtr,
        width,
        height,
        this.dictionary.dictPtr,
        greyPtr,
        thresPtr,
        binaryPtr,
        contoursHeaders,
        contoursPts,
        candidatesBuf1,
        candidatesBuf2,
        polyScratch,
        stackScratch,
        homographyBuf,
        markersOut,
      );

      const markers: Marker[] = [];
      for (let i = 0; i < numMarkers; i++) {
        const base = (markersOut + i * 40) >> 2;
        const id = i32View[base];
        const dist = i32View[base + 1];
        const corners: IPoint[] = [
          { x: i32View[base + 2], y: i32View[base + 3] },
          { x: i32View[base + 4], y: i32View[base + 5] },
          { x: i32View[base + 6], y: i32View[base + 7] },
          { x: i32View[base + 8], y: i32View[base + 9] },
        ];
        markers.push(new Marker(id, corners, dist));
      }
      return markers;
    });
  }

  detectStreamInit(width: number, height: number, callback: (image: IImage, markerList: Marker[]) => void): void {
    this.streamConfig = {
      width,
      height,
      imageSize: width * height * 4,
      index: 0,
      imageData: new Uint8ClampedArray(width * height * 4),
      callback: callback || function () {},
    };
  }

  detectStream(data: ArrayLike<number>): void {
    for (let i = 0; i < data.length; i++) {
      this.streamConfig.imageData[this.streamConfig.index] = data[i];
      this.streamConfig.index = (this.streamConfig.index + 1) % this.streamConfig.imageSize;
      if (this.streamConfig.index === 0) {
        const image: IImage = {
          width: this.streamConfig.width,
          height: this.streamConfig.height,
          data: this.streamConfig.imageData,
        };
        const markerList = this.detect(image);
        this.streamConfig.callback(image, markerList);
      }
    }
  }

  detectMJPEGStreamInit(
    width: number,
    height: number,
    callback: (image: IImage, markerList: Marker[]) => void,
    decoderFn: (jpeg: number[]) => Uint8Array,
  ): void {
    this.mjpeg = {
      decoderFn,
      chunks: [] as number[][],
      SOI: [0xff, 0xd8],
      EOI: [0xff, 0xd9],
    };
    this.detectStreamInit(width, height, callback);
  }

  detectMJPEGStream(chunk: number[]): void {
    const eoiPos = chunk.findIndex((el, idx, arr) => this.mjpeg.EOI[0] === el && arr.length > idx + 1 && this.mjpeg.EOI[1] === arr[idx + 1]);
    const soiPos = chunk.findIndex((el, idx, arr) => this.mjpeg.SOI[0] === el && arr.length > idx + 1 && this.mjpeg.SOI[1] === arr[idx + 1]);

    if (eoiPos === -1) {
      this.mjpeg.chunks.push(chunk);
    } else {
      const part1 = chunk.slice(0, eoiPos + 2);
      if (part1.length) this.mjpeg.chunks.push(part1);
      if (this.mjpeg.chunks.length) {
        const jpegImage = this.mjpeg.chunks.flat();
        const rgba = this.mjpeg.decoderFn(jpegImage);
        this.detectStream(rgba);
      }
      this.mjpeg.chunks = [];
    }
    if (soiPos > -1) {
      this.mjpeg.chunks = [chunk.slice(soiPos)];
    }
  }
}
