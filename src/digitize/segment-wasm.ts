import segmentWatBytes from "./segment-drawings.wat";
import type { Point2D } from "./types.ts";

export interface SegmentWasmExports {
  memory: WebAssembly.Memory;
  get_heap_ptr(): number;
  set_heap_ptr(ptr: number): void;
  alloc(size: number): number;
  scan_group_bbox(inkPtr: number, w: number, rx: number, ry: number, rw: number, rh: number, outPtr: number): number;
  extract_raster_crop(
    inkPtr: number,
    w: number,
    tightX: number,
    tightY: number,
    tightW: number,
    tightH: number,
    alphaMaskDst: number,
    defaultDataDst: number,
    activeDataDst: number,
    inkPixel32: number,
    activePixel32: number,
  ): void;
}

export interface ScanGroupResult {
  inkCount: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  pMinX: Point2D;
  pMaxX: Point2D;
  pMinY: Point2D;
  pMaxY: Point2D;
  pMinSum: Point2D;
  pMaxSum: Point2D;
  pMinDiff: Point2D;
  pMaxDiff: Point2D;
}

let wasm: SegmentWasmExports | null = null;
let u8View: Uint8Array | null = null;
let i32View: Int32Array | null = null;

function refreshViews(): void {
  if (!wasm) return;
  const buf = wasm.memory.buffer;
  if (!u8View || u8View.buffer !== buf) {
    u8View = new Uint8Array(buf);
    i32View = new Int32Array(buf);
  }
}

/**
 * Initializes the Segment WebAssembly module from a compiled binary or BufferSource.
 */
export async function initSegmentWasm(source: BufferSource | WebAssembly.Module = segmentWatBytes, imports?: WebAssembly.Imports): Promise<SegmentWasmExports> {
  if (wasm) return wasm;

  let instance: WebAssembly.Instance;
  if (source instanceof WebAssembly.Module) {
    instance = await WebAssembly.instantiate(source, imports);
  } else {
    const res = await WebAssembly.instantiate(source, imports);
    instance = res.instance;
  }

  wasm = instance.exports as unknown as SegmentWasmExports;
  refreshViews();
  return wasm;
}

/**
 * Synchronously retrieves or lazily instantiates the WebAssembly module.
 */
export function getSegmentWasm(): SegmentWasmExports {
  if (!wasm) {
    const module = new WebAssembly.Module(segmentWatBytes);
    const instance = new WebAssembly.Instance(module);
    wasm = instance.exports as unknown as SegmentWasmExports;
    refreshViews();
  }
  return wasm;
}

/**
 * Executes a function within a scratch allocation arena, restoring heap pointer on exit.
 */
export function withScratch<T>(fn: (alloc: (bytes: number) => number) => T): T {
  const instance = getSegmentWasm();
  const saved = instance.get_heap_ptr();
  try {
    return fn((bytes: number) => {
      const ptr = instance.alloc(bytes);
      refreshViews();
      return ptr;
    });
  } finally {
    instance.set_heap_ptr(saved);
  }
}

/**
 * Gets direct access to linear memory views.
 */
export function getMemoryViews(): { u8View: Uint8Array; i32View: Int32Array } {
  refreshViews();
  return { u8View: u8View!, i32View: i32View! };
}

/**
 * Invokes SIMD bounding-box scanner on a sub-rectangle of the binary ink image.
 */
export function wasmScanGroupBbox(inkPtr: number, w: number, rx: number, ry: number, rw: number, rh: number): ScanGroupResult {
  const instance = getSegmentWasm();

  return withScratch((alloc) => {
    const outPtr = alloc(84);
    const inkCount = instance.scan_group_bbox(inkPtr, w, rx, ry, rw, rh, outPtr);
    const { i32View } = getMemoryViews();
    const base = outPtr >> 2;

    return {
      inkCount,
      minX: i32View[base + 1],
      minY: i32View[base + 2],
      maxX: i32View[base + 3],
      maxY: i32View[base + 4],
      pMinX: { x: i32View[base + 5], y: i32View[base + 6] },
      pMaxX: { x: i32View[base + 7], y: i32View[base + 8] },
      pMinY: { x: i32View[base + 9], y: i32View[base + 10] },
      pMaxY: { x: i32View[base + 11], y: i32View[base + 12] },
      pMinSum: { x: i32View[base + 13], y: i32View[base + 14] },
      pMaxSum: { x: i32View[base + 15], y: i32View[base + 16] },
      pMinDiff: { x: i32View[base + 17], y: i32View[base + 18] },
      pMaxDiff: { x: i32View[base + 19], y: i32View[base + 20] },
    };
  });
}

/**
 * Invokes SIMD vector raster extraction to write directly to a canvas context
 * using a zero-copy Uint8ClampedArray view on WASM linear memory, completely
 * avoiding intermediate JS heap allocations and data copies.
 */
export function wasmExtractRasterCropToCanvas(
  inkPtr: number,
  w: number,
  tightX: number,
  tightY: number,
  tightW: number,
  tightH: number,
  inkPixel32: number,
  rasterCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null,
): Uint8Array {
  const instance = getSegmentWasm();
  const numPixels = tightW * tightH;

  return withScratch((alloc) => {
    const alphaMaskDst = alloc(numPixels);
    const defaultDataDst = alloc(numPixels * 4);

    instance.extract_raster_crop(
      inkPtr,
      w,
      tightX,
      tightY,
      tightW,
      tightH,
      alphaMaskDst,
      defaultDataDst,
      0, // activeDataDst = 0 (skip eager active canvas generation)
      inkPixel32,
      0,
    );

    if (rasterCtx) {
      // Direct zero-copy Uint8ClampedArray view into WASM memory
      const clampedView = new Uint8ClampedArray(instance.memory.buffer, defaultDataDst, numPixels * 4);
      const imgData = new ImageData(clampedView, tightW, tightH);
      rasterCtx.putImageData(imgData, 0, 0);
    }

    const { u8View } = getMemoryViews();
    return new Uint8Array(u8View.buffer, alphaMaskDst, numPixels).slice();
  });
}

/**
 * Invokes SIMD vector raster extraction to generate alpha mask and RGBA color buffers.
 */
export function wasmExtractRasterCrop(
  inkPtr: number,
  w: number,
  tightX: number,
  tightY: number,
  tightW: number,
  tightH: number,
  inkPixel32: number,
  activePixel32: number,
): { alphaMask: Uint8Array; defaultBytes: Uint8Array; activeBytes: Uint8Array } {
  const instance = getSegmentWasm();
  const numPixels = tightW * tightH;

  return withScratch((alloc) => {
    const alphaMaskDst = alloc(numPixels);
    const defaultDataDst = alloc(numPixels * 4);
    const activeDataDst = alloc(numPixels * 4);

    instance.extract_raster_crop(inkPtr, w, tightX, tightY, tightW, tightH, alphaMaskDst, defaultDataDst, activeDataDst, inkPixel32, activePixel32);

    const { u8View } = getMemoryViews();
    return {
      alphaMask: new Uint8Array(u8View.buffer, alphaMaskDst, numPixels).slice(),
      defaultBytes: new Uint8Array(u8View.buffer, defaultDataDst, numPixels * 4).slice(),
      activeBytes: new Uint8Array(u8View.buffer, activeDataDst, numPixels * 4).slice(),
    };
  });
}
