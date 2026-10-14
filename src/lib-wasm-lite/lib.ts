import initArucoWasm from "./aruco4x4.wasm?init";
import type { DetectedMarker, DetectionResult } from "./types.ts";

/**
 * Standard OpenCV DICT_4X4_50 ArUco marker codes.
 * Each marker has a 4x4 internal binary grid (16 bits).
 * 0 = black cell, 1 = white cell.
 */
export const DICT_4X4_50: number[] = [
  0xb532, 0x0f9a, 0x332d, 0x9946, 0x549e, 0x79cd, 0x9e2e, 0xc4f2, 0xfeda, 0xcf56, 0xf991, 0x11a7, 0x0eb7, 0x2a0f, 0x24b1, 0x263e, 0x4665, 0x6600, 0x6c5e,
  0x76af, 0x868b, 0xb02b, 0xccd5, 0xdd82, 0xfe47, 0x9471, 0xace4, 0xa554, 0x2123, 0x346f, 0x4415, 0x57b2, 0x9ecf, 0xf0cb, 0x08ae, 0x0929, 0x1875, 0x04ff,
  0x0df6, 0x1c5a, 0x1718, 0x2a28, 0x328c, 0x38b2, 0x24e8, 0x2eeb, 0x2d3f, 0x4b64, 0x502e, 0x5013,
];

interface ArucoWasmExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  ensureBuffers(width: number, height: number): number;
  getOutputPtr(): number;
  detect(width: number, height: number, maxHammingDistance: number): number;
}

export type ArucoWasmSource = BufferSource | WebAssembly.Module | string | URL;

export class Aruco4x4Detector {
  private exports: ArucoWasmExports | null = null;
  private wasmU8: Uint8Array | null = null;
  private wasmI32: Int32Array | null = null;
  private wasmF32: Float32Array | null = null;
  private lastBuffer: ArrayBufferLike | null = null;
  private wasmSource?: ArucoWasmSource;

  constructor(wasmSource?: ArucoWasmSource) {
    this.wasmSource = wasmSource;
    if (wasmSource instanceof WebAssembly.Module) {
      const instance = new WebAssembly.Instance(wasmSource);
      this.bindExports(instance.exports as ArucoWasmExports);
    } else if (wasmSource && typeof wasmSource === "object" && "byteLength" in (wasmSource as object)) {
      const module = new WebAssembly.Module(wasmSource as BufferSource);
      const instance = new WebAssembly.Instance(module);
      this.bindExports(instance.exports as ArucoWasmExports);
    }
  }

  async init(wasmSource?: ArucoWasmSource): Promise<void> {
    if (this.exports && !wasmSource) return;

    const source = wasmSource ?? this.wasmSource;
    let instance: WebAssembly.Instance;

    if (!source) {
      instance = await initArucoWasm();
    } else if (source instanceof WebAssembly.Module) {
      instance = await WebAssembly.instantiate(source);
    } else if (typeof source === "string" || ("href" in (source as object) && typeof (source as URL).href === "string")) {
      const response = fetch(source as string | URL);
      if ("instantiateStreaming" in WebAssembly) {
        const result = await WebAssembly.instantiateStreaming(response);
        instance = result.instance;
      } else {
        const bytes = await (await response).arrayBuffer();
        const result = await WebAssembly.instantiate(bytes);
        instance = result.instance;
      }
    } else {
      const result = await WebAssembly.instantiate(source as BufferSource);
      instance = result.instance;
    }

    this.bindExports(instance.exports as ArucoWasmExports);
  }

  private bindExports(exports: ArucoWasmExports): void {
    this.exports = exports;
    this.refreshViews();
  }

  private refreshViews(): void {
    if (!this.exports) return;
    const buf = this.exports.memory.buffer;
    if (buf !== this.lastBuffer) {
      this.lastBuffer = buf;
      this.wasmU8 = new Uint8Array(buf);
      this.wasmI32 = new Int32Array(buf);
      this.wasmF32 = new Float32Array(buf);
    }
  }

  detect(imageData: ImageData, maxHammingDistance = 1): DetectedMarker[] {
    if (!this.exports) {
      throw new Error("Aruco4x4Detector WASM module not initialized. Call await init() first.");
    }

    const { width, height, data } = imageData;

    // 1. Ensure WASM heap is sized for (width, height) and obtain RGBA input offset
    const rgbaPtr = this.exports.ensureBuffers(width, height);
    this.refreshViews();

    // 2. Single zero-allocation block copy of RGBA pixels into WASM linear memory
    this.wasmU8!.set(data, rgbaPtr);

    // 3. Run full SIMD detection pipeline inside WASM
    const count = this.exports.detect(width, height, maxHammingDistance);
    if (count === 0) return [];

    // 4. Read compact output structs (40 bytes / 10 x 32-bit words per marker)
    const outPtr = this.exports.getOutputPtr();
    const i32 = this.wasmI32!;
    const f32 = this.wasmF32!;
    const markers: DetectedMarker[] = new Array(count);

    let wordIdx = outPtr >> 2;
    for (let i = 0; i < count; i++) {
      const id = i32[wordIdx];
      const distance = i32[wordIdx + 1];
      markers[i] = {
        id,
        distance,
        corners: [
          { x: f32[wordIdx + 2], y: f32[wordIdx + 3] },
          { x: f32[wordIdx + 4], y: f32[wordIdx + 5] },
          { x: f32[wordIdx + 6], y: f32[wordIdx + 7] },
          { x: f32[wordIdx + 8], y: f32[wordIdx + 9] },
        ],
      };
      wordIdx += 10;
    }

    return markers;
  }

  destroy(): void {
    this.exports = null;
    this.wasmU8 = null;
    this.wasmI32 = null;
    this.wasmF32 = null;
    this.lastBuffer = null;
  }
}

export class ArucoTracker {
  private detector4x4: Aruco4x4Detector;
  private readonly currentDictionary = "4X4_50";
  private maxHammingDistance = 1;

  constructor(wasmSource?: ArucoWasmSource) {
    this.detector4x4 = new Aruco4x4Detector(wasmSource);
  }

  async init(maxHammingDistance = 1, wasmSource?: ArucoWasmSource): Promise<void> {
    this.maxHammingDistance = maxHammingDistance;
    await this.detector4x4.init(wasmSource);
  }

  getDictionary(): string {
    return this.currentDictionary;
  }

  detect(imageData: ImageData): DetectionResult {
    const { width, height } = imageData;
    const startTime = performance.now();

    const markers: DetectedMarker[] = this.detector4x4.detect(imageData, this.maxHammingDistance);
    const inferenceTimeMs = performance.now() - startTime;

    return {
      markers,
      inferenceTimeMs,
      imageWidth: width,
      imageHeight: height,
      timestamp: performance.now(),
    };
  }

  destroy(): void {
    this.detector4x4.destroy();
  }
}
