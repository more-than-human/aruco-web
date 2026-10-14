import { DICTIONARY_PACKED, type DictionaryConfig as SharedDictionaryConfig } from "../shared/dictionary-packed.ts";
import initArucoWasm from "./aruco4x4.wasm?init";
import type { DetectedMarker, DetectionResult, Point2D } from "./types.ts";

export type IPoint = Point2D;

export interface IImage {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array;
}

export interface IConfig {
  dictionaryName: "aruco" | "4x4_1000" | "5x5_1000" | "6x6_1000" | "mip_36h12" | string;
  maxHammingDistance?: number;
  limit?: number;
}

export interface DictionaryConfig {
  nBits: number;
  tau: number;
  codeList: (number | string | number[])[];
}

/**
 * Standard OpenCV DICT_4X4_50 ArUco marker codes.
 * Reused from DICTIONARY_PACKED["4x4_1000"].
 */
export const DICT_4X4_50: number[] = DICTIONARY_PACKED["4x4_1000"].codeList.slice(0, 50);

export interface ArucoWasmExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  get_heap_ptr(): number;
  set_heap_ptr(ptr: number): void;
  alloc(size: number): number;
  setDictionary(dictPtr: number): void;
  dict_calculate_tau(dictPtr: number): number;
  dict_find_bits(dictPtr: number, bitsMask: bigint): bigint;
  ensureBuffers(width: number, height: number): number;
  getOutputPtr(): number;
  detect(width: number, height: number, maxHammingDistance: number, dictPtr?: number): number;
}

export type ArucoWasmSource = BufferSource | WebAssembly.Module | string | URL;

let defaultWasm: ArucoWasmExports | null = null;
let u8View: Uint8Array | null = null;
let i32View: Int32Array | null = null;
let u64View: BigUint64Array | null = null;

function refreshViews(exports?: ArucoWasmExports | null): void {
  const active = exports ?? defaultWasm;
  if (!active) return;
  const buf = active.memory.buffer;
  if (!u8View || u8View.buffer !== buf) {
    u8View = new Uint8Array(buf);
    i32View = new Int32Array(buf);
    u64View = new BigUint64Array(buf);
  }
}

/**
 * Initializes the ArUco 4x4 WebAssembly module from a compiled binary or URL/BufferSource.
 */
export async function initArUcoWasm(source?: ArucoWasmSource, imports?: WebAssembly.Imports): Promise<ArucoWasmExports> {
  let instance: WebAssembly.Instance;

  if (!source) {
    instance = (await initArucoWasm(imports)) as unknown as WebAssembly.Instance;
  } else if (source instanceof WebAssembly.Module) {
    instance = await WebAssembly.instantiate(source, imports);
  } else if (typeof source === "string" || ("href" in (source as object) && typeof (source as URL).href === "string")) {
    const response = fetch(source as string | URL);
    if ("instantiateStreaming" in WebAssembly) {
      const result = await WebAssembly.instantiateStreaming(response, imports);
      instance = result.instance;
    } else {
      const bytes = await (await response).arrayBuffer();
      const result = await WebAssembly.instantiate(bytes, imports);
      instance = result.instance;
    }
  } else {
    const result = await WebAssembly.instantiate(source as BufferSource, imports);
    instance = result.instance;
  }

  defaultWasm = instance.exports as unknown as ArucoWasmExports;
  refreshViews(defaultWasm);
  return defaultWasm;
}

export class DICTIONARIES {
  static getDictionary(name: string): SharedDictionaryConfig {
    const key = name.toLowerCase();
    if (key === "4x4_50") {
      return {
        nBits: 16,
        tau: 0,
        codeList: DICTIONARY_PACKED["4x4_1000"].codeList.slice(0, 50),
      };
    }
    const dictionary = DICTIONARY_PACKED[key];
    if (dictionary && typeof dictionary === "object" && "nBits" in dictionary && "tau" in dictionary && "codeList" in dictionary) {
      return dictionary;
    }
    throw new Error(`The dictionary "${name}" is not recognized.`);
  }
}

export class Dictionary {
  public tau: number = 0;
  public nBits: number = 0;
  public markSize: number = 0;
  public limit?: number;
  public dictPtr: number = 0;
  public name: string;
  private wasm: ArucoWasmExports;

  constructor(dicName: string = "4x4_1000", limit?: number, wasmInstance?: ArucoWasmExports) {
    this.name = dicName;
    this.limit = limit;
    const inst = wasmInstance ?? defaultWasm;
    if (!inst) {
      throw new Error("Wasm instance not initialized for Dictionary. Pass wasm instance or call await initArUcoWasm() first.");
    }
    this.wasm = inst;
    this.initialize(dicName, limit);
  }

  private initialize(dicName: string, limit?: number): void {
    const dictionary = DICTIONARIES.getDictionary(dicName);
    this.nBits = dictionary.nBits;
    this.markSize = Math.sqrt(dictionary.nBits) + 2;
    const count = limit !== undefined && limit >= 0 ? Math.min(limit, dictionary.codeList.length) : dictionary.codeList.length;

    refreshViews(this.wasm);
    this.dictPtr = this.wasm.alloc(16 + count * 8);
    refreshViews(this.wasm);

    i32View![this.dictPtr >> 2] = this.nBits;
    i32View![(this.dictPtr >> 2) + 1] = this.markSize;
    i32View![(this.dictPtr >> 2) + 3] = count;

    const codesBase = (this.dictPtr + 16) >> 3;
    for (let i = 0; i < count; i++) {
      const binStr = this.parseCode(dictionary.codeList[i], dictionary.nBits);
      u64View![codesBase + i] = BigInt("0b" + binStr);
    }

    if (dictionary.tau) {
      this.tau = dictionary.tau;
      i32View![(this.dictPtr >> 2) + 2] = this.tau;
    } else {
      this.tau = this.wasm.dict_calculate_tau(this.dictPtr);
    }
  }

  public setTau(tau: number): void {
    this.tau = tau;
    refreshViews(this.wasm);
    i32View![(this.dictPtr >> 2) + 2] = tau;
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
    const res = this.wasm.dict_find_bits(this.dictPtr, mask);
    if (res < 0n) return null;
    return {
      id: Number(res & 0xffffffffn),
      distance: Number(res >> 32n),
    };
  }
}

export class Marker {
  public id: number;
  public corners: Point2D[];
  public hammingDistance: number;

  constructor(id: number, corners: Point2D[], hammingDistance: number) {
    this.id = id;
    this.corners = corners;
    this.hammingDistance = hammingDistance;
  }
}

export class Aruco4x4Detector {
  public config: IConfig;
  public dictionary: Dictionary | null = null;
  private exports: ArucoWasmExports | null = null;
  private wasmU8: Uint8Array | null = null;
  private wasmI32: Int32Array | null = null;
  private wasmF32: Float32Array | null = null;
  private lastBuffer: ArrayBufferLike | null = null;
  private wasmSource?: ArucoWasmSource;

  constructor(configOrSource?: string | IConfig | ArucoWasmSource, wasmSource?: ArucoWasmSource) {
    let config: IConfig | undefined;
    let source: ArucoWasmSource | undefined;

    if (typeof configOrSource === "string") {
      if (configOrSource in DICTIONARY_PACKED || configOrSource.toLowerCase() === "4x4_50" || configOrSource.startsWith("4x4_")) {
        config = { dictionaryName: configOrSource };
        source = wasmSource;
      } else {
        source = configOrSource;
      }
    } else if (
      configOrSource &&
      typeof configOrSource === "object" &&
      ("dictionaryName" in configOrSource || "limit" in configOrSource || "maxHammingDistance" in configOrSource)
    ) {
      config = configOrSource as IConfig;
      source = wasmSource;
    } else if (configOrSource) {
      source = configOrSource as ArucoWasmSource;
    }

    this.config = {
      dictionaryName: config?.dictionaryName ?? "4x4_1000",
      limit: config?.limit,
      maxHammingDistance: config?.maxHammingDistance ?? 1,
    };
    this.wasmSource = source;

    if (source instanceof WebAssembly.Module) {
      const instance = new WebAssembly.Instance(source);
      this.bindExports(instance.exports as unknown as ArucoWasmExports);
      this.initDictionary();
    } else if (source && typeof source === "object" && "byteLength" in (source as object)) {
      const module = new WebAssembly.Module(source as BufferSource);
      const instance = new WebAssembly.Instance(module);
      this.bindExports(instance.exports as unknown as ArucoWasmExports);
      this.initDictionary();
    }
  }

  async init(wasmSource?: ArucoWasmSource): Promise<void> {
    if (this.exports && !wasmSource) {
      if (!this.dictionary) {
        this.initDictionary();
      }
      return;
    }

    const source = wasmSource ?? this.wasmSource;
    let instance: WebAssembly.Instance;

    if (!source) {
      instance = (await initArucoWasm()) as unknown as WebAssembly.Instance;
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

    this.bindExports(instance.exports as unknown as ArucoWasmExports);
    this.initDictionary();
  }

  private initDictionary(): void {
    if (!this.exports) return;
    this.dictionary = new Dictionary(this.config.dictionaryName, this.config.limit, this.exports);
    if (this.config.maxHammingDistance != null) {
      this.dictionary.setTau(this.config.maxHammingDistance);
    }
    this.exports.setDictionary(this.dictionary.dictPtr);
  }

  setDictionary(dictionary: Dictionary | string, limit?: number): Dictionary {
    if (!this.exports) {
      throw new Error("Aruco4x4Detector WASM module not initialized. Call await init() first.");
    }
    if (dictionary instanceof Dictionary) {
      this.dictionary = dictionary;
    } else {
      this.dictionary = new Dictionary(dictionary, limit, this.exports);
    }
    this.exports.setDictionary(this.dictionary.dictPtr);
    return this.dictionary;
  }

  getDictionary(): Dictionary | null {
    return this.dictionary;
  }

  private bindExports(exports: ArucoWasmExports): void {
    this.exports = exports;
    defaultWasm = exports;
    this.refreshViews();
  }

  private refreshViews(): void {
    if (!this.exports) return;
    refreshViews(this.exports);
    const buf = this.exports.memory.buffer;
    if (buf !== this.lastBuffer) {
      this.lastBuffer = buf;
      this.wasmU8 = new Uint8Array(buf);
      this.wasmI32 = new Int32Array(buf);
      this.wasmF32 = new Float32Array(buf);
    }
  }

  detectImage(width: number, height: number, data: Uint8ClampedArray | Uint8Array, maxHammingDistance?: number): DetectedMarker[] {
    return this.detect({ width, height, data }, maxHammingDistance);
  }

  detect(image: ImageData | { width: number; height: number; data: Uint8ClampedArray | Uint8Array }, maxHammingDistance?: number): DetectedMarker[] {
    if (!this.exports) {
      throw new Error("Aruco4x4Detector WASM module not initialized. Call await init() first.");
    }
    if (!this.dictionary) {
      this.initDictionary();
    }

    const { width, height, data } = image;
    const maxHamming = maxHammingDistance ?? this.config.maxHammingDistance ?? 1;

    // 1. Ensure WASM heap is sized for (width, height) and obtain RGBA input offset
    const rgbaPtr = this.exports.ensureBuffers(width, height);
    this.refreshViews();

    // 2. Single zero-allocation block copy of RGBA pixels into WASM linear memory
    this.wasmU8!.set(data, rgbaPtr);

    // 3. Run full SIMD detection pipeline inside WASM
    const count = this.exports.detect(width, height, maxHamming, this.dictionary!.dictPtr);
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
    this.dictionary = null;
    this.wasmU8 = null;
    this.wasmI32 = null;
    this.wasmF32 = null;
    this.lastBuffer = null;
  }
}

export { Aruco4x4Detector as Detector };

export class ArucoTracker {
  private detector4x4: Aruco4x4Detector;
  private maxHammingDistance = 1;

  constructor(configOrSource?: string | IConfig | ArucoWasmSource, wasmSource?: ArucoWasmSource) {
    this.detector4x4 = new Aruco4x4Detector(configOrSource, wasmSource);
    if (configOrSource && typeof configOrSource === "object" && "maxHammingDistance" in configOrSource && configOrSource.maxHammingDistance != null) {
      this.maxHammingDistance = configOrSource.maxHammingDistance;
    }
  }

  async init(maxHammingDistance = 1, wasmSource?: ArucoWasmSource): Promise<void> {
    this.maxHammingDistance = maxHammingDistance;
    await this.detector4x4.init(wasmSource);
  }

  getDictionary(): string {
    return this.detector4x4.getDictionary()?.name ?? "4x4_1000";
  }

  setDictionary(dictName: string, limit?: number): void {
    this.detector4x4.setDictionary(dictName, limit);
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
