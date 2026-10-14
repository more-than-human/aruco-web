import { DICTIONARY_PACKED } from "../shared/dictionary-packed";

export interface IPoint {
  x: number;
  y: number;
}

export interface IImage {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array;
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
  dict_calculate_tau(dictPtr: number): number;
  dict_find_bits(dictPtr: number, bitsMask: bigint): bigint;
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
let u64View: BigUint64Array;

function refreshViews(): void {
  const buf = wasm.memory.buffer;
  if (!u8View || u8View.buffer !== buf) {
    u8View = new Uint8Array(buf);
    i32View = new Int32Array(buf);
    u64View = new BigUint64Array(buf);
  }
}

/**
 * Initializes the ArUco WebAssembly module from a compiled binary or URL/BufferSource.
 */
export async function initArUcoWasm(
  source: BufferSource | WebAssembly.Module | Response | Promise<Response>,
  imports?: WebAssembly.Imports,
): Promise<ArUcoWasmExports> {
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
 * Scratch Arena helper for temporary Wasm calls (restores heap pointer automatically)
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
  public tau: number = 0;
  public nBits: number = 0;
  public markSize: number = 0;
  public limit?: number;
  public dictPtr: number = 0;

  constructor(dicName: string, limit?: number) {
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
}
