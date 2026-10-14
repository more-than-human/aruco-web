interface ArucoWasmExports extends WebAssembly.Exports {
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

export const memory: WebAssembly.Memory;
export function get_heap_ptr(): number;
export function set_heap_ptr(ptr: number): void;
export function alloc(size: number): number;
export function setDictionary(dictPtr: number): void;
export function dict_calculate_tau(dictPtr: number): number;
export function dict_find_bits(dictPtr: number, bitsMask: bigint): bigint;
export function ensureBuffers(width: number, height: number): number;
export function getOutputPtr(): number;
export function detect(width: number, height: number, maxHammingDistance: number, dictPtr?: number): number;

declare const init: (importObject?: WebAssembly.Imports) => Promise<WebAssembly.Instance & { exports: ArucoWasmExports }>;
export default init;
