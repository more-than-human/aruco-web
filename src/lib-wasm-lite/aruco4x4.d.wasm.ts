interface ArucoWasmExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  ensureBuffers(width: number, height: number): number;
  getOutputPtr(): number;
  detect(width: number, height: number, maxHammingDistance: number): number;
}

export const memory: WebAssembly.Memory;
export function ensureBuffers(width: number, height: number): number;
export function getOutputPtr(): number;
export function detect(width: number, height: number, maxHammingDistance: number): number;

declare const init: (importObject?: WebAssembly.Imports) => Promise<WebAssembly.Instance & { exports: ArucoWasmExports }>;
export default init;
