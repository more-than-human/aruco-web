import cvPromise, { type CV } from "@techstark/opencv-js";

let cvInstance: CV | null = null;
let initPromise: Promise<CV> | null = null;

/**
 * Initializes and resolves the OpenCV.js WebAssembly runtime.
 * Safe to call multiple times (returns existing instance/promise).
 */
export async function initOpenCV(): Promise<CV> {
  if (cvInstance) {
    return cvInstance;
  }
  if (!initPromise) {
    initPromise = (async () => {
      // @techstark/opencv-js default export is a Promise-like object that resolves to the CV module
      const cv = ((cvPromise as unknown as { then?: (cb: (mod: CV) => void) => Promise<CV> }).then
        ? await (cvPromise as unknown as Promise<CV>)
        : cvPromise) as unknown as CV;

      cvInstance = cv;
      return cv;
    })();
  }
  return initPromise;
}

/**
 * Checks whether OpenCV.js has finished initializing.
 */
export function isOpenCVReady(): boolean {
  return cvInstance !== null;
}

/**
 * Synchronous accessor for the OpenCV module instance.
 * Throws if initOpenCV() has not yet resolved.
 */
export function getCv(): CV {
  if (!cvInstance) {
    throw new Error("OpenCV.js is not yet initialized. Call and await initOpenCV() first.");
  }
  return cvInstance;
}
