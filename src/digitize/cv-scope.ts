import cv, { type Mat, type MatVector } from "@techstark/opencv-js";

/**
 * RAII Scope Manager for OpenCV.js WebAssembly memory.
 * Ensures every allocated cv.Mat or OpenCV object is deleted upon scope exit.
 * Follows digitizer-v2.md Section 3.3.
 */
export class CvScope {
  private disposables: Array<{ delete(): void }> = [];

  /** Track an OpenCV object for automatic deletion */
  public manage<T extends { delete(): void }>(obj: T): T {
    this.disposables.push(obj);
    return obj;
  }

  /** Create a new Mat tracked by this scope */
  public mat(): Mat;
  public mat(rows: number, cols: number, type: number): Mat;
  public mat(rows?: number, cols?: number, type?: number): Mat {
    const m = rows !== undefined && cols !== undefined && type !== undefined ? new cv.Mat(rows, cols, type) : new cv.Mat();
    return this.manage(m);
  }

  /** Create a MatVector tracked by this scope */
  public matVector(): MatVector {
    return this.manage(new cv.MatVector());
  }

  /** Free all tracked objects in reverse order of allocation */
  public dispose(): void {
    while (this.disposables.length > 0) {
      const obj = this.disposables.pop()!;
      try {
        obj.delete();
      } catch (err) {
        console.warn("Failed to delete OpenCV object:", err);
      }
    }
  }

  /** Execute a function within a managed OpenCV memory scope */
  public static use<R>(block: (scope: CvScope) => R): R {
    const scope = new CvScope();
    try {
      return block(scope);
    } finally {
      scope.dispose();
    }
  }
}
