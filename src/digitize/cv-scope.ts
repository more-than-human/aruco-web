import type { CV, Mat, MatVector } from "@techstark/opencv-js";
import { getCv } from "./opencv.ts";

/**
 * RAII Scope Manager for OpenCV.js WebAssembly memory.
 * Ensures every allocated cv.Mat or OpenCV object is deleted upon scope exit.
 * Follows digitizer-v2.md Section 3.3.
 */
export class CvScope {
  public readonly cv: CV;
  private disposables: Array<{ delete(): void }> = [];

  constructor(cvInstance?: CV) {
    this.cv = cvInstance || getCv();
  }

  /** Track an OpenCV object for automatic deletion if it has a delete() method */
  public manage<T>(obj: T): T {
    if (obj && typeof (obj as { delete?: () => void }).delete === "function") {
      this.disposables.push(obj as unknown as { delete(): void });
    }
    return obj;
  }

  /** Create a new Mat tracked by this scope */
  public mat(): Mat;
  public mat(rows: number, cols: number, type: number): Mat;
  public mat(rows?: number, cols?: number, type?: number): Mat {
    const m = rows !== undefined && cols !== undefined && type !== undefined ? new this.cv.Mat(rows, cols, type) : new this.cv.Mat();
    return this.manage(m);
  }

  /** Create a MatVector tracked by this scope */
  public matVector(): MatVector {
    return this.manage(new this.cv.MatVector());
  }

  /** Free all tracked objects in reverse order of allocation */
  public dispose(): void {
    while (this.disposables.length > 0) {
      const obj = this.disposables.pop()!;
      try {
        const maybeDeletable = obj as unknown as { isDeleted?: () => boolean };
        if (typeof maybeDeletable.isDeleted === "function" && maybeDeletable.isDeleted()) {
          continue;
        }
        obj.delete();
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (!msg.includes("already deleted")) {
          console.warn("Failed to delete OpenCV object:", err);
        }
      }
    }
  }

  /** Execute a function within a managed OpenCV memory scope */
  public static use<R>(block: (scope: CvScope) => R, cvInstance?: CV): R {
    const scope = new CvScope(cvInstance);
    try {
      return block(scope);
    } finally {
      scope.dispose();
    }
  }
}
