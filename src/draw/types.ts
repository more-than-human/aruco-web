import type { IPoint } from "../lib/aruco.ts";

export interface IRect {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** Standard A4 dimensions in portrait orientation (aspect ratio 1 : √2) */
export const A4_PORTRAIT_WIDTH = 595;
export const A4_PORTRAIT_HEIGHT = 842;
export const A4_ASPECT_RATIO = A4_PORTRAIT_WIDTH / A4_PORTRAIT_HEIGHT; // ~0.707 (1 / √2)

/** Output of Stage 1 (PaperGeometry) */
export interface PaperWarpModel {
  /** 3x3 Homography mapping Screen (u,v) -> Camera (x,y) */
  screenToCameraH: Float64Array;
  /** 3x3 Homography mapping Camera (x,y) -> Screen (u,v) */
  cameraToScreenH: Float64Array;
  /** 4 axis-aligned marker exclusion zones (expanded by 0.1*D) in screen pixels */
  markerExclusionBoxes: IRect[];
  /** 4 outer paper corners in camera space (for debug overlay) */
  cameraPaperQuad: IPoint[];
}

/** Output of Stage 2 (PerspectiveWarper) */
export interface RectifiedFrame {
  width: number;
  height: number;
  /** Rectified grayscale luminance [0..255], length = width * height */
  gray: Uint8Array;
  /** 1 = drawable paper pixel, 0 = inside marker exclusion zone or off-camera */
  validMask: Uint8Array;
}
