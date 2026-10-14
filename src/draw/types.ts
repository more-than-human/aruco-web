import type { IPoint } from "../lib/aruco.ts";

export interface IRect {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** US Letter paper dimensions in landscape orientation (in mm: 279.4 x 215.9, or 11 x 8.5 in) */
export const LETTER_LANDSCAPE_WIDTH_MM = 279.4;
export const LETTER_LANDSCAPE_HEIGHT_MM = 215.9;
export const LETTER_ASPECT_RATIO = LETTER_LANDSCAPE_WIDTH_MM / LETTER_LANDSCAPE_HEIGHT_MM; // ~1.2941

/** Physical marker specifications (in mm) */
export const MARKER_SIZE_MM = 20; // 20mm x 20mm
export const PAPER_EDGE_PADDING_MM = 10; // 10mm padding to edge of paper
export const DIGITIZING_ZONE_EXTEND_MM = 15; // 15mm beyond marker center

/** Default screen/buffer pixel resolution matching US Letter landscape aspect ratio */
export const LETTER_LANDSCAPE_WIDTH = 1100;
export const LETTER_LANDSCAPE_HEIGHT = 850;

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
