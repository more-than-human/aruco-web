export { CvScope } from "./cv-scope.ts";
export {
  findHomography4Pts,
  invert3x3,
  isPointInPolygon,
  multiply3x3,
  projectPoint,
  projectPoints,
} from "./homography.ts";
export { PaperTracker } from "./paper-tracker.ts";
export { renderCorners, renderPaperBoundary, renderPaperStatus } from "./render.ts";
export {
  DEFAULT_SHEET_CONFIG,
  DIGITIZING_ZONE_EXTEND_MM,
  LETTER_ASPECT_RATIO,
  LETTER_LANDSCAPE_HEIGHT,
  LETTER_LANDSCAPE_HEIGHT_MM,
  LETTER_LANDSCAPE_WIDTH,
  LETTER_LANDSCAPE_WIDTH_MM,
  MARKER_SIZE_MM,
  PAPER_EDGE_PADDING_MM,
} from "./types.ts";
export type {
  PaperBoundary,
  PaperSheetConfig,
  PaperTrackingResult,
  Point2D,
  Rect,
} from "./types.ts";
