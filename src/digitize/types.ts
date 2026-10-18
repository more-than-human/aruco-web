export interface Point2D {
  x: number;
  y: number;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** US Letter paper dimensions in landscape orientation (in mm: 279.4 x 215.9, or 11 x 8.5 in) */
export const LETTER_LANDSCAPE_WIDTH_MM = 279.4;
export const LETTER_LANDSCAPE_HEIGHT_MM = 215.9;
export const LETTER_ASPECT_RATIO = LETTER_LANDSCAPE_WIDTH_MM / LETTER_LANDSCAPE_HEIGHT_MM; // ~1.2941

/** Default screen/canvas pixel resolution matching US Letter landscape aspect ratio (~3.937 px/mm) */
export const LETTER_LANDSCAPE_WIDTH = 1100;
export const LETTER_LANDSCAPE_HEIGHT = 850;

/** Physical marker specifications (in mm) */
export const MARKER_SIZE_MM = 20.0; // 20mm x 20mm
export const PAPER_EDGE_PADDING_MM = 10.0; // 10mm padding to edge of paper
export const DIGITIZING_ZONE_EXTEND_MM = 15.0; // 15mm beyond marker center

export interface PaperSheetConfig {
  sheetWidthMm: number;
  sheetHeightMm: number;
  canvasWidth: number;
  canvasHeight: number;
  markerSizeMm: number;
  paperEdgePaddingMm: number;
  markerMarginRatio: number;
  holdDurationMs: number;
  markerIds: [number, number, number, number]; // [TL, TR, BR, BL]
}

export const DEFAULT_SHEET_CONFIG: PaperSheetConfig = {
  sheetWidthMm: LETTER_LANDSCAPE_WIDTH_MM,
  sheetHeightMm: LETTER_LANDSCAPE_HEIGHT_MM,
  canvasWidth: LETTER_LANDSCAPE_WIDTH,
  canvasHeight: LETTER_LANDSCAPE_HEIGHT,
  markerSizeMm: MARKER_SIZE_MM,
  paperEdgePaddingMm: PAPER_EDGE_PADDING_MM,
  markerMarginRatio: 0.2,
  holdDurationMs: 250,
  markerIds: [0, 1, 2, 3],
};

export interface PaperBoundary {
  /** 4 corner points in camera image coordinates (TL, TR, BR, BL) */
  quadPx: Point2D[];
  /** Axis-aligned bounding box [x, y, w, h] in camera image space */
  bboxPx: [number, number, number, number];
  /** Whether tracking is active and stable */
  isLocked: boolean;
  /** Mean reprojection error in camera pixels */
  reprojErrPx: number;
}

export interface PaperTrackingResult {
  /** Whether the paper is currently tracked (either live or temporarily held) */
  isTracking: boolean;
  /** True if currently in PoseHold bridging state */
  isHeld: boolean;
  /** Number of currently visible markers */
  visibleCount: number;
  /** Projected boundary quad in camera space */
  boundary: PaperBoundary | null;
  /** 3x3 Homography: Camera image pixels -> Paper mm */
  cameraToMmH: Float64Array | null;
  /** 3x3 Homography: Paper mm -> Camera image pixels */
  mmToCameraH: Float64Array | null;
  /** 3x3 Homography: Camera image pixels -> Canvas pixels */
  cameraToCanvasH: Float64Array | null;
  /** 3x3 Homography: Canvas pixels -> Camera image pixels */
  canvasToCameraH: Float64Array | null;
  /** Marker exclusion zones in canvas pixel coordinates */
  markerExclusionBoxes: Rect[];
  /** Paper dimensions in mm [width, height] */
  sheetMm: [number, number];
  /** Canvas dimensions in pixels [width, height] */
  canvasSize: [number, number];
  /** Pixels per mm along X and Y */
  pxPerMm: [number, number];
}

export interface DrawingSegment {
  id: string;
  /** Bounding box [x, y, width, height] in canvas pixels */
  bboxPx: [number, number, number, number];
  /** Bounding box [x, y, width, height] in paper millimeters */
  bboxMm: [number, number, number, number];
  /** Total ink area in square millimeters */
  inkAreaMm2: number;
  /** SVG path d string in canvas coordinates */
  svgPath: string;
  /** Cached Path2D in canvas coordinates */
  path2d: Path2D;
  /** Whether the finger pointer currently intersects this segment */
  isActive: boolean;
}
