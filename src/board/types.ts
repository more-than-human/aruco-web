import type { Point2D } from "../digitize/types.ts";

export interface ScreenSegment {
  id: string;
  sourceId: string;
  /** Center X in screen viewport coordinates (CSS px) */
  x: number;
  /** Center Y in screen viewport coordinates (CSS px) */
  y: number;
  /** Visual scale factor applied to the original paper canvas path */
  scale: number;
  /** Original center X in paper canvas coordinates */
  centerX: number;
  /** Original center Y in paper canvas coordinates */
  centerY: number;
  /** Original bounding box [x, y, w, h] in paper canvas coordinates */
  bboxPx: [number, number, number, number];
  /** Bounding box width on screen in pixels */
  width: number;
  /** Bounding box height on screen in pixels */
  height: number;
  /** Original SVG path string */
  svgPath: string;
  /** Canvas Path2D in paper canvas coordinates */
  path2d: Path2D;
  /** Total ink area in square millimeters */
  inkAreaMm2: number;
  /** Whether mouse is currently hovering this screen segment */
  isHovered: boolean;
  /** Whether this segment is currently being dragged */
  isDragging: boolean;
}

export interface FloatingSegmentCopy {
  segment: ScreenSegment;
  /** Offset from mouse position to segment center when picked */
  grabOffset: Point2D;
}
