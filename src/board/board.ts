import type { DrawingSegment, Point2D } from "../digitize/types.ts";
import "./board.css";
import type { FloatingSegmentCopy, ScreenSegment } from "./types.ts";

const COLOR_INK = "#0f172a";
const COLOR_ACTIVE_FILL = "#1e3a8a";
const COLOR_ACTIVE_STROKE = "#2563eb";

/**
 * Calculates screen render scale and offset for an object-fit: contain canvas.
 */
export function getPaperCanvasScreenMetrics(canvas: HTMLCanvasElement): {
  scale: number;
  offsetX: number;
  offsetY: number;
  renderedW: number;
  renderedH: number;
} {
  const rect = canvas.getBoundingClientRect();
  const containerW = rect.width;
  const containerH = rect.height;
  const canvasW = canvas.width || 1100;
  const canvasH = canvas.height || 850;
  const canvasAspect = canvasW / canvasH;
  const containerAspect = containerW / containerH;

  let renderedW: number;
  let renderedH: number;
  let offsetX: number;
  let offsetY: number;

  if (containerAspect > canvasAspect) {
    renderedH = containerH;
    renderedW = containerH * canvasAspect;
    offsetX = rect.left + (containerW - renderedW) * 0.5;
    offsetY = rect.top;
  } else {
    renderedW = containerW;
    renderedH = containerW / canvasAspect;
    offsetX = rect.left;
    offsetY = rect.top + (containerH - renderedH) * 0.5;
  }

  const scale = canvasW > 0 ? renderedW / canvasW : 1.0;
  return { scale, offsetX, offsetY, renderedW, renderedH };
}

/**
 * Maps viewport screen coordinates to paper canvas coordinates.
 */
export function viewportToCanvasPoint(screenPos: Point2D, canvas: HTMLCanvasElement): Point2D | null {
  const { scale, offsetX, offsetY, renderedW, renderedH } = getPaperCanvasScreenMetrics(canvas);
  if (renderedW === 0 || renderedH === 0 || scale === 0) return null;

  const canvasW = canvas.width || 1100;
  const canvasH = canvas.height || 850;

  const cx = (screenPos.x - offsetX) / scale;
  const cy = (screenPos.y - offsetY) / scale;

  if (cx >= 0 && cx <= canvasW && cy >= 0 && cy <= canvasH) {
    return { x: cx, y: cy };
  }
  return null;
}

/**
 * Tests if a screen coordinate intersects a placed screen segment.
 */
export function isPointInScreenSegment(
  seg: ScreenSegment,
  point: Point2D,
  ctx: CanvasRenderingContext2D,
): boolean {
  const dx = point.x - seg.x;
  const dy = point.y - seg.y;
  const pad = 16;

  if (Math.abs(dx) > seg.width / 2 + pad || Math.abs(dy) > seg.height / 2 + pad) {
    return false;
  }

  const canvasX = dx / (seg.scale || 1) + seg.centerX;
  const canvasY = dy / (seg.scale || 1) + seg.centerY;

  ctx.save();
  ctx.resetTransform();
  ctx.lineWidth = 20 / (seg.scale || 1);
  const inPath =
    ctx.isPointInPath(seg.path2d, canvasX, canvasY, "evenodd") ||
    ctx.isPointInStroke(seg.path2d, canvasX, canvasY);
  ctx.restore();

  const isSmall = seg.width <= 48 && seg.height <= 48;
  return inPath || isSmall;
}

export class ScreenBoardManager {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private dpr = 1;

  private screenSegments: ScreenSegment[] = [];
  private floatingCopy: FloatingSegmentCopy | null = null;
  private draggingSegment: ScreenSegment | null = null;
  private dragOffset: Point2D = { x: 0, y: 0 };
  private dragStartPos: Point2D = { x: 0, y: 0 };
  private hasMovedDuringDrag = false;
  private mouseScreenPos: Point2D | null = null;
  private latestFingerScreenPos: Point2D | null = null;
  private nextCopyId = 1;

  // Cached paper context and segments for interaction testing
  private latestPaperCanvas: HTMLCanvasElement | null = null;
  private latestPaperSegments: DrawingSegment[] = [];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    this.resize();
    this.initEventListeners();
  }

  public resize(): void {
    this.dpr = window.devicePixelRatio || 1;
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  public reset(): void {
    this.screenSegments = [];
    this.floatingCopy = null;
    this.draggingSegment = null;
    this.mouseScreenPos = null;
    this.latestFingerScreenPos = null;
    this.canvas.style.cursor = "default";
    this.render();
  }

  public getMouseScreenPos(): Point2D | null {
    return this.mouseScreenPos;
  }

  public getMouseCanvasPoint(paperCanvas: HTMLCanvasElement): Point2D | null {
    if (!this.mouseScreenPos) return null;
    return viewportToCanvasPoint(this.mouseScreenPos, paperCanvas);
  }

  public isHoveringOrInteracting(): boolean {
    return (
      this.floatingCopy !== null ||
      this.draggingSegment !== null ||
      this.screenSegments.some((s) => s.isHovered)
    );
  }

  public getHoveredSegment(): ScreenSegment | null {
    return this.screenSegments.find((s) => s.isHovered) ?? null;
  }

  /**
   * Main per-frame update and render call.
   */
  public update(
    paperSegments: DrawingSegment[],
    paperCanvas: HTMLCanvasElement,
    fingerScreenPos?: Point2D | null,
  ): void {
    this.latestPaperCanvas = paperCanvas;
    this.latestPaperSegments = paperSegments;
    this.latestFingerScreenPos = fingerScreenPos ?? null;

    this.updateHoverStates(this.latestFingerScreenPos);
    this.render();
  }

  /**
   * Updates isHovered states for screen segments and adjusts cursor.
   * Both mouse and finger pointer have the same ability to hover over segments and make them blue.
   */
  private updateHoverStates(fingerScreenPos?: Point2D | null): void {
    let mouseHit: ScreenSegment | null = null;
    let fingerHit: ScreenSegment | null = null;

    for (let i = this.screenSegments.length - 1; i >= 0; i--) {
      const seg = this.screenSegments[i];
      const hitMouse =
        !mouseHit && this.mouseScreenPos
          ? isPointInScreenSegment(seg, this.mouseScreenPos, this.ctx)
          : false;
      const hitFinger =
        !fingerHit && fingerScreenPos
          ? isPointInScreenSegment(seg, fingerScreenPos, this.ctx)
          : false;

      if (hitMouse) mouseHit = seg;
      if (hitFinger) fingerHit = seg;

      seg.isHovered = seg === mouseHit || seg === fingerHit;
    }

    if (!this.draggingSegment && !this.floatingCopy) {
      if (mouseHit) {
        this.canvas.style.cursor = "grab";
      } else if (this.latestPaperCanvas && this.mouseScreenPos) {
        const mousePaperPt = this.getMouseCanvasPoint(this.latestPaperCanvas);
        const hitPaper = mousePaperPt && this.findPaperSegmentAt(mousePaperPt) !== null;
        this.canvas.style.cursor = hitPaper ? "pointer" : "default";
      } else {
        this.canvas.style.cursor = "default";
      }
    }
  }

  private initEventListeners(): void {
    window.addEventListener("resize", () => {
      this.resize();
      this.render();
    });

    this.canvas.addEventListener("mousemove", (e) => {
      this.mouseScreenPos = { x: e.clientX, y: e.clientY };

      if (this.draggingSegment) {
        this.draggingSegment.x = e.clientX - this.dragOffset.x;
        this.draggingSegment.y = e.clientY - this.dragOffset.y;
        if (Math.hypot(e.clientX - this.dragStartPos.x, e.clientY - this.dragStartPos.y) > 4) {
          this.hasMovedDuringDrag = true;
        }
        this.canvas.style.cursor = "grabbing";
        this.render();
        return;
      }

      if (this.floatingCopy) {
        this.floatingCopy.segment.x = e.clientX - this.floatingCopy.grabOffset.x;
        this.floatingCopy.segment.y = e.clientY - this.floatingCopy.grabOffset.y;
        this.canvas.style.cursor = "grabbing";
        this.render();
        return;
      }

      this.updateHoverStates(this.latestFingerScreenPos);
      this.render();
    });

    this.canvas.addEventListener("mousedown", (e) => {
      this.mouseScreenPos = { x: e.clientX, y: e.clientY };

      if (this.floatingCopy) {
        return;
      }

      // Check if user clicked on a placed screen segment to start dragging
      for (let i = this.screenSegments.length - 1; i >= 0; i--) {
        const seg = this.screenSegments[i];
        if (isPointInScreenSegment(seg, this.mouseScreenPos, this.ctx)) {
          this.draggingSegment = seg;
          seg.isDragging = true;
          this.dragOffset = { x: e.clientX - seg.x, y: e.clientY - seg.y };
          this.dragStartPos = { x: e.clientX, y: e.clientY };
          this.hasMovedDuringDrag = false;
          this.canvas.style.cursor = "grabbing";

          // Move dragged segment to top of render list
          this.screenSegments.splice(i, 1);
          this.screenSegments.push(seg);
          this.render();
          return;
        }
      }
    });

    this.canvas.addEventListener("mouseup", () => {
      if (this.draggingSegment) {
        this.draggingSegment.isDragging = false;
        this.draggingSegment = null;
        this.canvas.style.cursor = "grab";
        this.render();
      }
    });

    this.canvas.addEventListener("click", (e) => {
      this.mouseScreenPos = { x: e.clientX, y: e.clientY };

      // 1. If currently carrying a floating copy, clicking drops it onto the screen at full opacity!
      if (this.floatingCopy) {
        const droppedSeg = this.floatingCopy.segment;
        droppedSeg.x = e.clientX - this.floatingCopy.grabOffset.x;
        droppedSeg.y = e.clientY - this.floatingCopy.grabOffset.y;
        droppedSeg.isHovered = false;
        droppedSeg.isDragging = false;
        this.screenSegments.push(droppedSeg);
        this.floatingCopy = null;
        this.canvas.style.cursor = "default";
        this.render();
        return;
      }

      // If user just dragged a screen segment, don't trigger new pick action
      if (this.hasMovedDuringDrag) {
        this.hasMovedDuringDrag = false;
        return;
      }

      // 2. Check if user clicked on a paper segment to create a floating copy
      if (this.latestPaperCanvas) {
        const paperPt = viewportToCanvasPoint(this.mouseScreenPos, this.latestPaperCanvas);
        if (paperPt) {
          const clickedPaperSeg = this.findPaperSegmentAt(paperPt);
          if (clickedPaperSeg) {
            const metrics = getPaperCanvasScreenMetrics(this.latestPaperCanvas);
            const scale = metrics.scale;
            const [bx, by, bw, bh] = clickedPaperSeg.bboxPx;
            const centerX = bx + bw * 0.5;
            const centerY = by + bh * 0.5;

            const newScreenSeg: ScreenSegment = {
              id: `screen-${this.nextCopyId++}`,
              sourceId: clickedPaperSeg.id,
              x: e.clientX,
              y: e.clientY,
              scale,
              centerX,
              centerY,
              bboxPx: clickedPaperSeg.bboxPx,
              width: bw * scale,
              height: bh * scale,
              svgPath: clickedPaperSeg.svgPath,
              path2d: clickedPaperSeg.path2d,
              inkAreaMm2: clickedPaperSeg.inkAreaMm2,
              isHovered: false,
              isDragging: false,
            };

            this.floatingCopy = {
              segment: newScreenSeg,
              grabOffset: { x: 0, y: 0 },
            };

            this.canvas.style.cursor = "grabbing";
            this.render();
          }
        }
      }
    });

    this.canvas.addEventListener("mouseleave", () => {
      this.mouseScreenPos = null;
      this.updateHoverStates(this.latestFingerScreenPos);
      this.render();
    });
  }

  private findPaperSegmentAt(pt: Point2D): DrawingSegment | null {
    for (const seg of this.latestPaperSegments) {
      const [bx, by, bw, bh] = seg.bboxPx;
      const pad = 12;
      if (pt.x >= bx - pad && pt.x <= bx + bw + pad && pt.y >= by - pad && pt.y <= by + bh + pad) {
        return seg;
      }
    }
    return null;
  }

  /**
   * Renders placed screen segments and floating copies onto the screen canvas.
   */
  private render(): void {
    this.ctx.clearRect(0, 0, this.width, this.height);

    // 1. Render all placed screen segments at full opacity (1.0)
    for (const seg of this.screenSegments) {
      if (seg === this.draggingSegment) continue;
      this.drawSegment(seg, 1.0, seg.isHovered);
    }

    // 2. Render dragging segment on top
    if (this.draggingSegment) {
      this.drawSegment(this.draggingSegment, 1.0, true);
    }

    // 3. Render floating copy sticking to mouse at semi-transparent opacity (0.55)
    if (this.floatingCopy) {
      this.drawSegment(this.floatingCopy.segment, 0.55, true);
    }
  }

  private drawSegment(seg: ScreenSegment, opacity: number, isHighlighted: boolean): void {
    this.ctx.save();
    this.ctx.globalAlpha = opacity;
    this.ctx.translate(seg.x, seg.y);
    this.ctx.scale(seg.scale, seg.scale);
    this.ctx.translate(-seg.centerX, -seg.centerY);

    if (isHighlighted) {
      this.ctx.fillStyle = COLOR_ACTIVE_FILL;
      this.ctx.fill(seg.path2d, "evenodd");
      this.ctx.strokeStyle = COLOR_ACTIVE_STROKE;
      this.ctx.lineWidth = 3.5 / (seg.scale || 1);
      this.ctx.stroke(seg.path2d);

      // Render segment identifier label
      const [bx, by] = seg.bboxPx;
      this.ctx.save();
      this.ctx.fillStyle = COLOR_ACTIVE_STROKE;
      this.ctx.font = `600 ${Math.round(12 / (seg.scale || 1))}px system-ui, -apple-system, sans-serif`;
      this.ctx.fillText(seg.sourceId || seg.id, bx, Math.max(16, by - 6));
      this.ctx.restore();
    } else {
      this.ctx.fillStyle = COLOR_INK;
      this.ctx.fill(seg.path2d, "evenodd");
    }

    this.ctx.restore();
  }
}
