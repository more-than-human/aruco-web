import { TrackedMarker } from "../lib/tracker";
import type { PaperWarpModel } from "./types";

const STROKE_COLOR = "#00e676";
const LINE_WIDTH = 3;
const COASTING_DASH_PATTERN = [6, 4];

export function renderPaperBoundary(ctx: CanvasRenderingContext2D, warpModel: PaperWarpModel): void {
  const quad = warpModel.cameraPaperQuad;
  if (!quad || quad.length < 4) return;

  ctx.save();
  ctx.strokeStyle = "#2979ff";
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);

  ctx.beginPath();
  ctx.moveTo(quad[0].x, quad[0].y);
  for (let i = 1; i < quad.length; i++) {
    ctx.lineTo(quad[i].x, quad[i].y);
  }
  ctx.closePath();
  ctx.stroke();

  ctx.restore();
}

export function renderCorners(ctx: CanvasRenderingContext2D, markers: TrackedMarker[]): void {
  ctx.save();
  ctx.lineWidth = LINE_WIDTH;
  ctx.strokeStyle = STROKE_COLOR;

  for (const marker of markers) {
    const isPredicted = marker instanceof TrackedMarker && marker.isPredicted;
    ctx.setLineDash(isPredicted ? COASTING_DASH_PATTERN : []);

    ctx.beginPath();
    for (let j = 0; j < marker.corners.length; ++j) {
      const corner = marker.corners[j];
      if (j === 0) {
        ctx.moveTo(corner.x, corner.y);
      } else {
        ctx.lineTo(corner.x, corner.y);
      }
    }
    ctx.closePath();
    ctx.stroke();
  }

  ctx.restore();
}
