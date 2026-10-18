import type { TrackedMarker } from "../aruco-tracking/tracker.ts";
import type { PaperBoundary, PaperTrackingResult } from "./types.ts";

const COLOR_LOCKED = "#00e676";
const COLOR_HELD = "#ffab00";
const COLOR_CORNER = "#2979ff";

/**
 * Renders the tracked physical paper boundary quadrilateral on the camera canvas overlay.
 */
export function renderPaperBoundary(ctx: CanvasRenderingContext2D, boundary: PaperBoundary | null, isHeld: boolean = false): void {
  if (!boundary || !boundary.quadPx || boundary.quadPx.length < 4) return;

  const quad = boundary.quadPx;
  const strokeColor = isHeld ? COLOR_HELD : COLOR_LOCKED;

  ctx.save();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = strokeColor;
  ctx.setLineDash(isHeld ? [6, 4] : []);

  // Draw boundary quad
  ctx.beginPath();
  ctx.moveTo(quad[0].x, quad[0].y);
  for (let i = 1; i < 4; i++) {
    ctx.lineTo(quad[i].x, quad[i].y);
  }
  ctx.closePath();
  ctx.stroke();

  // Draw corner indicator dots
  ctx.setLineDash([]);
  ctx.fillStyle = strokeColor;
  for (let i = 0; i < 4; i++) {
    ctx.beginPath();
    ctx.arc(quad[i].x, quad[i].y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

/**
 * Renders ArUco marker boundaries and corners on the camera canvas.
 */
export function renderCorners(ctx: CanvasRenderingContext2D, markers: TrackedMarker[]): void {
  ctx.save();
  ctx.lineWidth = 2;

  for (const marker of markers) {
    if (!marker.corners || marker.corners.length < 4) continue;

    const isPredicted = marker.isPredicted;
    ctx.strokeStyle = isPredicted ? COLOR_HELD : COLOR_CORNER;
    ctx.setLineDash(isPredicted ? [4, 4] : []);

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

    // Draw first corner indicator (orientation)
    const c0 = marker.corners[0];
    ctx.fillStyle = "#ff1744";
    ctx.fillRect(c0.x - 2, c0.y - 2, 4, 4);
  }

  ctx.restore();
}

/**
 * Renders paper status and workspace feedback onto the paper canvas.
 */
export function renderPaperStatus(ctx: CanvasRenderingContext2D, tracking: PaperTrackingResult, width: number, height: number): void {
  // Clear paper canvas
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  if (!tracking.isTracking) {
    // Idle state: user guidance
    ctx.save();
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 8]);
    ctx.strokeRect(40, 40, width - 80, height - 80);

    ctx.fillStyle = "#94a3b8";
    ctx.font = "600 20px system-ui, -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Align paper sheet in camera view", width / 2, height / 2 - 14);

    ctx.font = "400 14px system-ui, -apple-system, sans-serif";
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText("Ensure 4 corner markers (0, 1, 2, 3) are visible", width / 2, height / 2 + 18);
    ctx.restore();
    return;
  }

  // Active tracking state: render clean paper surface with subtle layout guidelines
  ctx.save();

  // Subtle paper inner margin guideline
  ctx.strokeStyle = "#f1f5f9";
  ctx.lineWidth = 1;
  ctx.strokeRect(20, 20, width - 40, height - 40);

  // Marker exclusion zones
  for (const box of tracking.markerExclusionBoxes) {
    ctx.strokeStyle = "#f8fafc";
    ctx.fillStyle = "rgba(248, 250, 252, 0.7)";
    ctx.fillRect(box.x, box.y, box.width, box.height);
    ctx.strokeRect(box.x, box.y, box.width, box.height);
  }

  // Status Badge in top-right
  const badgeX = width - 260;
  const badgeY = 28;
  const badgeW = 232;
  const badgeH = 44;

  ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 8);
  ctx.fill();
  ctx.stroke();

  // Status indicator light
  const dotColor = tracking.isHeld ? COLOR_HELD : COLOR_LOCKED;
  ctx.fillStyle = dotColor;
  ctx.beginPath();
  ctx.arc(badgeX + 16, badgeY + badgeH / 2, 5, 0, Math.PI * 2);
  ctx.fill();

  // Status text
  ctx.fillStyle = "#1e293b";
  ctx.font = "600 12px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const title = tracking.isHeld ? `Tracking (Held: ${tracking.visibleCount}/4)` : `Paper Locked (${tracking.sheetMm[0]} × ${tracking.sheetMm[1]} mm)`;
  ctx.fillText(title, badgeX + 28, badgeY + 16);

  ctx.fillStyle = "#64748b";
  ctx.font = "400 10px system-ui, -apple-system, sans-serif";
  ctx.fillText("Phase 1: Basic Paper Tracking Ready", badgeX + 28, badgeY + 30);

  ctx.restore();
}
