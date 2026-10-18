// --- Configuration Constants ---
const EMA_ALPHA = 0.05; // Exponential moving average of frame duration -> smoothed FPS display.

/**
 * Tracks frame timings and displays a smoothed FPS readout in a DOM element.
 */
export class FpsTracker {
  private smoothedFrameMs: number | null = null;
  private lastFrameTime = performance.now();

  private readonly element: HTMLElement | null;

  constructor(elementId: string) {
    this.element = document.getElementById(elementId);
  }

  /** Record a new frame and update the FPS display. */
  update(): void {
    const now = performance.now();
    const frameMs = now - this.lastFrameTime;
    this.lastFrameTime = now;

    this.smoothedFrameMs =
      this.smoothedFrameMs === null ? frameMs : EMA_ALPHA * frameMs + (1 - EMA_ALPHA) * this.smoothedFrameMs;

    if (this.element && this.smoothedFrameMs > 0) {
      this.element.textContent = `${(1000 / this.smoothedFrameMs).toFixed(1)} FPS`;
    }
  }
}