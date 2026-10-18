import "./cursor.css";
import type { CursorConfig } from "./types.ts";

/**
 * On-screen cursor component responsible for:
 * 1. Rendering a lightweight colored circle representing a pointer (e.g. index fingertip).
 * 2. Positioning the circle smoothly on screen using hardware-accelerated transforms.
 */
export class Cursor {
  public readonly element: HTMLElement;
  private isVisible: boolean;
  private currentX = -100;
  private currentY = -100;

  constructor(config: CursorConfig = {}) {
    const { size = 20, color = "#ff0000", borderWidth = 0, borderColor = "transparent", container = document.body, className, visible = false } = config;

    this.element = document.createElement("div");
    this.element.className = `hand-cursor ${className ?? ""}`.trim();
    this.element.setAttribute("aria-hidden", "true");

    this.setSize(size);
    this.setColor(color);
    this.setBorder(borderWidth, borderColor);

    this.isVisible = visible;
    if (!this.isVisible) {
      this.element.classList.add("hand-cursor--hidden");
    }

    const parent = typeof container === "string" ? document.querySelector<HTMLElement>(container) : container;
    (parent ?? document.body).appendChild(this.element);
  }

  /**
   * Set cursor coordinates in screen space (pixels).
   */
  public setPosition(x: number, y: number): void {
    this.currentX = x;
    this.currentY = y;
    this.element.style.transform = `translate3d(${x}px, ${y}px, 0)`;
  }

  /**
   * Get current cursor coordinates.
   */
  public getPosition(): { x: number; y: number } {
    return { x: this.currentX, y: this.currentY };
  }

  /**
   * Show cursor on screen.
   */
  public show(): void {
    if (!this.isVisible) {
      this.isVisible = true;
      this.element.classList.remove("hand-cursor--hidden");
    }
  }

  /**
   * Hide cursor.
   */
  public hide(): void {
    if (this.isVisible) {
      this.isVisible = false;
      this.element.classList.add("hand-cursor--hidden");
    }
  }

  /**
   * Explicitly set cursor visibility.
   */
  public setVisible(visible: boolean): void {
    if (visible) {
      this.show();
    } else {
      this.hide();
    }
  }

  /**
   * Update cursor circle fill color.
   */
  public setColor(color: string): void {
    this.element.style.setProperty("--cursor-color", color);
  }

  /**
   * Update cursor circle diameter in pixels.
   */
  public setSize(size: number): void {
    this.element.style.setProperty("--cursor-size", `${size}px`);
  }

  /**
   * Update border styling.
   */
  public setBorder(width: number, color: string): void {
    this.element.style.setProperty("--cursor-border-width", `${width}px`);
    this.element.style.setProperty("--cursor-border-color", color);
  }

  /**
   * Remove cursor element from the DOM.
   */
  public destroy(): void {
    this.element.remove();
  }
}
