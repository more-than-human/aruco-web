export interface CursorConfig {
  /**
   * Diameter of the cursor in pixels.
   * @default 20
   */
  size?: number;

  /**
   * Fill color of the cursor.
   * @default "#ff0000"
   */
  color?: string;

  /**
   * Border width in pixels.
   * @default 0
   */
  borderWidth?: number;

  /**
   * Border color.
   * @default "transparent"
   */
  borderColor?: string;

  /**
   * Container element where cursor is appended.
   * @default document.body
   */
  container?: HTMLElement | string;

  /**
   * Optional custom CSS class name.
   */
  className?: string;

  /**
   * Initial visibility of the cursor.
   * @default false
   */
  visible?: boolean;
}
