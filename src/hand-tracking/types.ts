import type { HandLandmarkerResult, Landmark, NormalizedLandmark } from "@mediapipe/tasks-vision";

export interface Point2D {
  x: number;
  y: number;
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface HandTrackingResult {
  /**
   * Whether a hand is currently detected in this frame.
   */
  hasHand: boolean;

  /**
   * Normalized coordinates of the index fingertip (x, y in [0, 1], z relative depth).
   */
  indexFingerTip?: Point3D;

  /**
   * Smoothed position in camera canvas pixel coordinates (e.g. [0, 640] x [0, 480]).
   */
  cameraPosition?: Point2D;

  /**
   * Raw (unsmoothed) position in camera canvas pixel coordinates.
   */
  rawCameraPosition?: Point2D;

  /**
   * Smoothed screen position (in pixels) mapped to the viewport.
   */
  screenPosition?: Point2D;

  /**
   * Raw (unsmoothed) screen position (in pixels) mapped to the viewport.
   */
  rawScreenPosition?: Point2D;

  /**
   * Handedness ("Left" or "Right").
   */
  handedness?: string;

  /**
   * Handedness confidence score.
   */
  handednessScore?: number;

  /**
   * All 21 normalized hand landmarks for the primary detected hand.
   */
  landmarks?: NormalizedLandmark[];

  /**
   * All 21 world landmarks in 3D meters relative to geometric center.
   */
  worldLandmarks?: Landmark[];

  /**
   * Monotonic timestamp in ms for this frame.
   */
  timestamp: number;

  /**
   * Underlying raw MediaPipe HandLandmarkerResult.
   */
  rawResult?: HandLandmarkerResult;
}

export interface HandTrackerConfig {
  /**
   * Path or URL to the wasm assets root.
   * @default "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm"
   */
  wasmPath?: string;

  /**
   * Path or URL to the hand landmarker model asset.
   * @default "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
   */
  modelAssetPath?: string;

  /**
   * Compute delegate: "GPU" or "CPU".
   * @default "GPU"
   */
  delegate?: "GPU" | "CPU";

  /**
   * Maximum number of hands to detect.
   * @default 1
   */
  numHands?: number;

  /**
   * Minimum confidence score for hand detection.
   * @default 0.5
   */
  minHandDetectionConfidence?: number;

  /**
   * Minimum confidence score for hand presence.
   * @default 0.5
   */
  minHandPresenceConfidence?: number;

  /**
   * Minimum tracking confidence.
   * @default 0.5
   */
  minTrackingConfidence?: number;

  /**
   * Whether to mirror horizontal coordinate (recommended for front-facing webcam).
   * @default true
   */
  mirrored?: boolean;

  /**
   * Exponential smoothing factor between 0 (no movement) and 1 (instant/no smoothing).
   * @default 0.4
   */
  smoothing?: number;

  /**
   * Viewport dimensions provider for mapping normalized coords to screen px.
   * @default () => ({ width: window.innerWidth, height: window.innerHeight })
   */
  getViewportSize?: () => { width: number; height: number };

  /**
   * Reference to the camera canvas (or getter) for accurately mapping camera space
   * to screen space taking into account object-fit (cover/contain), dimensions, and mirroring.
   */
  cameraCanvas?: HTMLCanvasElement | (() => HTMLCanvasElement | null);

  /**
   * Optional custom mapping function from normalized camera coordinates to screen space (CSS pixels).
   */
  mapToScreen?: (normalizedPoint: Point2D) => Point2D;

  /**
   * Callback invoked whenever a tracking frame is processed.
   */
  onUpdate?: (result: HandTrackingResult) => void;

  /**
   * Callback invoked when no hand is detected in a frame.
   */
  onLost?: () => void;
}
