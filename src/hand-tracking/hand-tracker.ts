import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from "@mediapipe/tasks-vision";
import type { HandTrackerConfig, HandTrackingResult, Point2D } from "./types.ts";

/**
 * Standard MediaPipe Hand Landmark indices (21 landmarks).
 */
export const HAND_LANDMARKS = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_FINGER_MCP: 5,
  INDEX_FINGER_PIP: 6,
  INDEX_FINGER_DIP: 7,
  INDEX_FINGER_TIP: 8,
  MIDDLE_FINGER_MCP: 9,
  MIDDLE_FINGER_PIP: 10,
  MIDDLE_FINGER_DIP: 11,
  MIDDLE_FINGER_TIP: 12,
  RING_FINGER_MCP: 13,
  RING_FINGER_PIP: 14,
  RING_FINGER_DIP: 15,
  RING_FINGER_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20,
} as const;

export const HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;

const DEFAULT_WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm";
const DEFAULT_MODEL_PATH = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

/**
 * HandTracker uses MediaPipe's HandLandmarker in GPU VIDEO mode.
 *
 * It is completely decoupled from any webcam canvas/UI view:
 * it consumes the raw HTMLVideoElement stream directly and emits
 * normalized hand landmarks and screen-mapped cursor coordinates.
 */
export class HandTracker {
  private landmarker: HandLandmarker | null = null;
  private readonly config: HandTrackerConfig;

  private isRunning = false;
  private rafId: number | null = null;
  private videoElement: HTMLVideoElement | null = null;

  private lastVideoTime = -1;
  private lastTimestamp = -1;
  private smoothedPosition: Point2D | null = null;

  constructor(config: HandTrackerConfig = {}) {
    this.config = {
      wasmPath: DEFAULT_WASM_PATH,
      modelAssetPath: DEFAULT_MODEL_PATH,
      delegate: "GPU",
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      mirrored: true,
      smoothing: 0.4,
      ...config,
    };
  }

  /**
   * Initialize MediaPipe Wasm runtime and load the Hand Landmarker model
   * in GPU VIDEO mode.
   */
  public async init(): Promise<void> {
    if (this.landmarker) return;

    const wasmPath = this.config.wasmPath ?? DEFAULT_WASM_PATH;
    const modelAssetPath = this.config.modelAssetPath ?? DEFAULT_MODEL_PATH;
    const delegate = this.config.delegate ?? "GPU";

    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    try {
      this.landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath,
          delegate,
        },
        runningMode: "VIDEO",
        numHands: this.config.numHands ?? 1,
        minHandDetectionConfidence: this.config.minHandDetectionConfidence ?? 0.5,
        minHandPresenceConfidence: this.config.minHandPresenceConfidence ?? 0.5,
        minTrackingConfidence: this.config.minTrackingConfidence ?? 0.5,
      });
    } catch (err) {
      if (delegate === "GPU") {
        console.warn("HandTracker: GPU delegate initialization failed. Falling back to CPU delegate:", err);
        this.landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath,
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numHands: this.config.numHands ?? 1,
          minHandDetectionConfidence: this.config.minHandDetectionConfidence ?? 0.5,
          minHandPresenceConfidence: this.config.minHandPresenceConfidence ?? 0.5,
          minTrackingConfidence: this.config.minTrackingConfidence ?? 0.5,
        });
      } else {
        throw err;
      }
    }
  }

  /**
   * Start tracking hands from the video feed in an autonomous animation loop.
   */
  public start(video: HTMLVideoElement): void {
    if (!this.landmarker) {
      throw new Error("HandTracker must be initialized with init() before start()");
    }

    this.videoElement = video;
    this.isRunning = true;
    this.lastVideoTime = -1;
    this.lastTimestamp = -1;
    this.smoothedPosition = null;

    const loop = () => {
      if (!this.isRunning) return;

      if (this.videoElement) {
        this.processFrame(this.videoElement);
      }

      this.rafId = requestAnimationFrame(loop);
    };

    this.rafId = requestAnimationFrame(loop);
  }

  /**
   * Stop the autonomous tracking loop.
   */
  public stop(): void {
    this.isRunning = false;
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.smoothedPosition = null;
    this.videoElement = null;
  }

  /**
   * Process a single video frame manually. Can be used if orchestrating the
   * render loop externally instead of using start().
   */
  public processVideo(video: HTMLVideoElement, manualTimestamp?: number): HandTrackingResult | null {
    if (!this.landmarker) {
      return null;
    }

    return this.processFrame(video, manualTimestamp);
  }

  /**
   * Core frame processing logic.
   */
  private processFrame(video: HTMLVideoElement, manualTimestamp?: number): HandTrackingResult | null {
    if (!this.landmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.paused || video.ended) {
      return null;
    }

    // Only run MediaPipe inference when a new video frame is actually available
    if (video.currentTime === this.lastVideoTime) {
      return null;
    }
    this.lastVideoTime = video.currentTime;

    // Enforce strictly monotonically increasing timestamp for MediaPipe video mode
    const now = manualTimestamp ?? performance.now();
    const timestamp = now > this.lastTimestamp ? now : this.lastTimestamp + 1;
    this.lastTimestamp = timestamp;

    const rawResult = this.landmarker.detectForVideo(video, timestamp);
    const result = this.extractResult(rawResult, timestamp);

    if (result.hasHand) {
      this.config.onUpdate?.(result);
    } else {
      this.smoothedPosition = null;
      this.config.onLost?.();
      this.config.onUpdate?.(result);
    }

    return result;
  }

  /**
   * Maps a normalized camera coordinate (x, y in [0, 1]) to screen space (CSS pixels)
   * taking into account object-fit (cover), element bounding box, and mirroring.
   */
  private computeScreenPosition(normPoint: Point2D): Point2D {
    if (this.config.mapToScreen) {
      return this.config.mapToScreen(normPoint);
    }

    const canvas = typeof this.config.cameraCanvas === "function" ? this.config.cameraCanvas() : this.config.cameraCanvas;

    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      const containerW = rect.width;
      const containerH = rect.height;

      const intrinsicW = canvas.width || this.videoElement?.videoWidth || 640;
      const intrinsicH = canvas.height || this.videoElement?.videoHeight || 480;

      const scale = Math.max(containerW / intrinsicW, containerH / intrinsicH);
      const renderedW = intrinsicW * scale;
      const renderedH = intrinsicH * scale;

      const offsetX = rect.left + (containerW - renderedW) / 2;
      const offsetY = rect.top + (containerH - renderedH) / 2;

      const normX = this.config.mirrored ? 1 - normPoint.x : normPoint.x;
      const normY = normPoint.y;

      return {
        x: offsetX + normX * renderedW,
        y: offsetY + normY * renderedH,
      };
    }

    // Fallback: full viewport stretch
    const viewport = this.config.getViewportSize ? this.config.getViewportSize() : { width: window.innerWidth, height: window.innerHeight };

    const normX = this.config.mirrored ? 1 - normPoint.x : normPoint.x;
    const normY = normPoint.y;

    return {
      x: normX * viewport.width,
      y: normY * viewport.height,
    };
  }

  /**
   * Extract hand landmarks, index finger tip coordinate, and screen mapping.
   */
  private extractResult(raw: HandLandmarkerResult, timestamp: number): HandTrackingResult {
    const hasHand = Boolean(raw.landmarks && raw.landmarks.length > 0);

    if (!hasHand) {
      return {
        hasHand: false,
        timestamp,
        rawResult: raw,
      };
    }

    const landmarks = raw.landmarks[0];
    const worldLandmarks = raw.worldLandmarks?.[0];
    const indexTip = landmarks[HAND_LANDMARKS.INDEX_FINGER_TIP];

    const rawScreenPos = this.computeScreenPosition({ x: indexTip.x, y: indexTip.y });
    const rawScreenX = rawScreenPos.x;
    const rawScreenY = rawScreenPos.y;

    // Exponential smoothing for cursor stability
    let screenX = rawScreenX;
    let screenY = rawScreenY;
    const smoothing = this.config.smoothing ?? 0.4;

    if (smoothing > 0 && smoothing < 1 && this.smoothedPosition) {
      screenX = this.smoothedPosition.x + (rawScreenX - this.smoothedPosition.x) * smoothing;
      screenY = this.smoothedPosition.y + (rawScreenY - this.smoothedPosition.y) * smoothing;
      this.smoothedPosition = { x: screenX, y: screenY };
    } else {
      this.smoothedPosition = { x: screenX, y: screenY };
    }

    // Camera canvas pixel coordinates
    const intrinsicW = this.videoElement?.videoWidth || 640;
    const intrinsicH = this.videoElement?.videoHeight || 480;
    const normCameraX = this.config.mirrored ? 1 - indexTip.x : indexTip.x;
    const cameraX = normCameraX * intrinsicW;
    const cameraY = indexTip.y * intrinsicH;

    // Extract handedness
    let handedness: string | undefined;
    let handednessScore: number | undefined;
    if (raw.handedness && raw.handedness.length > 0 && raw.handedness[0].length > 0) {
      const topHand = raw.handedness[0][0];
      handedness = topHand.categoryName || topHand.displayName;
      handednessScore = topHand.score;
    }

    return {
      hasHand: true,
      indexFingerTip: {
        x: indexTip.x,
        y: indexTip.y,
        z: indexTip.z,
      },
      cameraPosition: {
        x: cameraX,
        y: cameraY,
      },
      rawCameraPosition: {
        x: cameraX,
        y: cameraY,
      },
      screenPosition: {
        x: screenX,
        y: screenY,
      },
      rawScreenPosition: {
        x: rawScreenX,
        y: rawScreenY,
      },
      handedness,
      handednessScore,
      landmarks,
      worldLandmarks,
      timestamp,
      rawResult: raw,
    };
  }

  /**
   * Release MediaPipe landmarker resources.
   */
  public destroy(): void {
    this.stop();
    if (this.landmarker) {
      this.landmarker.close();
      this.landmarker = null;
    }
  }
}
