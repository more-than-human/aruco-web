import arucoWatBytes from "./aruco-tracking/aruco-simd.wat";
import { Detector as WasmDetector, initArUcoWasm } from "./aruco-tracking/aruco-wasm.ts";
import { Tracker } from "./aruco-tracking/tracker.ts";
import { initCamera, setupCameraSelector } from "./camera.ts";
import { Cursor } from "./cursor/index.ts";
import { LETTER_LANDSCAPE_HEIGHT, LETTER_LANDSCAPE_WIDTH, PaperTracker, renderCorners, renderPaperBoundary, renderPaperStatus } from "./digitize/index.ts";
import { FpsTracker } from "./fps.ts";
import { HandTracker } from "./hand-tracking/index.ts";
import "./main.css";
import { SpeechBubble } from "./speech-bubble/index.ts";
import { SpeechRecognizer } from "./speech/index.ts";

// --- Configuration Constants ---
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;

// Render digitized paper in landscape US Letter dimensions
const DIGITIZED_PAPER_WIDTH = LETTER_LANDSCAPE_WIDTH;
const DIGITIZED_PAPER_HEIGHT = LETTER_LANDSCAPE_HEIGHT;

const DICTIONARY_NAME = "4x4_1000";

const TRACKER_CONFIG = {
  dictionaryName: DICTIONARY_NAME,
  limit: 50,
  maxHammingDistance: 2,
  maxMissedFrames: 20,
  processNoise: 1.5,
  measurementNoise: 8.0,
  coastVelocityDamping: 0.85,
} as const;

// --- Speech Recognition ---
const SPEECH_CONFIG = {
  lang: "en-US",
  continuous: true,
  interimResults: true,
  autoRestart: true,
} as const;

// --- Application Lifecycle ---
window.addEventListener("load", async () => {
  console.log("init ARuco app...");

  const video = document.querySelector<HTMLVideoElement>("#video")!;
  const cameraCanvas = document.querySelector<HTMLCanvasElement>("#camera-canvas")!;
  const paperCanvas = document.querySelector<HTMLCanvasElement>("#paper-canvas")!;

  video.width = VIDEO_WIDTH;
  video.height = VIDEO_HEIGHT;
  cameraCanvas.width = VIDEO_WIDTH;
  cameraCanvas.height = VIDEO_HEIGHT;
  paperCanvas.width = DIGITIZED_PAPER_WIDTH;
  paperCanvas.height = DIGITIZED_PAPER_HEIGHT;

  await initArUcoWasm(arucoWatBytes);

  const wasmDetector = new WasmDetector({
    dictionaryName: DICTIONARY_NAME,
    limit: TRACKER_CONFIG.limit,
    maxHammingDistance: TRACKER_CONFIG.maxHammingDistance,
  });

  const tracker = new Tracker({
    ...TRACKER_CONFIG,
    detector: wasmDetector,
  });

  const paperTracker = new PaperTracker({
    canvasWidth: DIGITIZED_PAPER_WIDTH,
    canvasHeight: DIGITIZED_PAPER_HEIGHT,
  });

  // Layer 1: Hand cursor on top DOM layer (simple red circle, minimum styling)
  const cursor = new Cursor({
    size: 20,
    color: "#ff0000",
    borderWidth: 0,
    borderColor: "transparent",
    visible: false,
  });

  // Hand tracking engine for index finger tracking
  const handTracker = new HandTracker({
    cameraCanvas,
    mirrored: true,
    smoothing: 0.5,
    onUpdate: (result) => {
      if (result.hasHand && result.screenPosition) {
        cursor.setPosition(result.screenPosition.x, result.screenPosition.y);
        cursor.show();
      } else {
        cursor.hide();
      }
    },
    onLost: () => {
      cursor.hide();
    },
  });

  // Voice -> Text speech recognition engine
  const speech = new SpeechRecognizer(SPEECH_CONFIG);

  // Speech bubble system responsible for displaying text and letting user toggle start/stop
  new SpeechBubble({
    speech,
  });

  try {
    await initCamera(video);
  } catch (err) {
    console.error("Error initializing camera:", err);
    return;
  }

  await setupCameraSelector(
    video,
    () => {
      tracker.reset();
      paperTracker.reset();
      cursor.hide();
    },
    {
      idealWidth: VIDEO_WIDTH,
      idealHeight: VIDEO_HEIGHT,
    },
  );

  startDetectionLoop(video, cameraCanvas, paperCanvas, tracker, paperTracker);

  // Initialize and start hand tracking asynchronously
  handTracker
    .init()
    .then(() => {
      handTracker.start(video);
    })
    .catch((err) => {
      console.error("Error initializing HandTracker:", err);
    });
});

// --- Core Rendering & Processing Loop ---
function startDetectionLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  paperCanvas: HTMLCanvasElement,
  tracker: Tracker,
  paperTracker: PaperTracker,
): void {
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  const paperCtx = paperCanvas.getContext("2d", { willReadFrequently: true })!;

  const fpsTracker = new FpsTracker("frame-counter");

  const update = () => {
    fpsTracker.update();

    ctx.drawImage(video, 0, 0, video.width, video.height);
    const imageData = ctx.getImageData(0, 0, video.width, video.height);
    const now = performance.now();
    const markers = tracker.detect(imageData, now);
    const tracking = paperTracker.update(markers, now);

    renderPaperStatus(paperCtx, tracking, DIGITIZED_PAPER_WIDTH, DIGITIZED_PAPER_HEIGHT);

    renderCorners(ctx, markers);
    if (tracking.isTracking) {
      renderPaperBoundary(ctx, tracking.boundary, tracking.isHeld);
    }

    requestAnimationFrame(update);
  };

  requestAnimationFrame(update);
}
