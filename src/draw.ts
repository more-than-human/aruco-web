import { initCamera, setupCameraSelector } from "./camera.ts";
import "./draw.css";
import { A4_PORTRAIT_HEIGHT, A4_PORTRAIT_WIDTH, PaperDigitizer } from "./draw/index.ts";
import { renderCorners, renderPaperBoundary } from "./draw/render.ts";
import arucoWatBytes from "./lib-wasm/aruco-simd.wat";
import { Detector as WasmDetector, initArUcoWasm } from "./lib-wasm/aruco-wasm.ts";
import { Tracker } from "./lib/tracker.ts";

// --- Configuration Constants ---
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;

// Render digitized paper in portrait A4 dimensions
const DIGITIZED_PAPER_WIDTH = A4_PORTRAIT_WIDTH;
const DIGITIZED_PAPER_HEIGHT = A4_PORTRAIT_HEIGHT;

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

// --- Application Lifecycle ---
window.addEventListener("load", async () => {
  console.log("init ARuco app...");

  const video = document.querySelector<HTMLVideoElement>("#video")!;
  const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
  const paperCanvas = document.querySelector<HTMLCanvasElement>("#paper-canvas")!;

  video.width = VIDEO_WIDTH;
  video.height = VIDEO_HEIGHT;
  canvas.width = VIDEO_WIDTH;
  canvas.height = VIDEO_HEIGHT;
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

  const digitizer = new PaperDigitizer(DIGITIZED_PAPER_WIDTH, DIGITIZED_PAPER_HEIGHT, {
    threshold: {
      thresholdLow: 8, // Pick up faint strokes under low contrast
      thresholdHigh: 22,
      windowSize: 31,
    },
    temporal: {
      motionThreshold: 16, // Tolerant to residual jitter
      requiredStableFrames: 4, // Faster ink confirmation
      blendAlpha: 0.6,
    },
  });

  await initCamera(video);
  await setupCameraSelector(
    video,
    () => {
      tracker.reset();
      digitizer.reset();
    },
    {
      idealWidth: VIDEO_WIDTH,
      idealHeight: VIDEO_HEIGHT,
    },
  );

  startDetectionLoop(video, canvas, paperCanvas, tracker, digitizer);
});

// --- Core Rendering & Processing Loop ---
function startDetectionLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  paperCanvas: HTMLCanvasElement,
  tracker: Tracker,
  digitizer: PaperDigitizer,
): void {
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  const paperCtx = paperCanvas.getContext("2d", { willReadFrequently: true })!;

  const update = () => {
    ctx.drawImage(video, 0, 0, video.width, video.height);
    const imageData = ctx.getImageData(0, 0, video.width, video.height);
    const markers = tracker.detect(imageData, performance.now());

    const { digitizedImage, warpModel } = digitizer.processFrame(imageData, markers);
    paperCtx.putImageData(digitizedImage, 0, 0);

    renderCorners(ctx, markers);
    if (warpModel) {
      renderPaperBoundary(ctx, warpModel);
    }

    requestAnimationFrame(update);
  };

  requestAnimationFrame(update);
}
