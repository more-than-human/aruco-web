import arucoWatBytes from "./lib-wasm/aruco-simd.wat";
import { Marker, Detector as WasmDetector, initArUcoWasm } from "./lib-wasm/aruco-wasm.ts";
import { TrackedMarker, Tracker } from "./lib/tracker.ts";
import "./style.css";

// --- Configuration Constants ---
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
const STROKE_COLOR = "#00e676";
const LINE_WIDTH = 3;
const COASTING_DASH_PATTERN = [6, 4];
const DICTIONARY_NAME = "4x4_1000";

const TRACKER_CONFIG = {
  dictionaryName: DICTIONARY_NAME,
  limit: 50,
  maxHammingDistance: 2,
  maxMissedFrames: 20,
  processNoise: 8.0,
  measurementNoise: 2.5,
  coastVelocityDamping: 0.85,
} as const;

// --- Application Lifecycle ---
window.addEventListener("load", async () => {
  console.log("init ARuco app...");

  const video = document.querySelector<HTMLVideoElement>("#video")!;
  const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;

  video.width = VIDEO_WIDTH;
  video.height = VIDEO_HEIGHT;
  canvas.width = VIDEO_WIDTH;
  canvas.height = VIDEO_HEIGHT;

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

  await initCamera(video);
  await setupCameraSelector(video, tracker);

  startDetectionLoop(video, canvas, tracker);
});

// --- Core Rendering & Processing Loop ---
function startDetectionLoop(video: HTMLVideoElement, canvas: HTMLCanvasElement, tracker: Tracker): void {
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

  const update = () => {
    ctx.drawImage(video, 0, 0, video.width, video.height);
    const imageData = ctx.getImageData(0, 0, video.width, video.height);
    const markers = tracker.detect(imageData, performance.now());

    drawCorners(ctx, markers);
    requestAnimationFrame(update);
  };

  requestAnimationFrame(update);
}

function drawCorners(ctx: CanvasRenderingContext2D, markers: (Marker | TrackedMarker)[]): void {
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

// --- Camera & Stream Management ---
async function initCamera(video: HTMLVideoElement): Promise<void> {
  const constraints: MediaStreamConstraints = {
    audio: false,
    video: {
      facingMode: "environment",
      width: 1280,
      height: 720,
    },
  };

  video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
  await video.play();

  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    await new Promise((resolve) => video.addEventListener("loadeddata", resolve, { once: true }));
  }
}

async function setupCameraSelector(videoElement: HTMLVideoElement, tracker: Tracker): Promise<void> {
  const select = document.getElementById("camera-select") as HTMLSelectElement | null;
  if (!select || !navigator.mediaDevices?.enumerateDevices) {
    return;
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((device) => device.kind === "videoinput");

    select.innerHTML = "";

    const activeTrack = videoElement.srcObject instanceof MediaStream ? videoElement.srcObject.getVideoTracks()[0] : null;
    const currentDeviceId = activeTrack?.getSettings().deviceId ?? null;

    videoDevices.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.text = device.label || `Camera ${index + 1}`;
      option.selected = device.deviceId === currentDeviceId;
      select.appendChild(option);
    });

    select.onchange = async () => {
      tracker.reset();

      if (videoElement.srcObject instanceof MediaStream) {
        videoElement.srcObject.getTracks().forEach((track) => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        audio: false,
        video: {
          deviceId: { exact: select.value },
          width: { ideal: VIDEO_WIDTH },
          height: { ideal: VIDEO_HEIGHT },
        },
      };

      try {
        videoElement.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
        await videoElement.play();
      } catch (err) {
        console.error("Error switching camera:", err);
      }
    };
  } catch (err) {
    console.error("Error enumerating devices:", err);
  }
}
