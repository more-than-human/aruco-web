import arucoWatBytes from "./lib-wasm/aruco-simd.wat";
import { Marker, Detector as WasmDetector, initArUcoWasm } from "./lib-wasm/aruco-wasm.ts";
import { TrackedMarker, Tracker } from "./lib/tracker.ts";
import "./style.css";

let sourceVideo: HTMLVideoElement;
let targetCanvas: HTMLCanvasElement;
let activeEngine: "wasm" | "ts" = "wasm";
let wasmDetector: WasmDetector | null = null;
let tracker: Tracker;

async function initCamera(): Promise<HTMLVideoElement> {
  const constraints: MediaStreamConstraints = {
    audio: false,
    video: {
      // using the "environment" rear camera
      facingMode: "environment",
      // using the "user" front camera
      // facingMode: "user",
      width: 1280,
      height: 720,
    },
  };

  // initialize video source
  const video = document.querySelector<HTMLVideoElement>("#video")!;
  video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

function initTargetCanvas(): void {
  // target canvas should overlap source video
  targetCanvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
  targetCanvas.width = sourceVideo.width;
  targetCanvas.height = sourceVideo.height;
}

export function getImageData(v: CanvasImageSource, w: number, h: number): ImageData {
  console.log(v);
  const ctx = targetCanvas.getContext("2d")!;
  ctx.drawImage(v, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

window.addEventListener("load", async () => {
  console.log("init ARuco app...");

  await initArUcoWasm(arucoWatBytes);
  wasmDetector = new WasmDetector({
    dictionaryName: "4x4_1000",
    limit: 50,
    maxHammingDistance: 2,
  });
  tracker = new Tracker({
    dictionaryName: "4x4_1000",
    limit: 50,
    maxHammingDistance: 2,
    detector: wasmDetector,
    maxMissedFrames: 20,
    processNoise: 8.0,
    measurementNoise: 2.5,
    coastVelocityDamping: 0.85,
  });

  setupEngineSelector();

  initCamera()
    .then((video) => {
      // start camera playback
      sourceVideo = video;
      sourceVideo.width = 640;
      sourceVideo.height = 480;
      sourceVideo.play();

      // init target canvas
      initTargetCanvas();

      // Populate camera selector after initial permission is granted
      setupCameraSelector(sourceVideo);

      return new Promise<HTMLVideoElement>((resolve) => {
        sourceVideo.addEventListener("loadeddata", (event) => {
          console.log("Camera is ready", event);
          resolve(sourceVideo);
        });
      });
    })
    .then((video) => {
      let imageData: ImageData;

      function update() {
        const ctx = targetCanvas.getContext("2d", {
          willReadFrequently: true,
        })!;
        ctx.drawImage(video, 0, 0, video.width, video.height);
        imageData = ctx.getImageData(0, 0, video.width, video.height);

        const markers = tracker.detect(imageData, performance.now());

        drawCorners(ctx, markers);
        updateStatusBadge(markers);
        requestAnimationFrame(update);
      }

      update();
    });
});

function setupEngineSelector(): void {
  const engineSelect = document.getElementById("engine-select") as HTMLSelectElement | null;
  if (!engineSelect) {
    return;
  }

  if (!wasmDetector) {
    const wasmOption = engineSelect.querySelector('option[value="wasm"]') as HTMLOptionElement | null;
    if (wasmOption) {
      wasmOption.disabled = true;
      wasmOption.text += " (Not Supported / Failed to load)";
    }
    engineSelect.value = "ts";
    activeEngine = "ts";
  } else {
    engineSelect.value = activeEngine;
  }

  engineSelect.addEventListener("change", () => {
    activeEngine = engineSelect.value as "wasm" | "ts";
    const detector = wasmDetector;
    if (detector) {
      tracker.setDetector(detector);
    }
    console.log(`Switched engine to: ${activeEngine}`);
  });
}

function updateStatusBadge(markers?: (Marker | TrackedMarker)[]): void {
  const statsEl = document.getElementById("tracker-stats");
  if (!statsEl) return;

  const activeCount = tracker ? tracker.activeTracksCount : 0;
  const coastingCount = markers ? markers.filter((m) => m instanceof TrackedMarker && m.isPredicted).length : 0;
  if (coastingCount > 0) {
    statsEl.textContent = `Kalman: ${activeCount} (${coastingCount} coasting)`;
    statsEl.style.color = "#ef6c00";
  } else {
    statsEl.textContent = `Kalman: ${activeCount} active`;
    statsEl.style.color = "#2e7d32";
  }
}

async function setupCameraSelector(videoElement: HTMLVideoElement): Promise<void> {
  const select = document.getElementById("camera-select") as HTMLSelectElement | null;
  if (!select) {
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    console.warn("enumerateDevices() not supported.");
    return;
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((device) => device.kind === "videoinput");

    select.innerHTML = "";

    // Identify currently active camera device
    let currentDeviceId: string | null = null;
    if (videoElement.srcObject instanceof MediaStream) {
      const videoTracks = videoElement.srcObject.getVideoTracks();
      if (videoTracks.length > 0) {
        currentDeviceId = videoTracks[0].getSettings().deviceId || null;
      }
    }

    videoDevices.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.text = device.label || `Camera ${index + 1}`;
      if (device.deviceId === currentDeviceId) {
        option.selected = true;
      }
      select.appendChild(option);
    });

    select.onchange = async () => {
      const selectedDeviceId = select.value;

      tracker.reset();

      // Stop existing video tracks before switching
      if (videoElement.srcObject instanceof MediaStream) {
        videoElement.srcObject.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        audio: false,
        video: {
          deviceId: { exact: selectedDeviceId },
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      };

      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;
        await videoElement.play();
        initTargetCanvas();
      } catch (err) {
        console.error("Error switching camera:", err);
      }
    };
  } catch (err) {
    console.error("Error enumerating devices:", err);
  }
}

function drawCorners(ctx: CanvasRenderingContext2D, markers: (Marker | TrackedMarker)[]): void {
  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#00e676";

  for (let i = 0; i < markers.length; ++i) {
    const m = markers[i];
    const corners = m.corners;
    const isPredicted = m instanceof TrackedMarker && m.isPredicted;

    if (isPredicted) {
      ctx.setLineDash([6, 4]);
    } else {
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    for (let j = 0; j < corners.length; ++j) {
      const corner = corners[j];
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
