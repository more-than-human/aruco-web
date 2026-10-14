import arucoWatBytes from "./lib-wasm/aruco-simd.wat";
import { Detector as WasmDetector, initArUcoWasm } from "./lib-wasm/aruco-wasm.ts";
import { Marker, Detector as TsDetector } from "./lib/aruco.ts";
import { TrackedMarker, Tracker } from "./lib/tracker.ts";
import "./style.css";

let sourceVideo: HTMLVideoElement;
let targetCanvas: HTMLCanvasElement;
let activeEngine: "wasm" | "ts" = "wasm";
let wasmDetector: WasmDetector | null = null;
let tsDetector: TsDetector | null = null;
let tracker: Tracker;
let trackingEnabled = true;

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

  try {
    await initArUcoWasm(arucoWatBytes);
    wasmDetector = new WasmDetector({
      dictionaryName: "4x4_1000",
      limit: 50,
      maxHammingDistance: 2,
    });
  } catch (err) {
    console.error("Failed to initialize WebAssembly ArUco engine:", err);
    activeEngine = "ts";
  }

  tsDetector = new TsDetector({
    dictionaryName: "4x4_1000",
    limit: 50,
    maxHammingDistance: 2,
  });

  const activeDetector = activeEngine === "wasm" && wasmDetector ? wasmDetector : tsDetector;

  tracker = new Tracker({
    dictionaryName: "4x4_1000",
    limit: 50,
    maxHammingDistance: 2,
    detector: activeDetector,
    maxMissedFrames: 5,
    processNoise: 15.0,
    measurementNoise: 2.5,
    coastVelocityDamping: 0.85,
  });

  setupEngineSelector();
  setupTrackingControls();

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

        const now = performance.now();
        let markers: (Marker | TrackedMarker)[];

        if (trackingEnabled) {
          markers = tracker.detect(imageData, now);
        } else {
          const detector = activeEngine === "wasm" && wasmDetector ? wasmDetector : tsDetector;
          markers = detector ? detector.detect(imageData) : [];
        }

        drawCorners(ctx, markers);
        drawId(ctx, markers);
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
    const detector = activeEngine === "wasm" && wasmDetector ? wasmDetector : tsDetector;
    if (detector) {
      tracker.setDetector(detector);
    }
    console.log(`Switched engine to: ${activeEngine}`);
  });
}

function setupTrackingControls(): void {
  const toggle = document.getElementById("tracking-toggle") as HTMLInputElement | null;
  if (toggle) {
    toggle.checked = trackingEnabled;
    toggle.addEventListener("change", () => {
      trackingEnabled = toggle.checked;
      if (!trackingEnabled) {
        tracker.reset();
      }
      updateStatusBadge();
    });
  }
}

function updateStatusBadge(markers?: (Marker | TrackedMarker)[]): void {
  const statsEl = document.getElementById("tracker-stats");
  if (!statsEl) return;

  if (!trackingEnabled) {
    statsEl.textContent = `Raw (${markers ? markers.length : 0})`;
    statsEl.style.color = "#d32f2f";
  } else {
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

  for (let i = 0; i < markers.length; ++i) {
    const m = markers[i];
    const corners = m.corners;
    const isTracked = m instanceof TrackedMarker;
    const isPredicted = isTracked && (m as TrackedMarker).isPredicted;

    if (isPredicted) {
      // Dropped/occluded in current frame: draw coasting prediction in dashed orange
      ctx.strokeStyle = "#ff9800";
      ctx.setLineDash([6, 4]);
      ctx.fillStyle = "rgba(255, 152, 0, 0.15)";
    } else {
      // Detected and filtered
      ctx.strokeStyle = isTracked ? "#00e676" : "#ff1744";
      ctx.setLineDash([]);
      ctx.fillStyle = isTracked ? "rgba(0, 230, 118, 0.1)" : "rgba(255, 23, 68, 0.1)";
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
    ctx.fill();
    ctx.stroke();

    // Mark Corner 0 to show marker orientation
    ctx.setLineDash([]);
    ctx.fillStyle = isPredicted ? "#ffb74d" : "#00e5ff";
    ctx.fillRect(corners[0].x - 3, corners[0].y - 3, 6, 6);

    // Render velocity vector arrow for tracked markers
    if (isTracked) {
      const tm = m as TrackedMarker;
      if (tm.velocity && tm.velocity.length === 4) {
        const vx = (tm.velocity[0].x + tm.velocity[1].x + tm.velocity[2].x + tm.velocity[3].x) * 0.25;
        const vy = (tm.velocity[0].y + tm.velocity[1].y + tm.velocity[2].y + tm.velocity[3].y) * 0.25;
        const speedSq = vx * vx + vy * vy;
        if (speedSq > 0.5) {
          const cx = (corners[0].x + corners[1].x + corners[2].x + corners[3].x) * 0.25;
          const cy = (corners[0].y + corners[1].y + corners[2].y + corners[3].y) * 0.25;
          const scale = 3.0;
          ctx.strokeStyle = "#ffea00";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(cx + vx * scale, cy + vy * scale);
          ctx.stroke();
          ctx.closePath();
          ctx.lineWidth = 3;
        }
      }
    }
  }

  ctx.restore();
}

function drawId(ctx: CanvasRenderingContext2D, markers: (Marker | TrackedMarker)[]): void {
  ctx.save();
  ctx.font = "bold 13px monospace";

  for (let i = 0; i < markers.length; ++i) {
    const m = markers[i];
    const corners = m.corners;
    const isTracked = m instanceof TrackedMarker;
    const isPredicted = isTracked && (m as TrackedMarker).isPredicted;

    let minX = Infinity;
    let minY = Infinity;
    for (let j = 0; j < corners.length; ++j) {
      minX = Math.min(minX, corners[j].x);
      minY = Math.min(minY, corners[j].y);
    }

    let label = `ID: ${m.id}`;
    if (isTracked) {
      const tm = m as TrackedMarker;
      if (isPredicted) {
        label += ` [coasting ${tm.missedFrames}]`;
      } else {
        label += ` (age: ${tm.age})`;
      }
    }

    const textWidth = ctx.measureText(label).width;
    const badgeX = minX;
    const badgeY = Math.max(18, minY - 8);

    // High legibility background pill
    ctx.fillStyle = isPredicted ? "rgba(230, 81, 0, 0.88)" : "rgba(0, 0, 0, 0.78)";
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(badgeX - 4, badgeY - 14, textWidth + 8, 18, 4);
    } else {
      ctx.rect(badgeX - 4, badgeY - 14, textWidth + 8, 18);
    }
    ctx.fill();

    ctx.fillStyle = isPredicted ? "#fff3e0" : "#ffffff";
    ctx.fillText(label, badgeX, badgeY);
  }

  ctx.restore();
}
