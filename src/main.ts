import arucoWatBytes from "./lib-wasm/aruco-simd.wat";
import { Detector as WasmDetector, initArUcoWasm } from "./lib-wasm/aruco-wasm.ts";
import { Marker, Detector as TsDetector } from "./lib/aruco.ts";
import "./style.css";

let sourceVideo: HTMLVideoElement;
let targetCanvas: HTMLCanvasElement;
let activeEngine: "wasm" | "ts" = "wasm";
let wasmDetector: WasmDetector | null = null;
let tsDetector: TsDetector | null = null;

async function initCamera(): Promise<HTMLVideoElement> {
  const constraints: MediaStreamConstraints = {
    audio: false,
    video: {
      // using the "environment" rear camera
      facingMode: "environment",
      // using the "user" front camera
      // facingMode: "user",
      width: 640,
      height: 480,
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
      dictionaryName: "6x6_1000",
      maxHammingDistance: 2,
    });
  } catch (err) {
    console.error("Failed to initialize WebAssembly ArUco engine:", err);
    activeEngine = "ts";
  }

  tsDetector = new TsDetector({
    dictionaryName: "6x6_1000",
    maxHammingDistance: 2,
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

        const detector = activeEngine === "wasm" && wasmDetector ? wasmDetector : tsDetector;
        const markers = detector ? detector.detect(imageData) : [];

        drawCorners(ctx, markers);
        drawId(ctx, markers);
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
    console.log(`Switched engine to: ${activeEngine}`);
  });
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

function drawCorners(ctx: CanvasRenderingContext2D, markers: Marker[]): void {
  let corners, corner, i, j;

  ctx.lineWidth = 3;

  for (i = 0; i !== markers.length; ++i) {
    corners = markers[i].corners;

    ctx.strokeStyle = "red";
    ctx.beginPath();

    for (j = 0; j !== corners.length; ++j) {
      corner = corners[j];
      ctx.moveTo(corner.x, corner.y);
      corner = corners[(j + 1) % corners.length];
      ctx.lineTo(corner.x, corner.y);
    }

    ctx.stroke();
    ctx.closePath();

    ctx.strokeStyle = "green";
    ctx.strokeRect(corners[0].x - 2, corners[0].y - 2, 4, 4);
  }
}

function drawId(ctx: CanvasRenderingContext2D, markers: Marker[]): void {
  let corners, corner, x, y, i, j;

  ctx.strokeStyle = "blue";
  ctx.lineWidth = 1;

  for (i = 0; i !== markers.length; ++i) {
    corners = markers[i].corners;

    x = Infinity;
    y = Infinity;

    for (j = 0; j !== corners.length; ++j) {
      corner = corners[j];

      x = Math.min(x, corner.x);
      y = Math.min(y, corner.y);
    }

    ctx.strokeText(String(markers[i].id), x, y);
  }
}
