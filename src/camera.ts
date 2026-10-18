export interface CameraConfig {
  idealWidth?: number;
  idealHeight?: number;
}

export async function initCamera(video: HTMLVideoElement): Promise<void> {
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

export async function setupCameraSelector(videoElement: HTMLVideoElement, onCameraChange?: () => void, config: CameraConfig = {}): Promise<void> {
  const select = document.getElementById("camera-select") as HTMLSelectElement | null;
  if (!select || !navigator.mediaDevices?.enumerateDevices) {
    return;
  }

  const { idealWidth = 640, idealHeight = 480 } = config;

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
      onCameraChange?.();

      if (videoElement.srcObject instanceof MediaStream) {
        videoElement.srcObject.getTracks().forEach((track) => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        audio: false,
        video: {
          deviceId: { exact: select.value },
          width: { ideal: idealWidth },
          height: { ideal: idealHeight },
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

export function setupControlButtons(): void {
  const printBtn = document.getElementById("print-btn");
  const helpBtn = document.getElementById("help-btn");

  printBtn?.addEventListener("click", () => {
    const link = document.createElement("a");
    link.href = `${import.meta.env.BASE_URL}tracking-paper.pdf`;
    link.download = "tracking-paper.pdf";
    link.click();
  });

  helpBtn?.addEventListener("click", () => {
    window.open("https://github.com/more-than-human/aruco-web/blob/master/README.md", "_blank", "noopener,noreferrer");
  });
}
