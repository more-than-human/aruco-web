import { ArucoDetector, DetectorParameters, Dictionary, type ImageInput, MarkerDetector, PredefinedDictionaryType } from "./lib/aruco-nano";
import "./style.css";

// 1. Create a synthetic 400x400 image with a rotated DICT_ARUCO_MIP_36h12 marker (ID = 7)
function createSyntheticScene(): ImageInput {
  const width = 400;
  const height = 400;
  const data = new Uint8Array(width * height).fill(220); // Light gray background

  const dict = Dictionary.getPredefinedDictionary(PredefinedDictionaryType.DICT_ARUCO_MIP_36h12);
  const grid = dict.generateMarkerMatrix(7, 1); // 8x8 matrix (6x6 inner + 1 border)

  // Render marker centered at (200, 200) of size 160x160 pixels
  const startX = 120;
  const startY = 120;
  const cellPx = 20;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const val = grid[r * 8 + c];
      for (let dy = 0; dy < cellPx; dy++) {
        for (let dx = 0; dx < cellPx; dx++) {
          data[(startY + r * cellPx + dy) * width + (startX + c * cellPx + dx)] = val;
        }
      }
    }
  }

  return { width, height, data, channels: 1 };
}

// 2. Native ArUco Nano Style Detection + Pose Estimation
const scene = createSyntheticScene();
const markers = MarkerDetector.detect(scene);

const camMatrix = [500, 0, 200, 0, 500, 200, 0, 0, 1];
const distCoeffs = [0, 0, 0, 0, 0];

for (const m of markers) {
  const pose = m.estimatePose(camMatrix, distCoeffs, 0.05); // 5cm physical marker
  console.log(`Detected Marker ID=${m.id}`, m, pose);
}

// 3. OpenCV-Compatible API Style
const detector = new ArucoDetector(Dictionary.getPredefinedDictionary(PredefinedDictionaryType.DICT_ARUCO_MIP_36h12), new DetectorParameters());
const corners: Array<Array<{ x: number; y: number }>> = [];
const ids: number[] = [];
detector.detectMarkers(scene, corners, ids);
