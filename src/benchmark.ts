import benchImageSrc from "../benchmark/bench-01.jpg";
import arucoWatBytes from "./lib-wasm/aruco-simd.wat";
import { Detector as WasmDetector, initArUcoWasm } from "./lib-wasm/aruco-wasm";
import { Detector as TsDetector } from "./lib/aruco";

async function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(e);
    img.src = src;
  });
}

function getImageData(img: HTMLImageElement): ImageData {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

interface BenchmarkMetrics {
  name: string;
  iterations: number;
  markersCount: number;
  avgLatency: number;
  minLatency: number;
  maxLatency: number;
  avgFps: number;
  totalTime: number;
}

function runBenchmarkOnDetector(name: string, detectFn: () => any[], iterations = 50, warmup = 5): BenchmarkMetrics {
  // Warmup
  for (let i = 0; i < warmup; i++) {
    detectFn();
  }

  const latencies: number[] = [];
  let markersCount = 0;
  const totalStartTime = performance.now();

  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    const markers = detectFn();
    const t1 = performance.now();
    latencies.push(t1 - t0);
    markersCount = markers.length;
  }
  const totalEndTime = performance.now();

  const totalTime = totalEndTime - totalStartTime;
  const avgLatency = latencies.reduce((sum, val) => sum + val, 0) / latencies.length;
  const avgFps = 1000 / avgLatency;
  const minLatency = Math.min(...latencies);
  const maxLatency = Math.max(...latencies);

  return {
    name,
    iterations,
    markersCount,
    avgLatency,
    minLatency,
    maxLatency,
    avgFps,
    totalTime,
  };
}

async function runBenchmark() {
  const statusEl = document.getElementById("status");
  const resultsEl = document.getElementById("results");

  try {
    if (statusEl) statusEl.textContent = "Loading WebAssembly module...";
    await initArUcoWasm(arucoWatBytes);

    if (statusEl) statusEl.textContent = "Loading bench-01.jpg...";
    const img = await loadImage(benchImageSrc);

    if (statusEl) statusEl.textContent = "Preparing image data...";
    const imageData = getImageData(img);

    const tsDetector = new TsDetector({
      dictionaryName: "6x6_1000",
      maxHammingDistance: 2,
    });

    const wasmDetector = new WasmDetector({
      dictionaryName: "6x6_1000",
      maxHammingDistance: 2,
    });

    if (statusEl) statusEl.textContent = "Running TypeScript benchmark...";
    await new Promise((resolve) => setTimeout(resolve, 50));
    const tsMetrics = runBenchmarkOnDetector("TypeScript", () => tsDetector.detect(imageData), 50);

    if (statusEl) statusEl.textContent = "Running WASM benchmark...";
    await new Promise((resolve) => setTimeout(resolve, 50));
    const wasmMetrics = runBenchmarkOnDetector("WebAssembly (SIMD)", () => wasmDetector.detect(imageData), 50);

    if (statusEl) statusEl.textContent = "Benchmark complete!";

    const speedup = tsMetrics.avgLatency / wasmMetrics.avgLatency;

    if (resultsEl) {
      resultsEl.innerHTML = `
        <h2>Benchmark Results (bench-01.jpg: ${imageData.width}x${imageData.height})</h2>
        <table border="1" cellpadding="6" style="border-collapse: collapse;">
          <thead>
            <tr>
              <th>Implementation</th>
              <th>Iterations</th>
              <th>Markers Detected</th>
              <th>Avg Latency</th>
              <th>Min Latency</th>
              <th>Max Latency</th>
              <th>Framerate (FPS)</th>
              <th>Total Time</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>${wasmMetrics.name}</strong></td>
              <td>${wasmMetrics.iterations}</td>
              <td>${wasmMetrics.markersCount}</td>
              <td>${wasmMetrics.avgLatency.toFixed(2)} ms</td>
              <td>${wasmMetrics.minLatency.toFixed(2)} ms</td>
              <td>${wasmMetrics.maxLatency.toFixed(2)} ms</td>
              <td><strong>${wasmMetrics.avgFps.toFixed(2)} FPS</strong></td>
              <td>${wasmMetrics.totalTime.toFixed(2)} ms</td>
            </tr>
            <tr>
              <td><strong>${tsMetrics.name}</strong></td>
              <td>${tsMetrics.iterations}</td>
              <td>${tsMetrics.markersCount}</td>
              <td>${tsMetrics.avgLatency.toFixed(2)} ms</td>
              <td>${tsMetrics.minLatency.toFixed(2)} ms</td>
              <td>${tsMetrics.maxLatency.toFixed(2)} ms</td>
              <td><strong>${tsMetrics.avgFps.toFixed(2)} FPS</strong></td>
              <td>${tsMetrics.totalTime.toFixed(2)} ms</td>
            </tr>
          </tbody>
        </table>
        <p><strong>WASM Speedup:</strong> ${speedup.toFixed(2)}x faster</p>
      `;
    }

    console.log("Benchmark Results:", {
      imageSize: `${imageData.width}x${imageData.height}`,
      wasm: wasmMetrics,
      typescript: tsMetrics,
      speedup: `${speedup.toFixed(2)}x`,
    });
  } catch (error) {
    if (statusEl) statusEl.textContent = `Error: ${error}`;
    console.error("Benchmark failed:", error);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  runBenchmark();
});
