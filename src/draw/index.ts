import type { Marker } from "../lib/aruco.ts";
import type { TrackedMarker } from "../lib/tracker.ts";
import { PaperGeometry } from "./homography.ts";
import { type TemporalFilterConfig, TemporalInkFilter } from "./temporal-filter.ts";
import { AdaptiveThreshold, type ThresholdConfig } from "./threshold.ts";
import { LETTER_LANDSCAPE_HEIGHT, LETTER_LANDSCAPE_WIDTH, type PaperWarpModel } from "./types.ts";
import { PerspectiveWarper } from "./warper.ts";

export interface PaperDigitizerConfig {
  threshold?: ThresholdConfig;
  temporal?: TemporalFilterConfig;
  outwardPaddingRatio?: number;
  markerMarginRatio?: number;
}

export class PaperDigitizer {
  public readonly width: number;
  public readonly height: number;
  private geometry: PaperGeometry;
  private warper: PerspectiveWarper;
  private threshold: AdaptiveThreshold;
  private temporalFilter: TemporalInkFilter;

  constructor(width: number = LETTER_LANDSCAPE_WIDTH, height: number = LETTER_LANDSCAPE_HEIGHT, config?: PaperDigitizerConfig) {
    this.width = width;
    this.height = height;
    this.geometry = new PaperGeometry(width, height, config?.outwardPaddingRatio, config?.markerMarginRatio);
    this.warper = new PerspectiveWarper(width, height);
    this.threshold = new AdaptiveThreshold(width, height, config?.threshold);
    this.temporalFilter = new TemporalInkFilter(width, height, config?.temporal);
  }

  /**
   * Runs the 4-stage pipeline when all 4 markers are stably tracked.
   * Returns the latest digitized ImageData and current warp model (for debug overlays).
   */
  public processFrame(
    cameraImage: ImageData,
    markers: (Marker | TrackedMarker)[],
  ): {
    digitizedImage: ImageData;
    warpModel: PaperWarpModel | null;
  } {
    const warpModel = this.geometry.computeWarpModel(markers);

    if (!warpModel) {
      return {
        digitizedImage: this.temporalFilter.getOutputImage(),
        warpModel: null,
      };
    }

    const rectified = this.warper.warp(cameraImage, warpModel);
    const instantInk = this.threshold.process(rectified);
    const digitizedImage = this.temporalFilter.update(rectified, instantInk);

    return {
      digitizedImage,
      warpModel,
    };
  }

  public reset(): void {
    this.temporalFilter.reset();
  }
}

export { PaperGeometry } from "./homography.ts";
export { renderCorners, renderPaperBoundary } from "./render.ts";
export { TemporalInkFilter } from "./temporal-filter.ts";
export { AdaptiveThreshold } from "./threshold.ts";
export {
  DIGITIZING_ZONE_EXTEND_MM,
  LETTER_ASPECT_RATIO,
  LETTER_LANDSCAPE_HEIGHT,
  LETTER_LANDSCAPE_HEIGHT_MM,
  LETTER_LANDSCAPE_WIDTH,
  LETTER_LANDSCAPE_WIDTH_MM,
  MARKER_SIZE_MM,
  PAPER_EDGE_PADDING_MM,
} from "./types.ts";
export type { IRect, PaperWarpModel, RectifiedFrame } from "./types.ts";
export { PerspectiveWarper } from "./warper.ts";
