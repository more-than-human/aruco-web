export interface Point2D {
  x: number;
  y: number;
}

export interface DetectedMarker {
  id: number;
  corners: Point2D[];
  distance: number;
}

export interface TrackedMarkerState {
  id: number;
  detected: boolean;
  lastSeenAt: number | null;
  corners: Point2D[] | null;
  center: Point2D | null;
  distance: number | null;
}

export interface DetectionResult {
  markers: DetectedMarker[];
  inferenceTimeMs: number;
  imageWidth: number;
  imageHeight: number;
  timestamp: number;
}
