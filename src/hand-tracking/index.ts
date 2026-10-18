/**
 * Hand Tracking Module (MediaPipe GPU Video Mode)
 *
 * This module is solely responsible for processing a video feed using
 * MediaPipe's HandLandmarker in GPU video mode and extracting
 * hand landmarks, index-finger position, and screen-mapped coordinates.
 * It is completely decoupled from any canvas or camera UI view.
 */

export * from "./hand-tracker.ts";
export * from "./types.ts";
