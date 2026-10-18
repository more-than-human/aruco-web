import type { SpeechRecognizerEvent } from "../speech/index.ts";

/**
 * Minimal contract required by the speech-bubble system from the speech engine.
 * `SpeechRecognizer` in `src/speech` satisfies this interface.
 */
export interface SpeechSource {
  readonly isSupported: boolean;
  readonly isListening: boolean;
  start(): void;
  stop(): void;
  toggle(): void;
  reset?(): void;
  on(handler: (event: SpeechRecognizerEvent) => void): () => void;
}

export interface SpeechBubbleConfig {
  /** The speech recognition source responsible for voice -> text. */
  speech: SpeechSource;
  /** Element (or CSS selector) to mount the bubble into. Defaults to `document.body`. */
  container?: HTMLElement | string;
  /** Auto-hide the transcript after this many ms of silence (0 or undefined to disable auto-hide). Defaults to 0. */
  autoHideDelayMs?: number;
  /** Placeholder hint shown when idle with no transcript. Defaults to "Click mic to speak". */
  placeholder?: string;
}
