/**
 * Types and interfaces for the Speech Recognition system (Voice -> Text).
 */

export type SpeechRecognitionErrorCodeLike =
  "no-speech" | "aborted" | "audio-capture" | "network" | "not-allowed" | "service-not-allowed" | "bad-grammar" | "language-not-supported";

export interface SpeechStartEvent {
  type: "start";
}

export interface SpeechResultEvent {
  type: "result";
  /** Finalized text for the current segment, or "" if the segment is still in-flight. */
  finalText: string;
  /** In-flight interim text for the current segment, or "" once finalized. */
  interimText: string;
  /** Whether the current segment has reached its final version. */
  isFinal: boolean;
  /** Full text of the current segment (interim text while in-flight, final text once finalized). */
  segmentText: string;
}

export interface SpeechErrorEvent {
  type: "error";
  error: SpeechRecognitionErrorCodeLike;
  message: string;
}

export interface SpeechEndEvent {
  type: "end";
}

export type SpeechRecognizerEvent = SpeechStartEvent | SpeechResultEvent | SpeechErrorEvent | SpeechEndEvent;

export type SpeechRecognizerEventHandler = (event: SpeechRecognizerEvent) => void;

export interface SpeechRecognizerConfig {
  /** BCP-47 language tag, e.g. "en-US". Defaults to document language or "en-US". */
  lang?: string;
  /** Keep listening across pauses. Defaults to true. */
  continuous?: boolean;
  /** Emit `interim` results while user is speaking. Defaults to true. */
  interimResults?: boolean;
  /** Number of alternative transcripts requested. Defaults to 1. */
  maxAlternatives?: number;
  /** Automatically restart after the browser ends a continuous session. Defaults to true. */
  autoRestart?: boolean;
  /** Initial segment text (for pre-populating transcript). Defaults to "". */
  initialSegmentText?: string;
}
