import type {
  SpeechEndEvent,
  SpeechErrorEvent,
  SpeechRecognitionErrorCodeLike,
  SpeechRecognizerConfig,
  SpeechRecognizerEvent,
  SpeechRecognizerEventHandler,
  SpeechResultEvent,
  SpeechStartEvent,
} from "./types.ts";

/**
 * Thin, typed wrapper around the Web Speech API's SpeechRecognition interface.
 *
 * The API is vendor-prefixed in Chromium-based browsers (`webkitSpeechRecognition`)
 * and not implemented in Firefox, so we resolve the constructor defensively and expose
 * `isSupported` for graceful degradation.
 *
 * @see https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API
 */

// Minimal structural typings for the Web Speech API surface.
interface SpeechRecognitionAlternativeLike {
  readonly transcript: string;
  readonly confidence: number;
}

interface SpeechRecognitionResultLike {
  readonly isFinal: boolean;
  readonly length: number;
  [index: number]: SpeechRecognitionAlternativeLike;
}

interface SpeechRecognitionResultListLike {
  readonly length: number;
  [index: number]: SpeechRecognitionResultLike;
}

interface SpeechRecognitionEventLike extends Event {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultListLike;
}

interface SpeechRecognitionErrorEventLike extends Event {
  readonly error: SpeechRecognitionErrorCodeLike;
  readonly message: string;
}

interface SpeechRecognitionLike extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
}

type SpeechRecognitionConstructorLike = new () => SpeechRecognitionLike;

function resolveConstructor(): SpeechRecognitionConstructorLike | null {
  if (typeof window === "undefined") {
    return null;
  }
  const candidate = window as unknown as {
    SpeechRecognition?: SpeechRecognitionConstructorLike;
    webkitSpeechRecognition?: SpeechRecognitionConstructorLike;
  };
  return candidate.SpeechRecognition ?? candidate.webkitSpeechRecognition ?? null;
}

/**
 * Speech recognition engine responsible purely for voice -> text transcription.
 */
export class SpeechRecognizer {
  private readonly Constructor = resolveConstructor();
  private recognition: SpeechRecognitionLike | null = null;

  private handlers: SpeechRecognizerEventHandler[] = [];
  private finalText = "";
  private interimText = "";
  private isFinal = false;
  private listening = false;
  private disposed = false;
  /** Set true while an intentional stop/abort is in flight to suppress auto-restart. */
  private stopping = false;

  public readonly lang: string;
  public readonly continuous: boolean;
  public readonly interimResults: boolean;
  public readonly maxAlternatives: number;
  public readonly autoRestart: boolean;

  constructor(config: SpeechRecognizerConfig = {}) {
    this.lang = config.lang ?? (typeof document !== "undefined" ? document.documentElement.lang : "en-US") ?? "en-US";
    this.continuous = config.continuous ?? true;
    this.interimResults = config.interimResults ?? true;
    this.maxAlternatives = config.maxAlternatives ?? 1;
    this.autoRestart = config.autoRestart ?? true;
    this.finalText = config.initialSegmentText ?? "";
    this.isFinal = this.finalText.length > 0;
  }

  /** Whether the current browser exposes the Web Speech API. */
  public get isSupported(): boolean {
    return this.Constructor !== null;
  }

  public get isListening(): boolean {
    return this.listening;
  }

  /** Finalized text of the current segment, or "" if still in-flight. */
  public get committedText(): string {
    return this.finalText;
  }

  /** Current in-flight interim text of the active segment, or "" if finalized. */
  public get pendingText(): string {
    return this.interimText;
  }

  /** Current segment text (interim text while in-flight, final text once finalized). */
  public get currentSegment(): string {
    return this.isFinal ? this.finalText : this.interimText;
  }

  /** Whether the current segment has reached its final version. */
  public get isSegmentFinal(): boolean {
    return this.isFinal;
  }

  /** Subscribe to recognizer events. Returns an unsubscribe function. */
  public on(handler: SpeechRecognizerEventHandler): () => void {
    this.handlers.push(handler);
    return () => {
      const index = this.handlers.indexOf(handler);
      if (index >= 0) {
        this.handlers.splice(index, 1);
      }
    };
  }

  public start(): void {
    if (!this.Constructor || this.disposed || this.listening) {
      return;
    }

    if (!this.recognition) {
      this.recognition = this.createRecognition();
    }

    this.stopping = false;
    try {
      this.recognition.start();
      this.handleStart();
    } catch {
      // Calling start() on an already-started instance throws InvalidStateError; safe to ignore.
    }
  }

  public stop(): void {
    if (!this.recognition || !this.listening) {
      return;
    }
    this.stopping = true;
    this.handleEnd();
    try {
      this.recognition.stop();
    } catch {
      /* ignore */
    }
  }

  public toggle(): void {
    if (this.listening) {
      this.stop();
    } else {
      this.start();
    }
  }

  /** Clear the current segment text without tearing down the recognizer. */
  public reset(): void {
    this.finalText = "";
    this.interimText = "";
    this.isFinal = false;
    this.emit({ type: "result", finalText: "", interimText: "", isFinal: false, segmentText: "" });
  }

  public dispose(): void {
    this.disposed = true;
    this.stopping = true;
    if (this.recognition) {
      this.recognition.onresult = null;
      this.recognition.onerror = null;
      this.recognition.onend = null;
      this.recognition.onstart = null;
      try {
        this.recognition.abort();
      } catch {
        /* ignore */
      }
      this.recognition = null;
    }
    this.handlers = [];
  }

  private createRecognition(): SpeechRecognitionLike {
    const recognition = new this.Constructor!();
    recognition.lang = this.lang;
    recognition.continuous = this.continuous;
    recognition.interimResults = this.interimResults;
    recognition.maxAlternatives = this.maxAlternatives;

    recognition.onstart = () => this.handleStart();
    recognition.onresult = (event) => this.handleResult(event);
    recognition.onerror = (event) => this.handleError(event);
    recognition.onend = () => this.handleEnd();

    return recognition;
  }

  private handleStart(): void {
    if (!this.listening) {
      this.listening = true;
      this.emit({ type: "start" });
    }
  }

  private handleResult(event: SpeechRecognitionEventLike): void {
    if (!event.results || event.results.length === 0) {
      return;
    }

    // Find the latest finalized result index.
    let lastFinalIndex = -1;
    for (let i = event.results.length - 1; i >= 0; i -= 1) {
      if (event.results[i].isFinal) {
        lastFinalIndex = i;
        break;
      }
    }

    let isFinal = false;
    let finalText = "";
    let interimText = "";

    if (lastFinalIndex === event.results.length - 1) {
      // The latest result in the list has reached its final version.
      isFinal = true;
      finalText = event.results[lastFinalIndex][0]?.transcript?.trim() ?? "";
      interimText = "";
    } else {
      // In-flight interim results after the last final result.
      isFinal = false;
      finalText = "";
      const interimParts: string[] = [];
      const startIndex = lastFinalIndex >= 0 ? lastFinalIndex + 1 : 0;
      for (let i = startIndex; i < event.results.length; i += 1) {
        const text = event.results[i][0]?.transcript ?? "";
        if (text) {
          interimParts.push(text);
        }
      }
      interimText = interimParts.join(" ").trim();
    }

    this.finalText = finalText;
    this.interimText = interimText;
    this.isFinal = isFinal;
    const segmentText = isFinal ? finalText : interimText;

    this.emit({
      type: "result",
      finalText,
      interimText,
      isFinal,
      segmentText,
    });
  }

  private handleError(event: SpeechRecognitionErrorEventLike): void {
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      this.stopping = true;
    }
    this.listening = false;
    this.emit({ type: "error", error: event.error, message: event.message });
  }

  private handleEnd(): void {
    const wasListening = this.listening;
    this.listening = false;
    if (wasListening) {
      this.emit({ type: "end" });
    }
    // In continuous mode Chromium ends the session on silence; restart unless we stopped on purpose.
    if (this.autoRestart && !this.stopping && !this.disposed) {
      window.setTimeout(() => {
        if (this.autoRestart && !this.stopping && !this.disposed) {
          this.start();
        }
      }, 50);
    }
  }

  private emit(event: SpeechRecognizerEvent): void {
    for (const handler of this.handlers.slice()) {
      handler(event);
    }
  }
}

export type {
  SpeechEndEvent,
  SpeechErrorEvent,
  SpeechRecognitionErrorCodeLike,
  SpeechRecognizerConfig,
  SpeechRecognizerEvent,
  SpeechRecognizerEventHandler,
  SpeechResultEvent,
  SpeechStartEvent,
};
