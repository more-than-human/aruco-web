import type { SpeechRecognitionErrorCode } from "../speech/index.ts";
import "./speech-bubble.css";
import type { SpeechBubbleConfig, SpeechSource } from "./types.ts";

/**
 * Speech Bubble component responsible for:
 * 1. Displaying voice-to-text transcription (committed text, in-flight interim speech, status messages).
 * 2. Providing user controls to explicitly toggle "start/stop" of speech recognition.
 */
export class SpeechBubble {
  public readonly root: HTMLElement;
  public readonly toggleBtn: HTMLButtonElement;
  public readonly clearBtn: HTMLButtonElement;
  public readonly contentEl: HTMLElement;
  public readonly statusEl: HTMLElement;
  public readonly bodyEl: HTMLElement;

  public readonly speech: SpeechSource;

  private readonly autoHideDelayMs: number;
  private readonly placeholder: string;

  private finalText = "";
  private interimText = "";
  private hideTimeout: number | null = null;
  private unsubscribeSpeech: (() => void) | null = null;

  constructor(config: SpeechBubbleConfig) {
    this.speech = config.speech;
    this.autoHideDelayMs = config.autoHideDelayMs ?? 0;
    this.placeholder = config.placeholder ?? "Click mic to speak";

    const target = typeof config.container === "string" ? document.querySelector<HTMLElement>(config.container) : config.container;
    const container = target ?? document.body;

    // Root speech bubble container
    this.root = document.createElement("div");
    this.root.className = "speech-bubble speech-bubble--empty";
    this.root.setAttribute("role", "region");
    this.root.setAttribute("aria-label", "Speech to text");

    // Start / Stop toggle button
    this.toggleBtn = document.createElement("button");
    this.toggleBtn.className = "speech-bubble__toggle";
    this.toggleBtn.type = "button";
    this.toggleBtn.innerHTML = `
      <svg class="speech-bubble__mic-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
        <line x1="12" x2="12" y1="19" y2="22"/>
        <line x1="8" x2="16" y1="22" y2="22"/>
      </svg>
    `;

    // Content container (status + transcript body)
    this.contentEl = document.createElement("div");
    this.contentEl.className = "speech-bubble__content";

    this.statusEl = document.createElement("div");
    this.statusEl.className = "speech-bubble__status";

    this.bodyEl = document.createElement("div");
    this.bodyEl.className = "speech-bubble__body";
    this.bodyEl.setAttribute("aria-live", "polite");
    this.bodyEl.hidden = true;

    this.contentEl.append(this.statusEl, this.bodyEl);

    // Clear transcript button
    this.clearBtn = document.createElement("button");
    this.clearBtn.className = "speech-bubble__clear";
    this.clearBtn.type = "button";
    this.clearBtn.setAttribute("aria-label", "Clear transcript");
    this.clearBtn.title = "Clear transcript";
    this.clearBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <line x1="18" x2="6" y1="6" y2="18"/>
        <line x1="6" x2="18" y1="6" y2="18"/>
      </svg>
    `;
    this.clearBtn.hidden = true;

    this.root.append(this.toggleBtn, this.contentEl, this.clearBtn);
    container.appendChild(this.root);

    this.setupListeners();
    this.bindSpeechEvents();
    this.updateToggleState();
    this.render();
  }

  /** Update displayed transcript with finalized and interim parts. */
  public setTranscript(finalText: string, interimText: string): void {
    this.finalText = finalText;
    this.interimText = interimText;
    this.render();

    if (this.hasTranscript()) {
      this.scheduleAutoHide();
    }
  }

  /** Display a status or error message in the bubble. */
  public setStatus(status: string): void {
    this.statusEl.textContent = status;
    this.statusEl.hidden = status.trim().length === 0;
  }

  /** Whether the bubble is currently displaying transcript text. */
  public hasTranscript(): boolean {
    return this.finalText.trim().length > 0 || this.interimText.trim().length > 0;
  }

  /** Clear the transcript and reset bubble state. */
  public reset(): void {
    if (this.hideTimeout !== null) {
      clearTimeout(this.hideTimeout);
      this.hideTimeout = null;
    }
    this.finalText = "";
    this.interimText = "";
    this.speech.reset?.();
    this.render();
  }

  /** Destroy elements and detach listeners. */
  public dispose(): void {
    if (this.hideTimeout !== null) {
      clearTimeout(this.hideTimeout);
      this.hideTimeout = null;
    }
    if (this.unsubscribeSpeech) {
      this.unsubscribeSpeech();
      this.unsubscribeSpeech = null;
    }
    this.root.remove();
  }

  private setupListeners(): void {
    this.toggleBtn.addEventListener("click", () => {
      if (!this.speech.isSupported) {
        return;
      }
      this.speech.toggle();
      this.updateToggleState();
    });

    this.clearBtn.addEventListener("click", () => {
      this.reset();
    });

    // Clicking status placeholder when idle starts speech recognition
    this.statusEl.addEventListener("click", () => {
      if (this.speech.isSupported && !this.speech.isListening && !this.hasTranscript()) {
        this.speech.start();
        this.updateToggleState();
      }
    });
  }

  private bindSpeechEvents(): void {
    this.unsubscribeSpeech = this.speech.on((event) => {
      switch (event.type) {
        case "start":
          this.updateToggleState();
          if (!this.hasTranscript()) {
            this.setStatus("Listening…");
          }
          break;
        case "result":
          this.setTranscript(event.finalText, event.interimText);
          break;
        case "error":
          this.updateToggleState();
          this.setStatus(this.describeError(event.error));
          break;
        case "end":
          this.updateToggleState();
          if (!this.hasTranscript()) {
            this.render();
          }
          break;
        default:
          break;
      }
    });
  }

  private updateToggleState(): void {
    if (!this.speech.isSupported) {
      this.toggleBtn.disabled = true;
      this.toggleBtn.classList.add("speech-bubble__toggle--unsupported");
      this.toggleBtn.classList.remove("speech-bubble__toggle--listening");
      this.toggleBtn.setAttribute("aria-label", "Speech recognition unsupported");
      this.toggleBtn.title = "The Web Speech API is not supported in this browser. Try Chrome or Edge.";
      return;
    }

    if (this.speech.isListening) {
      this.toggleBtn.disabled = false;
      this.toggleBtn.classList.remove("speech-bubble__toggle--unsupported");
      this.toggleBtn.classList.add("speech-bubble__toggle--listening");
      this.toggleBtn.setAttribute("aria-label", "Stop speech recognition");
      this.toggleBtn.title = "Stop speech recognition";
    } else {
      this.toggleBtn.disabled = false;
      this.toggleBtn.classList.remove("speech-bubble__toggle--unsupported");
      this.toggleBtn.classList.remove("speech-bubble__toggle--listening");
      this.toggleBtn.setAttribute("aria-label", "Start speech recognition");
      this.toggleBtn.title = "Start speech recognition";
    }
  }

  private render(): void {
    const hasText = this.hasTranscript();

    if (hasText) {
      this.bodyEl.textContent = "";

      if (this.finalText.length > 0) {
        this.bodyEl.appendChild(document.createTextNode(this.finalText));
      }

      if (this.interimText.length > 0) {
        if (this.finalText.length > 0) {
          this.bodyEl.appendChild(document.createTextNode(" "));
        }
        const interimSpan = document.createElement("span");
        interimSpan.className = "speech-bubble__interim";
        interimSpan.textContent = this.interimText;
        this.bodyEl.appendChild(interimSpan);
      }

      this.bodyEl.hidden = false;
      this.statusEl.hidden = true;
      this.clearBtn.hidden = false;
      this.root.classList.remove("speech-bubble--empty");
    } else {
      this.bodyEl.textContent = "";
      this.bodyEl.hidden = true;
      this.clearBtn.hidden = true;
      this.root.classList.add("speech-bubble--empty");

      if (!this.speech.isSupported) {
        this.setStatus("Speech recognition unsupported");
      } else if (this.speech.isListening) {
        this.setStatus("Listening…");
      } else {
        this.setStatus(this.placeholder);
      }
      this.statusEl.hidden = false;
    }
  }

  private scheduleAutoHide(): void {
    if (this.autoHideDelayMs <= 0) {
      return;
    }
    if (this.hideTimeout !== null) {
      clearTimeout(this.hideTimeout);
    }
    this.hideTimeout = window.setTimeout(() => {
      this.hideTimeout = null;
      if (!this.speech.isListening) {
        this.reset();
      }
    }, this.autoHideDelayMs);
  }

  private describeError(error: SpeechRecognitionErrorCode): string {
    switch (error) {
      case "not-allowed":
      case "service-not-allowed":
        return "Microphone access was blocked.";
      case "no-speech":
        return "Listening… (no speech detected)";
      case "audio-capture":
        return "No microphone was found.";
      case "network":
        return "Speech recognition network error.";
      case "aborted":
        return "Speech recognition stopped.";
      default:
        return "Speech recognition error.";
    }
  }
}
