"""VoiceService: non-blocking dictation and speech for the gaze UIs.

The UI calls ``poll()`` from its timer tick; all network and audio work runs
on background threads so gaze tracking never stalls.

Cartesia is used only when the user has opted into cloud features
(``privacy.cloud_features_enabled``) *and* a ``CARTESIA_API_KEY`` is set.
Otherwise speaking falls back to macOS ``say`` and dictation is unavailable.
"""

from __future__ import annotations

import enum
import logging
import subprocess
import threading
import time
from collections import deque

import numpy as np

from opengaze.voice.audio import MicRecorder
from opengaze.voice.cartesia import (
    TTS_SAMPLE_RATE,
    CartesiaClient,
    CartesiaConfig,
    CartesiaError,
)

log = logging.getLogger("opengaze.voice")

# Recordings shorter than this are almost certainly an accidental trigger.
_MIN_RECORDING_S = 0.3


class VoiceState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    SPEAKING = "speaking"


class VoiceService:
    def __init__(
        self,
        cloud_enabled: bool,
        cartesia: CartesiaConfig | None = None,
        recorder: MicRecorder | None = None,
    ) -> None:
        self._client = CartesiaClient(cartesia) if (cloud_enabled and cartesia) else None
        self._recorder = recorder or MicRecorder()
        self._lock = threading.Lock()
        self._transcripts: deque[str] = deque()
        self._state = VoiceState.IDLE
        self._speak_until = 0.0
        self._say_proc: subprocess.Popen | None = None
        self._gen = 0  # bumps on cancel so stale worker results are dropped
        if self._client:
            self.status = "Voice: Cartesia (Ink-Whisper + Sonic)"
        elif cartesia and not cloud_enabled:
            self.status = "Voice: offline — enable privacy.cloud_features_enabled for Cartesia"
        else:
            self.status = "Voice: offline — set CARTESIA_API_KEY for dictation"

    @property
    def engine(self) -> str:
        return "Cartesia" if self._client else "offline"

    @property
    def can_dictate(self) -> bool:
        return self._client is not None

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def mic_level(self) -> float:
        return self._recorder.level if self._state is VoiceState.LISTENING else 0.0

    # ------------------------------------------------------------------
    # Dictation
    # ------------------------------------------------------------------

    def toggle_dictation(self) -> None:
        if self._state is VoiceState.LISTENING:
            self._finish_listening()
        elif self._state is VoiceState.TRANSCRIBING:
            return
        else:
            self.start_dictation()

    def start_dictation(self) -> None:
        if not self.can_dictate:
            self.status = "Dictation needs Cartesia — see README › Voice"
            return
        self.stop_speaking()  # never record our own TTS output
        try:
            self._recorder.start()
        except Exception as exc:
            log.warning("mic start failed: %s", exc)
            self.status = f"Microphone unavailable: {exc}"
            return
        self._state = VoiceState.LISTENING
        self.status = "Listening… stops automatically when you pause"

    def _finish_listening(self) -> None:
        pcm = self._recorder.stop()
        heard = self._recorder.heard_speech
        seconds = len(pcm) / 2 / self._recorder.sample_rate
        if not heard or seconds < _MIN_RECORDING_S:
            self._state = VoiceState.IDLE
            self.status = "Didn't hear anything — try again"
            return
        self._state = VoiceState.TRANSCRIBING
        self.status = "Transcribing with Cartesia Ink-Whisper…"
        gen = self._gen
        threading.Thread(
            target=self._transcribe_worker, args=(pcm, gen), daemon=True,
        ).start()

    def _transcribe_worker(self, pcm: bytes, gen: int) -> None:
        assert self._client is not None
        t0 = time.monotonic()
        try:
            text = self._client.transcribe(pcm, self._recorder.sample_rate)
            status = (
                f"Heard in {time.monotonic() - t0:.1f}s" if text else "No words recognised"
            )
        except CartesiaError as exc:
            log.warning("transcribe failed: %s", exc)
            text, status = "", f"Transcription failed ({exc}) — type instead"
        with self._lock:
            if gen != self._gen:
                return
            if text:
                self._transcripts.append(text)
            self._state = VoiceState.IDLE
            self.status = status

    # ------------------------------------------------------------------
    # Speech
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self._state is VoiceState.LISTENING:
            self.cancel()
        self.stop_speaking()
        self._state = VoiceState.SPEAKING
        if self._client is None:
            self._speak_local(text)
            return
        self.status = "Speaking with Cartesia Sonic…"
        gen = self._gen
        threading.Thread(target=self._speak_worker, args=(text, gen), daemon=True).start()

    def _speak_worker(self, text: str, gen: int) -> None:
        assert self._client is not None
        try:
            pcm = self._client.synthesize(text)
        except CartesiaError as exc:
            log.warning("synthesize failed, falling back to say: %s", exc)
            if gen == self._gen:
                self._speak_local(text)
                self.status = f"Cartesia unavailable ({exc}) — used macOS voice"
            return
        if gen != self._gen or not pcm:
            return
        try:
            import sounddevice as sd

            samples = np.frombuffer(pcm, dtype="<i2")
            sd.play(samples, TTS_SAMPLE_RATE)
            self._speak_until = time.monotonic() + len(samples) / TTS_SAMPLE_RATE
        except Exception as exc:
            log.warning("playback failed: %s", exc)
            self._state = VoiceState.IDLE
            self.status = f"Audio output failed: {exc}"

    def _speak_local(self, text: str) -> None:
        try:
            self._say_proc = subprocess.Popen(["say", text])
            self.status = "Speaking (macOS voice)"
        except Exception as exc:
            self._state = VoiceState.IDLE
            self.status = f"Speech unavailable: {exc}"

    def stop_speaking(self) -> None:
        if self._say_proc is not None and self._say_proc.poll() is None:
            self._say_proc.terminate()
        self._say_proc = None
        if self._speak_until:
            try:
                import sounddevice as sd

                sd.stop()
            except Exception:
                pass
            self._speak_until = 0.0
        if self._state is VoiceState.SPEAKING:
            self._gen += 1
            self._state = VoiceState.IDLE

    # ------------------------------------------------------------------
    # UI integration
    # ------------------------------------------------------------------

    def poll(self) -> list[str]:
        """Advance timers; return any finished transcripts. Call from the UI thread."""
        if self._state is VoiceState.LISTENING and self._recorder.should_stop():
            self._finish_listening()
        elif self._state is VoiceState.SPEAKING:
            done_say = self._say_proc is not None and self._say_proc.poll() is not None
            done_pcm = self._speak_until and time.monotonic() >= self._speak_until
            if done_say or done_pcm:
                self._say_proc = None
                self._speak_until = 0.0
                self._state = VoiceState.IDLE
                if self.status.startswith("Speaking"):
                    self.status = f"Voice: {self.engine} — ready"
        with self._lock:
            out = list(self._transcripts)
            self._transcripts.clear()
        return out

    def cancel(self) -> None:
        with self._lock:
            self._gen += 1
        if self._state is VoiceState.LISTENING:
            self._recorder.stop()
        self.stop_speaking()
        self._state = VoiceState.IDLE

    def shutdown(self) -> None:
        self.cancel()
