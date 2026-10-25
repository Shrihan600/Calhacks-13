"""VoiceService: non-blocking dictation and speech for the gaze UIs.

The UI calls ``poll()`` from its timer tick; all network and audio work runs
on background threads or child processes so gaze tracking never stalls.

Playback goes through macOS ``afplay`` in its own process rather than a Python
audio callback: with MediaPipe and Qt painting holding the GIL, an in-process
callback underruns and the voice stutters on and off.

Cartesia is used only when the user has opted into cloud features
(``privacy.cloud_features_enabled``) *and* a ``CARTESIA_API_KEY`` is set.
Otherwise speaking falls back to macOS ``say`` and dictation is unavailable.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import os
import subprocess
import tempfile
import threading
import time
from collections import deque

from opengaze.voice.audio import MicRecorder
from opengaze.voice.cartesia import (
    TTS_SAMPLE_RATE,
    CartesiaClient,
    CartesiaConfig,
    CartesiaError,
    pcm16_to_wav,
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
        self._player: subprocess.Popen | None = None
        self._player_file: str | None = None
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
        gen = self._gen
        if self._client is None:
            self._start_player(["say", text], gen)
            self.status = "Speaking (macOS voice)"
            return
        self.status = "Speaking with Cartesia Sonic…"
        threading.Thread(target=self._speak_worker, args=(text, gen), daemon=True).start()

    def _speak_worker(self, text: str, gen: int) -> None:
        assert self._client is not None
        try:
            pcm = self._client.synthesize(text)
        except CartesiaError as exc:
            log.warning("synthesize failed, falling back to say: %s", exc)
            if self._start_player(["say", text], gen):
                self.status = f"Cartesia unavailable ({exc}) — used macOS voice"
            return
        if not pcm:
            return
        fd, path = tempfile.mkstemp(prefix="opengaze-tts-", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(pcm16_to_wav(pcm, TTS_SAMPLE_RATE))
        if not self._start_player(["afplay", path], gen, path):
            os.unlink(path)

    def _start_player(self, cmd: list[str], gen: int, path: str | None = None) -> bool:
        """Start playback unless cancelled meanwhile. Returns True if started."""
        with self._lock:
            if gen != self._gen:
                return False
            try:
                self._player = subprocess.Popen(cmd)
            except Exception as exc:
                log.warning("playback failed: %s", exc)
                self._state = VoiceState.IDLE
                self.status = f"Audio output failed: {exc}"
                return False
            self._player_file = path
            return True

    def _release_player(self) -> None:
        if self._player is not None and self._player.poll() is None:
            self._player.terminate()
        self._player = None
        if self._player_file:
            with contextlib.suppress(OSError):
                os.unlink(self._player_file)
            self._player_file = None

    def stop_speaking(self) -> None:
        with self._lock:
            self._release_player()
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
        with self._lock:
            if (
                self._state is VoiceState.SPEAKING
                and self._player is not None
                and self._player.poll() is not None
            ):
                self._release_player()
                self._state = VoiceState.IDLE
                if self.status.startswith("Speaking"):
                    self.status = f"Voice: {self.engine} — ready"
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
