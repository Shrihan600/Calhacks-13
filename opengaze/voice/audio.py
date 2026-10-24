"""Microphone capture with automatic end-of-speech detection.

The user may not be able to hit a "stop" key quickly, so recording ends by
itself after a pause in speech, after a hard time cap, or if nothing is heard.
"""

from __future__ import annotations

import threading
import time

import numpy as np

MIC_SAMPLE_RATE = 16000


class MicRecorder:
    def __init__(
        self,
        sample_rate: int = MIC_SAMPLE_RATE,
        speech_rms: float = 500.0,
        end_silence_s: float = 1.5,
        no_speech_timeout_s: float = 6.0,
        max_duration_s: float = 20.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.speech_rms = speech_rms
        self.end_silence_s = end_silence_s
        self.no_speech_timeout_s = no_speech_timeout_s
        self.max_duration_s = max_duration_s
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self._start_t = 0.0
        self._last_voice_t: float | None = None
        self.level: float = 0.0  # 0..1 for the UI meter

    @property
    def heard_speech(self) -> bool:
        return self._last_voice_t is not None

    def start(self) -> None:
        import sounddevice as sd

        self._chunks = []
        self._last_voice_t = None
        self._start_t = time.monotonic()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16",
            callback=lambda data, *_: self._on_chunk(data[:, 0].copy(), time.monotonic()),
        )
        self._stream.start()

    def _on_chunk(self, samples: np.ndarray, now: float) -> None:
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0
        with self._lock:
            self._chunks.append(samples)
        self.level = min(rms / (self.speech_rms * 4), 1.0)
        if rms >= self.speech_rms:
            self._last_voice_t = now

    def should_stop(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if now - self._start_t >= self.max_duration_s:
            return True
        if self._last_voice_t is None:
            return now - self._start_t >= self.no_speech_timeout_s
        return now - self._last_voice_t >= self.end_silence_s

    def stop(self) -> bytes:
        """Stop capture and return mono s16le PCM."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self.level = 0.0
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return b""
        return np.concatenate(chunks).astype("<i2").tobytes()
