"""Tests for the Cartesia voice layer. No network, microphone or speaker required."""

from __future__ import annotations

import io
import json
import time
import wave

import numpy as np
import pytest

from opengaze.voice import cartesia
from opengaze.voice.audio import MicRecorder
from opengaze.voice.cartesia import CartesiaClient, CartesiaConfig, CartesiaError
from opengaze.voice.service import VoiceService, VoiceState


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeRecorder(MicRecorder):
    """MicRecorder with the sounddevice stream replaced by injected chunks."""

    def start(self) -> None:
        self._chunks = []
        self._last_voice_t = None
        self._start_t = time.monotonic()

    def feed_speech(self, seconds: float) -> None:
        n = int(self.sample_rate * seconds)
        self._on_chunk(np.full(n, 3000, dtype=np.int16), time.monotonic())


def _wait(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_from_env_reads_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_VOICE_ID", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\nCARTESIA_API_KEY='sk_car_test'\nCARTESIA_VOICE_ID=\n")
    cfg = CartesiaConfig.from_env([env])
    assert cfg is not None
    assert cfg.api_key == "sk_car_test"
    assert cfg.voice_id == cartesia.DEFAULT_VOICE_ID


def test_from_env_real_env_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("CARTESIA_API_KEY=from_file\n")
    monkeypatch.setenv("CARTESIA_API_KEY", "from_env")
    assert CartesiaConfig.from_env([env]).api_key == "from_env"


def test_from_env_missing_key_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    assert CartesiaConfig.from_env([tmp_path / "nope.env"]) is None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def test_wav_roundtrip():
    pcm = np.arange(100, dtype="<i2").tobytes()
    with wave.open(io.BytesIO(cartesia.pcm16_to_wav(pcm, 16000))) as w:
        assert w.getframerate() == 16000
        assert w.readframes(100) == pcm


def test_transcribe_request_shape(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["body"] = req.data
        return _FakeResp(json.dumps({"type": "transcript", "text": " hello there "}).encode())

    monkeypatch.setattr(cartesia.urllib.request, "urlopen", fake_urlopen)
    client = CartesiaClient(CartesiaConfig(api_key="k"))
    assert client.transcribe(b"\x00\x00" * 1600, 16000) == "hello there"
    assert seen["url"].endswith("/stt")
    assert seen["headers"]["Cartesia-version"] == cartesia.API_VERSION
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert b'name="model"\r\n\r\nink-whisper' in seen["body"]
    assert b"RIFF" in seen["body"]


def test_synthesize_request_shape(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["json"] = json.loads(req.data)
        return _FakeResp(b"\x01\x00" * 10)

    monkeypatch.setattr(cartesia.urllib.request, "urlopen", fake_urlopen)
    pcm = CartesiaClient(CartesiaConfig(api_key="k", voice_id="v")).synthesize("hi")
    assert len(pcm) == 20
    assert seen["json"]["voice"] == {"id": "v"}
    assert seen["json"]["output_format"]["container"] == "raw"


def test_network_error_becomes_cartesia_error(monkeypatch):
    def boom(req, timeout):
        raise cartesia.urllib.error.URLError("offline")

    monkeypatch.setattr(cartesia.urllib.request, "urlopen", boom)
    with pytest.raises(CartesiaError):
        CartesiaClient(CartesiaConfig(api_key="k")).synthesize("hi")


# ---------------------------------------------------------------------------
# Recorder end-of-speech detection
# ---------------------------------------------------------------------------

def test_recorder_stops_after_pause():
    r = MicRecorder(end_silence_s=1.0, no_speech_timeout_s=5.0)
    r._start_t = 0.0
    r._on_chunk(np.full(160, 3000, dtype=np.int16), 0.5)
    assert r.heard_speech
    r._on_chunk(np.zeros(160, dtype=np.int16), 1.0)
    assert not r.should_stop(1.2)
    assert r.should_stop(1.6)


def test_recorder_gives_up_when_silent():
    r = MicRecorder(no_speech_timeout_s=3.0)
    r._start_t = 0.0
    r._on_chunk(np.zeros(160, dtype=np.int16), 1.0)
    assert not r.should_stop(2.0)
    assert r.should_stop(3.1)


def test_recorder_hard_cap():
    r = MicRecorder(max_duration_s=10.0, end_silence_s=100)
    r._start_t = 0.0
    r._on_chunk(np.full(160, 3000, dtype=np.int16), 9.9)
    assert r.should_stop(10.0)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

def test_cloud_disabled_never_uses_cartesia():
    svc = VoiceService(cloud_enabled=False, cartesia=CartesiaConfig(api_key="k"))
    assert svc.engine == "offline"
    svc.start_dictation()
    assert svc.state is VoiceState.IDLE
    assert "Cartesia" in svc.status


def test_dictation_flow(monkeypatch):
    monkeypatch.setattr(CartesiaClient, "transcribe", lambda self, pcm, sr: "I need water")
    rec = _FakeRecorder()
    svc = VoiceService(cloud_enabled=True, cartesia=CartesiaConfig(api_key="k"), recorder=rec)
    svc.toggle_dictation()
    assert svc.state is VoiceState.LISTENING
    rec.feed_speech(1.0)
    svc.toggle_dictation()
    got: list[str] = []
    assert _wait(lambda: got.extend(svc.poll()) or got)
    assert got == ["I need water"]
    assert svc.state is VoiceState.IDLE


def test_dictation_ignores_silence(monkeypatch):
    called = []
    monkeypatch.setattr(CartesiaClient, "transcribe", lambda *a: called.append(1) or "x")
    svc = VoiceService(
        cloud_enabled=True, cartesia=CartesiaConfig(api_key="k"), recorder=_FakeRecorder(),
    )
    svc.toggle_dictation()
    svc.toggle_dictation()
    assert svc.state is VoiceState.IDLE
    assert not called


def test_transcription_failure_reports_status(monkeypatch):
    def fail(self, pcm, sr):
        raise CartesiaError("HTTP 401")

    monkeypatch.setattr(CartesiaClient, "transcribe", fail)
    rec = _FakeRecorder()
    svc = VoiceService(cloud_enabled=True, cartesia=CartesiaConfig(api_key="k"), recorder=rec)
    svc.toggle_dictation()
    rec.feed_speech(1.0)
    svc.toggle_dictation()
    assert _wait(lambda: svc.state is VoiceState.IDLE)
    assert svc.poll() == []
    assert "401" in svc.status


def test_cancel_drops_inflight_transcript(monkeypatch):
    def slow(self, pcm, sr):
        time.sleep(0.2)
        return "late"

    monkeypatch.setattr(CartesiaClient, "transcribe", slow)
    rec = _FakeRecorder()
    svc = VoiceService(cloud_enabled=True, cartesia=CartesiaConfig(api_key="k"), recorder=rec)
    svc.toggle_dictation()
    rec.feed_speech(1.0)
    svc.toggle_dictation()
    svc.cancel()
    time.sleep(0.3)
    assert svc.poll() == []


def test_retries_once_on_timeout(monkeypatch):
    calls = []

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("stalled")
        return _FakeResp(json.dumps({"type": "transcript", "text": "ok"}).encode())

    monkeypatch.setattr(cartesia.urllib.request, "urlopen", flaky)
    assert CartesiaClient(CartesiaConfig(api_key="k")).transcribe(b"\0\0" * 160, 16000) == "ok"
    assert len(calls) == 2


def test_client_error_not_retried_and_message_parsed(monkeypatch):
    calls = []

    def unauthorized(req, timeout):
        calls.append(1)
        body = json.dumps({"error_code": None, "title": "Unauthorized",
                           "message": "Invalid API key.", "request_id": "x"}).encode()
        raise cartesia.urllib.error.HTTPError(req.full_url, 401, "", {}, io.BytesIO(body))

    monkeypatch.setattr(cartesia.urllib.request, "urlopen", unauthorized)
    with pytest.raises(CartesiaError, match="HTTP 401: Invalid API key."):
        CartesiaClient(CartesiaConfig(api_key="k")).synthesize("hi")
    assert len(calls) == 1
