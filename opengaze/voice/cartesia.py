"""Minimal Cartesia REST client: batch STT (Ink-Whisper) and TTS (Sonic).

Uses only the standard library so the voice features add no dependencies.
Only audio the user explicitly records and text the user explicitly asks to
speak is sent; camera data never goes through this module.
"""

from __future__ import annotations

import io
import json
import os
import ssl
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

API_BASE = "https://api.cartesia.ai"
API_VERSION = "2026-08-14"
STT_MODEL = "ink-whisper"
TTS_MODEL = "sonic-3.6"
# Cartesia's documented example voice; override with CARTESIA_VOICE_ID.
DEFAULT_VOICE_ID = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"
TTS_SAMPLE_RATE = 24000

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ssl_context() -> ssl.SSLContext:
    """System trust store, falling back to certifi when it is empty.

    python.org macOS builds ship with no CA certificates until the user runs
    "Install Certificates.command", which makes every HTTPS call fail with
    CERTIFICATE_VERIFY_FAILED.
    """
    ctx = ssl.create_default_context()
    if ctx.cert_store_stats().get("x509_ca", 0) == 0:
        try:
            import certifi
        except ImportError:
            return ctx
        ctx.load_verify_locations(certifi.where())
    return ctx


class CartesiaError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file. Missing file → empty dict."""
    out: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip("'\"")
    return out


@dataclass(frozen=True)
class CartesiaConfig:
    api_key: str
    voice_id: str = DEFAULT_VOICE_ID
    language: str = "en"

    @classmethod
    def from_env(cls, dotenv_paths: list[Path] | None = None) -> CartesiaConfig | None:
        """Real environment wins; otherwise fall back to .env in cwd or repo root."""
        if dotenv_paths is None:
            dotenv_paths = [Path.cwd() / ".env", _REPO_ROOT / ".env"]
        file_vals: dict[str, str] = {}
        for p in reversed(dotenv_paths):
            file_vals.update(read_dotenv(p))

        def get(name: str) -> str:
            return os.environ.get(name) or file_vals.get(name, "")

        key = get("CARTESIA_API_KEY")
        if not key:
            return None
        return cls(
            api_key=key,
            voice_id=get("CARTESIA_VOICE_ID") or DEFAULT_VOICE_ID,
            language=get("CARTESIA_LANGUAGE") or "en",
        )


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _multipart(fields: dict[str, str], file_name: str, file_bytes: bytes,
               file_type: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{file_name}"\r\nContent-Type: {file_type}\r\n\r\n'.encode()
        + file_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _error_message(code: int, raw: bytes) -> str:
    """Cartesia-Version 2026-03-01+ errors are JSON with ``title``/``message``."""
    try:
        err = json.loads(raw)
        return f"HTTP {code}: {err.get('message') or err.get('title')}"
    except (ValueError, AttributeError):
        return f"HTTP {code}: {raw[:200].decode(errors='replace').strip()}"


class CartesiaClient:
    def __init__(self, config: CartesiaConfig, timeout_s: float = 10.0, retries: int = 1) -> None:
        self.config = config
        self.timeout_s = timeout_s
        self.retries = retries

    def _post(self, path: str, body: bytes, content_type: str) -> bytes:
        """POST with one retry on network errors and 5xx (seen: occasional STT stalls)."""
        for attempt in range(self.retries + 1):
            try:
                return self._post_once(path, body, content_type)
            except CartesiaError as exc:
                if not exc.retryable or attempt == self.retries:
                    raise
        raise AssertionError("unreachable")

    def _post_once(self, path: str, body: bytes, content_type: str) -> bytes:
        req = urllib.request.Request(
            API_BASE + path,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Cartesia-Version": API_VERSION,
                "Content-Type": content_type,
            },
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout_s, context=_ssl_context(),
            ) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            raise CartesiaError(
                _error_message(exc.code, exc.read()), retryable=exc.code >= 500,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CartesiaError(f"network error: {exc}", retryable=True) from exc

    def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        """Speech → text with Ink-Whisper. ``pcm16`` is mono signed 16-bit LE."""
        body, ctype = _multipart(
            {"model": STT_MODEL, "language": self.config.language},
            "speech.wav", pcm16_to_wav(pcm16, sample_rate), "audio/wav",
        )
        raw = self._post("/stt", body, ctype)
        try:
            return str(json.loads(raw)["text"]).strip()
        except (ValueError, KeyError) as exc:
            raise CartesiaError("unexpected STT response") from exc

    def synthesize(self, text: str) -> bytes:
        """Text → mono s16le PCM at ``TTS_SAMPLE_RATE`` with Sonic."""
        payload = {
            "model_id": TTS_MODEL,
            "transcript": text,
            "voice": {"id": self.config.voice_id},
            "language": self.config.language,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": TTS_SAMPLE_RATE,
            },
        }
        return self._post("/tts/bytes", json.dumps(payload).encode(), "application/json")
