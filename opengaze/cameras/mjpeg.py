"""Blocking reader for MJPEG-over-HTTP streams (``multipart/x-mixed-replace``).

This is the transport the ESP32 camera web-server examples use, and the usual one for Pi
camera streamers. It is plain stdlib ``http.client``, not OpenCV/FFmpeg, for three reasons:

* the part headers carry device timestamps and sequence numbers, which FFmpeg drops;
* nothing is buffered beyond the socket, so a slow reader gets the next frame, not a queue;
* timeouts and errors are explicit, so a dropped Wi-Fi link becomes a clean reconnect.

ESP-IDF's ``httpd_resp_send_chunk`` uses HTTP chunked transfer encoding; ``HTTPResponse``
removes it, so the multipart parser below only sees the multipart body.
"""

from __future__ import annotations

import errno
import http.client
import socket
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from opengaze.cameras.types import CameraConnectionError, CameraDisconnectedError

DEFAULT_MAX_PART_BYTES = 2_000_000  # far above a 640×480 JPEG; guards against garbage
_MAX_HEADER_LINES = 32


@dataclass(frozen=True, slots=True)
class MjpegPart:
    """One multipart body (normally one JPEG) and its part headers (lower-cased names)."""

    data: bytes
    headers: dict[str, str]
    host_receive_ns: int

    @property
    def device_time_us(self) -> int | None:
        """Device capture time in microseconds, from ``X-Timestamp-Us`` or ``X-Timestamp``.

        ``X-Timestamp`` is ``seconds.microseconds`` as sent by Espressif's CameraWebServer.
        """
        raw = self.headers.get("x-timestamp-us")
        if raw is not None:
            try:
                return int(raw.strip())
            except ValueError:
                return None
        raw = self.headers.get("x-timestamp")
        if raw is None:
            return None
        sec, _, frac = raw.strip().partition(".")
        try:
            return int(sec) * 1_000_000 + (int(frac.ljust(6, "0")[:6]) if frac else 0)
        except ValueError:
            return None

    @property
    def sequence(self) -> int | None:
        raw = self.headers.get("x-sequence")
        try:
            return int(raw) if raw is not None else None
        except ValueError:
            return None


def boundary_from_content_type(content_type: str) -> str | None:
    """``multipart/x-mixed-replace;boundary=abc`` → ``abc`` (quotes and a leading ``--``
    removed, since some servers include them)."""
    main, *params = content_type.split(";")
    if not main.strip().lower().startswith("multipart/"):
        return None
    for param in params:
        key, _, value = param.strip().partition("=")
        if key.strip().lower() == "boundary":
            value = value.strip().strip('"')
            return value.removeprefix("--") or None
    return None


class MjpegStream:
    """One open stream. ``read_part()`` blocks until the next part arrives."""

    def __init__(
        self,
        url: str,
        *,
        connect_timeout_s: float,
        read_timeout_s: float,
        max_part_bytes: int = DEFAULT_MAX_PART_BYTES,
    ) -> None:
        parts = urlsplit(url)
        if parts.scheme != "http" or not parts.hostname:
            raise CameraConnectionError(f"not an http:// URL: {url!r}")
        self.url = url
        self._host = parts.hostname
        self._port = parts.port or 80
        self._path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        self._connect_timeout = connect_timeout_s
        self._read_timeout = read_timeout_s
        self._max_part = max_part_bytes
        self._conn: http.client.HTTPConnection | None = None
        self._resp: http.client.HTTPResponse | None = None
        self._boundary = b""
        self._at_headers = False  # True when the boundary line was already consumed
        self.response_headers: dict[str, str] = {}

    def open(self) -> None:
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self._connect_timeout)
        try:
            conn.request("GET", self._path, headers={"Accept": "multipart/x-mixed-replace"})
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()
            raise CameraConnectionError(f"{self.url}: {describe_network_error(exc)}") from exc
        if resp.status != 200:
            conn.close()
            raise CameraConnectionError(f"{self.url}: HTTP {resp.status} {resp.reason}")
        content_type = resp.getheader("Content-Type", "")
        boundary = boundary_from_content_type(content_type)
        if boundary is None:
            conn.close()
            raise CameraConnectionError(
                f"{self.url}: expected a multipart MJPEG stream, got Content-Type "
                f"{content_type!r} (is this the stream endpoint, usually :81/stream?)"
            )
        if conn.sock is not None:
            # Fail just before the caller's own read timeout, so a dead link surfaces as a
            # socket error on this thread rather than an abandoned worker.
            conn.sock.settimeout(self._read_timeout)
        self._conn, self._resp = conn, resp
        self._boundary = b"--" + boundary.encode("latin-1")
        self._at_headers = False
        self.response_headers = {k.lower(): v for k, v in resp.getheaders()}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = self._resp = None

    # -- parsing -----------------------------------------------------------------------

    def _readline(self) -> bytes:
        assert self._resp is not None
        try:
            line = self._resp.readline(self._max_part)
        except (OSError, http.client.HTTPException) as exc:
            message = f"stream read failed: {describe_network_error(exc)}"
            raise CameraDisconnectedError(message) from exc
        if not line:
            raise CameraDisconnectedError("stream closed by the device")
        return line

    def _read_exact(self, n: int) -> bytes:
        assert self._resp is not None
        try:
            data = self._resp.read(n)
        except (OSError, http.client.HTTPException) as exc:
            message = f"stream read failed: {describe_network_error(exc)}"
            raise CameraDisconnectedError(message) from exc
        if len(data) < n:
            raise CameraDisconnectedError("stream closed mid-frame")
        return data

    def _is_boundary(self, line: bytes) -> bool:
        return line.rstrip(b"\r\n").startswith(self._boundary)

    def read_part(self) -> MjpegPart:
        if self._resp is None:
            raise CameraDisconnectedError("stream not open")
        if not self._at_headers:
            for _ in range(_MAX_HEADER_LINES):
                line = self._readline()
                if self._is_boundary(line):
                    break
                if line.strip():
                    raise CameraDisconnectedError(f"garbage before boundary: {line[:40]!r}")
            else:
                raise CameraDisconnectedError("no multipart boundary in stream")
        self._at_headers = False

        headers: dict[str, str] = {}
        for _ in range(_MAX_HEADER_LINES):
            line = self._readline()
            if not line.strip():
                break
            name, sep, value = line.decode("latin-1").partition(":")
            if sep:
                headers[name.strip().lower()] = value.strip()
        else:
            raise CameraDisconnectedError("multipart part headers never ended")

        length = headers.get("content-length")
        if length is not None and length.strip().isdigit():
            n = int(length)
            if n > self._max_part:
                raise CameraDisconnectedError(
                    f"part of {n} bytes exceeds the {self._max_part} limit"
                )
            data = self._read_exact(n)
        else:
            # No length: the body runs until the next boundary line.
            chunks: list[bytes] = []
            size = 0
            while True:
                line = self._readline()
                if self._is_boundary(line):
                    self._at_headers = True
                    break
                chunks.append(line)
                size += len(line)
                if size > self._max_part:
                    raise CameraDisconnectedError("part without Content-Length never ended")
            data = b"".join(chunks).removesuffix(b"\r\n")
        return MjpegPart(data, headers, time.monotonic_ns())


def jpeg_is_complete(data: bytes) -> bool:
    """True if ``data`` starts with SOI and ends with EOI (ignoring trailing padding)."""
    return data[:2] == b"\xff\xd8" and data.rstrip(b"\x00\r\n ")[-2:] == b"\xff\xd9"


LOCAL_NETWORK_HINT = (
    "on macOS 15+, allow the app that launched OpenGaze (Terminal, VS Code, …) under System "
    "Settings → Privacy & Security → Local Network"
)


def describe_network_error(exc: BaseException) -> str:
    if isinstance(exc, OSError) and exc.errno == errno.EHOSTUNREACH and sys.platform == "darwin":
        return f"no route to host ({LOCAL_NETWORK_HINT}; or the device is on another network)"
    if isinstance(exc, socket.timeout | TimeoutError):
        return "timed out (device off, out of Wi-Fi range, or on another network?)"
    if isinstance(exc, ConnectionRefusedError):
        return "connection refused (device up, but nothing listening on that port)"
    if isinstance(exc, socket.gaierror):
        return f"host name not found ({exc})"
    return f"{type(exc).__name__}: {exc}"
