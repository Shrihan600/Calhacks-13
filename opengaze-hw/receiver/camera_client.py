"""MJPEG stream client for the OpenGaze ESP32-CAM firmware.

Why not cv2.VideoCapture? It hides per-frame headers (we need X-Timestamp-Us), buffers
internally (latency), and gives poor failure detection. This client parses the multipart
stream directly, keeps ONLY the newest decoded frame, and reconnects with back-off.
"""
from __future__ import annotations
import http.client
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import cv2
import numpy as np

from synchronization import ClockTracker


@dataclass
class Frame:
    img: np.ndarray
    dev_ts_us: int        # device esp_timer clock at capture (0 if header absent)
    host_ts: float        # time.monotonic() when the last byte arrived
    frame_id: int
    nbytes: int
    rel_delay_ms: Optional[float]


class MJPEGClient(threading.Thread):
    def __init__(self, name: str, host: str, port: int = 81, path: str = "/stream",
                 connect_timeout: float = 4.0, read_timeout: float = 3.0):
        super().__init__(daemon=True, name=f"mjpeg-{name}")
        self.label, self.host, self.port, self.path = name, host, port, path
        self.connect_timeout, self.read_timeout = connect_timeout, read_timeout
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._latest_taken = True
        self.clock = ClockTracker()
        # status / stats
        self.state = "connecting"
        self.last_error = ""
        self.reconnects = 0            # successful re-connections after a drop
        self.failures = 0              # connection attempts / streams that ended in an error
        self._ever_streamed = False
        self.frames_ok = 0
        self.decode_fail = 0
        self.overwritten = 0           # frames replaced before the display consumed them
        self.stalls = 0                # inter-frame gap > 3x recent median
        self.bytes_total = 0
        self.decode_ms = 0.0           # EMA
        self.rel_delay_ms: Optional[float] = None
        self._arrivals: Deque[float] = deque(maxlen=120)
        self._gaps: Deque[float] = deque(maxlen=30)
        self.last_frame_host_ts = 0.0
        self.resolution = (0, 0)

    # ---------------------------------------------------------------- public
    def stop(self):
        self._stop_evt.set()

    def get_latest(self) -> Optional[Frame]:
        with self._lock:
            self._latest_taken = True
            return self._latest

    @property
    def fps(self) -> float:
        a = self._arrivals
        if len(a) < 3:
            return 0.0
        span = a[-1] - a[0]
        if time.monotonic() - a[-1] > 2.0:
            return 0.0
        return (len(a) - 1) / span if span > 0 else 0.0

    @property
    def frame_age_s(self) -> float:
        return time.monotonic() - self.last_frame_host_ts if self.last_frame_host_ts else float("inf")

    # ---------------------------------------------------------------- thread
    def run(self):
        backoff = 0.5
        while not self._stop_evt.is_set():
            conn = None
            try:
                self.state = "connecting"
                conn = http.client.HTTPConnection(self.host, self.port, timeout=self.connect_timeout)
                conn.request("GET", self.path, headers={"Connection": "keep-alive"})
                resp = conn.getresponse()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                try:
                    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    conn.sock.settimeout(self.read_timeout)
                except Exception:
                    pass
                self.state = "streaming"
                if self._ever_streamed:
                    self.reconnects += 1
                self._ever_streamed = True
                self.last_error = ""
                backoff = 0.5
                self._read_loop(resp)
            except Exception as e:  # noqa: BLE001 - we want every failure to trigger a reconnect
                self.last_error = f"{type(e).__name__}: {e}"
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
            if self._stop_evt.is_set():
                break
            self.state = "reconnecting"
            self.failures += 1
            self._stop_evt.wait(backoff)
            backoff = min(backoff * 2, 5.0)
        self.state = "stopped"

    def _readline(self, resp) -> bytes:
        line = resp.readline(512)
        if not line:
            raise ConnectionError("stream closed by device")
        return line

    def _read_loop(self, resp):
        while not self._stop_evt.is_set():
            # --- boundary line (skip blank preamble lines)
            line = self._readline(resp).strip()
            while not line:
                line = self._readline(resp).strip()
            if not line.startswith(b"--"):
                continue  # resync
            # --- part headers
            length, dev_ts, fid = None, 0, -1
            while True:
                h = self._readline(resp).strip()
                if not h:
                    break
                k, _, v = h.partition(b":")
                k = k.strip().lower()
                v = v.strip()
                if k == b"content-length":
                    length = int(v)
                elif k == b"x-timestamp-us":
                    dev_ts = int(v)
                elif k == b"x-frame-id":
                    fid = int(v)
            if not length or length > 2_000_000:
                continue
            data = resp.read(length)
            if len(data) != length:
                raise ConnectionError("short read")
            now = time.monotonic()
            self._on_jpeg(data, dev_ts, fid, now)

    def _on_jpeg(self, data: bytes, dev_ts: int, fid: int, now: float):
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        dt = (time.perf_counter() - t0) * 1e3
        if img is None:
            self.decode_fail += 1
            return
        self.decode_ms = dt if self.decode_ms == 0 else 0.9 * self.decode_ms + 0.1 * dt
        rel = None
        if dev_ts:
            self.clock.update(dev_ts, now)
            rel = self.clock.rel_delay_ms(dev_ts, now)
            self.rel_delay_ms = rel if self.rel_delay_ms is None else 0.9 * self.rel_delay_ms + 0.1 * rel
        if self._arrivals:
            gap = now - self._arrivals[-1]
            if len(self._gaps) >= 10 and gap > 3 * sorted(self._gaps)[len(self._gaps) // 2]:
                self.stalls += 1
            self._gaps.append(gap)
        self._arrivals.append(now)
        self.last_frame_host_ts = now
        self.frames_ok += 1
        self.bytes_total += len(data)
        self.resolution = (img.shape[1], img.shape[0])
        f = Frame(img, dev_ts, now, fid, len(data), rel)
        with self._lock:
            if not self._latest_taken:
                self.overwritten += 1
            self._latest, self._latest_taken = f, False
