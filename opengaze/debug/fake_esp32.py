"""Emulated OpenGaze ESP32-CAM glasses, for developing the Mac side without hardware.

Serves the same interface as the reference firmware (``docs/hardware-interface.md``):

* per eye: ``GET /stream`` (MJPEG, chunked like ESP-IDF, with ``X-Sequence`` and
  ``X-Timestamp-Us`` part headers), ``GET /status`` (JSON) and ``GET /control``;
* right eye only: MPU-6050 JSON telemetry over UDP to whoever sends a subscribe datagram.

The emulated device clocks start at an arbitrary offset from the Mac's, as real ones do,
and faults (drops, jitter, stalls, disconnects, invalid readings) can be injected. Everything
it serves is marked emulated (``X-OpenGaze-Emulated: 1`` and ``"emulated": true``) and shows
up as such in diagnostics. Passing tests against it says nothing about real hardware.

Run it with ``opengaze fake-esp32`` and point ``configs/hardware/esp32-emulator.yaml`` at it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import random
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from opengaze.cameras.esp32 import FRAMESIZES
from opengaze.cameras.simulated import GazeTrajectory, eyelid_closure, render_eye, skin_background
from opengaze.imu.sources import SyntheticImuSource

log = logging.getLogger(__name__)

BOUNDARY = "123456789000000000000987654321"  # the one Espressif's example uses


@dataclass
class Faults:
    drop_rate: float = 0.0  # fraction of frames / IMU packets skipped (sequence still advances)
    jitter_ms: float = 0.0  # extra random delay per frame
    stall_every_s: float | None = None  # stop sending for stall_s, this often
    stall_s: float = 2.0
    disconnect_every_s: float | None = None  # close the stream connection this often
    invalid_rate: float = 0.0  # IMU readings flagged valid: false


@dataclass
class EyeSettings:
    role: str  # "left" | "right"
    resolution: tuple[int, int] = (320, 240)
    fps: float = 20.0
    quality: int = 80  # cv2 JPEG quality (0-100)
    content_length: bool = True  # false: parts delimited only by the boundary
    seed: int = 7
    faults: Faults = field(default_factory=Faults)


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:  # noqa: ANN001
        # Clients hanging up (keep-alive after /status, reconnect tests) is normal here.
        log.debug("fake ESP32: connection from %s ended with an error", client_address)


class _DeviceClock:
    """A device's microsecond clock: monotonic, arbitrary origin, slight drift."""

    def __init__(self, seed: int, drift_ppm: float = 20.0) -> None:
        rng = random.Random(seed)
        self._origin_ns = time.monotonic_ns() - rng.randint(10_000_000, 900_000_000) * 1000
        self._drift = 1.0 + drift_ppm * 1e-6

    def micros(self) -> int:
        return int((time.monotonic_ns() - self._origin_ns) * self._drift / 1000)


class FakeEyeCamera:
    """HTTP server for one emulated eye camera."""

    def __init__(self, settings: EyeSettings, host: str = "127.0.0.1", port: int = 0) -> None:
        self.settings = settings
        self.clock = _DeviceClock(settings.seed + (1 if settings.role == "right" else 0))
        self.trajectory = GazeTrajectory("saccades", settings.seed, 1.2)
        self._rng = random.Random(settings.seed)
        self._start = time.monotonic()
        self._sequence = 0
        self._lock = threading.Lock()
        self.frames_sent = 0
        self.clients = 0
        self.extra_status: dict[str, object] = {}
        self._server = _QuietServer((host, port), self._handler_class())
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> FakeEyeCamera:
        self._thread = threading.Thread(
            target=self._server.serve_forever, name=f"fake-esp32-{self.settings.role}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stopping.set()
        self._server.shutdown()
        self._server.server_close()

    # -- content -----------------------------------------------------------------------

    def render(self) -> tuple[bytes, int, int]:
        """(jpeg, sequence, device_time_us) for the next frame."""
        s = self.settings
        t = time.monotonic() - self._start
        w, h = s.resolution
        gaze = self.trajectory(t)
        image, _ = render_eye(skin_background(w, h, 0.3), gaze, eyelid_closure(t, 4.0, 0.15), 0.07)
        if s.role == "right":
            image = cv2.flip(image, 1)  # mounted mirror-image on the other side of the frame
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, s.quality])
        assert ok
        with self._lock:
            seq = self._sequence
            self._sequence += 1
        return jpeg.tobytes(), seq, self.clock.micros()

    def status(self) -> dict[str, object]:
        s = self.settings
        return {
            "device_id": f"{s.role}_esp32",
            "role": f"{s.role}_eye",
            "firmware": "opengaze-emulator",
            "framesize": FRAMESIZES.get(s.resolution),
            "resolution": list(s.resolution),
            "quality": s.quality,
            "fps_target": s.fps,
            "frames_sent": self.frames_sent,
            "clients": self.clients,
            "rssi": -55,
            "uptime_ms": self.clock.micros() // 1000,
            "emulated": True,
            **self.extra_status,
        }

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        camera = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: object) -> None:
                log.debug("fake %s: " + fmt, camera.settings.role, *args)

            def do_GET(self) -> None:  # noqa: N802 (http.server naming)
                url = urlsplit(self.path)
                if url.path == "/stream":
                    camera._serve_stream(self)
                elif url.path == "/status":
                    body = json.dumps(camera.status()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/control":
                    query = {k: v[0] for k, v in parse_qs(url.query).items()}
                    ok = camera._control(query.get("var", ""), query.get("val", ""))
                    self.send_response(200 if ok else 500)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    self.send_error(404)

        return Handler

    def _control(self, var: str, val: str) -> bool:
        try:
            value = int(val)
        except ValueError:
            return False
        if var == "framesize":
            sizes = {v: k for k, v in FRAMESIZES.items()}
            if value not in sizes:
                return False
            self.settings.resolution = sizes[value]
            return True
        if var == "quality":
            # ESP32 quality 4-63 (lower is better) → cv2 0-100 (higher is better), roughly.
            self.settings.quality = int(np.clip(100 - (value - 4) * 1.5, 5, 100))
            return True
        return False

    def _serve_stream(self, handler: BaseHTTPRequestHandler) -> None:
        s = self.settings
        f = s.faults
        handler.send_response(200)
        handler.send_header("Content-Type", f"multipart/x-mixed-replace;boundary={BOUNDARY}")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("X-OpenGaze-Emulated", "1")
        handler.end_headers()

        def chunk(data: bytes) -> None:
            handler.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")

        connected_at = time.monotonic()
        next_due = time.monotonic()
        self.clients += 1
        try:
            while not self._stopping.is_set():
                now = time.monotonic()
                if f.disconnect_every_s and now - connected_at > f.disconnect_every_s:
                    return  # drop the connection like a Wi-Fi glitch
                if f.stall_every_s:
                    phase = (now - self._start) % (f.stall_every_s + f.stall_s)
                    if phase > f.stall_every_s:
                        time.sleep(0.05)
                        next_due = time.monotonic()
                        continue
                if now < next_due:
                    time.sleep(next_due - now)
                next_due += 1.0 / s.fps
                jpeg, seq, device_us = self.render()
                if f.drop_rate and self._rng.random() < f.drop_rate:
                    continue
                if f.jitter_ms:
                    time.sleep(self._rng.uniform(0.0, f.jitter_ms) / 1000.0)
                headers = [
                    "Content-Type: image/jpeg",
                    f"X-Sequence: {seq}",
                    f"X-Timestamp-Us: {device_us}",
                    f"X-Timestamp: {device_us // 1_000_000}.{device_us % 1_000_000:06d}",
                ]
                if s.content_length:
                    headers.insert(1, f"Content-Length: {len(jpeg)}")
                part = (f"\r\n--{BOUNDARY}\r\n" + "\r\n".join(headers) + "\r\n\r\n").encode()
                chunk(part)
                chunk(jpeg)
                handler.wfile.flush()
                self.frames_sent += 1
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # the client went away
        finally:
            self.clients -= 1
            handler.close_connection = True
            # End the chunked body (ignored if the socket is already gone).
            with contextlib.suppress(OSError):
                handler.wfile.write(b"0\r\n\r\n")


class FakeImu:
    """MPU-6050 telemetry from the emulated right ESP32 over UDP.

    Listens on ``port`` for ``{"type": "subscribe"}`` datagrams and streams to the most
    recent subscriber, stopping when no subscribe arrives for ``subscription_timeout_s``.
    ``target`` sends to a fixed address instead (no handshake), like a firmware build with
    the Mac's IP compiled in.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        rate_hz: float = 100.0,
        batch: int = 1,
        motion: str = "sway_pause",
        target: tuple[str, int] | None = None,
        subscription_timeout_s: float = 5.0,
        clock: _DeviceClock | None = None,
        faults: Faults | None = None,
        seed: int = 3,
    ) -> None:
        self.rate_hz = rate_hz
        self.batch = max(1, batch)
        self.target = target
        self.subscription_timeout_s = subscription_timeout_s
        self.clock = clock or _DeviceClock(seed)
        self.faults = faults or Faults()
        self._rng = random.Random(seed)
        self._synthetic = SyntheticImuSource({"rate_hz": rate_hz, "motion": motion, "seed": seed})
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(0.0)
        self._subscriber: tuple[str, int] | None = None
        self._subscribed_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.sequence = 0
        self.packets_sent = 0
        self._t0_us = 0

    @property
    def port(self) -> int:
        return int(self._sock.getsockname()[1])

    def start(self) -> FakeImu:
        self._synthetic.open()
        self._t0_us = self.clock.micros()
        self._thread = threading.Thread(target=self._run, name="fake-esp32-imu", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
        self._sock.close()

    def _poll_subscribe(self) -> None:
        while True:
            try:
                data, sender = self._sock.recvfrom(2048)
            except (BlockingIOError, OSError):
                return
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            if isinstance(msg, dict) and msg.get("type") == "subscribe":
                if sender != self._subscriber:
                    log.info("fake IMU: streaming to %s:%d", *sender)
                self._subscriber, self._subscribed_at = sender, time.monotonic()

    def destination(self) -> tuple[str, int] | None:
        if self.target is not None:
            return self.target
        fresh = time.monotonic() - self._subscribed_at < self.subscription_timeout_s
        if self._subscriber and fresh:
            return self._subscriber
        return None

    def _reading(self) -> dict[str, object]:
        sample = self._synthetic.read(1.0)
        assert sample is not None
        seq = self.sequence
        self.sequence += 1
        record: dict[str, object] = {
            "device_id": "right_esp32",
            "sensor": "mpu6050",
            "sequence": seq,
            # Stamped at sample time (the firmware's timer tick), not at send time, so a
            # delayed burst of packets still carries evenly spaced timestamps.
            "timestamp_us": self._t0_us + int(round((sample.device_time_s or 0.0) * 1e6)),
            "emulated": True,
        }
        if self.faults.invalid_rate and self._rng.random() < self.faults.invalid_rate:
            record["valid"] = False
            return record
        record.update(
            accel_x=round(float(sample.accel[0]), 4),
            accel_y=round(float(sample.accel[1]), 4),
            accel_z=round(float(sample.accel[2]), 4),
            gyro_x=round(float(sample.gyro[0]), 5),
            gyro_y=round(float(sample.gyro[1]), 5),
            gyro_z=round(float(sample.gyro[2]), 5),
            temp_c=round(31.0 + 0.5 * math.sin(time.monotonic() / 60.0), 2),
            valid=True,
        )
        return record

    def _run(self) -> None:
        batch: list[dict[str, object]] = []
        while not self._stop.is_set():
            self._poll_subscribe()
            record = self._reading()  # paced by the synthetic source at rate_hz
            if self.faults.drop_rate and self._rng.random() < self.faults.drop_rate:
                continue
            batch.append(record)
            if len(batch) < self.batch:
                continue
            dest = self.destination()
            payload = json.dumps(batch[0] if len(batch) == 1 else batch).encode()
            batch = []
            if dest is None:
                continue
            try:
                self._sock.sendto(payload, dest)
                self.packets_sent += 1
            except OSError as exc:
                log.debug("fake IMU send failed: %s", exc)


@dataclass
class FakeGlasses:
    left: FakeEyeCamera
    right: FakeEyeCamera
    imu: FakeImu

    def stop(self) -> None:
        self.left.stop()
        self.right.stop()
        self.imu.stop()


def start_fake_glasses(
    *,
    host: str = "127.0.0.1",
    left_port: int = 0,
    right_port: int = 0,
    imu_port: int = 0,
    resolution: tuple[int, int] = (320, 240),
    fps: float = 20.0,
    imu_rate_hz: float = 100.0,
    imu_motion: str = "sway_pause",
    faults: Faults | None = None,
) -> FakeGlasses:
    """Start both emulated eye cameras and the right-side IMU (ports 0 = pick free ones)."""
    faults = faults or Faults()
    left = FakeEyeCamera(EyeSettings("left", resolution, fps, faults=faults), host, left_port)
    right = FakeEyeCamera(EyeSettings("right", resolution, fps, faults=faults), host, right_port)
    # The MPU-6050 hangs off the right ESP32, so it shares that device's clock.
    imu = FakeImu(
        host, imu_port, rate_hz=imu_rate_hz, motion=imu_motion, clock=right.clock, faults=faults
    )
    right.extra_status["imu"] = "mpu6050"
    right.extra_status["imu_port"] = imu.port
    return FakeGlasses(left.start(), right.start(), imu.start())
