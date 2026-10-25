"""ESP32-CAM (OV2640) eye cameras streaming MJPEG over Wi-Fi.

Works with the OpenGaze reference firmware (``firmware/opengaze_esp32cam``) and with
Espressif's stock ``CameraWebServer`` example. Both serve ``http://<ip>:81/stream`` and
``http://<ip>/status``. The reference firmware adds two part headers:

* ``X-Sequence``: frame counter, so frames lost on the device or the link show up as gaps;
* ``X-Timestamp-Us``: capture time on the ESP32's monotonic clock.

The stock example sends ``X-Timestamp`` (``seconds.microseconds``) and no sequence numbers.
Device timestamps are mapped to the host clock by ``DeviceClockSync``. The ESP32 and the Mac
clocks are never assumed to agree.

Settings such as resolution and JPEG quality live in the firmware. With
``apply_settings: true`` the adapter also asks the device to change them through the
``/control`` endpoint at connect time. Either way, the resolution actually delivered is
read back from the first frame and reported.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, ClassVar
from urllib.parse import urlsplit

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from opengaze.cameras.base import BlockingCaptureSource, ConfigIssue, RawCapture
from opengaze.cameras.mjpeg import DEFAULT_MAX_PART_BYTES, MjpegPart, MjpegStream, jpeg_is_complete
from opengaze.cameras.opencv_source import resolution_result
from opengaze.cameras.timing import DeviceClockSync
from opengaze.cameras.types import (
    CameraCapabilities,
    CameraDisconnectedError,
    ControlSupport,
    FrameStatus,
    PixelFormat,
    SettingResult,
    SourceKind,
    TimestampSource,
)
from opengaze.config.schema import CameraConfig

# esp32-camera ``framesize_t`` values (esp32-camera ≥ 2.0). Older releases number them
# differently, which is why the delivered resolution is always read back.
FRAMESIZES: dict[tuple[int, int], int] = {
    (96, 96): 0,
    (160, 120): 1,
    (176, 144): 2,
    (240, 176): 3,
    (240, 240): 4,
    (320, 240): 5,
    (400, 296): 6,
    (480, 320): 7,
    (640, 480): 8,
    (800, 600): 9,
    (1024, 768): 10,
    (1280, 720): 11,
    (1280, 1024): 12,
    (1600, 1200): 13,
}

# Undecodable JPEGs in a row before the stream counts as broken and is reconnected.
_MAX_BAD_FRAMES = 10


class ESP32Options(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Either host (+ port/path) or a full url. Hosts may be IPs or mDNS names
    # (the reference firmware announces opengaze-left.local / opengaze-right.local).
    host: str | None = None
    port: int = Field(default=81, gt=0, lt=65536)
    path: str = "/stream"
    url: str | None = None
    # Device status JSON (``/status`` on the web-server port); null = don't query it.
    status_port: int | None = Field(default=80, gt=0, lt=65536)
    status_path: str = "/status"
    # Send resolution / jpeg_quality to ``/control`` on connect (CameraWebServer API).
    apply_settings: bool = False
    jpeg_quality: int | None = Field(default=None, ge=4, le=63)  # lower = better quality
    # Map part timestamps onto the host clock. false = use host receive times only.
    device_clock: bool = True
    grayscale: bool = False  # decode straight to 8-bit gray (eye pipelines accept either)
    max_frame_bytes: int = Field(default=DEFAULT_MAX_PART_BYTES, gt=1000)

    @model_validator(mode="after")
    def _has_address(self) -> ESP32Options:
        if not self.host and not self.url:
            raise ValueError(
                "set host (the ESP32's IP address or mDNS name, e.g. opengaze-left.local) "
                "or url (e.g. http://192.168.1.42:81/stream)"
            )
        if self.url is not None and urlsplit(self.url).scheme != "http":
            raise ValueError("url must start with http://")
        if not self.path.startswith("/"):
            raise ValueError("path must start with /")
        return self

    def stream_url(self) -> str:
        if self.url:
            return self.url
        return f"http://{self.host}:{self.port}{self.path}"

    def device_host(self) -> str:
        return self.host or urlsplit(self.stream_url()).hostname or ""

    def http_url(self, port: int, path: str) -> str:
        return f"http://{self.device_host()}:{port}{path}"


class ESP32StreamSource(BlockingCaptureSource):
    source_type = "esp32"
    source_kind = SourceKind.LIVE
    Options: ClassVar[type[BaseModel]] = ESP32Options
    # The network address and transport settings don't move the image; changing an IP
    # must not invalidate a calibration.
    non_geometric_options = frozenset(
        {
            "host",
            "port",
            "path",
            "url",
            "status_port",
            "status_path",
            "apply_settings",
            "jpeg_quality",
            "device_clock",
            "max_frame_bytes",
        }
    )

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self._clock = DeviceClockSync()
        self._pending: RawCapture | None = None
        self._bad_in_a_row = 0
        self.corrupt_frames = 0
        self._emulated = False

    @classmethod
    def check_config(cls, config: CameraConfig) -> list[ConfigIssue]:
        issues = super().check_config(config)
        resolution = config.resolution
        if not issues and resolution is not None and tuple(resolution) not in FRAMESIZES:
            sizes = ", ".join(f"{w}x{h}" for w, h in FRAMESIZES)
            issues.append(
                ConfigIssue(
                    "warning",
                    f"{config.camera_id}: {resolution[0]}x{resolution[1]} "
                    f"is not an OV2640 frame size ({sizes})",
                )
            )
        return issues

    @classmethod
    def device_key(cls, config: CameraConfig) -> tuple[Any, ...] | None:
        try:
            options: ESP32Options = cls.parse_options(config)
        except Exception:
            return None
        parts = urlsplit(options.stream_url())
        return ("http", (parts.hostname or "").lower(), parts.port or 80, parts.path)

    # -- blocking driver ---------------------------------------------------------------

    def _open_blocking(self) -> Any:
        o: ESP32Options = self.options
        status = self._fetch_status() if o.status_port is not None else None
        results: list[SettingResult] = []
        if o.apply_settings:
            results += self._apply_settings()

        stream = MjpegStream(
            o.stream_url(),
            connect_timeout_s=self.config.connect_timeout_s,
            read_timeout_s=self.config.read_timeout_s,
            max_part_bytes=o.max_frame_bytes,
        )
        stream.open()
        self._emulated = stream.response_headers.get("x-opengaze-emulated") == "1"
        self._bad_in_a_row = 0
        try:
            first = self._capture(stream)
        except Exception:
            stream.close()
            raise
        self._pending = first
        actual = (int(first.image.shape[1]), int(first.image.shape[0]))
        results.insert(0, resolution_result(self.config.resolution, actual))

        notes = [
            f"stream {o.stream_url()}",
            "Wi-Fi MJPEG: expect jittery frame intervals and occasional drops",
        ]
        if self._emulated:
            notes.insert(0, "EMULATED device (opengaze fake-esp32), not real hardware")
        has_device_ts = first.timestamp_source is TimestampSource.DEVICE
        notes.append(
            "capture times: ESP32 clock mapped to host (min-offset sync)"
            if has_device_ts
            else "capture times: host receive (device sends no timestamps)"
        )
        device_name = None
        if status is not None:
            device_name = str(status.get("device_id") or status.get("role") or "") or None
            summary = ", ".join(
                f"{k}={status[k]}"
                for k in ("firmware", "role", "framesize", "quality", "rssi", "imu")
                if k in status
            )
            if summary:
                notes.append(f"status: {summary}")
        self._capabilities = CameraCapabilities(
            source_type=self.source_type,
            source_kind=self.source_kind,
            backend="mjpeg-http",
            device_name=device_name,
            resolution=actual,
            supported_resolutions=tuple(FRAMESIZES),
            reported_fps=None,
            pixel_formats=(PixelFormat.GRAY8 if o.grayscale else PixelFormat.BGR8,),
            controls={r.name: r.support for r in results},
            provides_device_timestamps=has_device_ts,
            intrinsics=self._intrinsics,
            distortion=self._distortion,
            setting_results=tuple(results),
            notes=tuple(notes),
        )
        return stream

    def _read_blocking(self, handle: Any) -> RawCapture:
        if self._pending is not None:
            raw, self._pending = self._pending, None
            return raw
        return self._capture(handle)

    def _close_blocking(self, handle: Any) -> None:
        self._pending = None
        handle.close()

    # -- helpers -----------------------------------------------------------------------

    def _capture(self, stream: MjpegStream) -> RawCapture:
        """Next decodable frame. Undecodable parts are skipped and counted."""
        while True:
            part = stream.read_part()
            raw = self._decode(part)
            if raw is not None:
                self._bad_in_a_row = 0
                return raw
            self.corrupt_frames += 1
            self._bad_in_a_row += 1
            if self._bad_in_a_row >= _MAX_BAD_FRAMES:
                raise CameraDisconnectedError(
                    f"{self._bad_in_a_row} undecodable JPEG frames in a row"
                )

    def _decode(self, part: MjpegPart) -> RawCapture | None:
        o: ESP32Options = self.options
        if not part.data:
            return None
        flags = cv2.IMREAD_GRAYSCALE if o.grayscale else cv2.IMREAD_COLOR
        image = cv2.imdecode(np.frombuffer(part.data, np.uint8), flags)
        if image is None or image.size == 0:
            return None
        warnings: list[str] = []
        status = FrameStatus.OK
        if not jpeg_is_complete(part.data):
            status = FrameStatus.PARTIAL
            warnings.append("truncated JPEG")
        if self._emulated:
            warnings.append("emulated device")

        capture_ns = None
        source = TimestampSource.HOST_RECEIVE
        uncertainty = None
        device_us = part.device_time_us if o.device_clock else None
        if device_us is not None:
            capture_ns = self._clock.update(device_us, part.host_receive_ns)
            source = TimestampSource.DEVICE
            uncertainty = self._clock.uncertainty_ns
        return RawCapture(
            image=image,
            host_receive_ns=part.host_receive_ns,
            capture_ns=capture_ns,
            timestamp_source=source,
            timestamp_uncertainty_ns=uncertainty,
            status=status,
            warnings=tuple(warnings),
            sequence=part.sequence,
        )

    def _fetch_status(self) -> dict[str, Any] | None:
        o: ESP32Options = self.options
        assert o.status_port is not None
        url = o.http_url(o.status_port, o.status_path)
        try:
            with urllib.request.urlopen(url, timeout=min(2.0, self.config.connect_timeout_s)) as r:
                data = json.loads(r.read(16_384))
        except (OSError, ValueError, urllib.error.URLError):
            return None  # optional: the stream itself is what matters
        return data if isinstance(data, dict) else None

    def _apply_settings(self) -> list[SettingResult]:
        o: ESP32Options = self.options
        port = o.status_port or 80
        requests: list[tuple[str, object, str, int]] = []
        if self.config.resolution is not None:
            size = tuple(self.config.resolution)
            if size in FRAMESIZES:
                requests.append(("framesize", size, "framesize", FRAMESIZES[size]))
        if o.jpeg_quality is not None:
            requests.append(("jpeg_quality", o.jpeg_quality, "quality", o.jpeg_quality))
        results = []
        for name, requested, var, val in requests:
            url = o.http_url(port, f"/control?var={var}&val={val}")
            try:
                with urllib.request.urlopen(url, timeout=2.0) as r:
                    ok = r.status == 200
            except (OSError, urllib.error.URLError) as exc:
                results.append(
                    SettingResult(name, requested, None, ControlSupport.UNSUPPORTED, str(exc))
                )
                continue
            support = ControlSupport.UNKNOWN if ok else ControlSupport.UNSUPPORTED
            message = "accepted by /control; delivered size is checked below" if ok else ""
            results.append(SettingResult(name, requested, None, support, message))
        if requests:
            time.sleep(0.3)  # the sensor drops a frame or two while it reconfigures
        return results
