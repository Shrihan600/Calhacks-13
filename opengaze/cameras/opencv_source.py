"""Shared implementation for cameras opened through OpenCV ``VideoCapture``.

On macOS this uses the AVFoundation backend, which serves both the built-in webcam and
UVC (USB) cameras. OpenCV can't report device names or exposure timestamps here, and many
``CAP_PROP_*`` requests are silently ignored, so every requested setting is read back and
reported instead of trusted.
"""

from __future__ import annotations

import math
import os
import sys
import time
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol

import cv2

# OpenCV's own permission prompt only works on the main thread, and cameras open on worker
# threads, so OpenGaze asks on the main thread instead (see ``permissions``).
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

from opengaze.cameras.base import BlockingCaptureSource, RawCapture
from opengaze.cameras.types import (
    CameraCapabilities,
    CameraConnectionError,
    CameraDisconnectedError,
    ControlSupport,
    PixelFormat,
    SettingResult,
    SourceKind,
)

_CONTROL_PROPS: dict[str, int] = {
    "fps": cv2.CAP_PROP_FPS,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "gain": cv2.CAP_PROP_GAIN,
    "focus": cv2.CAP_PROP_FOCUS,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
}

MACOS_PERMISSION_HINT = (
    "On macOS, check System Settings → Privacy & Security → Camera and allow the app that "
    "launched OpenGaze (Terminal, iTerm, VS Code, …), then restart it."
)


class CaptureLike(Protocol):
    """The subset of ``cv2.VideoCapture`` used for settings, so tests can substitute a fake."""

    def set(self, prop: int, value: float) -> bool: ...
    def get(self, prop: int) -> float: ...


def default_api_preference() -> int:
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def apply_capture_settings(
    cap: CaptureLike,
    resolution: tuple[int, int] | None,
    controls: Mapping[str, float | bool],
) -> list[SettingResult]:
    """Request each setting, read it back, and classify what the driver actually did.

    Resolution is only *requested* here; ``resolution_result`` confirms it from a real frame.
    """
    results: list[SettingResult] = []
    if resolution is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(resolution[0]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(resolution[1]))
    for name, value in controls.items():
        prop = _CONTROL_PROPS.get(name)
        if prop is None:
            results.append(
                SettingResult(name, value, None, ControlSupport.UNSUPPORTED, "no OpenCV property")
            )
            continue
        requested = float(value)
        accepted = bool(cap.set(prop, requested))
        actual = float(cap.get(prop))
        if not accepted:
            support, message = ControlSupport.UNSUPPORTED, "driver rejected the setting"
        elif name == "fps" and actual <= 0.0:
            support, message = ControlSupport.UNKNOWN, "driver does not report frame rate"
        elif math.isclose(actual, requested, rel_tol=0.01, abs_tol=0.01):
            support, message = ControlSupport.SUPPORTED, ""
        else:
            support, message = ControlSupport.IGNORED, f"driver reports {actual:g}"
        results.append(SettingResult(name, value, actual, support, message))
    return results


def resolution_result(requested: tuple[int, int] | None, actual: tuple[int, int]) -> SettingResult:
    if requested is None:
        return SettingResult("resolution", None, actual, ControlSupport.UNKNOWN, "driver default")
    if tuple(requested) == actual:
        return SettingResult("resolution", requested, actual, ControlSupport.SUPPORTED)
    return SettingResult(
        "resolution",
        requested,
        actual,
        ControlSupport.IGNORED,
        f"driver delivered {actual[0]}x{actual[1]}",
    )


class OpenCVCaptureSource(BlockingCaptureSource):
    source_kind = SourceKind.LIVE
    needs_camera_permission = True

    @abstractmethod
    def _device_index(self) -> int | str:
        """The OpenCV device index (or path) to open."""

    def _api_preference(self) -> int:
        return default_api_preference()

    def _open_blocking(self) -> Any:
        device = self._device_index()
        cap = cv2.VideoCapture(device, self._api_preference())
        if not cap.isOpened():
            cap.release()
            hint = f" {MACOS_PERMISSION_HINT}" if sys.platform == "darwin" else ""
            raise CameraConnectionError(f"could not open camera device {device!r}.{hint}")
        requested_controls: dict[str, float | bool] = dict(self.config.controls.requested())
        if self.config.fps is not None:
            requested_controls["fps"] = self.config.fps
        results = apply_capture_settings(cap, self.config.resolution, requested_controls)
        # The first frame confirms the stream works and gives the true delivered resolution.
        ok, image = cap.read()
        if not ok or image is None or image.size == 0:
            cap.release()
            raise CameraConnectionError(
                f"camera device {device!r} opened but delivered no frame (in use elsewhere?)"
            )
        actual = (int(image.shape[1]), int(image.shape[0]))
        results.insert(0, resolution_result(self.config.resolution, actual))
        reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._capabilities = CameraCapabilities(
            source_type=self.source_type,
            source_kind=self.source_kind,
            backend=cap.getBackendName(),
            device_name=None,  # OpenCV can't report device names
            resolution=actual,
            supported_resolutions=None,  # OpenCV can't enumerate modes
            reported_fps=reported_fps if reported_fps > 0 else None,
            pixel_formats=(PixelFormat.BGR8,),
            controls={r.name: r.support for r in results},
            provides_device_timestamps=False,
            intrinsics=self._intrinsics,
            distortion=self._distortion,
            setting_results=tuple(results),
            notes=(
                "capture timestamps are host receive times; exposure time is unknown",
                f"device index {device!r}; OpenCV indices can change when cameras are re-plugged",
            ),
        )
        return cap

    def _read_blocking(self, handle: Any) -> RawCapture:
        ok, image = handle.read()
        received = time.monotonic_ns()
        if not ok or image is None or image.size == 0:
            raise CameraDisconnectedError("camera stopped delivering frames")
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return RawCapture(image=image, host_receive_ns=received)

    def _close_blocking(self, handle: Any) -> None:
        handle.release()


def parse_device_index(device_id: int | str | None) -> int | str | None:
    """Accept integer indices, digit strings, and (on Linux) device paths like /dev/video2."""
    if isinstance(device_id, int) or device_id is None:
        return device_id
    if device_id.strip().isdigit():
        return int(device_id)
    return device_id
