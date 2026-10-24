"""The MacBook's built-in camera, through OpenCV's AVFoundation backend."""

from __future__ import annotations

import sys
from typing import Any

from opengaze.cameras.base import ConfigIssue
from opengaze.cameras.opencv_source import OpenCVCaptureSource, parse_device_index
from opengaze.config.schema import CameraConfig


class MacWebcamSource(OpenCVCaptureSource):
    """Built-in webcam. ``device_id: auto`` opens index 0.

    Index 0 is normally the built-in camera, but an iPhone Continuity Camera or a USB camera
    can take it; set an explicit index (see ``opengaze probe-cameras``) if the preview shows
    the wrong camera.
    """

    source_type = "mac_webcam"

    @staticmethod
    def _resolve(device_id: int | str | None) -> int | str | None:
        if device_id is None or (isinstance(device_id, str) and device_id.lower() == "auto"):
            return 0
        return parse_device_index(device_id)

    def _device_index(self) -> int | str:
        device = self._resolve(self.config.device_id)
        assert device is not None
        return device

    @classmethod
    def check_config(cls, config: CameraConfig) -> list[ConfigIssue]:
        issues = super().check_config(config)
        if isinstance(cls._resolve(config.device_id), str):
            issues.append(
                ConfigIssue("error", f"{config.camera_id}: device_id must be 'auto' or an integer")
            )
        if sys.platform != "darwin":
            issues.append(
                ConfigIssue(
                    "warning",
                    f"{config.camera_id}: mac_webcam is meant for macOS; elsewhere it opens "
                    "device index with the platform's default backend",
                )
            )
        return issues

    @classmethod
    def device_key(cls, config: CameraConfig) -> tuple[Any, ...] | None:
        return ("opencv", cls._resolve(config.device_id))
