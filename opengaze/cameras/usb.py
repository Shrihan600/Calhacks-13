"""Generic UVC (USB Video Class) cameras through OpenCV.

Resolution, frame rate and sensor controls (exposure, gain, focus) are requested from the
driver and read back; whatever the driver ignores is reported in the camera's capabilities.
"""

from __future__ import annotations

from typing import Any

from opengaze.cameras.base import ConfigIssue
from opengaze.cameras.opencv_source import OpenCVCaptureSource, parse_device_index
from opengaze.config.schema import CameraConfig


class USBCameraSource(OpenCVCaptureSource):
    source_type = "usb"

    def _device_index(self) -> int | str:
        device = parse_device_index(self.config.device_id)
        assert device is not None
        return device

    @classmethod
    def check_config(cls, config: CameraConfig) -> list[ConfigIssue]:
        issues = super().check_config(config)
        if config.device_id is None:
            issues.append(
                ConfigIssue(
                    "error",
                    f"{config.camera_id}: device_id is not set; run `opengaze probe-cameras` "
                    "and set it to the camera's index",
                )
            )
        return issues

    @classmethod
    def device_key(cls, config: CameraConfig) -> tuple[Any, ...] | None:
        if config.device_id is None:
            return None
        return ("opencv", parse_device_index(config.device_id))
