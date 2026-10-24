"""Settings negotiation, tested against a fake driver (no camera needed)."""

from __future__ import annotations

import cv2
import pytest

from opengaze.cameras.mac_webcam import MacWebcamSource
from opengaze.cameras.opencv_source import apply_capture_settings, resolution_result
from opengaze.cameras.types import ControlSupport
from opengaze.cameras.usb import USBCameraSource
from tests.helpers import camera_config


class FakeCapture:
    """Accepts some properties, silently clamps others, rejects the rest — like real drivers."""

    def __init__(self) -> None:
        self.values: dict[int, float] = {cv2.CAP_PROP_FPS: 30.0}

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_GAIN:
            return False
        if prop == cv2.CAP_PROP_FPS:
            self.values[prop] = 30.0  # accepted, but stays at 30
            return True
        self.values[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.values.get(prop, 0.0)


def test_classifies_each_setting() -> None:
    results = {
        r.name: r
        for r in apply_capture_settings(
            FakeCapture(), (640, 480), {"exposure": -6.0, "gain": 4.0, "fps": 60.0}
        )
    }
    assert results["exposure"].support is ControlSupport.SUPPORTED
    assert results["gain"].support is ControlSupport.UNSUPPORTED
    assert results["fps"].support is ControlSupport.IGNORED
    assert results["fps"].actual == 30.0


def test_fps_not_reported() -> None:
    class Silent(FakeCapture):
        def get(self, prop: int) -> float:
            return 0.0

    (result,) = apply_capture_settings(Silent(), None, {"fps": 60.0})
    assert result.support is ControlSupport.UNKNOWN


def test_resolution_result() -> None:
    assert resolution_result((640, 480), (640, 480)).support is ControlSupport.SUPPORTED
    ignored = resolution_result((1920, 1080), (1280, 720))
    assert ignored.support is ControlSupport.IGNORED and "1280x720" in ignored.message
    assert resolution_result(None, (1280, 720)).support is ControlSupport.UNKNOWN


@pytest.mark.parametrize(("device_id", "expected"), [("auto", 0), (None, 0), (2, 2), ("1", 1)])
def test_mac_webcam_device_resolution(device_id, expected) -> None:
    config = camera_config("head", source_type="mac_webcam", device_id=device_id)
    assert MacWebcamSource(config)._device_index() == expected


def test_usb_requires_device_id() -> None:
    config = camera_config("left_eye", source_type="usb")
    issues = USBCameraSource.check_config(config)
    assert any(i.severity == "error" and "device_id" in i.message for i in issues)


@pytest.mark.hardware
async def test_real_mac_webcam_streams() -> None:
    """Opens the real built-in camera. Run explicitly: pytest -m hardware"""
    source = MacWebcamSource(
        camera_config("head", source_type="mac_webcam", device_id="auto", resolution=(1280, 720))
    )
    async with source:
        frames = [await source.read_frame() for _ in range(10)]
    assert frames[-1].width > 0
    assert source.get_capabilities().resolution is not None
