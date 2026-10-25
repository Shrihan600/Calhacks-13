"""Maps a profile's ``source_type`` string to the adapter class that implements it."""

from __future__ import annotations

from opengaze.cameras.base import CameraSource
from opengaze.cameras.esp32 import ESP32StreamSource
from opengaze.cameras.mac_webcam import MacWebcamSource
from opengaze.cameras.recorded import RecordedVideoSource
from opengaze.cameras.simulated import SyntheticCameraSource
from opengaze.cameras.types import CameraConfigError
from opengaze.cameras.usb import USBCameraSource
from opengaze.cameras.wired import SerialEyeSource
from opengaze.config.schema import CameraConfig

_SOURCES: dict[str, type[CameraSource]] = {
    cls.source_type: cls
    for cls in (
        MacWebcamSource,
        USBCameraSource,
        RecordedVideoSource,
        SyntheticCameraSource,
        ESP32StreamSource,
        SerialEyeSource,
    )
}

# Accepted in profiles so they can be written ahead of time, but rejected by the validator
# with this explanation until an adapter exists. Nothing pretends to support them yet.
PLANNED_SOURCE_TYPES: dict[str, str] = {
    "raspberry_pi": "planned for Phase 5 (MJPEG-over-HTTP / RTSP network stream)",
}


def register_source(cls: type[CameraSource]) -> type[CameraSource]:
    """Register an additional adapter (usable as a class decorator)."""
    _SOURCES[cls.source_type] = cls
    return cls


def available_source_types() -> list[str]:
    return sorted(_SOURCES)


def get_source_class(source_type: str) -> type[CameraSource]:
    cls = _SOURCES.get(source_type)
    if cls is not None:
        return cls
    if source_type in PLANNED_SOURCE_TYPES:
        reason = PLANNED_SOURCE_TYPES[source_type]
        raise CameraConfigError(f"source_type '{source_type}' is not implemented yet: {reason}")
    raise CameraConfigError(
        f"unknown source_type '{source_type}'; available: {', '.join(available_source_types())}"
    )


def create_source(config: CameraConfig) -> CameraSource:
    return get_source_class(config.source_type)(config)
