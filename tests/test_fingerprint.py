"""Hardware changes must invalidate calibrations; harmless changes must not."""

from __future__ import annotations

import copy

import pytest

from opengaze.config.fingerprint import (
    check_calibration_compatibility,
    geometry_snapshot,
    profile_fingerprint,
)
from opengaze.config.loader import check_profile_data
from opengaze.config.schema import HardwareProfile

BASE = {
    "hardware_profile": {
        "id": "rig",
        "version": 1,
        "frame_design": "v1",
        "cameras": {
            "left_eye": {"source_type": "usb", "device_id": 1, "resolution": [640, 480], "fps": 30},
            "head": {"source_type": "mac_webcam", "device_id": "auto"},
        },
    }
}


def build(mutate=None) -> HardwareProfile:
    data = copy.deepcopy(BASE)
    if mutate:
        mutate(data["hardware_profile"])
    report = check_profile_data(data)
    assert report.profile is not None, report.issues
    return report.profile


def left(p):
    return p["cameras"]["left_eye"]


@pytest.mark.parametrize(
    "harmless",
    [
        lambda p: left(p).update(device_id=2),  # re-plugged into another port
        lambda p: left(p).update(fps=60),
        lambda p: left(p).update(controls={"exposure": -5}),
        lambda p: left(p).update(read_timeout_s=5),
        lambda p: p.update(description="new notes"),
    ],
)
def test_harmless_changes_keep_calibration(harmless) -> None:
    stored = geometry_snapshot(build())
    result = check_calibration_compatibility(stored, build(harmless))
    assert result.compatible, result.explain()


@pytest.mark.parametrize(
    ("change", "path"),
    [
        (lambda p: left(p).update(rotation_degrees=90), "cameras.left_eye.rotation_degrees"),
        (lambda p: left(p).update(mirror_horizontal=True), "cameras.left_eye.mirror_horizontal"),
        (lambda p: left(p).update(resolution=[1280, 720]), "cameras.left_eye.resolution"),
        (
            lambda p: left(p).update(mounting_position="above_eye"),
            "cameras.left_eye.mounting_position",
        ),
        (
            lambda p: left(p).update(crop={"x": 0, "y": 0, "width": 0.5, "height": 0.5}),
            "cameras.left_eye.crop",
        ),
        (
            lambda p: left(p).update(geometry={"camera_to_eye_mm": 25}),
            "cameras.left_eye.geometry.camera_to_eye_mm",
        ),
        (
            lambda p: left(p).update(optics={"focal_length_mm": 3.6}),
            "cameras.left_eye.optics.focal_length_mm",
        ),
        (
            lambda p: left(p).update(source_type="mac_webcam", device_id=3),
            "cameras.left_eye.source_type",
        ),
        (lambda p: p.update(frame_design="v2"), "frame_design"),
        (lambda p: p.update(version=2), "version"),
        (
            lambda p: p["cameras"].update(right_eye={"source_type": "usb", "device_id": 2}),
            "cameras.right_eye",
        ),
        (lambda p: left(p).update(enabled=False), "cameras.left_eye"),
    ],
)
def test_geometry_changes_invalidate_calibration(change, path) -> None:
    stored = geometry_snapshot(build())
    result = check_calibration_compatibility(stored, build(change))
    assert not result.compatible
    assert any(c.path == path for c in result.changes), result.explain()
    assert "recalibrate" in result.explain()


def test_fingerprint_is_stable_and_sensitive() -> None:
    assert profile_fingerprint(build()) == profile_fingerprint(build())
    rotated = build(lambda p: left(p).update(rotation_degrees=180))
    assert profile_fingerprint(rotated) != profile_fingerprint(build())


def test_synthetic_fault_options_are_not_geometry() -> None:
    def synthetic(options):
        return lambda p: p["cameras"].update(
            left_eye={"source_type": "synthetic", "options": options}
        )

    stored = geometry_snapshot(build(synthetic({"pattern": "eye"})))
    noisy = build(synthetic({"pattern": "eye", "drop_probability": 0.2, "latency_ms": 50}))
    assert check_calibration_compatibility(stored, noisy).compatible
    remounted = build(synthetic({"pattern": "eye", "mount_rotation_degrees": 90}))
    assert not check_calibration_compatibility(stored, remounted).compatible
