"""Hardware profile loading, validation, overlays and tracking-mode inference."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from opengaze.cameras.types import CameraRole
from opengaze.config.loader import (
    ProfileValidationError,
    check_hardware_profile,
    check_profile_data,
    deep_merge,
    load_app_config,
    load_hardware_profile,
)
from opengaze.modes import TrackingMode, infer_mode
from tests.helpers import HARDWARE_DIR, REPO_ROOT


def profile(**cameras) -> dict:
    return {"hardware_profile": {"id": "test", "cameras": cameras}}


def errors(report) -> list[str]:
    return [f"{i.location}: {i.message}" for i in report.errors]


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("webcam-only.yaml", TrackingMode.WEBCAM_ONLY),
        ("synthetic-dual-eye.yaml", TrackingMode.DUAL_EYE),
        ("synthetic-dual-eye-3d.yaml", TrackingMode.DUAL_EYE),
        ("synthetic-faults.yaml", TrackingMode.DUAL_EYE),
    ],
)
def test_shipped_profiles_are_valid(name: str, mode: TrackingMode) -> None:
    report = check_hardware_profile(HARDWARE_DIR / name)
    assert report.ok, errors(report)
    assert report.mode is mode


def test_usb_template_asks_for_device_id() -> None:
    report = check_hardware_profile(HARDWARE_DIR / "generic-usb.yaml")
    assert not report.ok
    assert any("device_id" in e for e in errors(report))
    assert report.mode is TrackingMode.SINGLE_EYE  # right_eye is disabled in the template


def test_default_app_config_resolves_profile_path() -> None:
    config = load_app_config(REPO_ROOT / "configs" / "default.yaml")
    assert config.app.hardware_profile == (HARDWARE_DIR / "webcam-only.yaml").resolve()
    assert config.privacy.allow_recording is False


def test_role_and_id_come_from_key() -> None:
    report = check_profile_data(profile(head={"source_type": "synthetic"}))
    cam = report.profile.cameras["head"]
    assert (cam.camera_id, cam.role) == ("head", CameraRole.HEAD)


def test_custom_key_needs_explicit_role() -> None:
    report = check_profile_data(profile(cam_a={"source_type": "synthetic"}))
    assert any("role" in e for e in errors(report))
    report = check_profile_data(profile(cam_a={"source_type": "synthetic", "role": "left_eye"}))
    assert report.ok


@pytest.mark.parametrize(
    ("camera", "needle"),
    [
        ({"source_type": "synthetic", "rotation_degrees": 45}, "right-angle"),
        ({"source_type": "synthetic", "rotatoin_degrees": 90}, "Extra inputs"),  # typo
        ({"source_type": "synthetic", "fps": -5}, "greater than 0"),
        (
            {"source_type": "synthetic", "crop": {"x": 0.6, "y": 0, "width": 0.5, "height": 1}},
            "inside the image",
        ),
        ({"source_type": "synthetic", "optics": {"distortion": [0.1, 0.2, 0.3]}}, "coefficients"),
        ({"source_type": "synthetic", "options": {"pattern": "spiral"}}, "options.pattern"),
        ({"source_type": "webcam9000"}, "unknown source_type"),
        ({"source_type": "esp32"}, "Phase 5"),
        ({"source_type": "raspberry_pi"}, "not implemented yet"),
    ],
)
def test_invalid_camera_settings_are_reported(camera: dict, needle: str) -> None:
    report = check_profile_data(profile(left_eye=camera))
    assert not report.ok
    assert any(needle in e for e in errors(report)), errors(report)


def test_duplicate_roles_rejected() -> None:
    report = check_profile_data(
        profile(
            a={"source_type": "synthetic", "role": "head"},
            b={"source_type": "synthetic", "role": "head"},
        )
    )
    assert any("role 'head'" in e for e in errors(report))


def test_same_device_twice_rejected() -> None:
    report = check_profile_data(
        profile(
            head={"source_type": "mac_webcam", "device_id": "auto"},
            left_eye={"source_type": "usb", "device_id": 0},
        )
    )
    assert any("same device" in e for e in errors(report))


def test_no_trackable_camera_rejected() -> None:
    report = check_profile_data(profile(misc={"source_type": "synthetic", "role": "other"}))
    assert any("nothing to track" in e for e in errors(report))


def test_eye_without_head_is_a_warning() -> None:
    report = check_profile_data(profile(left_eye={"source_type": "synthetic"}))
    assert report.ok
    assert any("head" in w.message for w in report.warnings)


def test_disabled_cameras_do_not_count() -> None:
    report = check_profile_data(
        profile(
            left_eye={"source_type": "synthetic"},
            right_eye={"source_type": "esp32", "enabled": False},
            head={"source_type": "synthetic", "options": {"pattern": "test_card"}},
        )
    )
    assert report.ok, errors(report)
    assert report.mode is TrackingMode.SINGLE_EYE


def test_geometric_estimation_needs_eye_cameras() -> None:
    data = profile(head={"source_type": "synthetic"})
    data["hardware_profile"]["calibration"] = {"geometry_estimation": "geometric"}
    assert any("needs eye cameras" in e for e in errors(check_profile_data(data)))


def test_geometric_estimation_needs_eye_3d_pipeline() -> None:
    data = profile(left_eye={"source_type": "synthetic"})
    data["hardware_profile"]["calibration"] = {"geometry_estimation": "geometric"}
    assert any("eye_3d" in e for e in errors(check_profile_data(data)))
    data["hardware_profile"]["gaze"] = {"pipeline": "eye_3d"}
    report = check_profile_data(data)
    assert report.ok, errors(report)


def test_gaze_components_are_validated() -> None:
    data = profile(left_eye={"source_type": "synthetic"})
    data["hardware_profile"]["gaze"] = {
        "pipeline": "eye_3d",
        "detector_2d": {"type": "nope"},
        "eye_model_3d": {"type": "sphere_fit", "options": {"buffer_sizee": 10}},
    }
    errs = errors(check_profile_data(data))
    assert any(e.startswith("gaze.detector_2d.type") for e in errs)
    assert any(e.startswith("gaze.eye_model_3d.options") for e in errs)


def test_user_overlay_merges_over_profile(tmp_path: Path) -> None:
    overlay = tmp_path / "user.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "hardware_profile": {
                    "cameras": {
                        "right_eye": {"crop": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8}}
                    }
                }
            }
        )
    )
    report = load_hardware_profile(HARDWARE_DIR / "synthetic-dual-eye.yaml", [overlay])
    right = report.profile.cameras["right_eye"]
    assert right.crop is not None and right.crop.x == 0.1
    assert right.rotation_degrees == 180  # untouched settings survive the merge


def test_load_raises_with_all_errors() -> None:
    with pytest.raises(ProfileValidationError) as excinfo:
        load_hardware_profile(HARDWARE_DIR / "generic-usb.yaml")
    assert "device_id" in str(excinfo.value)


def test_unreadable_yaml_is_reported(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("hardware_profile: [unclosed")
    assert not check_hardware_profile(bad).ok
    assert not check_hardware_profile(tmp_path / "missing.yaml").ok


def test_deep_merge() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"b": 5}, "d": None}) == {
        "a": {"b": 5, "c": 2},
        "d": None,
    }


@pytest.mark.parametrize(
    ("roles", "mode"),
    [
        ({CameraRole.HEAD}, TrackingMode.WEBCAM_ONLY),
        ({CameraRole.HEAD, CameraRole.LEFT_EYE}, TrackingMode.SINGLE_EYE),
        ({CameraRole.RIGHT_EYE}, TrackingMode.SINGLE_EYE),
        ({CameraRole.HEAD, CameraRole.LEFT_EYE, CameraRole.RIGHT_EYE}, TrackingMode.DUAL_EYE),
        ({CameraRole.OTHER}, None),
    ],
)
def test_infer_mode(roles, mode) -> None:
    assert infer_mode(roles) is mode
