"""Load and validate application config and hardware profiles.

Validation happens in two layers: the pydantic schema (types, ranges, typos), then semantic
checks that need the adapter registry (known source types, adapter options, duplicate roles
or devices, missing files). Neither layer opens a device.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from opengaze.cameras.registry import PLANNED_SOURCE_TYPES, get_source_class
from opengaze.cameras.types import CameraConfigError, CameraRole
from opengaze.config.schema import AppConfig, HardwareProfile
from opengaze.modes import TrackingMode, infer_mode

log = logging.getLogger(__name__)

Severity = Literal["error", "warning", "info"]
UNIQUE_ROLES = (CameraRole.LEFT_EYE, CameraRole.RIGHT_EYE, CameraRole.HEAD)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Severity
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity.upper():7} {self.location}: {self.message}"


@dataclass(frozen=True)
class ProfileReport:
    path: Path | None
    profile: HardwareProfile | None
    issues: tuple[ValidationIssue, ...] = ()
    mode: TrackingMode | None = None

    @property
    def ok(self) -> bool:
        return self.profile is not None and not self.errors

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


class ProfileValidationError(ValueError):
    def __init__(self, path: Path | None, issues: Sequence[ValidationIssue]) -> None:
        self.path = path
        self.issues = tuple(issues)
        super().__init__(str(self))

    def __str__(self) -> str:
        lines = [f"invalid hardware profile {self.path or '<data>'}:"]
        lines += [f"  {issue}" for issue in self.issues if issue.severity == "error"]
        return "\n".join(lines)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay values win; nested mappings merge key by key (used for per-user adjustments)."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _pydantic_issues(exc: ValidationError, prefix: str) -> list[ValidationIssue]:
    issues = []
    for err in exc.errors():
        location = ".".join([prefix, *(str(p) for p in err["loc"])])
        issues.append(ValidationIssue("error", location, err["msg"]))
    return issues


def check_profile_data(
    data: Mapping[str, Any], *, path: Path | None = None, base_dir: Path | None = None
) -> ProfileReport:
    """Validate an already-parsed ``{"hardware_profile": {...}}`` mapping. Never raises for
    content problems; everything is reported as issues."""
    raw = data.get("hardware_profile")
    if not isinstance(raw, Mapping):
        issue = ValidationIssue("error", "hardware_profile", "missing top-level 'hardware_profile'")
        return ProfileReport(path, None, (issue,))
    raw = dict(raw)
    if base_dir is not None and isinstance(raw.get("cameras"), Mapping):
        raw["cameras"] = {
            key: ({"base_dir": base_dir, **cam} if isinstance(cam, Mapping) else cam)
            for key, cam in raw["cameras"].items()
        }
    try:
        profile = HardwareProfile.model_validate(raw)
    except ValidationError as exc:
        return ProfileReport(path, None, tuple(_pydantic_issues(exc, "hardware_profile")))
    issues = semantic_issues(profile)
    mode = infer_mode(cam.role for cam in profile.enabled_cameras.values())
    return ProfileReport(path, profile, tuple(issues), mode)


def semantic_issues(profile: HardwareProfile) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    enabled = profile.enabled_cameras

    for cid, cam in profile.cameras.items():
        loc = f"cameras.{cid}"
        if not cam.enabled:
            issues.append(ValidationIssue("info", loc, "disabled"))
            continue
        try:
            cls = get_source_class(cam.source_type)
        except CameraConfigError as exc:
            issues.append(ValidationIssue("error", f"{loc}.source_type", str(exc)))
            continue
        for issue in cls.check_config(cam):
            message = issue.message.removeprefix(f"{cid}: ")  # location already names the camera
            issues.append(ValidationIssue(issue.severity, loc, message))  # type: ignore[arg-type]
        intrinsics = cam.optics.intrinsics
        if (
            intrinsics is not None
            and cam.resolution is not None
            and intrinsics.to_intrinsics().scaled_to(*cam.resolution) is None
        ):
            issues.append(
                ValidationIssue(
                    "warning",
                    f"{loc}.optics.intrinsics",
                    f"measured at {intrinsics.width}x{intrinsics.height}, a different aspect "
                    "ratio from the requested resolution; they will be ignored",
                )
            )

    for role in UNIQUE_ROLES:
        holders = [cid for cid, cam in enabled.items() if cam.role is role]
        if len(holders) > 1:
            issues.append(
                ValidationIssue("error", "cameras", f"role '{role.value}' assigned to {holders}")
            )

    devices: dict[tuple[Any, ...], str] = {}
    for cid, cam in enabled.items():
        if cam.source_type in PLANNED_SOURCE_TYPES:
            continue
        try:
            key = get_source_class(cam.source_type).device_key(cam)
        except CameraConfigError:
            continue
        if key is None:
            continue
        if key in devices:
            issues.append(
                ValidationIssue(
                    "error", f"cameras.{cid}", f"uses the same device as '{devices[key]}' {key}"
                )
            )
        else:
            devices[key] = cid

    roles = {cam.role for cam in enabled.values()}
    mode = infer_mode(roles)
    imu = profile.imu if profile.imu is not None and profile.imu.enabled else None
    if mode is None:
        issues.append(
            ValidationIssue("error", "cameras", "no enabled eye or head camera; nothing to track")
        )
    elif mode is not TrackingMode.WEBCAM_ONLY:
        has_head_cam = CameraRole.HEAD in roles
        if imu is None and not has_head_cam:
            issues.append(
                ValidationIssue(
                    "warning",
                    "cameras",
                    "no IMU and no head camera: head movement can't be compensated, so keep "
                    "the head still",
                )
            )
        elif imu is None:
            issues.append(
                ValidationIssue(
                    "info", "imu", "no IMU: head rotation comes from the webcam alone"
                )
            )
        elif not has_head_cam:
            issues.append(
                ValidationIssue(
                    "info",
                    "imu",
                    "IMU without a head camera: head rotation is tracked, head position isn't "
                    "(assumed fixed) and yaw may drift slowly; recenter while facing the screen",
                )
            )
    elif imu is not None:
        issues.append(
            ValidationIssue("warning", "imu", "ignored in Mode A: the webcam gives head pose")
        )
    if imu is not None:
        issues += _imu_issues(imu)
    issues += _gaze_issues(profile, mode)
    return issues


def _imu_issues(imu) -> list[ValidationIssue]:
    from opengaze.imu.sources import IMU_SOURCES

    cls = IMU_SOURCES.get(imu.source_type)
    if cls is None:
        return [
            ValidationIssue(
                "error",
                "imu.source_type",
                f"unknown {imu.source_type!r}; available: {sorted(IMU_SOURCES)}",
            )
        ]
    try:
        cls.Options.model_validate(imu.options)
    except ValidationError as exc:
        return _pydantic_issues(exc, "imu.options")
    if cls.source_kind == "synthetic":
        return [ValidationIssue("info", "imu", "SYNTHETIC IMU — simulated head motion")]
    return []


def _gaze_issues(profile: HardwareProfile, mode: TrackingMode | None) -> list[ValidationIssue]:
    from opengaze.gaze.detector2d import DETECTORS_2D
    from opengaze.gaze.eye_model3d import EYE_MODELS_3D

    issues: list[ValidationIssue] = []
    gaze = profile.gaze
    components = [("gaze.detector_2d", gaze.detector_2d, DETECTORS_2D)]
    if gaze.pipeline == "eye_3d":
        components.append(("gaze.eye_model_3d", gaze.eye_model_3d, EYE_MODELS_3D))
    for loc, component, registry in components:
        cls = registry.get(component.type)
        if cls is None:
            issues.append(
                ValidationIssue(
                    "error",
                    f"{loc}.type",
                    f"unknown {component.type!r}; available: {sorted(registry)}",
                )
            )
            continue
        try:
            cls.Options.model_validate(component.options)
        except ValidationError as exc:
            issues += _pydantic_issues(exc, f"{loc}.options")

    # Mode A's webcam eye features are gaze directions already; Modes B/C need eye_3d.
    if (
        profile.calibration.geometry_estimation == "geometric"
        and mode not in (None, TrackingMode.WEBCAM_ONLY)
        and gaze.pipeline != "eye_3d"
    ):
        issues.append(
            ValidationIssue(
                "error",
                "calibration.geometry_estimation",
                "geometric estimation needs a 3D gaze vector; set gaze.pipeline: eye_3d",
            )
        )
    return issues


def check_hardware_profile(path: Path, overlays: Iterable[Path] = ()) -> ProfileReport:
    """Read a profile (plus optional per-user overlay files) and report every problem."""
    path = Path(path)
    try:
        data = read_yaml(path)
        for overlay in overlays:
            data = deep_merge(data, read_yaml(Path(overlay)))
    except (OSError, yaml.YAMLError, ValueError) as exc:
        return ProfileReport(path, None, (ValidationIssue("error", str(path), str(exc)),))
    return check_profile_data(data, path=path, base_dir=path.resolve().parent)


def load_hardware_profile(path: Path, overlays: Iterable[Path] = ()) -> ProfileReport:
    """Like ``check_hardware_profile`` but raises ``ProfileValidationError`` on any error."""
    report = check_hardware_profile(path, overlays)
    if not report.ok:
        raise ProfileValidationError(report.path, report.issues)
    for issue in report.warnings:
        log.warning("%s: %s", path, issue)
    return report


def load_app_config(path: Path) -> AppConfig:
    """Load application settings; relative paths resolve against the config file."""
    path = Path(path)
    data = read_yaml(path)
    config = AppConfig.model_validate(data)
    profile_path = config.app.hardware_profile
    if not profile_path.is_absolute():
        profile_path = (path.resolve().parent / profile_path).resolve()
    return config.model_copy(
        update={"app": config.app.model_copy(update={"hardware_profile": profile_path})}
    )
