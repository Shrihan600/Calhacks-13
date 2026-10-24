"""Geometry fingerprints: decide whether a saved calibration still fits the hardware.

A calibration stores a *snapshot* of every camera setting that could change how eye features
map to the screen. When a profile is loaded, the snapshot is diffed against it; any change
makes the calibration incompatible, and the changes are listed so the UI can explain why.

The rule is conservative: everything is geometry-relevant except an explicit allowlist of
settings known not to move the image (device index, frame rate, sensor controls, timeouts).
Phase 3 also stores the resolution actually negotiated during calibration, because
``resolution: null`` means "driver default", which can change on its own.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from opengaze.cameras.registry import get_source_class
from opengaze.cameras.types import CameraConfigError
from opengaze.config.schema import CameraConfig, HardwareProfile

# Camera settings that cannot change the eye-to-screen mapping.
NON_GEOMETRIC_CAMERA_FIELDS = frozenset(
    {"device_id", "fps", "controls", "connect_timeout_s", "read_timeout_s", "base_dir", "enabled"}
)

_ABSENT = "<absent>"


def _non_geometric_options(camera: CameraConfig) -> frozenset[str]:
    try:
        return get_source_class(camera.source_type).non_geometric_options
    except CameraConfigError:
        return frozenset()  # unknown adapter: treat every option as geometry-relevant


def geometry_snapshot(profile: HardwareProfile) -> dict[str, Any]:
    """JSON-serializable description of the calibration-relevant hardware state."""
    cameras: dict[str, Any] = {}
    for cid, camera in sorted(profile.enabled_cameras.items()):
        entry = camera.model_dump(mode="json", exclude=set(NON_GEOMETRIC_CAMERA_FIELDS))
        skip = _non_geometric_options(camera)
        entry["options"] = {k: v for k, v in sorted(entry["options"].items()) if k not in skip}
        cameras[cid] = entry
    return {
        "profile_id": profile.id,
        "version": profile.version,
        "frame_design": profile.frame_design,
        "cameras": cameras,
    }


def fingerprint(snapshot: dict[str, Any]) -> str:
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def profile_fingerprint(profile: HardwareProfile) -> str:
    return fingerprint(geometry_snapshot(profile))


@dataclass(frozen=True, slots=True)
class GeometryChange:
    path: str
    before: Any
    after: Any

    def __str__(self) -> str:
        return f"{self.path}: {self.before!r} → {self.after!r}"


def diff_snapshots(old: Any, new: Any, path: str = "") -> list[GeometryChange]:
    if isinstance(old, dict) and isinstance(new, dict):
        changes: list[GeometryChange] = []
        for key in sorted(set(old) | set(new), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in old:
                changes.append(GeometryChange(child, _ABSENT, new[key]))
            elif key not in new:
                changes.append(GeometryChange(child, old[key], _ABSENT))
            else:
                changes.extend(diff_snapshots(old[key], new[key], child))
        return changes
    if old != new:
        return [GeometryChange(path, old, new)]
    return []


@dataclass(frozen=True, slots=True)
class CalibrationCompatibility:
    compatible: bool
    changes: tuple[GeometryChange, ...]

    def explain(self) -> str:
        if self.compatible:
            return "calibration matches the current hardware profile"
        lines = ["hardware changed since this calibration — recalibrate before use:"]
        lines += [f"  {change}" for change in self.changes]
        return "\n".join(lines)


def check_calibration_compatibility(
    stored_snapshot: dict[str, Any], profile: HardwareProfile
) -> CalibrationCompatibility:
    changes = diff_snapshots(stored_snapshot, geometry_snapshot(profile))
    return CalibrationCompatibility(not changes, tuple(changes))
