"""Save and load fitted gaze models, bound to a geometry fingerprint.

Storage format: a single .npz file per calibration.
Filename encodes the schema_id and fingerprint so different modes / hardware
profiles never collide and stale calibrations are easy to spot.

Layout inside the .npz:
  mapper_kind                                 — which ScreenMapper (absent = ridge, pre-Phase 5)
  <mapper arrays>                             — e.g. coef, intercept, feature_mean, feature_std
  schema_id                                   — byte-string scalar
  geometry_snapshot                           — JSON byte-string
  tracking_snapshot                           — JSON byte-string (Modes B/C): eye pipeline and
                                                calibration policy the model was fitted with
  eye_<side>__<name>                          — frozen 3D eye-model state per eye (eye_3d only)
  completed_at_ns                             — int64 scalar
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from opengaze.calibration.mappers import mapper_from_arrays
from opengaze.calibration.model import ScreenMapper
from opengaze.config.fingerprint import (
    CalibrationCompatibility,
    GeometryChange,
    check_calibration_compatibility,
    diff_snapshots,
    geometry_snapshot,
    profile_fingerprint,
)
from opengaze.config.schema import HardwareProfile
from opengaze.vision.observations import SchemaId

_STORE_DIR = Path.home() / ".config" / "opengaze" / "calibrations"
_SUFFIX = ".calib.npz"


def tracking_snapshot(schema_id: str, profile: HardwareProfile) -> dict[str, Any]:
    """Tracking settings a calibration depends on besides camera geometry."""
    snap: dict[str, Any] = {"geometry_estimation": profile.calibration.geometry_estimation}
    if schema_id != "A-webcam":
        snap["gaze"] = profile.gaze.model_dump(mode="json")
    return snap


@dataclass
class CalibrationBundle:
    model: ScreenMapper
    # Frozen per-eye 3D model state ("left"/"right" → arrays); empty for other pipelines.
    eye_states: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


def _filename(schema_id: SchemaId, fp: str) -> str:
    safe = schema_id.replace("-", "_")
    return f"{safe}__{fp}{_SUFFIX}"


class CalibrationStore:
    """Reads and writes .npz calibration files for a given profile."""

    def __init__(self, store_dir: Path | None = None) -> None:
        self.store_dir = store_dir or _STORE_DIR

    def _path_for(self, schema_id: SchemaId, fp: str) -> Path:
        return self.store_dir / _filename(schema_id, fp)

    def save(
        self,
        model: ScreenMapper,
        profile: HardwareProfile,
        completed_at_ns: int | None = None,
        eye_states: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> Path:
        """Persist model (and any frozen eye-model state) to disk.  Returns the path written."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        fp = profile_fingerprint(profile)
        path = self._path_for(model.schema_id, fp)

        arrays: dict[str, Any] = {}
        for key, value in model.to_arrays().items():
            arrays[key] = np.bytes_(value) if isinstance(value, str) else value
        for side, state in (eye_states or {}).items():
            for name, value in state.items():
                arrays[f"eye_{side}__{name}"] = value
        np.savez_compressed(
            path,
            **arrays,
            geometry_snapshot=np.bytes_(_dumps(geometry_snapshot(profile))),
            tracking_snapshot=np.bytes_(_dumps(tracking_snapshot(model.schema_id, profile))),
            completed_at_ns=np.int64(completed_at_ns or time.time_ns()),
        )
        return path

    def load(
        self,
        schema_id: SchemaId,
        profile: HardwareProfile,
    ) -> ScreenMapper:
        """Load and return the model for this schema + profile.

        Raises FileNotFoundError if no calibration exists.
        Raises ValueError if the stored geometry or tracking settings are incompatible.
        """
        return self.load_bundle(schema_id, profile).model

    def load_bundle(self, schema_id: SchemaId, profile: HardwareProfile) -> CalibrationBundle:
        """Like ``load`` but also returns the frozen eye-model state."""
        fp = profile_fingerprint(profile)
        path = self._path_for(schema_id, fp)
        if not path.exists():
            raise FileNotFoundError(
                f"no calibration found for schema={schema_id!r}, fingerprint={fp!r} "
                f"(expected {path})"
            )
        return self._load_path(path, profile)

    def _load_path(self, path: Path, profile: HardwareProfile) -> CalibrationBundle:
        data = dict(np.load(path, allow_pickle=False))
        compat = _compatibility(data, profile)
        if not compat.compatible:
            raise ValueError(compat.explain())
        eye_states: dict[str, dict[str, np.ndarray]] = {}
        for key, value in data.items():
            if key.startswith("eye_") and "__" in key:
                side, name = key[len("eye_") :].split("__", 1)
                eye_states.setdefault(side, {})[name] = value
        return CalibrationBundle(mapper_from_arrays(data), eye_states)

    def exists(self, schema_id: SchemaId, profile: HardwareProfile) -> bool:
        fp = profile_fingerprint(profile)
        return self._path_for(schema_id, fp).exists()

    def compatibility(
        self, schema_id: SchemaId, profile: HardwareProfile
    ) -> CalibrationCompatibility | None:
        """Return compatibility info, or None if no calibration file exists."""
        fp = profile_fingerprint(profile)
        path = self._path_for(schema_id, fp)
        if not path.exists():
            return None
        return _compatibility(dict(np.load(path, allow_pickle=False)), profile)

    def list_all(self) -> list[Path]:
        if not self.store_dir.exists():
            return []
        return sorted(self.store_dir.glob(f"*{_SUFFIX}"))

    def delete(self, schema_id: SchemaId, profile: HardwareProfile) -> bool:
        """Delete the calibration file.  Returns True if a file was deleted."""
        fp = profile_fingerprint(profile)
        path = self._path_for(schema_id, fp)
        if path.exists():
            path.unlink()
            return True
        return False


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _compatibility(data: dict[str, Any], profile: HardwareProfile) -> CalibrationCompatibility:
    stored_snap: dict[str, Any] = json.loads(bytes(data["geometry_snapshot"]).decode())
    compat = check_calibration_compatibility(stored_snap, profile)
    if "tracking_snapshot" not in data:  # written before Phase 5: Mode A ridge only
        return compat
    stored_tracking = json.loads(bytes(data["tracking_snapshot"]).decode())
    schema_id = bytes(data["schema_id"]).decode()
    changes = [
        GeometryChange(f"tracking.{c.path}", c.before, c.after)
        for c in diff_snapshots(stored_tracking, tracking_snapshot(schema_id, profile))
    ]
    all_changes = (*compat.changes, *changes)
    return CalibrationCompatibility(not all_changes, all_changes)
