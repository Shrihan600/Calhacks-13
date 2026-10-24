"""Save and load fitted gaze models, bound to a geometry fingerprint.

Storage format: a single .npz file per calibration.
Filename encodes the schema_id and fingerprint so different modes / hardware
profiles never collide and stale calibrations are easy to spot.

Layout inside the .npz:
  coef, intercept, feature_mean, feature_std  — model weights
  schema_id                                   — byte-string scalar
  alpha                                       — float64 scalar
  geometry_snapshot                           — JSON byte-string
  completed_at_ns                             — int64 scalar
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from opengaze.calibration.model import GazeModel, SchemaId
from opengaze.config.fingerprint import (
    CalibrationCompatibility,
    check_calibration_compatibility,
    geometry_snapshot,
    profile_fingerprint,
)
from opengaze.config.schema import HardwareProfile

_STORE_DIR = Path.home() / ".config" / "opengaze" / "calibrations"
_SUFFIX = ".calib.npz"


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
        model: GazeModel,
        profile: HardwareProfile,
        completed_at_ns: int | None = None,
    ) -> Path:
        """Persist model to disk.  Returns the path written."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        fp = profile_fingerprint(profile)
        path = self._path_for(model.schema_id, fp)

        arrays = model.to_arrays()
        snap = geometry_snapshot(profile)

        np.savez_compressed(
            path,
            coef=arrays["coef"],
            intercept=arrays["intercept"],
            feature_mean=arrays["feature_mean"],
            feature_std=arrays["feature_std"],
            schema_id=np.bytes_(model.schema_id),
            alpha=np.float64(model.alpha),
            geometry_snapshot=np.bytes_(json.dumps(snap, sort_keys=True, separators=(",", ":"))),
            completed_at_ns=np.int64(completed_at_ns or time.time_ns()),
        )
        return path

    def load(
        self,
        schema_id: SchemaId,
        profile: HardwareProfile,
    ) -> GazeModel:
        """Load and return the model for this schema + profile.

        Raises FileNotFoundError if no calibration exists.
        Raises ValueError if the stored geometry snapshot is incompatible.
        """
        fp = profile_fingerprint(profile)
        path = self._path_for(schema_id, fp)
        if not path.exists():
            raise FileNotFoundError(
                f"no calibration found for schema={schema_id!r}, fingerprint={fp!r} "
                f"(expected {path})"
            )
        return self._load_path(path, profile)

    def _load_path(self, path: Path, profile: HardwareProfile) -> GazeModel:
        data = np.load(path, allow_pickle=False)
        snap_json = bytes(data["geometry_snapshot"]).decode()
        stored_snap: dict[str, Any] = json.loads(snap_json)
        compat = check_calibration_compatibility(stored_snap, profile)
        if not compat.compatible:
            raise ValueError(compat.explain())
        return GazeModel.from_arrays(dict(data))

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
        data = np.load(path, allow_pickle=False)
        snap_json = bytes(data["geometry_snapshot"]).decode()
        stored_snap: dict[str, Any] = json.loads(snap_json)
        return check_calibration_compatibility(stored_snap, profile)

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
