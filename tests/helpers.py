from __future__ import annotations

from pathlib import Path
from typing import Any

from opengaze.config.schema import CameraConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_DIR = REPO_ROOT / "configs" / "hardware"


def camera_config(camera_id: str = "left_eye", **fields: Any) -> CameraConfig:
    fields.setdefault(
        "role", camera_id if camera_id in ("left_eye", "right_eye", "head") else "other"
    )
    fields.setdefault("source_type", "synthetic")
    return CameraConfig(camera_id=camera_id, **fields)
