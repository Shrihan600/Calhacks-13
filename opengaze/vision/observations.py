"""Observation data types produced by vision detectors.

All x/y coordinates are normalised to [0, 1] within the frame image unless noted.
Nothing is invented: a field is ``None`` when genuinely unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np


@dataclass(frozen=True, slots=True)
class EyeObservation:
    """Pupil / iris features for one eye extracted from one frame."""

    camera_id: str
    eye: Literal["left", "right"]
    t_ns: int
    t_uncertainty_ns: int | None
    # Normalised to [0, 1] within the normalised camera image.
    pupil_x_norm: float
    pupil_y_norm: float
    pupil_radius_norm: float  # fraction of the image's shorter side
    blink: bool
    confidence: float  # 0 … 1
    detector: str  # e.g. "mediapipe_face_landmarker"


@dataclass(frozen=True, slots=True, eq=False)
class HeadObservation:
    """Head pose and face-centre position in one frame.

    ``eq=False``: the optional ``transform_4x4`` numpy array cannot be compared
    element-wise with ``==`` without raising an ambiguity error.
    """

    t_ns: int
    t_uncertainty_ns: int | None
    yaw_deg: float    # positive = turning right
    pitch_deg: float  # positive = looking up
    roll_deg: float   # positive = tilting right
    face_center_x_norm: float  # midpoint between irises, normalised [0, 1]
    face_center_y_norm: float
    relative_scale: float  # inter-ocular distance / reference; 1.0 at calibration distance
    confidence: float  # 0 … 1
    transform_4x4: np.ndarray | None  # 4×4 head-to-camera transform from MediaPipe, if available


# Canonical order for Mode A. Calibration models are trained to this fixed layout.
FEATURE_NAMES_MODE_A: tuple[str, ...] = (
    "left_pupil_x",
    "left_pupil_y",
    "left_pupil_conf",
    "right_pupil_x",
    "right_pupil_y",
    "right_pupil_conf",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "head_center_x",
    "head_center_y",
    "head_scale",
)


# One schema per mode × eye pipeline. A calibration is fitted for exactly one of these.
SchemaId = Literal["A-webcam", "B-pupil2d", "C-pupil2d", "B-eye3d", "C-eye3d"]
SCHEMA_IDS: tuple[str, ...] = get_args(SchemaId)

_HEAD_NAMES = (
    "head_yaw",
    "head_pitch",
    "head_roll",
    "head_center_x",
    "head_center_y",
    "head_scale",
)

# Modes B/C, 2D pupil pipeline: pupil centre in the normalised eye-camera image. Same layout
# as Mode A, but the pupil coordinates come from a different image, so models never transfer.
FEATURE_NAMES_PUPIL_2D: tuple[str, ...] = FEATURE_NAMES_MODE_A

# Modes B/C, 3D eye-model pipeline: unit gaze vector in eye-camera coordinates.
FEATURE_NAMES_EYE_3D: tuple[str, ...] = (
    "left_gaze_x",
    "left_gaze_y",
    "left_gaze_z",
    "left_gaze_conf",
    "right_gaze_x",
    "right_gaze_y",
    "right_gaze_z",
    "right_gaze_conf",
    *_HEAD_NAMES,
)


@dataclass(frozen=True, slots=True)
class FeatureLayout:
    """Where each eye's values, each eye's confidence, and the head pose live in a vector."""

    names: tuple[str, ...]
    eye_values: dict[str, tuple[int, ...]]  # "left"/"right" -> value columns (no confidence)
    eye_conf: dict[str, int]
    head: tuple[int, ...]  # yaw, pitch, roll, center x, center y, scale
    eye_kind: Literal["position", "direction"]

    @property
    def width(self) -> int:
        return len(self.names)


_LAYOUT_2D = FeatureLayout(
    names=FEATURE_NAMES_PUPIL_2D,
    eye_values={"left": (0, 1), "right": (3, 4)},
    eye_conf={"left": 2, "right": 5},
    head=tuple(range(6, 12)),
    eye_kind="position",
)
_LAYOUT_3D = FeatureLayout(
    names=FEATURE_NAMES_EYE_3D,
    eye_values={"left": (0, 1, 2), "right": (4, 5, 6)},
    eye_conf={"left": 3, "right": 7},
    head=tuple(range(8, 14)),
    eye_kind="direction",
)

FEATURE_LAYOUTS: dict[str, FeatureLayout] = {
    "A-webcam": _LAYOUT_2D,
    "B-pupil2d": _LAYOUT_2D,
    "C-pupil2d": _LAYOUT_2D,
    "B-eye3d": _LAYOUT_3D,
    "C-eye3d": _LAYOUT_3D,
}


@dataclass(frozen=True, slots=True, eq=False)
class FeatureVector:
    """Fixed-length feature sample ready for a calibration or gaze model.

    ``valid[i]`` is ``False`` when that feature was unavailable (blink, no face
    detected, missing camera). Models must consult the validity mask before using
    a value; invalid slots hold 0.0.

    ``eq=False``: contains numpy arrays.
    """

    t_ns: int
    values: np.ndarray  # float32[N]
    valid: np.ndarray   # bool[N]
    schema_id: SchemaId
    max_skew_ns: int    # time gap between the newest and oldest contributing observation
    head_t_ns: int | None
    left_eye_t_ns: int | None
    right_eye_t_ns: int | None
