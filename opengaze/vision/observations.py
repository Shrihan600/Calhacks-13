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
    eye: Literal["left", "right"]  # the user's left / right eye
    t_ns: int
    t_uncertainty_ns: int | None
    # Normalised to [0, 1] within the normalised camera image.
    pupil_x_norm: float
    pupil_y_norm: float
    pupil_radius_norm: float  # fraction of the image's shorter side
    blink: bool
    confidence: float  # 0 … 1
    detector: str  # e.g. "mediapipe_face_landmarker"
    # Webcam only: unit gaze direction in the head frame (see vision/head_pose.py), from the
    # iris position within the eye opening, so it doesn't move when the head does.
    gaze_in_head: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True, eq=False)
class HeadObservation:
    """Head rotation and position in the world frame (see ``vision/head_pose.py``).

    ``eq=False``: the numpy arrays cannot be compared element-wise with ``==`` without
    raising an ambiguity error.
    """

    t_ns: int
    t_uncertainty_ns: int | None
    yaw_deg: float    # positive = turning right
    pitch_deg: float  # positive = looking up
    roll_deg: float   # positive = tilting right
    face_center_x_norm: float  # midpoint between the eyes in the head image, [0, 1]
    face_center_y_norm: float
    relative_scale: float  # eye-corner span in the head image, fraction of image width
    confidence: float  # 0 … 1
    transform_4x4: np.ndarray | None  # 4×4 head-to-camera transform from MediaPipe, if available
    # Head position (eye midpoint) in cm, world frame; None when unknown (e.g. IMU only).
    position_cm: tuple[float, float, float] | None = None
    rotation: np.ndarray | None = None  # 3×3 head → world, consistent with the angles
    source: Literal["webcam", "imu", "imu+webcam"] = "webcam"


HEAD_FEATURE_NAMES: tuple[str, ...] = (
    "head_yaw",
    "head_pitch",
    "head_roll",
    "head_x_cm",
    "head_y_cm",
    "head_z_cm",
)

# Mode A: eye-in-head gaze direction per eye (from iris within the eye opening) + head pose.
FEATURE_NAMES_MODE_A: tuple[str, ...] = (
    "left_gaze_x",
    "left_gaze_y",
    "left_gaze_z",
    "left_gaze_conf",
    "right_gaze_x",
    "right_gaze_y",
    "right_gaze_z",
    "right_gaze_conf",
    *HEAD_FEATURE_NAMES,
)

# Modes B/C, 2D pupil pipeline: pupil centre in the normalised eye-camera image.
FEATURE_NAMES_PUPIL_2D: tuple[str, ...] = (
    "left_pupil_x",
    "left_pupil_y",
    "left_pupil_conf",
    "right_pupil_x",
    "right_pupil_y",
    "right_pupil_conf",
    *HEAD_FEATURE_NAMES,
)

# Modes B/C, 3D eye-model pipeline: unit gaze vector in eye-camera coordinates.
FEATURE_NAMES_EYE_3D: tuple[str, ...] = FEATURE_NAMES_MODE_A

# One schema per mode × eye pipeline. A calibration is fitted for exactly one of these.
SchemaId = Literal["A-webcam", "B-pupil2d", "C-pupil2d", "B-eye3d", "C-eye3d"]
SCHEMA_IDS: tuple[str, ...] = get_args(SchemaId)
# Bump when any layout or the meaning of a feature changes; stored calibrations are tied to it.
FEATURE_LAYOUT_VERSION = 2


@dataclass(frozen=True, slots=True)
class FeatureLayout:
    """Where each eye's values, each eye's confidence, and the head pose live in a vector."""

    names: tuple[str, ...]
    eye_values: dict[str, tuple[int, ...]]  # "left"/"right" -> value columns (no confidence)
    eye_conf: dict[str, int]
    head_rot: tuple[int, ...]  # yaw, pitch, roll (degrees)
    head_pos: tuple[int, ...]  # x, y, z (cm, world frame)
    eye_kind: Literal["position", "direction"]
    # Frame the eye directions are in: the eye camera's, or the head's (webcam, Mode A).
    eye_frame: Literal["image", "eye_camera", "head"]

    @property
    def width(self) -> int:
        return len(self.names)

    @property
    def head(self) -> tuple[int, ...]:
        return self.head_rot + self.head_pos


def _layout(names, kind, frame) -> FeatureLayout:
    per_eye = 2 if kind == "position" else 3
    stride = per_eye + 1
    return FeatureLayout(
        names=names,
        eye_values={
            "left": tuple(range(per_eye)),
            "right": tuple(range(stride, stride + per_eye)),
        },
        eye_conf={"left": per_eye, "right": stride + per_eye},
        head_rot=tuple(range(2 * stride, 2 * stride + 3)),
        head_pos=tuple(range(2 * stride + 3, 2 * stride + 6)),
        eye_kind=kind,
        eye_frame=frame,
    )


_LAYOUT_2D = _layout(FEATURE_NAMES_PUPIL_2D, "position", "image")
_LAYOUT_3D = _layout(FEATURE_NAMES_EYE_3D, "direction", "eye_camera")

FEATURE_LAYOUTS: dict[str, FeatureLayout] = {
    "A-webcam": _layout(FEATURE_NAMES_MODE_A, "direction", "head"),
    "B-pupil2d": _LAYOUT_2D,
    "C-pupil2d": _LAYOUT_2D,
    "B-eye3d": _LAYOUT_3D,
    "C-eye3d": _LAYOUT_3D,
}


def head_values(head: HeadObservation) -> tuple[tuple[float, ...], bool, tuple[float, ...], bool]:
    """(rotation values, rotation valid, position values, position valid) for feature slots."""
    rot = (head.yaw_deg, head.pitch_deg, head.roll_deg)
    if head.position_cm is None:
        return rot, True, (0.0, 0.0, 0.0), False
    return rot, True, tuple(head.position_cm), True


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
