"""Detects face landmarks, head pose, and iris positions using MediaPipe FaceLandmarker.

Requires the ``tracking`` extra:  pip install -e '.[tracking]'

The FaceLandmarker model (~5 MB) is downloaded automatically to
``~/.cache/opengaze/face_landmarker.task`` on first use, or you can supply a
custom path.  Run ``opengaze download-models`` to pre-fetch it.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from opengaze.cameras.types import CameraFrame, PixelFormat
from opengaze.vision.observations import EyeObservation, HeadObservation

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_MODEL_CACHE = Path.home() / ".cache" / "opengaze"
_MODEL_FILENAME = "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# MediaPipe FaceMesh landmark indices used here.
_IRIS_LEFT = 468
_IRIS_RIGHT = 473
_EYE_OUTER_LEFT = 33
_EYE_OUTER_RIGHT = 263

_BLINK_LEFT = "eyeBlinkLeft"
_BLINK_RIGHT = "eyeBlinkRight"
_BLINK_THRESHOLD = 0.4  # blendshape score above this counts as a blink


def default_model_path() -> Path:
    return _MODEL_CACHE / _MODEL_FILENAME


def ensure_model(model_path: Path | None = None) -> Path:
    """Return the model path, downloading it from the MediaPipe CDN if absent."""
    path = model_path or default_model_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s to %s …", _MODEL_FILENAME, path)

    def _progress(count: int, block: int, total: int) -> None:
        if total > 0:
            pct = min(100, count * block * 100 // total)
            print(f"\r  {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(_MODEL_URL, path, reporthook=_progress)
    print()
    log.info("downloaded %.1f MB", path.stat().st_size / 1e6)
    return path


class FaceLandmarkDetector:
    """Detects head pose and iris positions from a normalised CameraFrame.

    Uses MediaPipe FaceLandmarker in VIDEO mode for temporal smoothing.  Frames
    must be passed in monotonically increasing timestamp order.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.core import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for face tracking. "
                "Install with:  pip install -e '.[tracking]'"
            ) from exc

        resolved = ensure_model(model_path)
        self._mp = mp
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved)),
            running_mode=RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms: int = -1

    def detect(
        self, frame: CameraFrame
    ) -> tuple[HeadObservation | None, list[EyeObservation]]:
        """Run detection on a normalised frame; return (head, [left_eye, right_eye]).

        Returns (None, []) when no face is detected.
        """
        rgb = _to_rgb(frame)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(frame.timestamp_ns // 1_000_000, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return _parse_result(result, frame)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceLandmarkDetector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_rgb(frame: CameraFrame) -> np.ndarray:
    import cv2

    img = frame.image
    if frame.pixel_format is PixelFormat.BGR8:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif frame.pixel_format is PixelFormat.GRAY8:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return np.ascontiguousarray(img)


def _parse_result(
    result: object,
    frame: CameraFrame,
) -> tuple[HeadObservation | None, list[EyeObservation]]:
    if not result.face_landmarks:
        return None, []

    landmarks = result.face_landmarks[0]
    t_ns = frame.timestamp_ns
    t_unc = frame.timestamp_uncertainty_ns
    cam = frame.camera_id

    # Blink scores from blendshapes (0 = open, 1 = fully closed).
    blink_l = blink_r = 0.0
    if result.face_blendshapes:
        for shape in result.face_blendshapes[0]:
            if shape.category_name == _BLINK_LEFT:
                blink_l = shape.score
            elif shape.category_name == _BLINK_RIGHT:
                blink_r = shape.score

    iris_l = landmarks[_IRIS_LEFT]
    iris_r = landmarks[_IRIS_RIGHT]

    # Inter-ocular distance (normalised image units) used for radius proxy + scale.
    outer_l = landmarks[_EYE_OUTER_LEFT]
    outer_r = landmarks[_EYE_OUTER_RIGHT]
    iod = float(np.hypot(outer_l.x - outer_r.x, outer_l.y - outer_r.y))
    pupil_radius = iod * 0.07  # iris ≈ 7% of IOD (heuristic)

    eyes = [
        EyeObservation(
            camera_id=cam,
            eye="left",
            t_ns=t_ns,
            t_uncertainty_ns=t_unc,
            pupil_x_norm=float(iris_l.x),
            pupil_y_norm=float(iris_l.y),
            pupil_radius_norm=pupil_radius,
            blink=blink_l > _BLINK_THRESHOLD,
            confidence=max(0.0, 1.0 - blink_l),
            detector="mediapipe_face_landmarker",
        ),
        EyeObservation(
            camera_id=cam,
            eye="right",
            t_ns=t_ns,
            t_uncertainty_ns=t_unc,
            pupil_x_norm=float(iris_r.x),
            pupil_y_norm=float(iris_r.y),
            pupil_radius_norm=pupil_radius,
            blink=blink_r > _BLINK_THRESHOLD,
            confidence=max(0.0, 1.0 - blink_r),
            detector="mediapipe_face_landmarker",
        ),
    ]

    head: HeadObservation | None = None
    mats = result.facial_transformation_matrixes
    if mats:
        mat = np.asarray(mats[0], dtype=np.float64)
        if mat.shape == (4, 4):
            yaw, pitch, roll = _rotation_to_euler_deg(mat[:3, :3])
            cx = (float(iris_l.x) + float(iris_r.x)) / 2
            cy = (float(iris_l.y) + float(iris_r.y)) / 2
            head = HeadObservation(
                t_ns=t_ns,
                t_uncertainty_ns=t_unc,
                yaw_deg=yaw,
                pitch_deg=pitch,
                roll_deg=roll,
                face_center_x_norm=cx,
                face_center_y_norm=cy,
                relative_scale=iod,
                confidence=1.0,
                transform_4x4=mat,
            )

    return head, eyes


def _rotation_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler angles (yaw, pitch, roll) in degrees from a 3×3 rotation matrix."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:  # gimbal-lock fallback
        yaw = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0
    return float(yaw), float(pitch), float(roll)
