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
from opengaze.vision.head_pose import (
    MP_LEFT_EYE_OUTER,
    MP_LEFT_IRIS,
    MP_RIGHT_EYE_OUTER,
    MP_RIGHT_IRIS,
    estimate_from_landmarks,
    focal_px,
)
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

# Blendshapes are named after the face MediaPipe sees (like the landmarks).
_BLINK_MP_LEFT = "eyeBlinkLeft"
_BLINK_MP_RIGHT = "eyeBlinkRight"
_BLINK_THRESHOLD = 0.4  # blendshape score above this counts as a blink
# During a wink the open eye's score rises too (the cheek squints it, and the model couples
# the eyes), often past the threshold, which turns every wink into a blink. An eye counts as
# open while the other eye's score is at least this much higher.
_WINK_MARGIN = 0.25


def closed_from_scores(left: float, right: float) -> tuple[bool, bool]:
    """(left closed, right closed) from the two eyes' blink scores (0 = open, 1 = closed)."""
    return (
        left > _BLINK_THRESHOLD and right - left < _WINK_MARGIN,
        right > _BLINK_THRESHOLD and left - right < _WINK_MARGIN,
    )


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

    ``mirrored``: the head camera's profile flips the image horizontally (as the webcam
    profile does), which decides which eye is the user's left. ``hfov_deg`` sets the focal
    length when the frame carries no intrinsics; it scales the head-position estimate.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        mirrored: bool = False,
        hfov_deg: float = 60.0,
    ) -> None:
        self.mirrored = mirrored
        self.hfov_deg = hfov_deg
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision as mp_vision
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp_vision.FaceLandmarker
            FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
            RunningMode = mp_vision.RunningMode
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
        return _parse_result(result, frame, self.mirrored, self.hfov_deg)

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
    mirrored: bool,
    hfov_deg: float,
) -> tuple[HeadObservation | None, list[EyeObservation]]:
    if not result.face_landmarks:
        return None, []

    landmarks = result.face_landmarks[0]
    lm = np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float64)
    t_ns = frame.timestamp_ns
    t_unc = frame.timestamp_uncertainty_ns
    cam = frame.camera_id
    w, h = frame.width, frame.height

    if frame.intrinsics is not None:
        f, pp = frame.intrinsics.fx, (frame.intrinsics.cx, frame.intrinsics.cy)
        if mirrored:  # intrinsics describe the unmirrored image
            pp = (w - pp[0], pp[1])
    else:
        f, pp = focal_px(w, hfov_deg), None
    pose = estimate_from_landmarks(lm, w, h, mirrored=mirrored, focal_length_px=f,
                                   principal_point=pp)

    # Blink scores from blendshapes (0 = open, 1 = fully closed), named like the landmarks.
    blink_mp = {"left": 0.0, "right": 0.0}
    if result.face_blendshapes:
        for shape in result.face_blendshapes[0]:
            if shape.category_name == _BLINK_MP_LEFT:
                blink_mp["left"] = shape.score
            elif shape.category_name == _BLINK_MP_RIGHT:
                blink_mp["right"] = shape.score

    # User's eye → (MediaPipe iris index, MediaPipe side). Mirrored, MediaPipe's right is the
    # user's left.
    if mirrored:
        user_eyes = {"left": (MP_RIGHT_IRIS, "right"), "right": (MP_LEFT_IRIS, "left")}
    else:
        user_eyes = {"left": (MP_LEFT_IRIS, "left"), "right": (MP_RIGHT_IRIS, "right")}

    outer_l, outer_r = landmarks[MP_RIGHT_EYE_OUTER], landmarks[MP_LEFT_EYE_OUTER]
    iod = float(np.hypot(outer_l.x - outer_r.x, outer_l.y - outer_r.y))
    pupil_radius = iod * 0.07  # iris ≈ 7% of the eye-corner span (heuristic)

    closed_l, closed_r = closed_from_scores(blink_mp["left"], blink_mp["right"])
    closed_mp = {"left": closed_l, "right": closed_r}

    eyes = []
    for side, (iris_idx, mp_side) in user_eyes.items():
        iris = landmarks[iris_idx]
        score = blink_mp[mp_side]
        gaze = None
        if pose is not None and side in pose.eye_dirs:
            gaze = tuple(float(c) for c in pose.eye_dirs[side])
        eyes.append(
            EyeObservation(
                camera_id=cam,
                eye=side,  # type: ignore[arg-type]
                t_ns=t_ns,
                t_uncertainty_ns=t_unc,
                pupil_x_norm=float(iris.x),
                pupil_y_norm=float(iris.y),
                pupil_radius_norm=pupil_radius,
                blink=closed_mp[mp_side],
                confidence=max(0.0, 1.0 - score),
                detector="mediapipe_face_landmarker",
                gaze_in_head=gaze,
            )
        )

    head: HeadObservation | None = None
    if pose is not None:
        mats = result.facial_transformation_matrixes
        mat = np.asarray(mats[0], dtype=np.float64) if mats else None
        head = HeadObservation(
            t_ns=t_ns,
            t_uncertainty_ns=t_unc,
            yaw_deg=pose.yaw_deg,
            pitch_deg=pose.pitch_deg,
            roll_deg=pose.roll_deg,
            face_center_x_norm=float((outer_l.x + outer_r.x) / 2),
            face_center_y_norm=float((outer_l.y + outer_r.y) / 2),
            relative_scale=iod,
            confidence=1.0,
            transform_4x4=mat if mat is not None and mat.shape == (4, 4) else None,
            position_cm=pose.position_cm,
            rotation=pose.rotation,
            source="webcam",
        )

    return head, eyes
