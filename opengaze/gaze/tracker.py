"""GazeTracker: the per-frame vision front end shared by every UI, for Modes A, B and C.

Each ``step()`` reads the newest frame of every camera, runs the stages the profile calls
for, and returns a ``TrackerFrame``: the feature vector for the profile's schema, each eye's
open/closed state (for gestures), and preview images.

* Mode A: MediaPipe on the head camera gives both the head pose and iris features.
* Modes B/C: each eye camera goes through its own eye pipeline (``gaze.pipeline``:
  ``pupil_2d`` or ``eye_3d``); MediaPipe on the head camera, if there is one, gives head pose.

A camera whose newest frame was already processed is not processed again; its last result is
reused, which also keeps duplicate frames out of the 3D eye model's fit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from opengaze.cameras.types import CameraFrame, CameraRole, PixelFormat
from opengaze.config.schema import AppConfig, HardwareProfile
from opengaze.gaze.features import build_eye_features
from opengaze.gaze.pipelines import EyePipeline, EyeResult, EyeSide, make_eye_pipeline
from opengaze.modes import TrackingMode, infer_mode, profile_schema
from opengaze.vision.feature_extractor import build_mode_a
from opengaze.vision.observations import EyeObservation, FeatureVector, HeadObservation
from opengaze.vision.preprocessing import FrameNormalizer

log = logging.getLogger(__name__)

_ROLE_SIDE: dict[CameraRole, EyeSide] = {
    CameraRole.LEFT_EYE: "left",
    CameraRole.RIGHT_EYE: "right",
}


@dataclass(eq=False)
class TrackerFrame:
    features: FeatureVector | None  # None until any camera has delivered a frame
    head: HeadObservation | None
    left: EyeObservation | None
    right: EyeObservation | None
    # None = that eye isn't observed (no camera / face lost), which is not the same as open.
    left_closed: bool | None
    right_closed: bool | None
    head_image: np.ndarray | None  # BGR, normalised, with iris overlay in Mode A
    eye_images: dict[str, np.ndarray] = field(default_factory=dict)  # side → BGR with overlay
    eye_results: dict[str, EyeResult] = field(default_factory=dict)  # Modes B/C only


class GazeTracker:
    def __init__(
        self,
        profile: HardwareProfile,
        config: AppConfig,
        *,
        face_detector: object | None = None,
    ) -> None:
        self.profile = profile
        self.mode = infer_mode(cam.role for cam in profile.enabled_cameras.values())
        self.schema_id = profile_schema(profile)
        self.normalizer = FrameNormalizer()

        enabled = profile.enabled_cameras
        self.head_ids = [cid for cid, cam in enabled.items() if cam.role is CameraRole.HEAD]
        self.eye_ids: dict[EyeSide, str] = {
            _ROLE_SIDE[cam.role]: cid for cid, cam in enabled.items() if cam.role in _ROLE_SIDE
        }
        self.pipelines: dict[EyeSide, EyePipeline] = {
            side: make_eye_pipeline(profile.gaze) for side in self.eye_ids
        }

        if face_detector is not None:
            self.face_detector, self._owns_face = face_detector, False
        elif self.head_ids:
            self.face_detector, self._owns_face = make_face_detector(config), True
        else:
            self.face_detector, self._owns_face = None, False

        self._last_seq: dict[str, tuple[int, int]] = {}
        self._head_cache: tuple[HeadObservation | None, list[EyeObservation], np.ndarray] | None
        self._head_cache = None
        self._eye_cache: dict[EyeSide, tuple[EyeResult, np.ndarray]] = {}

    @property
    def is_mode_a(self) -> bool:
        return self.mode in (None, TrackingMode.WEBCAM_ONLY)

    # ------------------------------------------------------------------

    def _fresh(self, cid: str, frame: CameraFrame) -> bool:
        key = (frame.sequence, frame.host_receive_ns)
        if self._last_seq.get(cid) == key:
            return False
        self._last_seq[cid] = key
        return True

    def step(self, manager: object) -> TrackerFrame:
        head: HeadObservation | None = None
        mp_eyes: list[EyeObservation] = []
        head_image: np.ndarray | None = None
        any_frame = False

        for cid in self.head_ids:
            frame = manager.latest(cid)  # type: ignore[attr-defined]
            if frame is None:
                continue
            any_frame = True
            if self._fresh(cid, frame) or self._head_cache is None:
                frame = self.normalizer(frame)
                h, eyes = None, []
                if self.face_detector is not None:
                    try:
                        h, eyes = self.face_detector.detect(frame)  # type: ignore[attr-defined]
                    except Exception as exc:
                        log.debug("face detection failed: %s", exc)
                image = _draw_iris(_to_bgr(frame), eyes if self.is_mode_a else [])
                self._head_cache = (h, eyes, image)
            head, mp_eyes, head_image = self._head_cache
            break  # one head camera

        if self.is_mode_a:
            left = next((e for e in mp_eyes if e.eye == "left"), None)
            right = next((e for e in mp_eyes if e.eye == "right"), None)
            fv = build_mode_a(head, left, right) if any_frame else None
            return TrackerFrame(
                fv,
                head,
                left,
                right,
                left.blink if left is not None else None,
                right.blink if right is not None else None,
                head_image,
            )

        results: dict[str, EyeResult] = {}
        eye_images: dict[str, np.ndarray] = {}
        for side, cid in self.eye_ids.items():
            frame = manager.latest(cid)  # type: ignore[attr-defined]
            if frame is None:
                continue
            any_frame = True
            if self._fresh(cid, frame) or side not in self._eye_cache:
                frame = self.normalizer(frame)
                result = self.pipelines[side].process(
                    _to_gray(frame),
                    eye=side,
                    camera_id=cid,
                    t_ns=frame.timestamp_ns,
                    t_uncertainty_ns=frame.timestamp_uncertainty_ns,
                )
                self._eye_cache[side] = (result, _draw_pupil(_to_bgr(frame), result))
            results[side], eye_images[side] = self._eye_cache[side]

        left_r, right_r = results.get("left"), results.get("right")
        fv = build_eye_features(self.schema_id, head, left_r, right_r) if any_frame else None
        return TrackerFrame(
            fv,
            head,
            left_r.observation() if left_r else None,
            right_r.observation() if right_r else None,
            left_r.closed if left_r else None,
            right_r.closed if right_r else None,
            head_image,
            eye_images,
            results,
        )

    def refeature(self, frame: TrackerFrame) -> FeatureVector | None:
        """Rebuild a past frame's features under the current eye-model state.

        Calibration calls this after freezing the eye models so that every sample is
        expressed in the same, final eyeball geometry. Mode A features are returned as is.
        """
        if self.is_mode_a or frame.features is None:
            return frame.features
        results = {
            side: self.pipelines[side].reproject(r)  # type: ignore[index]
            for side, r in frame.eye_results.items()
        }
        fv = build_eye_features(
            self.schema_id, frame.head, results.get("left"), results.get("right")
        )
        return replace(fv, t_ns=frame.features.t_ns)

    # ------------------------------------------------------------------
    # Learned eye-model state (eye_3d): freeze after calibration, persist with it
    # ------------------------------------------------------------------

    def freeze(self, frozen: bool = True) -> None:
        for p in self.pipelines.values():
            p.freeze(frozen)

    def eye_states(self) -> dict[str, dict[str, np.ndarray]]:
        return {s: st for s, p in self.pipelines.items() if (st := p.state_arrays())}

    def load_eye_states(self, states: dict[str, dict[str, np.ndarray]]) -> None:
        for side, state in states.items():
            if side in self.pipelines:
                self.pipelines[side].load_state_arrays(state)  # type: ignore[index]

    def close(self) -> None:
        if self._owns_face and self.face_detector is not None:
            self.face_detector.close()  # type: ignore[attr-defined]
            self.face_detector = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_face_detector(config: AppConfig) -> object | None:
    """MediaPipe FaceLandmarker if enabled, installed and downloaded; otherwise None."""
    if not config.vision.enable_face_tracking:
        return None
    try:
        from opengaze.vision.mediapipe_detector import FaceLandmarkDetector, default_model_path

        path = config.vision.face_landmarker_model_path or default_model_path()
        if not path.exists():
            log.warning("MediaPipe model not found — run 'opengaze download-models'")
            return None
        return FaceLandmarkDetector(path)
    except ImportError:
        log.warning("mediapipe not installed; no face tracking / head pose")
        return None
    except Exception as exc:
        log.error("could not create face detector: %s", exc)
        return None


def _to_bgr(frame: CameraFrame) -> np.ndarray:
    if frame.pixel_format is PixelFormat.GRAY8:
        return cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
    if frame.pixel_format is PixelFormat.RGB8:
        return cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
    return frame.image


def _to_gray(frame: CameraFrame) -> np.ndarray:
    if frame.pixel_format is PixelFormat.GRAY8:
        return frame.image
    code = cv2.COLOR_RGB2GRAY if frame.pixel_format is PixelFormat.RGB8 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(frame.image, code)


def _draw_iris(image: np.ndarray, eyes: list[EyeObservation]) -> np.ndarray:
    if not eyes:
        return image
    image = image.copy()
    h, w = image.shape[:2]
    for eye in eyes:
        if eye.blink:
            continue
        cx = round(eye.pupil_x_norm * w)
        cy = round(eye.pupil_y_norm * h)
        r = max(4, round(eye.pupil_radius_norm * min(w, h)))
        cv2.circle(image, (cx, cy), r, (80, 200, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(image, (cx, cy), (80, 200, 255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
    return image


def _draw_pupil(image: np.ndarray, result: EyeResult) -> np.ndarray:
    d = result.detection
    image = image.copy()
    if d is None:
        return image
    color = (90, 90, 230) if result.closed else (80, 200, 255)
    center = (round(d.center_x), round(d.center_y))
    axes = (max(1, round(d.minor_px / 2)), max(1, round(d.major_px / 2)))
    cv2.ellipse(image, center, axes, d.minor_axis_angle_deg, 0, 360, color, 2, cv2.LINE_AA)
    g = result.gaze
    if g is not None:
        ex, ey = g.eye_center_px
        cv2.circle(image, (round(ex), round(ey)), round(g.eye_radius_px), (60, 160, 60), 1)
        tip = (round(d.center_x + 40 * g.direction[0]), round(d.center_y + 40 * g.direction[1]))
        cv2.arrowedLine(image, center, tip, (60, 220, 60), 2, cv2.LINE_AA)
    return image
