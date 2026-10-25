"""Per-eye pipelines: one eye-camera image in, one ``EyeResult`` out.

Two interchangeable pipelines, chosen by ``gaze.pipeline`` in the hardware profile:

* ``pupil_2d``: 2D pupil detector → pupil centre (normalised eye-image coordinates).
* ``eye_3d``:   2D pupil detector → 3D eye model → unit gaze vector (eye-camera coordinates).

Both report the same eye-closed signal (pupil confidence below a threshold), which is what
gestures use in Modes B/C. Each eye camera gets its own pipeline instance, because a 3D eye
model is specific to one eye seen from one camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol

import numpy as np

from opengaze.config.schema import GazePipelineConfig
from opengaze.gaze.detector2d import PupilDetection2D, PupilDetector2D, make_pupil_detector
from opengaze.gaze.eye_model3d import EyeModel3D, GazeVector3D, make_eye_model
from opengaze.vision.observations import EyeObservation

EyeSide = Literal["left", "right"]


@dataclass(frozen=True, slots=True, eq=False)
class EyeResult:
    eye: EyeSide
    camera_id: str
    t_ns: int
    t_uncertainty_ns: int | None
    detection: PupilDetection2D | None
    gaze: GazeVector3D | None  # eye_3d only, and only once the eye model is fitted
    closed: bool
    confidence: float  # confidence of the feature this pipeline contributes

    def observation(self) -> EyeObservation | None:
        """The detection in the common ``EyeObservation`` form (diagnostics, overlays)."""
        d = self.detection
        if d is None:
            return None
        x, y = d.center_norm
        return EyeObservation(
            camera_id=self.camera_id,
            eye=self.eye,
            t_ns=self.t_ns,
            t_uncertainty_ns=self.t_uncertainty_ns,
            pupil_x_norm=x,
            pupil_y_norm=y,
            pupil_radius_norm=d.diameter_norm / 2,
            blink=self.closed,
            confidence=d.confidence,
            detector=d.detector,
        )


class EyePipeline(Protocol):
    kind: ClassVar[str]

    def process(
        self,
        gray: np.ndarray,
        *,
        eye: EyeSide,
        camera_id: str,
        t_ns: int,
        t_uncertainty_ns: int | None = None,
    ) -> EyeResult: ...

    def reproject(self, result: EyeResult) -> EyeResult:
        """Recompute a past result's feature under the current (e.g. just-frozen) state."""
        ...

    def freeze(self, frozen: bool = True) -> None: ...

    def state_arrays(self) -> dict[str, np.ndarray]: ...

    def load_state_arrays(self, arrays: dict[str, np.ndarray]) -> None: ...


class Pupil2DPipeline:
    kind: ClassVar[str] = "pupil_2d"

    def __init__(self, detector: PupilDetector2D, closed_below_confidence: float = 0.3) -> None:
        self.detector = detector
        self.closed_below = closed_below_confidence

    def process(
        self,
        gray: np.ndarray,
        *,
        eye: EyeSide,
        camera_id: str,
        t_ns: int,
        t_uncertainty_ns: int | None = None,
    ) -> EyeResult:
        d = self.detector.detect(gray)
        conf = d.confidence if d is not None else 0.0
        return EyeResult(
            eye, camera_id, t_ns, t_uncertainty_ns, d, None, conf < self.closed_below, conf
        )

    # A 2D pipeline has no learned state.
    def reproject(self, result: EyeResult) -> EyeResult:
        return result

    def freeze(self, frozen: bool = True) -> None:
        pass

    def state_arrays(self) -> dict[str, np.ndarray]:
        return {}

    def load_state_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        pass


class Eye3DPipeline:
    kind: ClassVar[str] = "eye_3d"

    def __init__(
        self,
        detector: PupilDetector2D,
        eye_model: EyeModel3D,
        closed_below_confidence: float = 0.3,
    ) -> None:
        self.detector = detector
        self.eye_model = eye_model
        self.closed_below = closed_below_confidence

    def process(
        self,
        gray: np.ndarray,
        *,
        eye: EyeSide,
        camera_id: str,
        t_ns: int,
        t_uncertainty_ns: int | None = None,
    ) -> EyeResult:
        d = self.detector.detect(gray)
        closed = d is None or d.confidence < self.closed_below
        gaze = None
        if d is not None and not closed:
            gaze = self.eye_model.update(d, t_ns)
        conf = gaze.confidence if gaze is not None else 0.0
        return EyeResult(eye, camera_id, t_ns, t_uncertainty_ns, d, gaze, closed, conf)

    def reproject(self, result: EyeResult) -> EyeResult:
        if result.detection is None or result.closed:
            return result
        gaze = self.eye_model.project(result.detection)
        conf = gaze.confidence if gaze is not None else 0.0
        return EyeResult(
            result.eye,
            result.camera_id,
            result.t_ns,
            result.t_uncertainty_ns,
            result.detection,
            gaze,
            result.closed,
            conf,
        )

    def freeze(self, frozen: bool = True) -> None:
        self.eye_model.freeze(frozen)

    def state_arrays(self) -> dict[str, np.ndarray]:
        return self.eye_model.state_arrays()

    def load_state_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        self.eye_model.load_state_arrays(arrays)
        self.eye_model.freeze(True)


def make_eye_pipeline(config: GazePipelineConfig) -> EyePipeline:
    detector = make_pupil_detector(config.detector_2d.type, config.detector_2d.options)
    if config.pipeline == "pupil_2d":
        return Pupil2DPipeline(detector, config.closed_below_confidence)
    model = make_eye_model(config.eye_model_3d.type, config.eye_model_3d.options)
    return Eye3DPipeline(detector, model, config.closed_below_confidence)
