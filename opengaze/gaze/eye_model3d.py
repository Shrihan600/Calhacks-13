"""3D eye models: a stream of 2D pupil ellipses in, a 3D gaze direction out.

Models are swappable behind ``EyeModel3D``. The built-in ``sphere_fit`` model is a
weak-perspective version of the Świrski/Dodgson approach that ``pye3d`` implements:

* The pupil is a disc on the surface of the eyeball. Seen from the camera, its ellipse's
  minor axis points toward the projected eyeball centre, so the centre is the least-squares
  intersection of the minor-axis lines of many ellipses.
* An ellipse with axis ratio ``minor/major = cos θ`` is a disc tilted by θ, and its centre lies
  ``R·sin θ`` from the eyeball centre, which gives the projected eyeball radius ``R``.
* The gaze direction for a new ellipse is then ``((p − c)/R, −√(1 − |p − c|²/R²))`` in eye-camera
  coordinates (x right, y down, z from the camera toward the eye), so gaze points back toward
  the camera with a negative z.

The model keeps refitting from a rolling buffer until ``freeze()``; calibrations store the
frozen state (``state_arrays``) so the gaze vectors don't drift between calibration and use.
An adapter for ``pye3d.detector_3d.Detector3D`` only has to turn its ``circle_3d.normal`` and
``confidence`` into a ``GazeVector3D`` and register here.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from opengaze.gaze.detector2d import PupilDetection2D


@dataclass(frozen=True, slots=True, eq=False)
class GazeVector3D:
    """Unit gaze direction in eye-camera coordinates (x right, y down, z into the scene)."""

    direction: np.ndarray  # float64[3], unit length
    eye_center_px: tuple[float, float]  # projected eyeball centre in the eye image
    eye_radius_px: float
    confidence: float  # 0 … 1, detection confidence × model confidence
    model_confidence: float  # 0 … 1, how well the eyeball fit is constrained
    model: str


class EyeModel3D(Protocol):
    name: ClassVar[str]

    def update(self, detection: PupilDetection2D, t_ns: int) -> GazeVector3D | None:
        """Feed one detection; return the gaze for it, or None while the model is unfitted."""
        ...

    def project(self, detection: PupilDetection2D) -> GazeVector3D | None:
        """Gaze for a detection under the current model, without learning from it."""
        ...

    def freeze(self, frozen: bool = True) -> None: ...

    @property
    def fitted(self) -> bool: ...

    def state_arrays(self) -> dict[str, np.ndarray]: ...

    def load_state_arrays(self, arrays: dict[str, np.ndarray]) -> None: ...


class SphereFitOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    buffer_size: int = Field(default=400, ge=20)
    min_observations: int = Field(default=25, ge=5)
    # Detections below this confidence never enter the fit.
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    # Near-circular ellipses carry no direction information.
    max_axis_ratio: float = Field(default=0.97, gt=0.0, lt=1.0)
    refit_every: int = Field(default=15, ge=1)
    # Spatial bins (per image axis) used to keep the buffer from being dominated by one
    # fixation. Each bin keeps at most buffer_size / bins² observations.
    bins: int = Field(default=6, ge=1)


class SphereFitEyeModel:
    name: ClassVar[str] = "sphere_fit"
    Options: ClassVar[type[BaseModel]] = SphereFitOptions

    def __init__(self, options: SphereFitOptions | None = None) -> None:
        self.options = options or SphereFitOptions()
        o = self.options
        per_bin = max(2, o.buffer_size // (o.bins * o.bins))
        self._bins: dict[tuple[int, int], collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=per_bin)
        )
        self._since_fit = 0
        self._frozen = False
        self._center: np.ndarray | None = None
        self._radius: float | None = None
        self._model_conf = 0.0

    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self._center is not None and self._radius is not None

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self, frozen: bool = True) -> None:
        self._frozen = frozen

    def reset(self) -> None:
        self._bins.clear()
        self._center = self._radius = None
        self._model_conf = 0.0
        self._frozen = False

    def update(self, detection: PupilDetection2D, t_ns: int) -> GazeVector3D | None:
        o = self.options
        if not self._frozen:
            self._observe(detection)
            self._since_fit += 1
            if self._since_fit >= o.refit_every or not self.fitted:
                self._since_fit = 0
                self._fit()
        return self.project(detection)

    def project(self, detection: PupilDetection2D) -> GazeVector3D | None:
        return self._gaze(detection) if self.fitted else None

    # ------------------------------------------------------------------

    def _observe(self, d: PupilDetection2D) -> None:
        o = self.options
        if d.confidence < o.min_confidence or d.axis_ratio > o.max_axis_ratio:
            return
        bx = min(o.bins - 1, int(d.center_x / d.image_width * o.bins))
        by = min(o.bins - 1, int(d.center_y / d.image_height * o.bins))
        nx, ny = d.minor_axis_dir
        sin_t = math.sqrt(max(0.0, 1.0 - d.axis_ratio**2))
        self._bins[(bx, by)].append((d.center_x, d.center_y, nx, ny, sin_t, d.confidence))

    def _fit(self) -> None:
        rows = [r for dq in self._bins.values() for r in dq]
        if len(rows) < self.options.min_observations:
            return
        obs = np.asarray(rows, dtype=np.float64)
        p, n, sin_t, w = obs[:, 0:2], obs[:, 2:4], obs[:, 4], obs[:, 5] * obs[:, 4]
        # Least-squares intersection of lines through p along n:
        # minimise Σ w ‖(I − nnᵀ)(c − p)‖².
        a = np.zeros((2, 2))
        b = np.zeros(2)
        for pi, ni, wi in zip(p, n, w, strict=True):
            proj = np.eye(2) - np.outer(ni, ni)
            a += wi * proj
            b += wi * proj @ pi
        if np.linalg.cond(a) > 1e6:
            return  # all minor axes parallel: centre unconstrained
        center = np.linalg.solve(a, b)
        tilted = sin_t > 0.25
        if tilted.sum() < 5:
            return
        radii = np.linalg.norm(p[tilted] - center, axis=1) / sin_t[tilted]
        radius = float(np.median(radii))
        if not math.isfinite(radius) or radius <= 0:
            return
        spread = float(np.median(np.abs(radii - radius)) / radius)
        # Conditioning: eigenvalue ratio of the line-intersection system (1 = all directions).
        eig = np.linalg.eigvalsh(a)
        isotropy = float(eig[0] / eig[1]) if eig[1] > 0 else 0.0
        self._center, self._radius = center, radius
        self._model_conf = float(
            np.clip(min(1.0, 3.0 * isotropy) * max(0.0, 1.0 - 2.0 * spread), 0.0, 1.0)
        )

    def _gaze(self, d: PupilDetection2D) -> GazeVector3D:
        assert self._center is not None and self._radius is not None
        v = (np.array([d.center_x, d.center_y]) - self._center) / self._radius
        norm = float(np.linalg.norm(v))
        if norm > 1.0:
            v = v / norm
            norm = 1.0
        direction = np.array([v[0], v[1], -math.sqrt(max(0.0, 1.0 - norm * norm))])
        direction /= np.linalg.norm(direction)
        return GazeVector3D(
            direction=direction,
            eye_center_px=(float(self._center[0]), float(self._center[1])),
            eye_radius_px=self._radius,
            confidence=float(d.confidence * self._model_conf),
            model_confidence=self._model_conf,
            model=self.name,
        )

    # ------------------------------------------------------------------
    # Persistence (stored alongside a calibration; numpy only, no pickle)
    # ------------------------------------------------------------------

    def state_arrays(self) -> dict[str, np.ndarray]:
        if not self.fitted:
            return {}
        assert self._center is not None and self._radius is not None
        return {
            "center": self._center.astype(np.float64),
            "radius": np.float64(self._radius),
            "model_confidence": np.float64(self._model_conf),
        }

    def load_state_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        self._center = np.asarray(arrays["center"], dtype=np.float64)
        self._radius = float(arrays["radius"])
        self._model_conf = float(arrays["model_confidence"])


EYE_MODELS_3D: dict[str, type] = {SphereFitEyeModel.name: SphereFitEyeModel}


def make_eye_model(type_name: str, options: dict[str, Any] | None = None) -> EyeModel3D:
    try:
        cls = EYE_MODELS_3D[type_name]
    except KeyError:
        raise ValueError(
            f"unknown 3D eye model {type_name!r}; available: {sorted(EYE_MODELS_3D)}"
        ) from None
    return cls(cls.Options.model_validate(options or {}))
