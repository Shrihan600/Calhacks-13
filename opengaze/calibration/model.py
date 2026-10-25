"""Gaze models ("screen mappers"): fit feature vectors to screen coordinates.

Every mapper implements ``ScreenMapper``. ``GazeModel`` (linear ridge regression) is the
Mode A model — fast, numerically stable, and reasonable for the ~9–18 samples a calibration
session collects. Modes B/C use the mappers in ``calibration/mappers.py``.
A mapper refuses vectors whose schema_id doesn't match what it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from opengaze.vision.observations import FEATURE_LAYOUTS, SchemaId

# Features used for each schema (indices into the FeatureVector.values array).
# Mode A: 12 features — left pupil x/y/conf, right pupil x/y/conf,
#         head yaw/pitch/roll, head center x/y, head scale.
SCHEMA_FEATURE_INDICES: dict[str, list[int]] = {
    sid: list(range(layout.width)) for sid, layout in FEATURE_LAYOUTS.items()
}


@dataclass(frozen=True, slots=True)
class GazeSample:
    """Model output: predicted gaze point on the screen."""

    t_ns: int
    screen_x_norm: float  # [0, 1], 0 = left edge
    screen_y_norm: float  # [0, 1], 0 = top edge
    confidence: float     # 0 … 1; derived from input feature validity fraction
    schema_id: SchemaId
    # Unclipped prediction: outside [0, 1] when looking past the screen edge (glances).
    raw_x_norm: float | None = None
    raw_y_norm: float | None = None


class ScreenMapper:
    """Common interface: fit on representative feature rows, predict screen positions.

    Subclasses implement ``fit``, ``predict_xy``, ``to_arrays`` and ``from_arrays``, and set
    ``kind`` (stored in calibration files to pick the class when loading).
    """

    kind: ClassVar[str]
    schema_id: SchemaId

    def fit(self, X: np.ndarray, y: np.ndarray, valid: np.ndarray | None = None) -> None:
        raise NotImplementedError

    def predict_xy(
        self, x: np.ndarray, valid: np.ndarray | None = None
    ) -> tuple[float, float] | None:
        """Unclipped prediction, or None when the valid features can't support one."""
        raise NotImplementedError

    def predict_raw(self, x: np.ndarray, valid: np.ndarray | None = None) -> tuple[float, float]:
        xy = self.predict_xy(x, valid)
        if xy is None:
            raise ValueError("not enough valid features to predict")
        return float(np.clip(xy[0], 0.0, 1.0)), float(np.clip(xy[1], 0.0, 1.0))

    def confidence(self, fv: object) -> float:
        from opengaze.vision.observations import FeatureVector

        assert isinstance(fv, FeatureVector)
        return float(fv.valid.mean())

    def predict(self, fv: object) -> GazeSample:
        """Predict from a FeatureVector.  Checks schema_id."""
        from opengaze.vision.observations import FeatureVector

        assert isinstance(fv, FeatureVector)
        if fv.schema_id != self.schema_id:
            raise ValueError(
                f"model trained on {self.schema_id!r}, got vector with {fv.schema_id!r}"
            )
        xy = self.predict_xy(fv.values, fv.valid)
        if xy is None:
            return GazeSample(fv.t_ns, 0.5, 0.5, 0.0, self.schema_id)
        return GazeSample(
            t_ns=fv.t_ns,
            screen_x_norm=float(np.clip(xy[0], 0.0, 1.0)),
            screen_y_norm=float(np.clip(xy[1], 0.0, 1.0)),
            confidence=self.confidence(fv),
            schema_id=self.schema_id,
            raw_x_norm=float(xy[0]),
            raw_y_norm=float(xy[1]),
        )

    def to_arrays(self) -> dict[str, np.ndarray | str | float]:
        raise NotImplementedError

    @classmethod
    def from_arrays(cls, arrays: dict) -> ScreenMapper:
        raise NotImplementedError


def decode_str(raw: object) -> str:
    """A string stored in an .npz (np.load may return a 0-d ndarray of bytes)."""
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    return raw.decode() if isinstance(raw, (bytes, np.bytes_)) else str(raw)


class GazeModel(ScreenMapper):
    """Ridge regression from FeatureVector → (screen_x_norm, screen_y_norm).

    Usage::

        model = GazeModel(schema_id="A-webcam")
        model.fit(feature_matrix, target_xy)   # arrays, not FeatureVector objects
        sample = model.predict(feature_vector)
    """

    kind: ClassVar[str] = "ridge"

    def __init__(self, schema_id: SchemaId, alpha: float = 1.0) -> None:
        self.schema_id = schema_id
        self.alpha = alpha
        self._fitted = False
        self._coef: np.ndarray | None = None    # shape (2, n_features)
        self._intercept: np.ndarray | None = None  # shape (2,)
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._feature_indices = SCHEMA_FEATURE_INDICES[schema_id]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, valid: np.ndarray | None = None) -> None:
        """Fit on already-extracted feature matrix (``valid`` is ignored: invalid slots are 0).

        Args:
            X: float32 array of shape (n_samples, n_features).
               Only the columns in _feature_indices matter; pass the full 12-col
               matrix and this method selects the right columns.
            y: float32 array of shape (n_samples, 2) — screen (x_norm, y_norm).
        """
        if X.shape[0] < 2:
            raise ValueError("need at least 2 samples to fit")
        if y.shape != (X.shape[0], 2):
            raise ValueError(f"y must be (n_samples, 2), got {y.shape}")

        Xs = X[:, self._feature_indices].astype(np.float64)
        ys = y.astype(np.float64)

        # z-score normalise features so ridge penalty is scale-invariant.
        self._feature_mean = Xs.mean(axis=0)
        std = Xs.std(axis=0)
        std[std < 1e-9] = 1.0
        self._feature_std = std
        Xs = (Xs - self._feature_mean) / self._feature_std

        # Add bias column.
        ones = np.ones((Xs.shape[0], 1))
        Xb = np.hstack([Xs, ones])

        # Closed-form ridge: (XᵀX + αI)⁻¹ Xᵀy
        A = Xb.T @ Xb
        A[:-1, :-1] += self.alpha * np.eye(len(self._feature_indices))
        w = np.linalg.solve(A, Xb.T @ ys)  # shape (n_features+1, 2)

        self._coef = w[:-1].T        # (2, n_features)
        self._intercept = w[-1]      # (2,)
        self._fitted = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_xy(
        self, x: np.ndarray, valid: np.ndarray | None = None
    ) -> tuple[float, float] | None:
        """Predict from a raw 1-D feature array (length = full schema width)."""
        if not self._fitted:
            raise RuntimeError("model not fitted")
        assert self._coef is not None
        assert self._intercept is not None
        assert self._feature_mean is not None
        assert self._feature_std is not None

        xs = x[self._feature_indices].astype(np.float64)
        xs = (xs - self._feature_mean) / self._feature_std
        pred = self._coef @ xs + self._intercept
        return float(pred[0]), float(pred[1])

    def confidence(self, fv: object) -> float:
        return float(fv.valid[self._feature_indices].mean())  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Serialisation (numpy arrays only — no pickle)
    # ------------------------------------------------------------------

    def to_arrays(self) -> dict[str, np.ndarray | str | float]:
        if not self._fitted:
            raise RuntimeError("model not fitted")
        assert self._coef is not None
        assert self._intercept is not None
        assert self._feature_mean is not None
        assert self._feature_std is not None
        return {
            "coef": self._coef,
            "intercept": self._intercept,
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
            "schema_id": self.schema_id,
            "alpha": np.float64(self.alpha),
            "mapper_kind": self.kind,
        }

    @classmethod
    def from_arrays(cls, arrays: dict) -> GazeModel:
        schema_id = decode_str(arrays["schema_id"])
        model = cls(schema_id=schema_id, alpha=float(arrays["alpha"]))  # type: ignore[arg-type]
        model._coef = arrays["coef"]
        model._intercept = arrays["intercept"]
        model._feature_mean = arrays["feature_mean"]
        model._feature_std = arrays["feature_std"]
        model._fitted = True
        return model
