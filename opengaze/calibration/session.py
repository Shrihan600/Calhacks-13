"""Calibration session: collect FeatureVector samples at known screen targets.

A session is mode-agnostic — the caller feeds FeatureVectors and screen coords.
The UI (calibration_ui.py) drives the flow; this module handles only data collection
and model fitting so it can be tested without a display.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from opengaze.calibration.model import GazeModel, ScreenMapper
from opengaze.vision.observations import FEATURE_LAYOUTS, FeatureVector, SchemaId

MapperFactory = Callable[[], ScreenMapper]

# Default 9-point grid: (x_norm, y_norm) covering the screen evenly.
GRID_9: tuple[tuple[float, float], ...] = (
    (0.05, 0.05), (0.5, 0.05), (0.95, 0.05),
    (0.05, 0.5),  (0.5, 0.5),  (0.95, 0.5),
    (0.05, 0.95), (0.5, 0.95), (0.95, 0.95),
)

# Calibration grid min/max used to remap model output to full [0,1] screen range.
GRID_LO: float = 0.05
GRID_HI: float = 0.95


def _has_eye(fv: FeatureVector) -> bool:
    layout = FEATURE_LAYOUTS.get(fv.schema_id)
    if layout is None or layout.width != len(fv.valid):
        return bool(fv.valid.any())
    return any(bool(fv.valid[list(cols)].all()) for cols in layout.eye_values.values())


@dataclass
class CalibrationPoint:
    """One target dot: its position and all raw samples collected while the user looked at it."""

    screen_x_norm: float
    screen_y_norm: float
    samples: list[FeatureVector] = field(default_factory=list)
    accepted: bool = False  # set True after the point passes quality check

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def _valid_samples(self) -> list[FeatureVector]:
        """Samples with at least one eye feature: head pose alone (e.g. an IMU that is always
        valid) says nothing about where the user was looking."""
        return [s for s in self.samples if _has_eye(s)]

    def representative(self) -> np.ndarray | None:
        """Per-feature median over the samples where that feature was valid, as a 1-D float32
        array (0.0 where it never was), or None if there are too few usable samples.

        Masking matters: a blink frame holds 0.0 in its eye slots, which would otherwise
        drag the median toward zero.
        """
        valid = self._valid_samples()
        if len(valid) < 3:
            return None
        values = np.stack([s.values for s in valid]).astype(np.float64)  # (n, features)
        mask = np.stack([s.valid for s in valid])
        masked = np.where(mask, values, np.nan)
        out = np.zeros(values.shape[1], dtype=np.float32)
        cols = mask.any(axis=0)
        out[cols] = np.nanmedian(masked[:, cols], axis=0)
        return out

    def representative_valid(self) -> np.ndarray | None:
        """Which features of ``representative()`` are trustworthy (valid in ≥ half the samples)."""
        valid = self._valid_samples()
        if len(valid) < 3:
            return None
        return np.stack([s.valid for s in valid]).mean(axis=0) >= 0.5

    def mean_confidence(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.mean([s.valid.mean() for s in self.samples]))


@dataclass
class SessionResult:
    """Output of a completed calibration session."""

    model: ScreenMapper
    points: list[CalibrationPoint]
    schema_id: SchemaId
    completed_at_ns: int
    mean_train_error_norm: float    # normalised screen units
    mean_train_error_px: float | None  # pixels, if screen size was provided


class CalibrationSession:
    """Collects samples for one calibration run then fits a GazeModel.

    Typical flow::

        session = CalibrationSession(schema_id="A-webcam")
        for i, (sx, sy) in enumerate(session.targets):
            session.start_point(i)
            # ... call session.add_sample(fv) for each webcam frame ...
            session.finish_point(i)
        result = session.fit()
    """

    def __init__(
        self,
        schema_id: SchemaId,
        targets: tuple[tuple[float, float], ...] = GRID_9,
        min_samples_per_point: int = 15,
        collect_duration_s: float = 1.5,
        settle_time_s: float = 0.0,
        alpha: float = 1.0,
        mapper_factory: MapperFactory | None = None,
    ) -> None:
        """``mapper_factory`` builds the model to fit (default: Mode A ridge ``GazeModel``)."""
        self.schema_id = schema_id
        self.targets = targets
        self.min_samples_per_point = min_samples_per_point
        self.collect_duration_s = collect_duration_s
        self.settle_time_s = settle_time_s  # skip this many seconds after dot appears
        self._alpha = alpha
        self._mapper_factory = mapper_factory or (
            lambda: GazeModel(schema_id=self.schema_id, alpha=self._alpha)
        )
        self.points: list[CalibrationPoint] = [
            CalibrationPoint(sx, sy) for sx, sy in targets
        ]
        self._current_index: int | None = None
        self._collect_start: float | None = None

    # ------------------------------------------------------------------
    # Sample ingestion
    # ------------------------------------------------------------------

    def start_point(self, index: int) -> None:
        self._current_index = index
        self._collect_start = time.monotonic()
        self.points[index].samples.clear()

    def add_sample(self, fv: FeatureVector) -> bool:
        """Add a sample to the current point.  Returns True if still collecting."""
        if self._current_index is None or self._collect_start is None:
            return False
        elapsed = time.monotonic() - self._collect_start
        total = self.settle_time_s + self.collect_duration_s
        if elapsed > total:
            return False
        if elapsed < self.settle_time_s:
            return True  # eye still moving to the dot — discard frame, keep timer running
        if fv.schema_id != self.schema_id:
            return True  # wrong schema — skip silently, keep collecting
        self.points[self._current_index].samples.append(fv)
        return True

    def finish_point(self, index: int) -> bool:
        """Mark the point done.  Returns True if it has enough good samples."""
        pt = self.points[index]
        ok = pt.n_samples >= self.min_samples_per_point and pt.representative() is not None
        pt.accepted = ok
        self._current_index = None
        return ok

    @property
    def collecting(self) -> bool:
        if self._current_index is None or self._collect_start is None:
            return False
        total = self.settle_time_s + self.collect_duration_s
        return (time.monotonic() - self._collect_start) <= total

    @property
    def collect_progress(self) -> float:
        """0.0 → 1.0 fill fraction for the dwell ring (covers settle + collect window)."""
        if self._current_index is None or self._collect_start is None:
            return 0.0
        elapsed = time.monotonic() - self._collect_start
        total = self.settle_time_s + self.collect_duration_s
        return min(elapsed / total, 1.0)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, screen_width_px: int | None = None, screen_height_px: int | None = None) -> SessionResult:
        """Fit a GazeModel from the collected samples.

        Raises ValueError if fewer than 4 accepted points (can't generalise).
        """
        good = [(i, p) for i, p in enumerate(self.points) if p.accepted]
        if len(good) < 4:
            raise ValueError(
                f"only {len(good)} accepted calibration point(s); need at least 4"
            )

        X, y, valid = stack_points([p for _, p in good])
        model = self._mapper_factory()
        model.fit(X, y, valid)

        # Training error (leave-nothing-out — just a sanity check).
        preds = np.array([model.predict_raw(x, v) for x, v in zip(X, valid, strict=True)])
        errors = np.linalg.norm(preds - y, axis=1)
        mean_err_norm = float(errors.mean())

        mean_err_px = None
        if screen_width_px and screen_height_px:
            px_errors = np.sqrt(
                (errors[:, 0] * screen_width_px) ** 2  # type: ignore[index]
                + (errors[:, 1] * screen_height_px) ** 2  # type: ignore[index]
            ) if False else (  # errors is 1-D after linalg.norm
                np.sqrt(
                    ((preds[:, 0] - y[:, 0]) * screen_width_px) ** 2
                    + ((preds[:, 1] - y[:, 1]) * screen_height_px) ** 2
                )
            )
            mean_err_px = float(px_errors.mean())

        return SessionResult(
            model=model,
            points=self.points,
            schema_id=self.schema_id,
            completed_at_ns=time.time_ns(),
            mean_train_error_norm=mean_err_norm,
            mean_train_error_px=mean_err_px,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Literal["idle", "collecting", "done"]:
        if self._current_index is not None:
            return "collecting"
        if all(p.accepted for p in self.points):
            return "done"
        return "idle"

    @property
    def n_accepted(self) -> int:
        return sum(1 for p in self.points if p.accepted)

    @property
    def n_targets(self) -> int:
        return len(self.points)


def stack_points(
    points: list[CalibrationPoint],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, valid) from points that have a representative sample."""
    X_rows, y_rows, v_rows = [], [], []
    for pt in points:
        rep, rep_valid = pt.representative(), pt.representative_valid()
        if rep is None or rep_valid is None:
            continue
        X_rows.append(rep)
        v_rows.append(rep_valid)
        y_rows.append([pt.screen_x_norm, pt.screen_y_norm])
    return (
        np.stack(X_rows).astype(np.float32),
        np.array(y_rows, dtype=np.float32),
        np.stack(v_rows),
    )
