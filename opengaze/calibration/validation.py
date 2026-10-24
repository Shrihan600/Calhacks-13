"""Leave-one-out cross-validation for a fitted calibration session.

Each calibration point is held out in turn; we refit on the remaining points
and measure error at the held-out point.  This gives an honest estimate of
generalisation error without needing separate validation targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opengaze.calibration.model import GazeModel
from opengaze.calibration.session import CalibrationPoint, MapperFactory, stack_points
from opengaze.vision.observations import SchemaId


@dataclass(frozen=True, slots=True)
class PointError:
    screen_x_norm: float
    screen_y_norm: float
    pred_x_norm: float
    pred_y_norm: float
    error_norm: float           # Euclidean distance in normalised [0,1]² screen space
    error_px: float | None      # pixels, if screen size supplied


@dataclass(frozen=True, slots=True)
class ValidationReport:
    schema_id: SchemaId
    point_errors: tuple[PointError, ...]
    mean_error_norm: float
    max_error_norm: float
    mean_error_px: float | None
    max_error_px: float | None
    passed: bool                # True if mean_error_norm < threshold

    def summary(self) -> str:
        px_part = (
            f"  {self.mean_error_px:.0f} px mean / {self.max_error_px:.0f} px max"
            if self.mean_error_px is not None
            else ""
        )
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] LOO CV — mean {self.mean_error_norm:.4f} norm "
            f"/ max {self.max_error_norm:.4f} norm{px_part}"
        )


def validate_calibration(
    points: list[CalibrationPoint],
    schema_id: SchemaId,
    alpha: float = 1.0,
    pass_threshold_norm: float = 0.08,
    screen_width_px: int | None = None,
    screen_height_px: int | None = None,
    mapper_factory: MapperFactory | None = None,
) -> ValidationReport:
    """Leave-one-out cross-validation over accepted calibration points.

    Requires at least 5 accepted points (so each LOO fold still has ≥ 4).
    """
    accepted = [p for p in points if p.accepted and p.representative() is not None]

    if len(accepted) < 5:
        raise ValueError(
            f"need at least 5 accepted points for LOO CV, got {len(accepted)}"
        )

    factory = mapper_factory or (lambda: GazeModel(schema_id=schema_id, alpha=alpha))
    reps, targets, valid = stack_points(accepted)

    point_errors: list[PointError] = []
    for i in range(len(accepted)):
        mask = np.ones(len(accepted), dtype=bool)
        mask[i] = False
        y_test = targets[i]

        m = factory()
        try:
            m.fit(reps[mask], targets[mask], valid[mask])
            pred = m.predict_xy(reps[i], valid[i])
        except ValueError:
            pred = None  # this fold can't be fitted (e.g. the held-out point was needed)
        if pred is None:
            continue
        px_norm = float(np.clip(pred[0], 0.0, 1.0))
        py_norm = float(np.clip(pred[1], 0.0, 1.0))

        err_norm = float(np.sqrt((px_norm - y_test[0]) ** 2 + (py_norm - y_test[1]) ** 2))
        err_px = None
        if screen_width_px and screen_height_px:
            err_px = float(
                np.sqrt(
                    ((px_norm - y_test[0]) * screen_width_px) ** 2
                    + ((py_norm - y_test[1]) * screen_height_px) ** 2
                )
            )
        point_errors.append(
            PointError(
                screen_x_norm=float(y_test[0]),
                screen_y_norm=float(y_test[1]),
                pred_x_norm=px_norm,
                pred_y_norm=py_norm,
                error_norm=err_norm,
                error_px=err_px,
            )
        )

    if not point_errors:
        raise ValueError("no leave-one-out fold could be fitted")
    errs_norm = [e.error_norm for e in point_errors]
    errs_px = [e.error_px for e in point_errors if e.error_px is not None]

    return ValidationReport(
        schema_id=schema_id,
        point_errors=tuple(point_errors),
        mean_error_norm=float(np.mean(errs_norm)),
        max_error_norm=float(np.max(errs_norm)),
        mean_error_px=float(np.mean(errs_px)) if errs_px else None,
        max_error_px=float(np.max(errs_px)) if errs_px else None,
        passed=float(np.mean(errs_norm)) < pass_threshold_norm,
    )
