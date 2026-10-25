"""Adaptive calibration: a continuous stream of targets, each one a test or a training point.

Instead of a fixed grid, targets keep coming until the model is good enough:

1. Seed: the first ``n_seed`` targets (centre + corners) are always training points.
2. Every later target is first a *test*: the current model predicts where the user is
   looking. Within ``tolerance_norm`` of the target → pass, the model is unchanged. Otherwise
   the point becomes training data and the model is refitted.
3. The run stops after ``required_passes`` consecutive passes (those passes were all scored
   against the final model, so their error is an honest held-out estimate), or at
   ``max_points``.

The next target goes where the model is least supported: far from every target shown so far,
weighted toward regions where tests failed.

Every point — target, representative features, every raw sample, prediction, error and
decision — is kept in an ``AdaptiveLog`` and saved to ``calibration_logs/`` whether or not the
calibration itself is kept. ``points_needed_curve`` replays a log to show how the error falls
with the number of training points.

Like ``CalibrationSession``, this module has no timing or UI: the caller collects a
``CalibrationPoint`` for each target and hands it to ``submit``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from opengaze.calibration.model import GazeModel, ScreenMapper, decode_str
from opengaze.calibration.session import (
    GRID_HI,
    GRID_LO,
    CalibrationPoint,
    MapperFactory,
    stack_points,
)
from opengaze.vision.observations import FeatureVector, SchemaId

# Centre first, then the corners: enough to fit every mapper (poly ridge needs 5).
SEED_TARGETS: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),
    (GRID_LO, GRID_LO), (GRID_HI, GRID_LO),
    (GRID_LO, GRID_HI), (GRID_HI, GRID_HI),
)

Phase = Literal["seed", "test"]
Action = Literal["train", "pass", "reject"]

_LOG_DIR = Path.home() / ".config" / "opengaze" / "calibration_logs"
_LOG_SUFFIX = ".adaptive.npz"


@dataclass(frozen=True, slots=True)
class PointRecord:
    """What happened at one target."""

    index: int                   # order shown, 0-based (rejected attempts included)
    target_x: float
    target_y: float
    phase: Phase
    action: Action               # train = added to the model; pass = test passed; reject = bad data
    pred_x: float | None         # model prediction before this point (test phase only)
    pred_y: float | None
    error_norm: float | None
    error_px: float | None
    n_samples: int
    mean_confidence: float
    n_training_after: int        # training points in the model after this point
    t_ns: int


@dataclass
class AdaptiveResult:
    model: ScreenMapper
    log: AdaptiveLog
    converged: bool              # stopped on a pass streak, not on max_points
    final_error_norm: float | None  # mean error of the closing pass streak
    final_error_px: float | None


@dataclass
class AdaptiveLog:
    """Everything recorded during one adaptive run; saved to disk for later analysis."""

    schema_id: SchemaId
    geometry_estimation: str
    settings: dict[str, Any]
    screen_px: tuple[int, int] | None
    records: list[PointRecord] = field(default_factory=list)
    # Per record (same order): representative features + validity, or None when rejected.
    reps: list[np.ndarray | None] = field(default_factory=list)
    rep_valid: list[np.ndarray | None] = field(default_factory=list)
    # Raw samples per record, for debugging the representative/median step.
    samples: list[list[FeatureVector]] = field(default_factory=list)
    started_at_ns: int = field(default_factory=time.time_ns)
    profile_fingerprint: str | None = None

    # ------------------------------------------------------------------

    @property
    def n_training(self) -> int:
        return sum(1 for r in self.records if r.action == "train")

    def summary(self) -> str:
        tests = [r for r in self.records if r.phase == "test" and r.error_norm is not None]
        passes = sum(1 for r in tests if r.action == "pass")
        rejects = sum(1 for r in self.records if r.action == "reject")
        return (
            f"{len(self.records)} targets: {self.n_training} training, "
            f"{passes}/{len(tests)} tests passed, {rejects} rejected"
        )

    # ------------------------------------------------------------------
    # Serialisation (.npz, no pickle)
    # ------------------------------------------------------------------

    def to_arrays(self) -> dict[str, Any]:
        width = next((r.shape[0] for r in self.reps if r is not None), 0)
        n = len(self.records)
        reps = np.zeros((n, width), dtype=np.float32)
        rep_valid = np.zeros((n, width), dtype=bool)
        has_rep = np.zeros(n, dtype=bool)
        for i, (r, v) in enumerate(zip(self.reps, self.rep_valid, strict=True)):
            if r is not None and v is not None:
                reps[i], rep_valid[i], has_rep[i] = r, v, True
        flat = [(i, s) for i, group in enumerate(self.samples) for s in group]
        meta = {
            "schema_id": self.schema_id,
            "geometry_estimation": self.geometry_estimation,
            "settings": self.settings,
            "screen_px": list(self.screen_px) if self.screen_px else None,
            "started_at_ns": self.started_at_ns,
            "profile_fingerprint": self.profile_fingerprint,
            "records": [asdict(r) for r in self.records],
        }
        return {
            "meta_json": np.bytes_(json.dumps(meta, sort_keys=True)),
            "reps": reps,
            "rep_valid": rep_valid,
            "has_rep": has_rep,
            "sample_point": np.array([i for i, _ in flat], dtype=np.int32),
            "sample_t_ns": np.array([s.t_ns for _, s in flat], dtype=np.int64),
            "sample_values": (
                np.stack([s.values for _, s in flat]).astype(np.float32)
                if flat else np.zeros((0, width), dtype=np.float32)
            ),
            "sample_valid": (
                np.stack([s.valid for _, s in flat]) if flat else np.zeros((0, width), dtype=bool)
            ),
        }

    @classmethod
    def from_arrays(cls, data: dict[str, Any]) -> AdaptiveLog:
        meta = json.loads(decode_str(data["meta_json"]))
        records = [PointRecord(**r) for r in meta["records"]]
        n = len(records)
        reps = [data["reps"][i] if data["has_rep"][i] else None for i in range(n)]
        rep_valid = [data["rep_valid"][i] if data["has_rep"][i] else None for i in range(n)]
        samples: list[list[FeatureVector]] = [[] for _ in range(n)]
        for pi, t, values, valid in zip(
            data["sample_point"], data["sample_t_ns"],
            data["sample_values"], data["sample_valid"], strict=True,
        ):
            samples[int(pi)].append(FeatureVector(
                t_ns=int(t), values=values, valid=valid, schema_id=meta["schema_id"],
                max_skew_ns=0, head_t_ns=None, left_eye_t_ns=None, right_eye_t_ns=None,
            ))
        screen = meta.get("screen_px")
        return cls(
            schema_id=meta["schema_id"],
            geometry_estimation=meta["geometry_estimation"],
            settings=meta["settings"],
            screen_px=(int(screen[0]), int(screen[1])) if screen else None,
            records=records,
            reps=reps,
            rep_valid=rep_valid,
            samples=samples,
            started_at_ns=int(meta["started_at_ns"]),
            profile_fingerprint=meta.get("profile_fingerprint"),
        )

    def save(self, log_dir: Path | None = None) -> Path:
        log_dir = log_dir or _LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.started_at_ns / 1e9))
        name = f"{self.schema_id.replace('-', '_')}__{self.profile_fingerprint or 'nofp'}__{stamp}"
        path = log_dir / f"{name}{_LOG_SUFFIX}"
        np.savez_compressed(path, **self.to_arrays())
        return path

    @classmethod
    def load(cls, path: Path) -> AdaptiveLog:
        return cls.from_arrays(dict(np.load(path, allow_pickle=False)))


def list_logs(log_dir: Path | None = None) -> list[Path]:
    """Saved adaptive logs, oldest first."""
    log_dir = log_dir or _LOG_DIR
    if not log_dir.exists():
        return []
    return sorted(log_dir.glob(f"*{_LOG_SUFFIX}"), key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class AdaptiveCalibrationSession:
    """Chooses targets, scores each collected point, and refits when a test fails.

    Typical flow::

        session = AdaptiveCalibrationSession(schema_id="A-webcam")
        while not session.done:
            sx, sy = session.next_target()
            point = CalibrationPoint(sx, sy)
            # ... append FeatureVectors to point.samples while the user looks at it ...
            record = session.submit(point)
        result = session.result()
    """

    def __init__(
        self,
        schema_id: SchemaId,
        mapper_factory: MapperFactory | None = None,
        geometry_estimation: str = "data_driven",
        tolerance_norm: float = 0.08,
        required_passes: int = 4,
        max_points: int = 30,
        min_samples_per_point: int = 10,
        seed_targets: tuple[tuple[float, float], ...] = SEED_TARGETS,
        train_on_pass: bool = False,
        error_weight: float = 2.0,
        grid_steps: int = 9,
        screen_px: tuple[int, int] | None = None,
        rng_seed: int | None = None,
    ) -> None:
        """
        Args:
            tolerance_norm: a test passes when its error (normalised screen units) is at most this.
            required_passes: consecutive passes that end the run.
            max_points: hard stop, counting accepted targets (rejected attempts don't count).
            train_on_pass: also add passing points to the model (they then stop being held out).
            error_weight: how strongly failed regions attract new targets (0 = pure coverage).
            grid_steps: candidate targets per axis between GRID_LO and GRID_HI.
        """
        self.schema_id = schema_id
        self._mapper_factory = mapper_factory or (lambda: GazeModel(schema_id=schema_id))
        self.tolerance_norm = tolerance_norm
        self.required_passes = required_passes
        self.max_points = max_points
        self.min_samples_per_point = min_samples_per_point
        self.seed_targets = seed_targets
        self.train_on_pass = train_on_pass
        self.error_weight = error_weight
        self.screen_px = screen_px
        self._rng = np.random.default_rng(rng_seed)

        axis = np.linspace(GRID_LO, GRID_HI, grid_steps)
        self._candidates = np.array([(x, y) for y in axis for x in axis])

        self.training: list[CalibrationPoint] = []
        self._training_log_idx: list[int] = []  # log record of each training point
        self.model: ScreenMapper | None = None
        self.consecutive_passes = 0
        self._shown: list[tuple[float, float, float]] = []  # (x, y, badness 0…2)
        self._pending: tuple[float, float] | None = None
        self.log = AdaptiveLog(
            schema_id=schema_id,
            geometry_estimation=geometry_estimation,
            settings={
                "tolerance_norm": tolerance_norm,
                "required_passes": required_passes,
                "max_points": max_points,
                "min_samples_per_point": min_samples_per_point,
                "n_seed": len(seed_targets),
                "train_on_pass": train_on_pass,
                "error_weight": error_weight,
                "grid_steps": grid_steps,
            },
            screen_px=screen_px,
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def n_accepted(self) -> int:
        return sum(1 for r in self.log.records if r.action != "reject")

    @property
    def in_seed_phase(self) -> bool:
        return self.model is None

    @property
    def converged(self) -> bool:
        return self.model is not None and self.consecutive_passes >= self.required_passes

    @property
    def done(self) -> bool:
        return self.converged or self.n_accepted >= self.max_points

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def next_target(self) -> tuple[float, float]:
        """Target for the next point (the same one again until ``submit`` accepts it)."""
        if self._pending is not None:
            return self._pending
        n_seeded = sum(1 for r in self.log.records if r.phase == "seed" and r.action == "train")
        if n_seeded < len(self.seed_targets):
            target = self.seed_targets[n_seeded]
        else:
            target = self._pick_adaptive()
        self._pending = target
        return target

    def _pick_adaptive(self) -> tuple[float, float]:
        """Candidate farthest from every shown target, boosted near failed tests."""
        shown = np.array([(x, y) for x, y, _ in self._shown])
        badness = np.array([b for _, _, b in self._shown])
        d = np.linalg.norm(self._candidates[:, None, :] - shown[None, :, :], axis=2)
        nearest = d.min(axis=1)
        # Each failed test pulls targets toward it, fading over ~a quarter screen.
        pull = (badness[None, :] * np.exp(-((d / 0.25) ** 2))).max(axis=1)
        score = nearest * (1.0 + self.error_weight * pull)
        score[nearest < 1e-6] = -np.inf  # never repeat an exact target
        score += self._rng.uniform(0, 1e-6, score.shape)  # break ties
        x, y = self._candidates[int(np.argmax(score))]
        return float(x), float(y)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def submit(self, point: CalibrationPoint) -> PointRecord:
        """Score a collected point, train on it if needed, and log it."""
        rep, rep_valid = point.representative(), point.representative_valid()
        ok = point.n_samples >= self.min_samples_per_point and rep is not None
        phase: Phase = "seed" if self.in_seed_phase else "test"
        pred = err_norm = err_px = None

        if not ok:
            action: Action = "reject"
        elif phase == "seed":
            action = "train"
        else:
            assert self.model is not None and rep is not None
            xy = self.model.predict_xy(rep, rep_valid)
            if xy is not None:
                pred = (float(np.clip(xy[0], 0.0, 1.0)), float(np.clip(xy[1], 0.0, 1.0)))
                err_norm, err_px = self._error(pred, point)
            action = "pass" if err_norm is not None and err_norm <= self.tolerance_norm else "train"

        if action == "reject":
            pass  # keep _pending: the same target is retried
        else:
            point.accepted = True
            self._pending = None
            # How far past tolerance, capped so one bad region can't swallow the whole stream.
            badness = 0.0 if err_norm is None else min(err_norm / self.tolerance_norm - 1, 2.0)
            badness = max(badness, 0.0)
            self._shown.append((point.screen_x_norm, point.screen_y_norm, badness))
            if action == "pass":
                self.consecutive_passes += 1
                if self.train_on_pass:
                    self._add_training(point)
                    self._refit()
            else:
                self.consecutive_passes = 0
                self._add_training(point)
                if phase == "test" or len(self.training) >= len(self.seed_targets):
                    self._refit()

        record = PointRecord(
            index=len(self.log.records),
            target_x=point.screen_x_norm,
            target_y=point.screen_y_norm,
            phase=phase,
            action=action,
            pred_x=pred[0] if pred else None,
            pred_y=pred[1] if pred else None,
            error_norm=err_norm,
            error_px=err_px,
            n_samples=point.n_samples,
            mean_confidence=point.mean_confidence(),
            n_training_after=len(self.training),
            t_ns=time.time_ns(),
        )
        self.log.records.append(record)
        self.log.reps.append(rep if ok else None)
        self.log.rep_valid.append(rep_valid if ok else None)
        self.log.samples.append(list(point.samples))
        return record

    def _add_training(self, point: CalibrationPoint) -> None:
        self.training.append(point)
        self._training_log_idx.append(len(self.log.records))  # the record about to be logged

    def _error(
        self, pred: tuple[float, float], point: CalibrationPoint
    ) -> tuple[float, float | None]:
        dx, dy = pred[0] - point.screen_x_norm, pred[1] - point.screen_y_norm
        err_px = None
        if self.screen_px:
            err_px = float(np.hypot(dx * self.screen_px[0], dy * self.screen_px[1]))
        return float(np.hypot(dx, dy)), err_px

    def _refit(self) -> None:
        X, y, valid = stack_points(self.training)
        model = self._mapper_factory()
        try:
            model.fit(X, y, valid)
        except ValueError:
            return  # not enough usable data yet (e.g. one eye never valid); keep the old model
        self.model = model

    def refeature(self, fn: Callable[[CalibrationPoint], list[FeatureVector]]) -> None:
        """Replace every training point's samples with ``fn(point)`` and refit.

        Used with the 3D eye pipeline: once the eye models are frozen, earlier samples are
        re-expressed under the final eyeball geometry. The log is updated to match.
        """
        for point, li in zip(self.training, self._training_log_idx, strict=True):
            point.samples = fn(point)
            self.log.samples[li] = list(point.samples)
            self.log.reps[li] = point.representative()
            self.log.rep_valid[li] = point.representative_valid()
        self._refit()

    # ------------------------------------------------------------------

    def result(self) -> AdaptiveResult:
        if self.model is None:
            raise ValueError(f"no model: only {len(self.training)} usable training point(s)")
        accepted = [r for r in self.log.records if r.action != "reject"]
        streak = accepted[len(accepted) - self.consecutive_passes:]
        errs = [r.error_norm for r in streak if r.error_norm is not None]
        errs_px = [r.error_px for r in streak if r.error_px is not None]
        return AdaptiveResult(
            model=self.model,
            log=self.log,
            converged=self.converged,
            final_error_norm=float(np.mean(errs)) if errs else None,
            final_error_px=float(np.mean(errs_px)) if errs_px else None,
        )


# ---------------------------------------------------------------------------
# Offline analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CurvePoint:
    n_training: int
    mean_error_norm: float
    max_error_norm: float
    mean_error_px: float | None
    n_eval: int


def points_needed_curve(
    log: AdaptiveLog, mapper_factory: MapperFactory | None = None
) -> list[CurvePoint]:
    """Error vs number of training points, replayed from a log.

    Every accepted point of the run is used, in the order shown: for each k, fit on the first
    k points and evaluate on all the others. Uses passing points too, so it answers "how many
    points would a fixed sequence like this one need" rather than repeating the run's decisions.
    """
    if mapper_factory is None:
        from opengaze.calibration.mappers import make_mapper

        def mapper_factory() -> ScreenMapper:
            return make_mapper(log.schema_id, log.geometry_estimation)

    idx = [i for i, r in enumerate(log.records) if r.action != "reject" and log.reps[i] is not None]
    if len(idx) < 3:
        return []
    X =np.stack([log.reps[i] for i in idx]).astype(np.float32)  # type: ignore[misc]
    V = np.stack([log.rep_valid[i] for i in idx])  # type: ignore[misc]
    Y = np.array([[log.records[i].target_x, log.records[i].target_y] for i in idx])
    curve: list[CurvePoint] = []
    for k in range(2, len(idx)):
        model = mapper_factory()
        try:
            model.fit(X[:k], Y[:k].astype(np.float32), V[:k])
        except ValueError:
            continue
        errs, errs_px = [], []
        for x, v, y in zip(X[k:], V[k:], Y[k:], strict=True):
            xy = model.predict_xy(x, v)
            if xy is None:
                continue
            dx = float(np.clip(xy[0], 0, 1)) - y[0]
            dy = float(np.clip(xy[1], 0, 1)) - y[1]
            errs.append(float(np.hypot(dx, dy)))
            if log.screen_px:
                errs_px.append(float(np.hypot(dx * log.screen_px[0], dy * log.screen_px[1])))
        if errs:
            curve.append(CurvePoint(
                n_training=k,
                mean_error_norm=float(np.mean(errs)),
                max_error_norm=float(np.max(errs)),
                mean_error_px=float(np.mean(errs_px)) if errs_px else None,
                n_eval=len(errs),
            ))
    return curve
