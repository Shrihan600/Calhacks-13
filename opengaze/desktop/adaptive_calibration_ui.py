"""Adaptive calibration wizard: a continuous stream of dots until the model is accurate.

Flow
----
1. Idle screen — press Space or click to begin.
2. Each dot appears, waits ``_SETTLE_S`` for the eyes to land, then collects samples.
3. The first dots (centre + corners) train the model. Every later dot is a test: the
   prediction is shown as a cross. Close enough → green (pass); too far → amber, the dot
   becomes training data and the model is refitted. Too few samples → red, retry.
4. Stops after several passes in a row (or the point cap), then shows every dot and
   prediction. Space/click saves; Escape discards.

The run's log (every target, sample, prediction and decision) is written to
``~/.config/opengaze/calibration_logs/`` whether or not the calibration is saved, so
``opengaze calibration-log`` can show how many points were needed.

With the 3D eye pipeline the eye models learn during the seed dots and are frozen once the
first model is fitted; seed samples are then re-expressed under the frozen geometry.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication

from opengaze.calibration.adaptive import (
    AdaptiveCalibrationSession,
    AdaptiveResult,
    PointRecord,
)
from opengaze.calibration.model import ScreenMapper
from opengaze.calibration.session import CalibrationPoint
from opengaze.calibration.storage import CalibrationStore
from opengaze.cameras.manager import CameraManager
from opengaze.cameras.runtime import CaptureRuntime
from opengaze.config.fingerprint import profile_fingerprint
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.debug import puppet
from opengaze.desktop.calibration_ui import (
    _BG,
    _DOT_ACTIVE,
    _DOT_R,
    _FLASH_ERR,
    _FLASH_OK,
    _RING,
    _RING_W,
    CalibrationWindow,
)
from opengaze.gaze.tracker import GazeTracker, TrackerFrame

log = logging.getLogger("opengaze.adaptive_calibration_ui")

_SETTLE_S = 0.35     # saccade + fixation before collecting
_COLLECT_S = 1.0
_FLASH_S = 0.6       # outcome display
_FLASH_TRAIN = QColor(240, 180, 41, 180)
_PRED = QColor("#7fb3ff")
_DOT_PASS = QColor("#2e9e5b")
_DOT_TRAIN = QColor("#f0b429")


class _AState:
    IDLE = "idle"
    SETTLE = "settle"
    COLLECTING = "collecting"
    FLASH = "flash"
    RESULT = "result"
    DONE = "done"


class AdaptiveCalibrationWindow(CalibrationWindow):
    def __init__(
        self,
        runtime: CaptureRuntime,
        report: ProfileReport,
        config: AppConfig,
        store: CalibrationStore | None = None,
        owns_runtime: bool = True,
        tracker: GazeTracker | None = None,
        log_dir: Path | None = None,
    ) -> None:
        super().__init__(runtime, report, config, store, owns_runtime, tracker)
        self.setWindowTitle("OpenGaze — Adaptive calibration")
        self._log_dir = log_dir
        self._estimation = report.profile.calibration.geometry_estimation  # type: ignore[union-attr]
        self._asession: AdaptiveCalibrationSession | None = None
        self._astate = _AState.IDLE
        self._point: CalibrationPoint | None = None
        self._frames: dict[int, list[TrackerFrame]] = {}  # id(point) → frames behind its samples
        self._phase_start = 0.0
        self._last: PointRecord | None = None
        self._aresult: AdaptiveResult | None = None
        self._log_path: Path | None = None
        self._error: str | None = None

    # ------------------------------------------------------------------
    # Flow
    # ------------------------------------------------------------------

    def _begin(self) -> None:
        self.tracker.freeze(False)
        self._asession = AdaptiveCalibrationSession(
            schema_id=self._schema_id,
            mapper_factory=self._mapper_factory,
            geometry_estimation=self._estimation,
            # Same bars as the grid wizard's LOO check; Mode A (webcam iris) is coarse.
            tolerance_norm=0.15 if self.tracker.is_mode_a else 0.08,
            screen_px=(self.width(), self.height()),
        )
        self._asession.log.profile_fingerprint = profile_fingerprint(self.profile)
        self._frames.clear()
        self._log_path = None
        self._error = None
        self._show_next()

    def _show_next(self) -> None:
        assert self._asession is not None
        self._point = CalibrationPoint(*self._asession.next_target())
        self._frames[id(self._point)] = []
        self._enter(_AState.SETTLE)

    def _enter(self, state: str) -> None:
        self._astate = state
        self._phase_start = time.monotonic()

    def _elapsed(self) -> float:
        return time.monotonic() - self._phase_start

    def _tick(self) -> None:
        state = self._astate
        if state in (_AState.SETTLE, _AState.COLLECTING, _AState.FLASH) and self._point:
            puppet.look_at(self._point.screen_x_norm, self._point.screen_y_norm)
        else:
            puppet.release()
        if state == _AState.IDLE:
            self._warm_up()
        elif state in (_AState.SETTLE, _AState.FLASH):
            self.tracker.step(self.runtime.manager)  # keep the pipeline (and 3D model) warm
        if state == _AState.SETTLE and self._elapsed() >= _SETTLE_S:
            self._enter(_AState.COLLECTING)
        elif state == _AState.COLLECTING:
            self._collect()
            if self._elapsed() >= _COLLECT_S:
                self._score()
        elif state == _AState.FLASH and self._elapsed() >= _FLASH_S:
            assert self._asession is not None
            if self._asession.done:
                self._finish()
            else:
                self._show_next()
        self.update()

    def _collect(self) -> None:
        frame = self.tracker.step(self.runtime.manager)
        fv = frame.features
        if fv is None or fv.schema_id != self._schema_id or self._point is None:
            return
        self._point.samples.append(fv)
        self._frames[id(self._point)].append(frame)

    def _score(self) -> None:
        assert self._asession is not None and self._point is not None
        had_model = self._asession.model is not None
        self._last = self._asession.submit(self._point)
        log.debug("adaptive point %s", self._last)
        if not had_model and self._asession.model is not None and not self.tracker.is_mode_a:
            # First model fitted: freeze the eye geometry and re-express the seed samples.
            self.tracker.freeze(True)
            self._asession.refeature(self._refeatured)
        self._enter(_AState.FLASH)

    def _refeatured(self, point: CalibrationPoint):
        return [
            fv for r in self._frames.get(id(point), [])
            if (fv := self.tracker.refeature(r)) is not None
        ]

    def _finish(self) -> None:
        assert self._asession is not None
        self.tracker.freeze(True)
        self._save_log()
        try:
            self._aresult = self._asession.result()
        except ValueError as exc:
            log.warning("adaptive calibration failed: %s", exc)
            self._error = str(exc)
            self.tracker.freeze(False)
            self._astate = _AState.IDLE
            return
        self._astate = _AState.RESULT

    def _save_log(self) -> None:
        if self._asession is None or not self._asession.log.records or self._log_path:
            return
        try:
            self._log_path = self._asession.log.save(self._log_dir)
            log.info("adaptive calibration log saved to %s", self._log_path)
        except Exception as exc:
            log.error("failed to save calibration log: %s", exc)

    def _fitted_model(self) -> ScreenMapper | None:
        return self._aresult.model if self._aresult is not None else None

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _on_advance(self) -> None:
        if self._astate == _AState.IDLE:
            if self._eye_ready:
                self._begin()
        elif self._astate == _AState.RESULT:
            self._save_and_close()
            self._astate = _AState.DONE
        elif self._astate == _AState.DONE:
            self.close()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), _BG)
        state = self._astate
        if state == _AState.IDLE:
            msg = "Look at each dot until the next one appears.\n\n"
            msg += "Dots keep coming until tracking is accurate.\n\n" + self._start_hint()
            if self._error:
                msg = f"Calibration failed: {self._error}\n\n" + msg
            self._draw_text(p, msg, 0.5, 0.5)
        elif state in (_AState.SETTLE, _AState.COLLECTING, _AState.FLASH):
            self._draw_stream(p)
        elif state == _AState.RESULT:
            self._draw_adaptive_result(p)
        elif state == _AState.DONE:
            self._draw_done(p)
        p.end()

    def _px(self, x: float, y: float) -> tuple[int, int]:
        return int(x * self.width()), int(y * self.height())

    def _draw_stream(self, p: QPainter) -> None:
        assert self._asession is not None and self._point is not None
        sx, sy = self._px(self._point.screen_x_norm, self._point.screen_y_norm)

        if self._astate == _AState.FLASH and self._last is not None:
            color = {"pass": _FLASH_OK, "train": _FLASH_TRAIN, "reject": _FLASH_ERR}[
                self._last.action
            ]
            color = QColor(color)
            color.setAlpha(int(color.alpha() * (1.0 - min(self._elapsed() / _FLASH_S, 1.0))))
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(sx - _DOT_R * 2, sy - _DOT_R * 2, _DOT_R * 4, _DOT_R * 4)
            if self._last.pred_x is not None and self._last.pred_y is not None:
                self._draw_cross(p, *self._px(self._last.pred_x, self._last.pred_y))

        if self._astate == _AState.COLLECTING:
            prog = min(self._elapsed() / _COLLECT_S, 1.0)
            pen = QPen(_RING, _RING_W)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            r = _DOT_R + 8
            p.drawArc(sx - r, sy - r, r * 2, r * 2, 90 * 16, int(-prog * 360 * 16))

        p.setBrush(_DOT_ACTIVE)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(sx - _DOT_R, sy - _DOT_R, _DOT_R * 2, _DOT_R * 2)

        s = self._asession
        phase = "learning" if s.in_seed_phase else (
            f"testing — {s.consecutive_passes}/{s.required_passes} in a row"
        )
        self._draw_text(
            p, f"{s.n_accepted + 1} · {phase} · {len(s.training)} training", 0.5, 0.97, size=14
        )
        if self._last is not None and self._last.action == "reject":
            if self._last.mean_confidence == 0.0:
                why = "No face or eyes detected — check the camera, lighting and face tracking"
            elif self._last.n_samples >= s.min_samples_per_point:
                why = "no usable eye features in the samples — pupil not detected (D in " \
                      "`opengaze run` shows each stage)"
            else:
                why = f"Only {self._last.n_samples} samples — hold still on the dot"

            self._draw_text(p, f"Retrying: {why}", 0.5, 0.91, size=14)

    def _draw_cross(self, p: QPainter, x: int, y: int) -> None:
        p.setPen(QPen(_PRED, 2))
        p.drawLine(x - 10, y, x + 10, y)
        p.drawLine(x, y - 10, x, y + 10)

    def _draw_adaptive_result(self, p: QPainter) -> None:
        assert self._aresult is not None
        res = self._aresult
        for rec in res.log.records:
            if rec.action == "reject":
                continue
            tx, ty = self._px(rec.target_x, rec.target_y)
            if rec.pred_x is not None and rec.pred_y is not None:
                px, py = self._px(rec.pred_x, rec.pred_y)
                p.setPen(QPen(_PRED, 1))
                p.drawLine(tx, ty, px, py)
                self._draw_cross(p, px, py)
            p.setBrush(_DOT_PASS if rec.action == "pass" else _DOT_TRAIN)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(tx - 7, ty - 7, 14, 14)
            if rec.error_px is not None:
                self._draw_small_text(p, f"{rec.error_px:.0f}px", tx, ty + 20)

        if res.final_error_px is not None:
            accuracy = f"Accuracy on the last {res.log.settings['required_passes']} dots: "
            accuracy += f"{res.final_error_px:.0f} px mean"
        else:
            accuracy = "Never reached the accuracy target — try better lighting or sit closer"
        lines = [
            "Calibration complete" if res.converged else "Stopped at the point limit",
            res.log.summary(),
            accuracy,
            "",
            "green = test passed  amber = used for training  cross = prediction",
            "Press Space or click to save.  Escape to discard.",
        ]
        if self._log_path:
            lines.append(f"log: {self._log_path.name}")
        self._draw_text(p, "\n".join(lines), 0.5, 0.12)

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_log()  # keep debugging data from runs abandoned midway
        super().closeEvent(event)


def run_adaptive_calibration(report: ProfileReport, config: AppConfig) -> int:
    assert report.profile is not None
    app = QApplication.instance() or QApplication(sys.argv[:1])
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()
    window = AdaptiveCalibrationWindow(runtime, report, config)
    window.show()
    try:
        return app.exec()
    finally:
        runtime.stop()
