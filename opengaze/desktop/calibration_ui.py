"""Calibration wizard: fullscreen dot sequence + live feature collection.

Flow
----
1. Idle screen — press Space or click to begin.
2. For each target dot:
   a. Dot appears.  User looks at it.
   b. Dwell ring fills over collect_duration_s.
   c. Point is accepted (green flash) or rejected (red flash, retry).
3. After all dots: fit model, run LOO validation, show error map.
4. Save on Space/click; discard on Escape.

The window polls a ``GazeTracker`` (Mode A, B or C, per the profile) on a QTimer so the UI
never blocks on camera or vision.

With the 3D eye pipeline, the eye models keep learning on the intro screen and during
collection, then are frozen before fitting; every sample is re-expressed under the frozen
eyeball geometry (``GazeTracker.refeature``), and that geometry is saved with the calibration.
"""

from __future__ import annotations

import logging
import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from opengaze.calibration.mappers import make_mapper
from opengaze.calibration.session import GRID_9, CalibrationSession, SessionResult
from opengaze.calibration.storage import CalibrationStore
from opengaze.calibration.validation import ValidationReport, validate_calibration
from opengaze.cameras.manager import CameraManager
from opengaze.cameras.runtime import CaptureRuntime
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.gaze.tracker import GazeTracker, TrackerFrame

log = logging.getLogger("opengaze.calibration_ui")

_DOT_R = 18          # target dot radius, px
_RING_W = 6          # dwell ring stroke width
_FLASH_MS = 350      # accept/reject flash duration
_POLL_MS = 33        # ~30 fps timer


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_BG = QColor("#0d0f12")
_DOT_IDLE = QColor("#e8e8e8")
_DOT_DONE = QColor("#2e9e5b")
_DOT_ACTIVE = QColor("#f0b429")
_RING = QColor("#f0b429")
_FLASH_OK = QColor(46, 158, 91, 180)
_FLASH_ERR = QColor(214, 69, 65, 180)
_TEXT = QColor("#c8c8c8")
_ERROR_LOW = QColor("#2e9e5b")
_ERROR_HIGH = QColor("#d64541")


def _lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(c1.red() + (c2.red() - c1.red()) * t),
        int(c1.green() + (c2.green() - c1.green()) * t),
        int(c1.blue() + (c2.blue() - c1.blue()) * t),
    )


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class _State:
    IDLE = "idle"
    COLLECTING = "collecting"
    FLASH = "flash"
    RESULT = "result"
    DONE = "done"


class CalibrationWindow(QWidget):
    def __init__(
        self,
        runtime: CaptureRuntime,
        report: ProfileReport,
        config: AppConfig,
        store: CalibrationStore | None = None,
        owns_runtime: bool = True,
        tracker: GazeTracker | None = None,
    ) -> None:
        super().__init__()
        assert report.profile is not None
        self.runtime = runtime
        self.profile = report.profile
        self.config = config
        self.store = store or CalibrationStore()
        self._owns_runtime = owns_runtime

        # Vision pipeline
        self._owns_tracker = tracker is None
        self.tracker = tracker or GazeTracker(report.profile, config)
        self.tracker.freeze(False)  # recalibrating: let a 3D eye model keep learning
        self._schema_id = self.tracker.schema_id
        estimation = report.profile.calibration.geometry_estimation
        self._mapper_factory = lambda: make_mapper(self._schema_id, estimation)

        self.session = CalibrationSession(
            schema_id=self._schema_id,
            targets=GRID_9,
            collect_duration_s=1.5,
            mapper_factory=self._mapper_factory,
        )
        self._records: dict[int, list[TrackerFrame]] = {}

        # State
        self._state = _State.IDLE
        self._current_point = 0
        self._flash_start: float | None = None
        self._flash_ok = True
        self._result: SessionResult | None = None
        self._validation: ValidationReport | None = None
        self._saved = False

        # Qt setup
        self.setWindowTitle("OpenGaze — Calibration")
        self.setStyleSheet("background: #0d0f12;")
        self.showFullScreen()

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------
    # Timer tick — vision + state transitions
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        if self._state == _State.IDLE:
            self.tracker.step(self.runtime.manager)  # warm up a 3D eye model
        elif self._state == _State.COLLECTING:
            self._collect_frame()
            if not self.session.collecting:
                ok = self.session.finish_point(self._current_point)
                self._flash_ok = ok
                self._flash_start = time.monotonic()
                self._state = _State.FLASH

        elif self._state == _State.FLASH:
            assert self._flash_start is not None
            if (time.monotonic() - self._flash_start) * 1000 > _FLASH_MS:
                if self._flash_ok:
                    self._current_point += 1
                    if self._current_point >= self.session.n_targets:
                        self._finish()
                    else:
                        self._start_collecting()
                else:
                    # Retry the same point
                    self._start_collecting()

        self.update()

    def _collect_frame(self) -> None:
        frame = self.tracker.step(self.runtime.manager)
        if frame.features is None:
            return
        if self.session.add_sample(frame.features):
            self._records.setdefault(self._current_point, []).append(frame)

    def _start_collecting(self) -> None:
        self.session.start_point(self._current_point)
        self._records[self._current_point] = []
        self._state = _State.COLLECTING

    def _finish(self) -> None:
        w, h = self.width(), self.height()
        # Freeze learned eye geometry and re-express every sample under it.
        self.tracker.freeze(True)
        for i, pt in enumerate(self.session.points):
            records = self._records.get(i, [])
            if len(records) == len(pt.samples):
                pt.samples = [fv for r in records if (fv := self.tracker.refeature(r)) is not None]
        try:
            self._result = self.session.fit(screen_width_px=w, screen_height_px=h)
        except ValueError as exc:
            log.warning("calibration fit failed: %s", exc)
            self.tracker.freeze(False)
            self._state = _State.IDLE
            return
        try:
            self._validation = validate_calibration(
                self.session.points,
                schema_id=self._schema_id,
                # Mode A (webcam-only) is inherently coarse — 0.15 norm ≈ 150–290px
                # depending on screen size, which is realistic for iris-from-webcam.
                pass_threshold_norm=0.15 if self.tracker.is_mode_a else 0.08,
                screen_width_px=w,
                screen_height_px=h,
                mapper_factory=self._mapper_factory,
            )
        except ValueError:
            self._validation = None
        self._state = _State.RESULT

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.close()
        elif key in (Qt.Key.Key_Space, Qt.Key.Key_Return):
            self._on_advance()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._on_advance()

    def _on_advance(self) -> None:
        if self._state == _State.IDLE:
            self._current_point = 0
            self._start_collecting()
        elif self._state == _State.RESULT:
            self._save_and_close()
        elif self._state == _State.DONE:
            self.close()

    def _save_and_close(self) -> None:
        if self._result is not None and not self._saved:
            try:
                path = self.store.save(
                    self._result.model, self.profile, eye_states=self.tracker.eye_states()
                )
                log.info("calibration saved to %s", path)
                self._saved = True
            except Exception as exc:
                log.error("failed to save calibration: %s", exc)
        self._state = _State.DONE
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), _BG)

        if self._state == _State.IDLE:
            self._draw_idle(p)
        elif self._state in (_State.COLLECTING, _State.FLASH):
            self._draw_collecting(p)
        elif self._state == _State.RESULT:
            self._draw_result(p)
        elif self._state == _State.DONE:
            self._draw_done(p)

        p.end()

    def _draw_idle(self, p: QPainter) -> None:
        self._draw_text(p, "Look at each dot and hold your gaze.\n\nPress Space or click to begin.", 0.5, 0.5)

    def _draw_collecting(self, p: QPainter) -> None:
        w, h = self.width(), self.height()

        # Draw all done dots small and grey
        for i, pt in enumerate(self.session.points):
            if i == self._current_point:
                continue
            sx = int(pt.screen_x_norm * w)
            sy = int(pt.screen_y_norm * h)
            color = _DOT_DONE if pt.accepted else _DOT_IDLE
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(sx - 6, sy - 6, 12, 12)

        # Active dot
        pt = self.session.points[self._current_point]
        sx = int(pt.screen_x_norm * w)
        sy = int(pt.screen_y_norm * h)

        if self._state == _State.FLASH:
            color = _FLASH_OK if self._flash_ok else _FLASH_ERR
            assert self._flash_start is not None
            age = (time.monotonic() - self._flash_start) * 1000 / _FLASH_MS
            alpha = int(255 * (1.0 - min(age, 1.0)))
            color.setAlpha(alpha)
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(sx - _DOT_R * 2, sy - _DOT_R * 2, _DOT_R * 4, _DOT_R * 4)

        # Dwell ring
        prog = self.session.collect_progress
        if prog > 0 and self._state == _State.COLLECTING:
            pen = QPen(_RING, _RING_W)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            span = int(-prog * 360 * 16)
            p.drawArc(sx - _DOT_R - 8, sy - _DOT_R - 8, (_DOT_R + 8) * 2, (_DOT_R + 8) * 2, 90 * 16, span)

        # Dot
        p.setBrush(_DOT_ACTIVE if self._state == _State.COLLECTING else _DOT_DONE)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(sx - _DOT_R, sy - _DOT_R, _DOT_R * 2, _DOT_R * 2)

        # Counter
        self._draw_text(p, f"{self._current_point + 1} / {self.session.n_targets}", 0.5, 0.97, size=14)

    def _draw_result(self, p: QPainter) -> None:
        w, h = self.width(), self.height()
        report = self._validation
        result = self._result

        # Error heatmap dots
        if report:
            max_err = max((e.error_norm for e in report.point_errors), default=0.01)
            for err in report.point_errors:
                sx = int(err.screen_x_norm * w)
                sy = int(err.screen_y_norm * h)
                t = err.error_norm / max(max_err, 1e-6)
                color = _lerp_color(_ERROR_LOW, _ERROR_HIGH, t)
                r = 6 + int(t * 14)
                p.setBrush(color)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(sx - r, sy - r, r * 2, r * 2)
                if err.error_px is not None:
                    self._draw_small_text(p, f"{err.error_px:.0f}px", sx, sy + r + 12)

        # Summary
        if result and report:
            quality = (
                "Good accuracy" if report.passed
                else "Low accuracy — try better lighting or sit closer"
            )
            lines = [
                "Calibration complete",
                f"Mean error: {report.mean_error_px:.0f} px" if report.mean_error_px else
                f"Mean error: {report.mean_error_norm:.3f} norm",
                quality,
                "",
                "Dots: green = accurate  red = less accurate",
                "Press Space or click to save.  Escape to discard.",
            ]
        else:
            lines = ["Calibration complete", "", "Press Space to save.  Escape to discard."]
        self._draw_text(p, "\n".join(lines), 0.5, 0.12)

    def _draw_done(self, p: QPainter) -> None:
        msg = "Saved!  Press Escape or click to close." if self._saved else "Closed without saving."
        self._draw_text(p, msg, 0.5, 0.5)

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _draw_text(self, p: QPainter, text: str, fx: float, fy: float, size: int = 20) -> None:
        p.setPen(_TEXT)
        font = QFont("Helvetica Neue", size)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        p.setFont(font)
        r = self.rect()
        p.drawText(
            int(r.width() * (fx - 0.4)),
            int(r.height() * (fy - 0.15)),
            int(r.width() * 0.8),
            int(r.height() * 0.3),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
            text,
        )

    def _draw_small_text(self, p: QPainter, text: str, cx: int, cy: int) -> None:
        p.setPen(_TEXT)
        font = QFont("Menlo", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        p.setFont(font)
        p.drawText(cx - 30, cy - 8, 60, 16, Qt.AlignmentFlag.AlignHCenter, text)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        if self._owns_runtime:
            self.runtime.stop()
        if self._owns_tracker:
            self.tracker.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_calibration(report: ProfileReport, config: AppConfig) -> int:
    assert report.profile is not None
    app = QApplication.instance() or QApplication(sys.argv[:1])
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()
    window = CalibrationWindow(runtime, report, config)
    window.show()
    try:
        return app.exec()
    finally:
        runtime.stop()
