"""Gaze test environment: fullscreen window showing gaze dot, camera feed, blink state.

Works in every tracking mode (A webcam, B/C eye cameras) through ``GazeTracker``.

Flow
----
1. If no calibration: run calibration wizard first, then return here.
2. Fullscreen window opens.
3. Gaze dot tracks eye position.
4. Eye gestures (blinks, winks, glances) fire their keybinds from ``gestures:`` in the app
   config; the default binds a double blink to a left click.
5. Space (or Tab, or a long right wink) pauses/resumes: while paused the real cursor isn't
   driven and key/mouse bindings don't act. Starts paused.
6. C re-runs calibration; A runs adaptive calibration (dots until accurate).
7. D shows the pipeline debug overlay: every stage's health over the last 3 s, and why the
   cursor isn't moving when it isn't. T starts/stops a trace (``opengaze trace-report``).
8. Escape quits.

``--trace`` records a trace from the start; ``--record-video`` also saves what each camera
gave the tracker, so the session can be re-run offline (``opengaze replay``).
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from opengaze.calibration.storage import CalibrationStore
from opengaze.cameras.manager import CameraManager
from opengaze.cameras.runtime import CaptureRuntime
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.debug.analyze import analyze
from opengaze.debug.trace import CursorTick, TraceWriter, run_header, tick_record, truth_of
from opengaze.gaze.loop import GazeLoop
from opengaze.gaze.tracker import GazeTracker
from opengaze.gestures.bindings import ActionDispatcher
from opengaze.modes import profile_schema

log = logging.getLogger("opengaze.run_ui")

_BG = QColor("#111318")
_GAZE_RING = QColor("#00d4ff")
_GAZE_CENTER = QColor("#ffffff")
_BLINK_ACTIVE = QColor("#f0b429")
_BLINK_INACTIVE = QColor("#333")
_CLICK_FLASH = QColor(0, 212, 255, 160)
_TEXT = QColor("#c8c8c8")
_DIM = QColor("#555")
_CURSOR_ON = QColor("#2e9e5b")
_CURSOR_OFF = QColor("#555")
_ERROR = QColor("#d64541")
_ERROR_BG = QColor(214, 69, 65, 40)
_PERMISSION_RECHECK_TICKS = 60  # ~2 s: notice permission granted while the window is open

_POLL_MS = 33   # ~30 fps
_OVERLAY_TICKS = 90  # debug overlay analyses the last ~3 s
_STATUS_COLORS = {"ok": QColor("#2e9e5b"), "warn": QColor("#f0b429"),
                  "fail": QColor("#d64541"), "n/a": QColor("#555")}
_TRUTH = QColor("#ff7ad9")
_REC = QColor("#d64541")


def _to_qimage(image: np.ndarray) -> QImage:
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    fmt = QImage.Format.Format_BGR888
    return QImage(image.data, w, h, image.strides[0], fmt).copy()


class GazeTestWindow(QWidget):
    def __init__(
        self,
        runtime: CaptureRuntime,
        report: ProfileReport,
        config: AppConfig,
        schema_id: str,
        *,
        trace: bool = False,
        record_video: bool = False,
    ) -> None:
        super().__init__()
        assert report.profile is not None

        self.runtime = runtime
        self.profile = report.profile
        self.config = config
        self.schema_id = schema_id

        # Vision pipeline (Mode A, B or C, per the profile)
        self._tracker = GazeTracker(report.profile, config)
        self._loop = GazeLoop(self._tracker, None, config.gestures)

        # Load calibration model (and frozen eye-model state, for the 3D pipeline)
        self._store = CalibrationStore()
        self._model = None
        self._load_model()

        # Cursor driver (optional); it is also the system input backend for keybinds.
        # Without one, ``_cursor_problem`` says why, and the window shows it as an error.
        self._cursor, self._cursor_problem = _try_make_cursor()
        if self._cursor_problem:
            print(f"ERROR: cursor control unavailable — {self._cursor_problem}")
        self._cursor_enabled = False  # paused by default — toggle with Space
        self._ticks = 0
        self._error_flash_until = 0.0
        self._banner_bottom = 0
        self._dispatcher = ActionDispatcher(
            config.gestures.bindings,
            self._cursor,
            {
                # Pausing and turning cursor control off are the same thing.
                "pause": self._toggle_cursor,
                "toggle_cursor": self._toggle_cursor,
                "recalibrate": lambda: QTimer.singleShot(0, self._run_calibration),
                "recenter": self._recenter,
                "keyboard": lambda: QTimer.singleShot(0, lambda: self._open_activity("keyboard")),
                "practice": lambda: QTimer.singleShot(0, lambda: self._open_activity("practice")),
                "quit": lambda: QTimer.singleShot(0, self.close),
            },
        )
        self._last_gesture: str = ""
        self._last_gesture_until: float = 0.0

        # Display state
        self._gaze_x: float = 0.5
        self._gaze_y: float = 0.5
        self._confidence: float = 0.0
        self._left_blink = False
        self._right_blink = False
        self._blink_progress: float = 0.0
        self._click_flash_until: float = 0.0
        self._camera_frame: np.ndarray | None = None
        self._clicks = 0
        self._status = "Press C to calibrate, Space to start/pause the cursor, Esc to quit"

        # Debugging: overlay, trace, session recording
        self._debug = False
        self._recent: deque[dict] = deque(maxlen=_OVERLAY_TICKS)
        self._last_rec: dict | None = None
        self._truth: tuple[float, float] | None = None  # simulated rigs: true gaze target
        self._trace: TraceWriter | None = None
        self._recorder = None
        if record_video:
            from opengaze.debug.harness import SessionRecorder
            self._recorder = SessionRecorder.new(self.profile, 1000 / _POLL_MS)
        if trace or record_video:
            QTimer.singleShot(0, self._start_trace)  # once fullscreen: header has screen size

        # Qt
        self.setWindowTitle("OpenGaze — Test Environment")
        self.setStyleSheet("background: #111318;")
        self.showFullScreen()

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            bundle = self._store.load_bundle(self.schema_id, self.profile)  # type: ignore[arg-type]
        except (FileNotFoundError, ValueError):
            self._model = None
        else:
            self._model = bundle.model
            self._tracker.load_eye_states(bundle.eye_states)
        self._loop.model = self._model

    # ------------------------------------------------------------------
    # Timer tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        t = time.monotonic()
        self._ticks += 1
        if self._cursor is None and self._ticks % _PERMISSION_RECHECK_TICKS == 0:
            self._retry_cursor()
        # A held wink holds the button down: keep tracking the open eye so it drags.
        self._loop.track_one_eye = self._button_held()
        state = self._loop.tick(self.runtime.manager, t)
        frame = state.frame
        self._camera_frame = _preview(frame)
        self._left_blink = bool(frame.left_closed)
        self._right_blink = bool(frame.right_closed)
        self._blink_progress = max(state.gesture.blink_progress, state.gesture.wink_progress)
        self._gaze_x, self._gaze_y = state.gaze_x, state.gaze_y
        self._confidence = state.confidence

        self._observe(state, t, self._drive_cursor(state))
        self._handle_gestures(state)
        self.update()

    def _cursor_tick(self, cmd: tuple[int, int] | None, blocked: str | None) -> CursorTick:
        # Read back *before* any move this tick: CGEventPost is asynchronous.
        actual = None
        if self._cursor is not None and (self._trace or self._debug):
            try:
                actual = self._cursor.position()
            except Exception:
                actual = None
        return CursorTick(self._cursor_enabled, self._cursor is not None, cmd, actual, blocked)

    def _drive_cursor(self, state) -> CursorTick:
        if self._cursor is None:
            return self._cursor_tick(None, "no_backend")
        if not self._cursor_enabled:
            return self._cursor_tick(None, "cursor_off")
        if not state.updated:
            return self._cursor_tick(None, f"gaze:{state.drop_reason}")
        tick = self._cursor_tick(None, None)
        tick.commanded = self._cursor.norm_to_px(self._gaze_x, self._gaze_y)
        self._cursor.move(*tick.commanded)
        return tick

    def _observe(self, state, t: float, cursor: CursorTick) -> None:
        """Feed the debug overlay, the trace and the session recording."""
        self._truth = truth_of(state.frame)
        if not (self._debug or self._trace or self._recorder):
            self._last_rec = None
            return
        rec = tick_record(state, t, cursor=cursor, imu=self._tracker.imu_health())
        self._last_rec = rec
        self._recent.append(rec)
        if self._trace is not None:
            self._trace.tick(rec)
        if self._recorder is not None:
            self._recorder.add(state.frame)

    def _header(self) -> dict:
        return run_header(
            self.profile, self._tracker, self._loop,
            screen_px=[self.width(), self.height()],
            calibrated=self._model is not None,
        )

    def _start_trace(self) -> None:
        trace_dir = self._recorder.dir if self._recorder is not None else None
        self._trace = TraceWriter.new(self._header(), trace_dir, name="run")
        self._status = f"Tracing to {self._trace.path}"
        log.info("trace: %s", self._trace.path)

    def _stop_trace(self) -> None:
        if self._trace is None:
            return
        self._trace.close()
        self._status = f"Trace saved ({self._trace.n_ticks} ticks): {self._trace.path}"
        print(f"trace saved: {self._trace.path}")
        self._trace = None

    def _event(self, message: str) -> None:
        if self._trace is not None:
            self._trace.event(time.monotonic(), message)

    def _sync_display(self) -> None:
        """Drive the display this window is on: gaze is calibrated against the window."""
        screen = self.screen()
        if self._cursor is not None and screen is not None and hasattr(self._cursor, "set_display"):
            g = screen.geometry()
            self._cursor.set_display(g.x(), g.y(), g.width(), g.height())

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._sync_display()

    def _handle_gestures(self, state) -> None:
        """Dispatch gesture events at the frozen pre-blink gaze position."""
        self._dispatcher.system_input_enabled = (
            self._cursor_enabled and self._model is not None
        )
        gaze_px = self._cursor.norm_to_px(self._gaze_x, self._gaze_y) if self._cursor else None
        for event in state.gesture.events:
            result = self._dispatcher.dispatch(event, gaze_px)
            binding = result.binding.describe() if result.binding else "unbound"
            name = event.gesture.value
            if event.count > 1:
                name += f" ×{event.count}"
            if event.phase:
                name += f" ({event.phase})"
            self._last_gesture = f"{name} → {binding} ({result.outcome})"
            self._last_gesture_until = time.monotonic() + 1.5
            is_click = (
                result.outcome == "done"
                and result.binding is not None
                and result.binding.mouse == "click"
            )
            if is_click:
                self._clicks += 1
                self._click_flash_until = time.monotonic() + 0.3

    def _recenter(self) -> None:
        self._tracker.recenter()
        self._status = "Head heading recentred — facing the screen"
        self._event("recenter")

    def _button_held(self) -> bool:
        return bool(getattr(self._cursor, "pressed", False))

    def _release_button(self) -> None:
        """Let go of a held mouse button (a drag) — whenever control is taken away."""
        if self._button_held():
            self._cursor.release()  # type: ignore[union-attr]
            self._event("button released")

    def _retry_cursor(self) -> bool:
        """Try again to get a cursor driver (permission may have been granted meanwhile)."""
        if self._cursor is not None:
            return True
        self._cursor, self._cursor_problem = _try_make_cursor()
        if self._cursor is None:
            return False
        self._dispatcher.backend = self._cursor
        self._sync_display()
        self._status = "Accessibility permission granted — cursor control available (Space)"
        self._event("cursor driver available")
        return True

    def _toggle_cursor(self) -> None:
        if not self._cursor_enabled and not self._retry_cursor():
            self._status = "Can't resume cursor control — see the error above"
            self._error_flash_until = time.monotonic() + 1.0
            return
        self._cursor_enabled = not self._cursor_enabled
        if not self._cursor_enabled:
            self._release_button()
        self._loop.reset()
        self._sync_display()
        self._status = "Cursor control ON" if self._cursor_enabled else "Paused"
        self._event("resumed" if self._cursor_enabled else "paused")

    def _open_permission_settings(self) -> None:
        try:
            from opengaze.control.cursor import open_accessibility_prefs
            open_accessibility_prefs()
        except Exception as exc:
            self._status = f"Couldn't open System Settings: {exc}"

    def _toggle_debug(self) -> None:
        self._debug = not self._debug
        self._recent.clear()

    def _toggle_trace(self) -> None:
        if self._recorder is not None:
            self._status = "Trace stays on while video is being recorded"
        elif self._trace is None:
            self._start_trace()
        else:
            self._stop_trace()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        k = event.key()
        if k == Qt.Key.Key_Escape:
            self.close()
        elif k in (Qt.Key.Key_Space, Qt.Key.Key_Tab):
            self._toggle_cursor()
        elif k == Qt.Key.Key_C:
            self._run_calibration()
        elif k == Qt.Key.Key_R:
            self._recenter()
        elif k == Qt.Key.Key_A:
            self._run_calibration(adaptive=True)
        elif k == Qt.Key.Key_K:
            self._open_activity("keyboard")
        elif k == Qt.Key.Key_G:
            self._open_activity("practice")
        elif k == Qt.Key.Key_D:
            self._toggle_debug()
        elif k == Qt.Key.Key_T:
            self._toggle_trace()
        elif k == Qt.Key.Key_P and self._cursor_problem:
            self._open_permission_settings()
        else:
            super().keyPressEvent(event)

    def _open_activity(self, kind: str) -> None:
        if self._model is None:
            self._status = "No calibration — press C first"
            return
        self._release_button()
        self._timer.stop()
        try:
            if kind == "keyboard":
                from opengaze.desktop.keyboard_ui import GazeKeyboardWindow
                win = GazeKeyboardWindow(
                    self.runtime, _make_report(self.profile),
                    self.config, self.schema_id, self._model,
                    tracker=self._tracker,
                )
            else:
                from opengaze.desktop.practice_ui import PracticeWindow
                win = PracticeWindow(
                    self.runtime, _make_report(self.profile),
                    self.config, self.schema_id, self._model,
                    tracker=self._tracker,
                )
            win.show()
            app = QApplication.instance()
            while win.isVisible():
                app.processEvents()
                time.sleep(0.03)
        except Exception as exc:
            log.warning("%s activity failed: %s", kind, exc)
        self._loop.reset()
        self._timer.start()

    def _run_calibration(self, adaptive: bool = False) -> None:
        self._release_button()
        self._timer.stop()
        try:
            win = _calibration_window_class(adaptive)(
                self.runtime, _make_report(self.profile), self.config, self._store,
                owns_runtime=False, tracker=self._tracker,
            )
            win.show()
            # Wait for it to close
            app = QApplication.instance()
            while win.isVisible():
                app.processEvents()
                time.sleep(0.03)
        except Exception as exc:
            log.warning("calibration failed: %s", exc)
        self._load_model()
        self._event(f"calibration {'loaded' if self._model is not None else 'missing'}")
        self._loop.reset()
        self._timer.start()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), _BG)

        self._draw_truth(p, w, h)
        self._draw_gaze(p, w, h)
        self._draw_camera_preview(p, w, h)
        self._draw_blink_indicators(p, w, h)
        self._draw_hud(p, w, h)
        if self._debug:
            self._draw_debug(p, w, h)
        self._draw_recording(p, w)
        p.end()

    def _draw_truth(self, p: QPainter, w: int, h: int) -> None:
        """Simulated rigs: where the synthetic eyes really look (pink square)."""
        if self._truth is None:
            return
        tx, ty = int(self._truth[0] * w), int(self._truth[1] * h)
        p.setPen(QPen(_TRUTH, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(tx - 12, ty - 12, 24, 24)

    def _draw_recording(self, p: QPainter, w: int) -> None:
        labels = []
        if self._recorder is not None:
            labels.append("● REC VIDEO")
        if self._trace is not None:
            labels.append("● TRACE")
        if not labels:
            return
        p.setPen(_REC)
        p.setFont(QFont("Menlo", 13, QFont.Weight.Bold))
        p.drawText(w // 2 - 100, 28, "   ".join(labels))

    def _draw_debug(self, p: QPainter, w: int, h: int) -> None:
        """Stage-by-stage health of the last few seconds, plus why the cursor isn't moving."""
        if len(self._recent) < 2:
            return
        report = analyze(list(self._recent), self._header())
        x0, lh = 16, 18
        y0 = max(80, self._banner_bottom + 28)  # below the cursor error banner, if any
        p.setBrush(QColor(0, 0, 0, 190))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(x0 - 8, y0 - 18, min(w - 16, 900), lh * (len(report.stages) * 2 + 4))
        p.setFont(QFont("Menlo", 11))
        y = y0
        for st in report.stages:
            p.setPen(_STATUS_COLORS[st.status])
            p.drawText(x0, y, f"{st.status.upper():4} {st.name}")
            p.setPen(_TEXT)
            metrics = "  ".join(
                f"{k}={v}" for k, v in st.metrics.items() if not isinstance(v, dict)
            )
            nested = "  ".join(
                f"{k}[" + " ".join(f"{a}={b}" for a, b in v.items()) + "]"
                for k, v in st.metrics.items() if isinstance(v, dict)
            )
            p.drawText(x0 + 120, y, f"{metrics}  {nested}"[:120])
            y += lh
            if st.findings:
                p.setPen(_STATUS_COLORS[st.status])
                p.drawText(x0 + 120, y, st.findings[0][:120])
                y += lh
        cur = (self._last_rec or {}).get("cursor") or {}
        blocked = cur.get("blocked")
        why = {
            "no_backend": "no cursor driver (Accessibility permission?)",
            "cursor_off": "paused (Space)",
        }.get(blocked or "", blocked)
        p.setPen(_STATUS_COLORS["ok" if not blocked else "warn"])
        p.drawText(x0, y + 6, "cursor: moving" if not blocked else f"cursor not moving: {why}")

    def _draw_gaze(self, p: QPainter, w: int, h: int) -> None:
        gx = int(self._gaze_x * w)
        gy = int(self._gaze_y * h)

        # Click flash
        if time.monotonic() < self._click_flash_until:
            p.setBrush(_CLICK_FLASH)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(gx - 60, gy - 60, 120, 120)

        if self._model is None:
            # No model — show message
            self._draw_centered(p, "No calibration — press C to calibrate", w // 2, h // 2)
            return

        # Long-blink progress ring
        if self._blink_progress > 0:
            pen = QPen(QColor("#f0b429"), 6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            span = int(-self._blink_progress * 360 * 16)
            p.drawArc(gx - 44, gy - 44, 88, 88, 90 * 16, span)

        # Outer ring
        pen = QPen(_GAZE_RING, 3)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(gx - 36, gy - 36, 72, 72)

        # Cross
        pen2 = QPen(_GAZE_RING, 2)
        p.setPen(pen2)
        p.drawLine(gx - 50, gy, gx - 38, gy)
        p.drawLine(gx + 38, gy, gx + 50, gy)
        p.drawLine(gx, gy - 50, gx, gy - 38)
        p.drawLine(gx, gy + 38, gx, gy + 50)

        # Center dot
        p.setBrush(_GAZE_CENTER)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(gx - 4, gy - 4, 8, 8)

    def _draw_camera_preview(self, p: QPainter, w: int, h: int) -> None:
        if self._camera_frame is None:
            return
        pw, ph = 240, 160
        px, py = w - pw - 16, h - ph - 60
        img = cv2.resize(self._camera_frame, (pw, ph))
        qimg = _to_qimage(img)
        p.drawImage(px, py, qimg)
        # Border
        pen = QPen(QColor("#333"), 1)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(px, py, pw, ph)

    def _draw_blink_indicators(self, p: QPainter, w: int, h: int) -> None:
        bh = h - 44
        for i, (label, active) in enumerate([("L", self._left_blink), ("R", self._right_blink)]):
            cx = w - 240 - 16 - 60 + i * 50
            cy = bh
            color = _BLINK_ACTIVE if active else _BLINK_INACTIVE
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(cx - 14, cy - 14, 28, 28)
            p.setPen(_TEXT if active else _DIM)
            font = QFont("Helvetica Neue", 11, QFont.Weight.Bold)
            p.setFont(font)
            p.drawText(cx - 6, cy + 5, label)

    def _draw_hud(self, p: QPainter, w: int, h: int) -> None:
        font = QFont("Menlo", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        p.setFont(font)
        p.setPen(_TEXT)

        if self._cursor is None:
            cursor_color, cursor_state = _ERROR, "NO CURSOR CONTROL"
        elif self._cursor_enabled and self._button_held():
            cursor_color, cursor_state = _BLINK_ACTIVE, "HOLDING (drag)"
        elif self._cursor_enabled:
            cursor_color, cursor_state = _CURSOR_ON, "CURSOR ON"
        else:
            cursor_color, cursor_state = _CURSOR_OFF, "PAUSED (Space)"

        # Top bar
        p.drawText(16, 28,
            f"OpenGaze  |  {self.schema_id}  |  conf {self._confidence:.2f}"
            f"  |  clicks {self._clicks}"
        )
        p.setPen(cursor_color)
        p.drawText(w - 200, 28, cursor_state)
        p.setPen(_TEXT)

        # Bottom bar
        p.drawText(
            16, h - 16,
            "C=calibrate  A=adaptive  R=recenter  K=keyboard  G=practice  "
            "D=debug  T=trace  Space=pause/resume  Esc=quit",
        )
        p.setPen(QColor("#f0b429"))
        p.drawText(16, h - 36, self._status)
        if time.monotonic() < self._last_gesture_until:
            p.drawText(16, 52, f"gesture: {self._last_gesture}")

        self._banner_bottom = self._draw_cursor_error(p, w)

        # Long blink hint
        if self._blink_progress > 0.05:
            p.setPen(QColor("#f0b429"))
            font2 = QFont("Helvetica Neue", 18, QFont.Weight.Bold)
            p.setFont(font2)
            p.drawText(w // 2 - 80, h - 80, "Hold blink...")

    def _draw_cursor_error(self, p: QPainter, w: int) -> int:
        """Red banner while the system cursor can't be driven, with how to fix it.

        Returns the banner's bottom edge (0 when there is none) so the debug overlay can sit
        below it.
        """
        if self._cursor is not None or not self._cursor_problem:
            return 0
        flash = time.monotonic() < self._error_flash_until
        bw = min(w - 32, 980)
        x, y, pad = (w - bw) // 2, 44, 14
        body_font = QFont("Helvetica Neue", 12)
        flags = Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap
        body = QFontMetrics(body_font).boundingRect(
            0, 0, bw - 2 * pad, 1000, flags, self._cursor_problem
        )
        bh = pad + 20 + body.height() + pad
        p.setBrush(_ERROR if flash else _ERROR_BG)
        p.setPen(QPen(_ERROR, 2))
        p.drawRoundedRect(x, y, bw, bh, 6, 6)
        p.setPen(QColor("#ffffff") if flash else _ERROR)
        p.setFont(QFont("Helvetica Neue", 14, QFont.Weight.Bold))
        p.drawText(x + pad, y + pad + 14, "Cursor control unavailable")
        p.setFont(body_font)
        p.setPen(QColor("#ffffff") if flash else _TEXT)
        p.drawText(x + pad, y + pad + 22, bw - 2 * pad, body.height(), flags, self._cursor_problem)
        return y + bh

    def _draw_centered(self, p: QPainter, text: str, cx: int, cy: int) -> None:
        p.setPen(_TEXT)
        font = QFont("Helvetica Neue", 22)
        p.setFont(font)
        p.drawText(cx - 300, cy - 20, 600, 40,
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, text)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        self._release_button()
        self._stop_trace()
        if self._recorder is not None:
            calib = self._store.path_for(self.schema_id, self.profile)  # type: ignore[arg-type]
            path = self._recorder.close(calib)
            print(f"session recorded: {path}  (replay: opengaze replay {path})")
        self.runtime.stop()
        self._tracker.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preview(frame) -> np.ndarray | None:
    """Head camera image, or the eye-camera images side by side (Modes B/C)."""
    eyes = [frame.eye_images[s] for s in ("left", "right") if s in frame.eye_images]
    if not eyes:
        return frame.head_image
    h = min(img.shape[0] for img in eyes)
    tiles = [cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h)) for img in eyes]
    return np.hstack(tiles)


def _launching_app() -> str:
    """The app macOS needs Accessibility permission for: whatever launched this terminal."""
    import os

    return {
        "Apple_Terminal": "Terminal",
        "iTerm.app": "iTerm",
        "vscode": "Visual Studio Code",
        "WarpTerminal": "Warp",
        "ghostty": "Ghostty",
    }.get(os.environ.get("TERM_PROGRAM", ""), "the app you launched opengaze from")


def _try_make_cursor() -> tuple[object | None, str | None]:
    """A cursor driver, or None and a user-facing explanation of what's missing."""
    if sys.platform != "darwin":
        return None, "moving the system cursor is only supported on macOS"
    try:
        from opengaze.control.cursor import QuartzCursorDriver, is_accessibility_granted
    except ImportError as exc:
        return None, f"pyobjc Quartz is missing ({exc}) — pip install -e '.[desktop]'"
    if not is_accessibility_granted():
        return None, (
            f"Accessibility permission not granted. System Settings › Privacy & Security › "
            f"Accessibility: turn on {_launching_app()}, then quit and reopen it. "
            f"Press P to open the settings."
        )
    try:
        return QuartzCursorDriver(), None
    except Exception as exc:
        return None, f"cursor driver failed to start: {exc}"


def _make_report(profile) -> object:
    """Minimal ProfileReport-like object for CalibrationWindow."""
    from opengaze.config.loader import ProfileReport
    from opengaze.modes import infer_mode
    mode = infer_mode({cam.role for cam in profile.enabled_cameras.values()})
    return ProfileReport(path=None, profile=profile, issues=[], mode=mode)  # type: ignore[arg-type]


def _calibration_window_class(adaptive: bool) -> type:
    if adaptive:
        from opengaze.desktop.adaptive_calibration_ui import AdaptiveCalibrationWindow
        return AdaptiveCalibrationWindow
    from opengaze.desktop.calibration_ui import CalibrationWindow
    return CalibrationWindow


def ensure_calibration(
    runtime: CaptureRuntime,
    report: ProfileReport,
    config: AppConfig,
    schema_id: str,
    adaptive: bool = False,
) -> None:
    """Run the calibration wizard (blocking) if this hardware has no calibration."""
    app = QApplication.instance()
    store = CalibrationStore()
    if store.exists(schema_id, report.profile):  # type: ignore[arg-type]
        return
    print("No calibration found — running calibration first…")
    cal_win = _calibration_window_class(adaptive)(
        runtime, report, config, store, owns_runtime=False
    )
    cal_win.show()
    while cal_win.isVisible():
        app.processEvents()
        time.sleep(0.03)
    if not store.exists(schema_id, report.profile):  # type: ignore[arg-type]
        print("Calibration was not saved — continuing without a model.")


def run_gaze_test(
    report: ProfileReport,
    config: AppConfig,
    adaptive: bool = False,
    *,
    trace: bool = False,
    record_video: bool = False,
) -> int:
    assert report.profile is not None
    app = QApplication.instance() or QApplication(sys.argv[:1])

    schema_id = profile_schema(report.profile)

    # Start runtime once — shared between calibration and gaze test so the camera
    # stays on continuously (no pause/restart between phases).
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()

    ensure_calibration(runtime, report, config, schema_id, adaptive)

    window = GazeTestWindow(
        runtime, report, config, schema_id, trace=trace, record_video=record_video
    )
    window.show()

    try:
        return app.exec()
    finally:
        runtime.stop()
