"""Gaze test environment: fullscreen window showing gaze dot, camera feed, blink state.

Works in every tracking mode (A webcam, B/C eye cameras) through ``GazeTracker``.

Flow
----
1. If no calibration: run calibration wizard first, then return here.
2. Fullscreen window opens.
3. Gaze dot tracks eye position.
4. Eye gestures (blinks, winks, glances) fire their keybinds from ``gestures:`` in the app
   config; the default binds a long blink to a click.
5. Tab toggles real cursor control on/off (key/mouse bindings only act while it's on).
6. C re-runs calibration.
7. Space pauses/resumes.
8. Escape quits.
"""

from __future__ import annotations

import logging
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from opengaze.calibration.storage import CalibrationStore
from opengaze.cameras.manager import CameraManager
from opengaze.cameras.runtime import CaptureRuntime
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.gaze.loop import GazeLoop
from opengaze.gaze.tracker import GazeTracker
from opengaze.gestures.bindings import ActionDispatcher
from opengaze.gestures.types import Gesture
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

_POLL_MS = 33   # ~30 fps


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

        # Cursor driver (optional); it is also the system input backend for keybinds
        self._cursor = _try_make_cursor()
        self._cursor_enabled = False  # off by default — toggle with Tab
        self._dispatcher = ActionDispatcher(
            config.gestures.bindings,
            self._cursor,
            {
                "pause": self._toggle_pause,
                "toggle_cursor": self._toggle_cursor,
                "recalibrate": lambda: QTimer.singleShot(0, self._run_calibration),
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
        self._paused = False
        self._click_flash_until: float = 0.0
        self._camera_frame: np.ndarray | None = None
        self._clicks = 0
        self._status = "Press C to calibrate, Tab to enable cursor, Space to pause, Esc to quit"

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
        if self._paused:
            # Keep gestures alive so an app binding (e.g. pause) can un-pause hands-free.
            state = self._loop.tick(self.runtime.manager)
            self._handle_gestures(state)
            self.update()
            return

        state = self._loop.tick(self.runtime.manager)
        frame = state.frame
        self._camera_frame = _preview(frame)
        self._left_blink = bool(frame.left_closed)
        self._right_blink = bool(frame.right_closed)
        self._blink_progress = max(state.gesture.blink_progress, state.gesture.wink_progress)
        self._gaze_x, self._gaze_y = state.gaze_x, state.gaze_y
        self._confidence = state.confidence

        if state.updated and self._cursor_enabled and self._cursor:
            px, py = self._cursor.norm_to_px(self._gaze_x, self._gaze_y)
            self._cursor.move(px, py)

        self._handle_gestures(state)
        self.update()

    def _handle_gestures(self, state) -> None:
        """Dispatch gesture events at the frozen pre-blink gaze position."""
        self._dispatcher.system_input_enabled = (
            self._cursor_enabled and not self._paused and self._model is not None
        )
        gaze_px = self._cursor.norm_to_px(self._gaze_x, self._gaze_y) if self._cursor else None
        for event in state.gesture.events:
            result = self._dispatcher.dispatch(event, gaze_px)
            binding = result.binding.describe() if result.binding else "unbound"
            self._last_gesture = f"{event.gesture.value} → {binding} ({result.outcome})"
            self._last_gesture_until = time.monotonic() + 1.5
            is_click = (
                result.binding.mouse == "click"
                if result.binding is not None
                else event.gesture is Gesture.LONG_BLINK
            )
            if is_click:
                self._clicks += 1
                self._click_flash_until = time.monotonic() + 0.3

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self._loop.reset()

    def _toggle_cursor(self) -> None:
        self._cursor_enabled = not self._cursor_enabled
        self._status = f"Cursor control {'ON' if self._cursor_enabled else 'OFF'}"

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        k = event.key()
        if k == Qt.Key.Key_Escape:
            self.close()
        elif k == Qt.Key.Key_Space:
            self._toggle_pause()
        elif k == Qt.Key.Key_Tab:
            self._toggle_cursor()
        elif k == Qt.Key.Key_C:
            self._run_calibration()
        elif k == Qt.Key.Key_K:
            self._open_activity("keyboard")
        elif k == Qt.Key.Key_G:
            self._open_activity("practice")
        else:
            super().keyPressEvent(event)

    def _open_activity(self, kind: str) -> None:
        if self._model is None:
            self._status = "No calibration — press C first"
            return
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

    def _run_calibration(self) -> None:
        self._timer.stop()
        try:
            from opengaze.desktop.calibration_ui import CalibrationWindow
            win = CalibrationWindow(
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

        if self._paused:
            self._draw_centered(p, "PAUSED — press Space to resume", w // 2, h // 2)
            if time.monotonic() < self._last_gesture_until:
                p.setPen(QColor("#f0b429"))
                p.drawText(16, 52, f"gesture: {self._last_gesture}")
            p.end()
            return

        self._draw_gaze(p, w, h)
        self._draw_camera_preview(p, w, h)
        self._draw_blink_indicators(p, w, h)
        self._draw_hud(p, w, h)
        p.end()

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

        cursor_color = _CURSOR_ON if self._cursor_enabled else _CURSOR_OFF
        cursor_state = "CURSOR ON" if self._cursor_enabled else "CURSOR OFF (Tab)"

        # Top bar
        p.drawText(16, 28,
            f"OpenGaze  |  {self.schema_id}  |  conf {self._confidence:.2f}"
            f"  |  clicks {self._clicks}"
        )
        p.setPen(cursor_color)
        p.drawText(w - 200, 28, cursor_state)
        p.setPen(_TEXT)

        # Bottom bar
        p.drawText(16, h - 16, "C=calibrate  K=keyboard  G=practice  Tab=cursor  Space=pause  Esc=quit")
        p.setPen(QColor("#f0b429"))
        p.drawText(16, h - 36, self._status)
        if time.monotonic() < self._last_gesture_until:
            p.drawText(16, 52, f"gesture: {self._last_gesture}")

        # Long blink hint
        if self._blink_progress > 0.05:
            p.setPen(QColor("#f0b429"))
            font2 = QFont("Helvetica Neue", 18, QFont.Weight.Bold)
            p.setFont(font2)
            p.drawText(w // 2 - 80, h - 80, "Hold blink...")

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


def _try_make_cursor() -> object | None:
    try:
        from opengaze.control.cursor import QuartzCursorDriver, is_accessibility_granted
        if is_accessibility_granted():
            return QuartzCursorDriver()
        log.info("Accessibility permission not granted — cursor control disabled")
    except Exception:
        pass
    return None


def _make_report(profile) -> object:
    """Minimal ProfileReport-like object for CalibrationWindow."""
    from opengaze.config.loader import ProfileReport
    from opengaze.modes import infer_mode
    mode = infer_mode({cam.role for cam in profile.enabled_cameras.values()})
    return ProfileReport(path=None, profile=profile, issues=[], mode=mode)  # type: ignore[arg-type]


def run_gaze_test(report: ProfileReport, config: AppConfig) -> int:
    assert report.profile is not None
    app = QApplication.instance() or QApplication(sys.argv[:1])

    schema_id = profile_schema(report.profile)

    # Start runtime once — shared between calibration and gaze test so the camera
    # stays on continuously (no pause/restart between phases).
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()

    store = CalibrationStore()
    if not store.exists(schema_id, report.profile):  # type: ignore[arg-type]
        print("No calibration found — running calibration first…")
        from opengaze.desktop.calibration_ui import CalibrationWindow
        cal_win = CalibrationWindow(runtime, report, config, store, owns_runtime=False)
        cal_win.show()
        while cal_win.isVisible():
            app.processEvents()
            time.sleep(0.03)
        if not store.exists(schema_id, report.profile):  # type: ignore[arg-type]
            print("Calibration was not saved — continuing without a model.")

    window = GazeTestWindow(runtime, report, config, schema_id)
    window.show()

    try:
        return app.exec()
    finally:
        runtime.stop()
