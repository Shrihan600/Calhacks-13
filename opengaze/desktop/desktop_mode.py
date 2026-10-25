"""System-wide gaze control: drive the real cursor over every app.

No fullscreen window. The same ``GazeLoop`` as the test environment runs in the
background; the macOS cursor follows your gaze and eye gestures fire their
``gestures:`` bindings (by default a long blink clicks, a held wink drags) in
whichever app is under the cursor. A click-through overlay draws the gesture
progress ring around the cursor, and the voice bar floats on top so dictation
and read-aloud are a gaze-click away.

Safety: moving the physical mouse/trackpad hands control back to the person
for 2 s; ⌃⌥G, the bar's 👁 button, or a gesture bound to ``pause`` pauses gaze.
"""

from __future__ import annotations

import logging
import signal
import sys
import time

from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from opengaze.calibration.storage import CalibrationStore
from opengaze.cameras.manager import CameraManager
from opengaze.cameras.runtime import CaptureRuntime
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.control.cursor import QuartzCursorDriver
from opengaze.control.safety import UserOverride
from opengaze.desktop.dictation_ui import DictationBar
from opengaze.desktop.macwin import become_accessory_app, float_everywhere
from opengaze.desktop.run_ui import ensure_calibration
from opengaze.gaze.loop import GazeLoop
from opengaze.gaze.tracker import GazeTracker
from opengaze.gestures.bindings import ActionDispatcher
from opengaze.modes import profile_schema
from opengaze.voice.service import VoiceService

log = logging.getLogger("opengaze.desktop_mode")

_RING = QColor("#f0b429")
_FLASH = QColor(0, 212, 255, 200)
_WARN = QColor("#f0b429")
_R = 26          # gesture ring radius around the cursor
_PAD = _R + 8    # repaint margin
# Show "face not found" only after this long, so ordinary blinks don't flicker it.
_NO_FACE_WARN_S = 1.0


class GazeCursorOverlay(QWidget):
    """Transparent, click-through, full-screen layer that also runs the gaze loop."""

    def __init__(
        self,
        runtime: CaptureRuntime,
        loop: GazeLoop,
        cursor: QuartzCursorDriver,
        config: AppConfig,
    ) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        # Calibration targets cover the primary display, so gaze maps onto it.
        cursor.set_display(screen.x(), screen.y(), screen.width(), screen.height())

        self._runtime = runtime
        self._loop = loop
        self._cursor = cursor
        self._override = UserOverride()
        self._dispatcher = ActionDispatcher(
            config.gestures.bindings,
            cursor,
            {
                "pause": self.toggle,
                "toggle_cursor": self.toggle,
                "recenter": loop.tracker.recenter,
                "quit": QApplication.quit,
            },
        )
        self.enabled = True

        self._pos = QPoint(-100, -100)
        self._progress = 0.0
        self._flash_until = 0.0
        self._last_face = time.monotonic()
        self._mouse_hold = False
        self._painted = QRect()

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def toggle(self) -> None:
        self.enabled = not self.enabled
        self._release_button()
        self._loop.reset()
        self._progress = 0.0
        self.update()

    def _release_button(self) -> None:
        """Let go of a held mouse button (a drag) whenever control is taken away."""
        if self._cursor.pressed:
            self._cursor.release()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        float_everywhere(self, click_through=True)

    def _tick(self) -> None:
        t = time.monotonic()
        # A held wink holds the button down: keep tracking the open eye so it drags.
        self._loop.track_one_eye = self._cursor.pressed
        state = self._loop.tick(self._runtime.manager, t)
        frame = state.frame
        if frame.left_closed is not None or frame.right_closed is not None:
            self._last_face = t

        allowed = self.enabled and self._override.gaze_allowed(self._cursor.position(), t)
        self._mouse_hold = self.enabled and not allowed
        if self._mouse_hold:
            self._release_button()

        px, py = self._cursor.norm_to_px(state.gaze_x, state.gaze_y)
        if allowed and state.updated:
            self._cursor.move(px, py)
            self._override.moved_to(px, py)

        # App bindings (pause …) always run, so a gesture can un-pause hands-free.
        self._dispatcher.system_input_enabled = allowed
        for event in state.gesture.events:
            result = self._dispatcher.dispatch(event, (px, py))
            if result.outcome == "done" and result.binding is not None and result.binding.mouse:
                self._override.moved_to(px, py)
                self._flash_until = t + 0.3

        self._progress = (
            max(state.gesture.blink_progress, state.gesture.wink_progress) if allowed else 0.0
        )
        self._pos = self.mapFromGlobal(QPoint(px, py))
        self._repaint_dirty()

    def _repaint_dirty(self) -> None:
        """Repaint only around the old and new ring, not the whole screen."""
        ring = QRect(self._pos.x() - _PAD, self._pos.y() - _PAD, _PAD * 2, _PAD * 2)
        self.update(ring.united(self._painted))
        self.update(QRect(self.width() - 360, 0, 360, 60))  # status corner
        self._painted = ring

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self.enabled:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self._pos.x(), self._pos.y()
        p.setBrush(Qt.BrushStyle.NoBrush)
        if time.monotonic() < self._flash_until:
            p.setPen(QPen(_FLASH, 4))
            p.drawEllipse(QPoint(cx, cy), _R, _R)
        elif self._progress > 0:
            pen = QPen(_RING, 4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawArc(cx - _R, cy - _R, _R * 2, _R * 2,
                      90 * 16, int(-self._progress * 360 * 16))

        note = ""
        if time.monotonic() - self._last_face > _NO_FACE_WARN_S:
            note = "👁 face not found — look at the screen"
        elif self._mouse_hold:
            note = "🖱 mouse in control"
        if note:
            p.setFont(QFont("Helvetica Neue", 13))
            p.setPen(_WARN)
            p.drawText(QRect(self.width() - 350, 30, 340, 24),
                       Qt.AlignmentFlag.AlignRight, note)
        p.end()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        self._release_button()
        super().closeEvent(event)


def run_desktop_mode(report: ProfileReport, config: AppConfig, voice: VoiceService) -> int:
    assert report.profile is not None
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C in the terminal quits
    app = QApplication.instance() or QApplication(sys.argv[:1])

    schema_id = profile_schema(report.profile)
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()
    tracker: GazeTracker | None = None
    try:
        ensure_calibration(runtime, report, config, schema_id)
        tracker = GazeTracker(report.profile, config)
        try:
            bundle = CalibrationStore().load_bundle(schema_id, report.profile)  # type: ignore[arg-type]
        except (FileNotFoundError, ValueError):
            print("No usable calibration — run `opengaze calibrate` first.")
            return 1
        tracker.load_eye_states(bundle.eye_states)
        loop = GazeLoop(tracker, bundle.model, config.gestures)

        # Only after calibration: its fullscreen window needs normal activation.
        become_accessory_app()
        app.setQuitOnLastWindowClosed(False)
        overlay = GazeCursorOverlay(runtime, loop, QuartzCursorDriver(), config)
        overlay.show()
        bar = DictationBar(voice, gaze=overlay)
        bar.show()
        return app.exec()
    finally:
        runtime.stop()
        if tracker is not None:
            tracker.close()
