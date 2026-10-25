"""System-wide gaze control: drive the real cursor over every app.

No fullscreen window. Gaze tracking runs in the background; the macOS cursor
follows your gaze and a long blink clicks wherever you were looking. A
click-through overlay draws the blink-progress ring around the cursor, and the
voice bar floats on top so dictation and read-aloud are a blink-click away.

Safety: moving the physical mouse/trackpad hands control back to the person
for 2 s; ⌃⌥G or the bar's 👁 button pauses gaze entirely.
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
from opengaze.cameras.types import CameraRole
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.control.cursor import QuartzCursorDriver
from opengaze.control.tracker import GazeTracker, UserOverride
from opengaze.desktop.dictation_ui import DictationBar
from opengaze.desktop.macwin import become_accessory_app, float_everywhere
from opengaze.desktop.run_ui import _make_detector, ensure_calibration, schema_for
from opengaze.voice.service import VoiceService

log = logging.getLogger("opengaze.desktop_mode")

_RING = QColor("#f0b429")
_FLASH = QColor(0, 212, 255, 200)
_WARN = QColor("#f0b429")
_R = 26          # blink ring radius around the cursor
_PAD = _R + 8    # repaint margin
# Show "face not found" only after this long, so ordinary blinks don't flicker it.
_NO_FACE_WARN_S = 1.0


class GazeCursorOverlay(QWidget):
    """Transparent, click-through, full-screen layer that also runs the gaze loop."""

    def __init__(self, tracker: GazeTracker, cursor: QuartzCursorDriver) -> None:
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
        self.setGeometry(QApplication.primaryScreen().geometry())

        self._tracker = tracker
        self._cursor = cursor
        self._override = UserOverride()
        self.enabled = True

        self._pos = QPoint(-100, -100)
        self._blink_prog = 0.0
        self._flash_until = 0.0
        self._last_face = time.monotonic()
        self._mouse_hold = False
        self._painted: QRect = QRect()

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def toggle(self) -> None:
        self.enabled = not self.enabled
        self._tracker.reset()
        self._blink_prog = 0.0
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        float_everywhere(self, click_through=True)

    def _tick(self) -> None:
        u = self._tracker.step()
        if u is None:
            return
        now = time.monotonic()
        if u.face:
            self._last_face = now
        if not self.enabled:
            return

        allowed = self._override.gaze_allowed(self._cursor.position(), now)
        self._mouse_hold = not allowed
        px, py = self._cursor.norm_to_px(u.x, u.y)
        if allowed and u.fresh:
            self._cursor.move(px, py)
            self._override.moved_to(px, py)
        if allowed and u.blink.clicked:
            self._cursor.click(px, py)
            self._override.moved_to(px, py)
            self._flash_until = now + 0.3

        self._blink_prog = u.blink.blink_progress if allowed else 0.0
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
        elif self._blink_prog > 0:
            pen = QPen(_RING, 4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawArc(cx - _R, cy - _R, _R * 2, _R * 2,
                      90 * 16, int(-self._blink_prog * 360 * 16))

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
        super().closeEvent(event)


def run_desktop_mode(report: ProfileReport, config: AppConfig, voice: VoiceService) -> int:
    assert report.profile is not None
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C in the terminal quits
    app = QApplication.instance() or QApplication(sys.argv[:1])

    schema_id = schema_for(report)
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()
    detector = None
    try:
        ensure_calibration(runtime, report, config, schema_id)
        try:
            model = CalibrationStore().load(schema_id, report.profile)  # type: ignore[arg-type]
        except (FileNotFoundError, ValueError):
            print("No usable calibration — run `opengaze calibrate` first.")
            return 1
        detector = _make_detector(config)
        if detector is None:
            print("Face tracking unavailable — run `opengaze download-models`.")
            return 1

        head_ids = {
            cid for cid, cam in report.profile.enabled_cameras.items()
            if cam.role is CameraRole.HEAD
        }
        tracker = GazeTracker(runtime.manager, head_ids, detector, model)

        # Only after calibration: its fullscreen window needs normal activation.
        become_accessory_app()
        app.setQuitOnLastWindowClosed(False)
        overlay = GazeCursorOverlay(tracker, QuartzCursorDriver())
        overlay.show()
        bar = DictationBar(voice, gaze=overlay)
        bar.show()
        return app.exec()
    finally:
        runtime.stop()
        if detector is not None:
            detector.close()
