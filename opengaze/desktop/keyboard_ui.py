"""Gaze-controlled on-screen keyboard.

Dwell (0.8 s) on any key to type it.  Blink immediately types the
highlighted key.  SPEAK uses macOS ``say`` for TTS.  Esc returns to
the gaze test window.
"""

from __future__ import annotations

import logging
import math
import subprocess
import time

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from opengaze.cameras.runtime import CaptureRuntime
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.gaze.loop import GazeLoop, clicked
from opengaze.gaze.tracker import GazeTracker

log = logging.getLogger("opengaze.keyboard_ui")

ROWS: list[list[str]] = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["SPACE", "⌫", "CLEAR", "SPEAK"],
]
_SPECIAL_W = {"SPACE": 4, "⌫": 2, "CLEAR": 2, "SPEAK": 2}
_KEY_DWELL_S = 0.8
_KEY_COOL_S = 0.5

def _nearest_key(rects: dict[str, QRect], gx: int, gy: int, kb_top: int) -> str | None:
    """Return the key whose centre is closest to (gx, gy), or None if above keyboard."""
    if gy < kb_top - 30:
        return None
    best, best_d = None, float("inf")
    for key, rect in rects.items():
        d = math.hypot(gx - rect.center().x(), gy - rect.center().y())
        if d < best_d:
            best_d, best = d, key
    return best

_BG         = QColor("#0d0f12")
_TEXT_BG    = QColor("#080a0d")
_KEY_IDLE   = QColor("#1c2030")
_KEY_HOVER  = QColor("#252c40")
_KEY_RING   = QColor("#f0b429")
_KEY_HIT    = QColor("#2e9e5b")
_TEXT_BRIGHT = QColor("#e8e8e8")
_DIM        = QColor("#555")
_GAZE       = QColor(0, 212, 255, 180)
_CURSOR_COL = QColor("#00d4ff")


def _build_rects(w: int, h: int) -> dict[str, QRect]:
    rects: dict[str, QRect] = {}
    n_rows = len(ROWS)
    kb_top = int(h * 0.32)
    kb_h   = int(h * 0.62)
    row_h  = kb_h // n_rows
    max_cols = max(len(r) for r in ROWS[:-1])
    key_w = int(w * 0.86 / max_cols)
    gap = 8

    for ri, row in enumerate(ROWS):
        y  = kb_top + ri * row_h + gap
        kh = row_h - gap * 2
        if ri < n_rows - 1:
            total = len(row) * key_w + (len(row) - 1) * gap
            x = (w - total) // 2
            for key in row:
                rects[key] = QRect(x, y, key_w, kh)
                x += key_w + gap
        else:
            widths = {k: key_w * _SPECIAL_W.get(k, 2) for k in row}
            total = sum(widths.values()) + (len(row) - 1) * gap
            x = (w - total) // 2
            for key in row:
                rects[key] = QRect(x, y, widths[key], kh)
                x += widths[key] + gap
    return rects


class GazeKeyboardWindow(QWidget):
    def __init__(
        self,
        runtime: CaptureRuntime,
        report: ProfileReport,
        config: AppConfig,
        schema_id: str,
        model: object,
        tracker: GazeTracker | None = None,
    ) -> None:
        super().__init__()
        assert report.profile is not None
        self._runtime = runtime
        self._model = model
        self._owns_tracker = tracker is None
        self._tracker = tracker or GazeTracker(report.profile, config)
        self._gestures = config.gestures
        self._loop = GazeLoop(self._tracker, model, config.gestures)  # type: ignore[arg-type]

        self._gaze_x: float = 0.5
        self._gaze_y: float = 0.5
        self._blink_prog: float = 0.0

        self._text = ""
        self._rects: dict[str, QRect] = {}
        self._hovered: str | None = None
        self._dwell_key: str | None = None
        self._dwell_start: float = 0.0
        self._dwell_prog: float = 0.0
        self._cool_until: float = 0.0
        self._hit_key: str | None = None
        self._hit_until: float = 0.0

        self.setWindowTitle("OpenGaze — Keyboard")
        self.setStyleSheet("background: #0d0f12;")
        self.showFullScreen()

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------
    # Gaze tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        st = self._loop.tick(self._runtime.manager)
        self._blink_prog = st.gesture.blink_progress
        self._gaze_x, self._gaze_y = st.gaze_x, st.gaze_y
        click = clicked(st, self._gestures)

        w, h = self.width(), self.height()
        gx = int(self._gaze_x * w)
        gy = int(self._gaze_y * h)
        kb_top = int(h * 0.32)
        hovered = _nearest_key(self._rects, gx, gy, kb_top)

        now = time.monotonic()
        in_cool = now < self._cool_until

        if hovered != self._dwell_key:
            self._dwell_key = hovered
            self._dwell_start = now
            self._dwell_prog = 0.0
        elif hovered and not in_cool:
            elapsed = now - self._dwell_start
            self._dwell_prog = min(elapsed / _KEY_DWELL_S, 1.0)
            if elapsed >= _KEY_DWELL_S:
                self._activate(hovered, now)

        self._hovered = hovered

        if click and hovered and not in_cool:
            self._activate(hovered, now)

        self.update()

    def _activate(self, key: str, now: float) -> None:
        if key == "SPACE":
            self._text += " "
        elif key == "⌫":
            self._text = self._text[:-1]
        elif key == "CLEAR":
            self._text = ""
        elif key == "SPEAK":
            if self._text.strip():
                try:
                    subprocess.Popen(["say", self._text])
                except Exception:
                    pass
            return
        else:
            self._text += key
        self._hit_key = key
        self._hit_until = now + 0.35
        self._dwell_key = None
        self._dwell_prog = 0.0
        self._cool_until = now + _KEY_COOL_S

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._rects = _build_rects(self.width(), self.height())
        super().resizeEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        self._rects = _build_rects(self.width(), self.height())
        super().showEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), _BG)
        self._draw_text_area(p, w, h)
        self._draw_keys(p)
        self._draw_gaze(p, w, h)
        p.end()

    def _draw_text_area(self, p: QPainter, w: int, h: int) -> None:
        th = int(h * 0.28)
        p.fillRect(0, 0, w, th, _TEXT_BG)
        p.setPen(QPen(QColor("#1a1f2a"), 1))
        p.drawLine(0, th, w, th)

        p.setFont(QFont("Menlo", 11))
        p.setPen(_DIM)
        p.drawText(32, 22, "TYPED TEXT  —  dwell 0.8s or blink to type  •  Esc = back")

        p.setFont(QFont("Helvetica Neue", 36))
        p.setPen(_CURSOR_COL if self._text else _DIM)
        display = self._text or "Start typing..."
        # Trim to fit
        fm = p.fontMetrics()
        max_w = w - 80
        while len(display) > 1 and fm.horizontalAdvance(display) > max_w:
            display = display[1:]
        p.drawText(40, int(th * 0.7), display)

        if self._text:
            cx = 40 + fm.horizontalAdvance(display) + 4
            cy_top = int(th * 0.45)
            p.setPen(QPen(_CURSOR_COL, 2))
            p.drawLine(cx, cy_top, cx, cy_top + 40)

    def _draw_keys(self, p: QPainter) -> None:
        now = time.monotonic()
        for key, rect in self._rects.items():
            is_hit = key == self._hit_key and now < self._hit_until
            is_hov = key == self._hovered
            bg = _KEY_HIT if is_hit else (_KEY_HOVER if is_hov else _KEY_IDLE)

            p.setBrush(bg)
            p.setPen(QPen(QColor("#2a3040"), 1))
            p.drawRoundedRect(rect, 10, 10)

            if key == self._dwell_key and self._dwell_prog > 0:
                pen = QPen(_KEY_RING, 5)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                cx, cy = rect.center().x(), rect.center().y()
                r = min(rect.width(), rect.height()) // 2 - 4
                p.drawArc(cx - r, cy - r, r * 2, r * 2,
                          90 * 16, int(-self._dwell_prog * 360 * 16))

            fsz = 13 if len(key) > 1 else 22
            p.setFont(QFont("Helvetica Neue", fsz,
                            QFont.Weight.Bold if is_hov else QFont.Weight.Normal))
            p.setPen(_KEY_RING if is_hov else (_TEXT_BRIGHT if is_hit else QColor("#9ba8c0")))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, key)

    def _draw_gaze(self, p: QPainter, w: int, h: int) -> None:
        gx = int(self._gaze_x * w)
        gy = int(self._gaze_y * h)
        if self._blink_prog > 0:
            pen = QPen(_KEY_RING, 3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(gx - 20, gy - 20, 40, 40,
                      90 * 16, int(-self._blink_prog * 360 * 16))
        p.setBrush(_GAZE)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(gx - 6, gy - 6, 12, 12)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        if self._owns_tracker:
            self._tracker.close()
        super().closeEvent(event)

