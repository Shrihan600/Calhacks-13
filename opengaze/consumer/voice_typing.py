"""Voice typing: a mic button beside the text caret in any app.

When a text field in another app has focus, a small round mic floats just past the caret
(found through the Accessibility API). Click it (or triple blink) and speak; recording ends
by itself after a pause, and the words are typed where the caret is. A "Listening" pill
shows under the mic while it records.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QWidget

from opengaze.consumer import theme
from opengaze.control.ax import FocusProbe, TextFocus
from opengaze.control.cursor import is_accessibility_granted
from opengaze.control.keystrokes import FocusTracker, set_clipboard, type_text
from opengaze.desktop.macwin import float_everywhere
from opengaze.voice.service import VoiceService, VoiceState

log = logging.getLogger("opengaze.consumer.voice_typing")

_TICK_MS = 50
_PROBE_EVERY = 5      # ticks between caret lookups (~4 per second)
_BTN = 44
_NOTE_S = 4.0

_FLOATING = (
    Qt.WindowType.FramelessWindowHint
    | Qt.WindowType.WindowStaysOnTopHint
    | Qt.WindowType.Tool
    | Qt.WindowType.WindowDoesNotAcceptFocus
)


def _mic_path(c: QPointF, s: float) -> QPainterPath:
    """The mic glyph (24-unit icon) centred on ``c`` at scale ``s``."""
    path = QPainterPath()
    ox, oy = c.x() - 12 * s, c.y() - 12 * s
    path.addRoundedRect(QRectF(ox + 9 * s, oy + 3 * s, 6 * s, 11 * s), 3 * s, 3 * s)
    path.moveTo(ox + 5 * s, oy + 11 * s)
    path.arcTo(QRectF(ox + 5 * s, oy + 4 * s, 14 * s, 14 * s), 180, 180)
    path.moveTo(ox + 12 * s, oy + 18 * s)
    path.lineTo(ox + 12 * s, oy + 21 * s)
    return path


class MicButton(QWidget):
    def __init__(self, on_click: Callable[[], None]) -> None:
        super().__init__(None, _FLOATING)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setFixedSize(_BTN, _BTN)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName("Voice typing")
        self._on_click = on_click
        self.live = False

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        float_everywhere(self)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_click()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        if self.live:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.qcolor(theme.PURPLE, 0.2))
            p.drawEllipse(c, 21, 21)
            p.setBrush(theme.qcolor(theme.PURPLE))
        else:
            p.setPen(QPen(theme.qcolor(theme.DOT_OFF), 1.5))
            p.setBrush(theme.qcolor("#FFFFFF"))
        p.drawEllipse(c, 16, 16)
        pen = QPen(theme.qcolor("#FFFFFF" if self.live else theme.PURPLE), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(_mic_path(c, 0.72))
        p.end()


class ListeningPill(QWidget):
    """Dark status pill under the mic: "Listening" with a level meter, or a short note."""

    def __init__(self) -> None:
        super().__init__(None, _FLOATING | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.text = ""
        self.level: float | None = None  # None: no meter
        self.setFixedHeight(36)

    def show_text(self, text: str, level: float | None) -> None:
        self.text, self.level = text, level
        width = 28 + (38 if level is not None else 0) + _text_width(text)
        self.setFixedWidth(int(width))
        self.update()
        if not self.isVisible():
            self.show()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        float_everywhere(self, click_through=True)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(theme.INK))
        p.drawRoundedRect(QRectF(self.rect()), 12, 12)
        x = 14.0
        if self.level is not None:
            p.setBrush(theme.qcolor(theme.LIME))
            flicker = int(time.monotonic() * 8)
            for i, base in enumerate((8, 14, 10, 16, 6)):
                swing = min(1.0, self.level * 3 + 0.2 * ((i + flicker) % 2))
                h = base * (0.45 + 0.55 * swing)
                p.drawRoundedRect(QRectF(x + i * 6, self.height() / 2 - h / 2, 3, h), 1.5, 1.5)
            x += 38
        p.setPen(theme.qcolor(theme.CREAM))
        p.setFont(theme.text_font(13, 600))
        p.drawText(QRectF(x, 0, self.width() - x, self.height()),
                   Qt.AlignmentFlag.AlignVCenter, self.text)
        p.end()


def _text_width(text: str) -> int:
    return QFontMetrics(theme.text_font(13, 600)).horizontalAdvance(text)


class VoiceTyping:
    """Runs the mic button: follows the caret, records, and types what was said."""

    def __init__(self, voice: VoiceService) -> None:
        self.voice = voice
        self.enabled = False
        self._probe = FocusProbe()
        self._focus = FocusTracker()
        self.button = MicButton(self.toggle)
        self.pill = ListeningPill()
        self._target: TextFocus | None = None
        self._note = ""
        self._note_until = 0.0
        self._ticks = 0
        self._timer = QTimer()
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_voice(self, voice: VoiceService) -> None:
        self.voice.shutdown()
        self.voice = voice

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        if not on:
            self.voice.cancel()
            self.button.hide()
            self.pill.hide()

    def toggle(self) -> None:
        """Start or stop listening (the mic button, a triple blink, ⌃⌥D)."""
        if not self.enabled:
            self._say("Voice typing is off. Turn it on in the menu bar.")
            return
        if not self.voice.can_dictate:
            self._say("Voice typing needs a Cartesia API key")
            return
        self._focus.update()
        self.voice.toggle_dictation()

    def _say(self, note: str) -> None:
        self._note, self._note_until = note, time.monotonic() + _NOTE_S

    def _deliver(self, transcript: str) -> None:
        if not is_accessibility_granted():
            set_clipboard(transcript)
            self._say("Copied. Press ⌘V to paste")
            return
        self._focus.restore()
        type_text(transcript + " ")

    def _tick(self) -> None:
        self._ticks += 1
        self._focus.update()
        for transcript in self.voice.poll():
            self._deliver(transcript)
        state = self.voice.state
        busy = state in (VoiceState.LISTENING, VoiceState.TRANSCRIBING)
        if self.enabled and self._ticks % _PROBE_EVERY == 0 and not busy:
            self._target = self._probe.text_focus()
        self._place(busy)

    def _place(self, busy: bool) -> None:
        now = time.monotonic()
        note = self._note if now < self._note_until else ""
        if not self.enabled or (self._target is None and not busy and not note):
            self.button.hide()
            self.pill.hide()
            return
        if self._target is not None:
            self.button.move(*self._button_pos(self._target))
        elif not self.button.isVisible():
            screen = QApplication.primaryScreen().availableGeometry()
            self.button.move(screen.center().x() - _BTN // 2, screen.bottom() - 120)
        live = self.voice.state is VoiceState.LISTENING
        if self.button.live != live:
            self.button.live = live
            self.button.update()
        self.button.show()

        if live:
            self.pill.show_text("Listening", self.voice.mic_level)
        elif self.voice.state is VoiceState.TRANSCRIBING:
            self.pill.show_text("Typing…", None)
        elif note:
            self.pill.show_text(note, None)
        else:
            self.pill.hide()
            return
        b = self.button.geometry()
        self.pill.move(b.center().x() - self.pill.width() // 2, b.bottom() + 8)

    def _button_pos(self, t: TextFocus) -> tuple[int, int]:
        screen = QApplication.primaryScreen().geometry()
        cy = t.y + t.height / 2
        if t.caret:
            x = t.x + t.width + 10
            if x + _BTN > screen.right():
                x = t.x - _BTN - 10
        else:
            x = t.x + t.width - _BTN - 6
        return int(x), int(cy - _BTN / 2)

    def shutdown(self) -> None:
        self._timer.stop()
        self.button.close()
        self.pill.close()
        self.voice.shutdown()
