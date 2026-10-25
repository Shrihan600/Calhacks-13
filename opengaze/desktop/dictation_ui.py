"""System-wide voice bar: dictate into any app, read any selection aloud.

A small always-on-top bar floats over every app and Space.

* ⌃⌥D, or click 🎤: dictate with Cartesia Ink-Whisper. The transcript is
  typed into whichever app had focus; recording stops after a pause.
* ⌃⌥S, or click 🔊: read the selected text in the focused app aloud with
  Cartesia Sonic. Press again to stop.
* ⌃⌥G, or click 👁 (when launched with gaze): pause/resume the gaze cursor.
* Drag to move, right-click to quit.

Clicks work with a real mouse or with OpenGaze's gaze cursor + blink click,
so the bar is usable hands-free.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Protocol

from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QApplication, QMenu, QWidget

from opengaze.control.cursor import is_accessibility_granted, open_accessibility_prefs
from opengaze.control.keystrokes import FocusTracker, copy_selection, set_clipboard, type_text
from opengaze.desktop.macwin import become_accessory_app, float_everywhere
from opengaze.voice.service import VoiceService, VoiceState

log = logging.getLogger("opengaze.dictation_ui")

_H = 76
_BTN = 52

_BG       = QColor(13, 15, 18, 235)
_BORDER   = QColor("#2a3040")
_IDLE     = QColor("#1c2030")
_LIVE     = QColor("#c0392b")
_BUSY     = QColor("#6c4ab6")
_SPEAKING = QColor("#2e9e5b")
_TEXT     = QColor("#e8e8e8")
_DIM      = QColor("#7a8499")
_ACCENT   = QColor("#00d4ff")

# NSEvent modifier flags and key codes (US layout positions).
_CTRL, _OPT, _CMD = 1 << 18, 1 << 19, 1 << 20
_KEY_D, _KEY_S, _KEY_G = 2, 1, 5


class GazeSwitch(Protocol):
    enabled: bool

    def toggle(self) -> None: ...


class DictationBar(QWidget):
    def __init__(self, voice: VoiceService, gaze: GazeSwitch | None = None) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        width = 540 if gaze else 460
        self.setFixedSize(width, _H)
        self.setWindowTitle("OpenGaze Voice")

        self._voice = voice
        self._gaze = gaze
        self._focus = FocusTracker()
        self._can_type = is_accessibility_granted()
        self._note = ""
        self._note_until = 0.0
        self._last_heard = ""
        self._drag_from: QPoint | None = None
        self._monitors: list[object] = []

        top = (_H - _BTN) // 2
        self._mic_rect = QRect(12, top, _BTN, _BTN)
        self._spk_rect = QRect(width - 12 - _BTN, top, _BTN, _BTN)
        self._eye_rect = QRect(self._spk_rect.left() - 10 - _BTN, top, _BTN, _BTN)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.center().x() - width // 2, screen.top() + 12)

        if not self._can_type:
            self._say(
                "Accessibility off: transcripts go to the clipboard (⌘V). "
                "Enable it for typing + hotkeys.", 12,
            )

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------
    # macOS window behaviour + hotkeys
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        float_everywhere(self)
        if not self._monitors:
            self._install_hotkeys()

    def _install_hotkeys(self) -> None:
        try:
            import AppKit
        except ImportError:
            return

        def matches(ev) -> str | None:
            flags = int(ev.modifierFlags())
            if flags & _CTRL and flags & _OPT and not flags & _CMD:
                return {_KEY_D: "dictate", _KEY_S: "speak", _KEY_G: "gaze"}.get(
                    int(ev.keyCode())
                )
            return None

        def on_global(ev) -> None:
            action = matches(ev)
            if action:
                self._trigger(action)

        def on_local(ev):
            action = matches(ev)
            if action:
                self._trigger(action)
                return None
            return ev

        mask = AppKit.NSEventMaskKeyDown
        self._monitors = [
            AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, on_global),
            AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, on_local),
        ]

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _trigger(self, action: str) -> None:
        if action == "gaze":
            if self._gaze is not None:
                self._gaze.toggle()
                self._say(f"Gaze cursor {'on' if self._gaze.enabled else 'paused'}")
        elif action == "dictate":
            self._voice.toggle_dictation()
        elif action == "speak":
            if self._voice.state is VoiceState.SPEAKING:
                self._voice.stop_speaking()
                return
            self._focus.restore()
            if not self._can_type:
                self._say("Reading selections needs Accessibility permission")
                return
            text = copy_selection().strip()
            if text:
                self._voice.speak(text)
            else:
                self._say(f"Select some text in {self._focus.last_name} first")
        self.update()

    def _deliver(self, transcript: str) -> None:
        self._last_heard = transcript
        if not self._can_type:
            set_clipboard(transcript)
            self._say("Copied — press ⌘V to paste")
            return
        self._focus.restore()
        type_text(transcript + " ")
        self._say(f"Typed into {self._focus.last_name}")

    def _say(self, note: str, seconds: float = 4.0) -> None:
        self._note = note
        self._note_until = time.monotonic() + seconds

    def _tick(self) -> None:
        self._focus.update()
        for transcript in self._voice.poll():
            self._deliver(transcript)
        self.update()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position().toPoint()
        if event.button() == Qt.MouseButton.RightButton:
            self._context_menu(event.globalPosition().toPoint())
        elif self._mic_rect.contains(pos):
            self._trigger("dictate")
        elif self._spk_rect.contains(pos):
            self._trigger("speak")
        elif self._gaze is not None and self._eye_rect.contains(pos):
            self._trigger("gaze")
        else:
            self._drag_from = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_from is not None:
            self.move(event.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag_from = None

    def _context_menu(self, at: QPoint) -> None:
        menu = QMenu()
        if not self._can_type:
            menu.addAction("Open Accessibility settings…", open_accessibility_prefs)
        menu.addAction("Quit OpenGaze Voice", QApplication.quit)
        menu.exec(at)

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        state = self._voice.state

        p.setBrush(_BG)
        p.setPen(QPen(_LIVE if state is VoiceState.LISTENING else _BORDER, 2))
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), _H // 2, _H // 2)

        mic_bg = {VoiceState.LISTENING: _LIVE, VoiceState.TRANSCRIBING: _BUSY}.get(state, _IDLE)
        self._draw_button(p, self._mic_rect, mic_bg,
                          "■" if state is VoiceState.LISTENING else "🎤")
        if state is VoiceState.LISTENING:
            lvl = self._voice.mic_level
            r = self._mic_rect.adjusted(-4, -4, 4, 4)
            p.setPen(QPen(QColor(192, 57, 43, int(80 + 175 * lvl)), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(r)
        self._draw_button(p, self._spk_rect,
                          _SPEAKING if state is VoiceState.SPEAKING else _IDLE,
                          "■" if state is VoiceState.SPEAKING else "🔊")
        if self._gaze is not None:
            self._draw_button(p, self._eye_rect,
                              _SPEAKING if self._gaze.enabled else _IDLE,
                              "👁" if self._gaze.enabled else "⏸")

        tx = self._mic_rect.right() + 14
        right = self._eye_rect if self._gaze is not None else self._spk_rect
        tw = right.left() - 14 - tx
        now = time.monotonic()
        line1 = self._note if now < self._note_until else self._voice.status
        if self._last_heard and state is VoiceState.IDLE and now >= self._note_until:
            line1 = f"“{self._last_heard}”"
        p.setFont(QFont("Helvetica Neue", 13))
        p.setPen(_TEXT)
        fm = p.fontMetrics()
        p.drawText(tx, 32, fm.elidedText(line1, Qt.TextElideMode.ElideRight, tw))

        p.setFont(QFont("Menlo", 10))
        p.setPen(_DIM)
        gaze_hint = " · ⌃⌥G gaze" if self._gaze is not None else ""
        hint = f"⌃⌥D dictate · ⌃⌥S read{gaze_hint} · → {self._focus.last_name}"
        p.drawText(tx, 54, p.fontMetrics().elidedText(hint, Qt.TextElideMode.ElideRight, tw))
        p.end()

    def _draw_button(self, p: QPainter, rect: QRect, bg: QColor, label: str) -> None:
        p.setBrush(bg)
        p.setPen(QPen(_BORDER, 1))
        p.drawEllipse(rect)
        p.setPen(_ACCENT if bg == _IDLE else _TEXT)
        p.setFont(QFont("Helvetica Neue", 20))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        try:
            import AppKit

            for m in self._monitors:
                AppKit.NSEvent.removeMonitor_(m)
        except Exception:
            pass
        self._monitors = []
        self._voice.shutdown()
        super().closeEvent(event)


def run_dictation_bar(voice: VoiceService) -> int:
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C in the terminal quits
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    become_accessory_app()
    bar = DictationBar(voice)
    bar.show()
    return app.exec()
