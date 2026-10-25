"""EyeMouse's gaze keyboard: a floating panel that types into whichever app has focus.

It never takes focus itself (a non-activating panel), so the text field you were in keeps
it. Look at a key and double blink: the gaze cursor clicks the key like any button. Word
suggestions complete the word being typed on this keyboard.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from opengaze.consumer import theme
from opengaze.control.keystrokes import KEY_DELETE, KEY_RETURN, FocusTracker, tap_key, type_text
from opengaze.desktop.macwin import float_everywhere

log = logging.getLogger("opengaze.consumer.keyboard")

_H = 420
_MARGIN = 40
_BOTTOM = 28
_N_SUGGESTIONS = 3
# How often to note which app has focus while the keyboard is open.
_FOCUS_POLL_MS = 250

LETTERS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
SYMBOLS = ["1234567890", "-/:;()&@\"", "?!'#%*+"]
_STARTERS = ["I", "Thanks", "Yes"]

# Common words, most useful first, for completing the word being typed.
WORDS = (  # noqa: SIM905 (a word list reads better as prose)
    "I you the and to a yes no please thanks thank it is in my me what that this can "
    "need want help water bathroom now later okay ok good great sure not of on for with "
    "have be are was do don't i'm it's there here when where how why who will would "
    "could should love tired hungry thirsty hot cold pain hurts more less stop wait come "
    "go home bed tea coffee food eat drink phone call text tomorrow today tonight morning "
    "afternoon evening night hello hi bye see soon again sorry fine well bad better "
    "doctor nurse medicine family friend mom dad sister brother bring big small umbrella "
    "some any also just about from up down out off over into after before then them they "
    "we us our your his her their all one two three time day week minute hour know think "
    "feel like look watch read music tv open close turn light window door chair cushion "
    "wheelchair car outside inside walk sit move again because maybe really very much"
).split()


def suggest(prefix: str, n: int = _N_SUGGESTIONS) -> list[str]:
    """Up to ``n`` words that complete ``prefix`` (keeping its capital); starters if empty."""
    if not prefix:
        return _STARTERS[:n]
    lower = prefix.lower()
    out: list[str] = []
    for word in WORDS:
        if word.lower().startswith(lower) and word.lower() != lower:
            shown = word[0].upper() + word[1:] if prefix[0].isupper() else word
            if shown not in out:
                out.append(shown)
        if len(out) == n:
            break
    return out


def _key_style() -> str:
    return f"""
QPushButton {{
    border: 0; border-radius: 16px; min-height: 72px; color: {theme.CREAM};
    background: {theme.NIGHT_KEY}; font-family: "{theme.TEXT}"; font-size: 26px;
    font-weight: 500;
}}
QPushButton:hover {{ background: {theme.NIGHT_HOVER}; }}
QPushButton[kind="mod"] {{ background: {theme.NIGHT_MOD}; font-size: 17px; font-weight: 600; }}
QPushButton[kind="mod"]:hover {{ background: {theme.NIGHT_HOVER}; }}
QPushButton[kind="mod"][on="true"] {{ background: {theme.PURPLE}; }}
QPushButton[kind="go"] {{ background: {theme.PURPLE}; font-size: 17px; font-weight: 600; }}
QPushButton[kind="go"]:hover {{ background: #6A2DF5; }}
QPushButton[kind="chip"] {{
    min-height: 48px; padding: 0 22px; border-radius: 14px; background: {theme.CREAM};
    color: {theme.INK}; font-size: 18px; font-weight: 600;
}}
QPushButton[kind="chip"]:hover {{ background: {theme.LIME}; }}
QPushButton[kind="close"] {{
    min-height: 48px; min-width: 48px; border-radius: 14px; background: transparent;
    font-size: 22px;
}}
QPushButton[kind="close"]:hover {{ background: {theme.NIGHT_KEY}; }}
QLabel {{ color: {theme.FAINT}; font-family: "{theme.TEXT}"; font-size: 15px; }}
"""


class GazeKeyboard(QWidget):
    def __init__(self, on_close: Callable[[], None] | None = None) -> None:
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
        self.setWindowTitle(f"{theme.APP_NAME} Keyboard")
        self.setStyleSheet(_key_style())
        self._on_close = on_close
        # The app to type into. Noted continuously, not on a key press: by then a click on
        # the keyboard may already have brought this app to the front.
        self._focus = FocusTracker()
        self._focus_timer = QTimer(self)
        self._focus_timer.setInterval(_FOCUS_POLL_MS)
        self._focus_timer.timeout.connect(self._focus.update)
        self._word = ""       # the word typed so far on this keyboard
        self._shift = False
        self._symbols = False

        col = QVBoxLayout(self)
        col.setContentsMargins(16, 16, 16, 16)
        col.setSpacing(10)
        top = QHBoxLayout()
        top.setContentsMargins(4, 0, 4, 2)
        top.setSpacing(10)
        self._chips = [self._button("", "chip", None) for _ in range(_N_SUGGESTIONS)]
        for chip in self._chips:
            chip.clicked.connect(lambda _=False, c=chip: self._pick(c.text()))
            top.addWidget(chip)
        top.addStretch(1)
        top.addWidget(QLabel("Look at a key, then double blink to type it."))
        close = self._button("✕", "close", self.close_keyboard)
        close.setAccessibleName("Close keyboard")
        top.addWidget(close)
        col.addLayout(top)

        self._rows: list[list[QPushButton]] = []
        for r, n in enumerate((10, 9, 7)):
            grid = QGridLayout()
            grid.setSpacing(10)
            keys = []
            offset = 0
            if r == 2:
                self._shift_btn = self._button("⇧  Shift", "mod", self._toggle_shift)
                grid.addWidget(self._shift_btn, 0, 0, 1, 3)
                offset = 3
            for i in range(n):
                key = self._button("", "", None)
                key.clicked.connect(lambda _=False, k=key: self._type(k.text()))
                grid.addWidget(key, 0, offset + i * 2, 1, 2)
                keys.append(key)
            if r == 2:
                delete = self._button("⌫", "mod", self._delete)
                delete.setAccessibleName("Delete")
                grid.addWidget(delete, 0, offset + n * 2, 1, 3)
            if r == 1:  # the middle row sits half a key in
                grid.setContentsMargins(40, 0, 40, 0)
            col.addLayout(grid)
            self._rows.append(keys)

        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        self._mode_btn = self._button("123", "mod", self._toggle_symbols)
        bottom.addWidget(self._mode_btn, 3)
        bottom.addWidget(self._button(",", "", lambda: self._type(",")), 2)
        bottom.addWidget(self._button("space", "mod", lambda: self._type(" ")), 12)
        bottom.addWidget(self._button(".", "", lambda: self._type(".")), 2)
        bottom.addWidget(self._button("Send", "go", self._send), 4)
        col.addLayout(bottom)
        self._relabel()

    def _button(self, text: str, kind: str, on_click: Callable[[], None] | None) -> QPushButton:
        btn = QPushButton(text)
        if kind:
            btn.setProperty("kind", kind)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        if on_click is not None:
            btn.clicked.connect(on_click)
        return btn

    # -- window ------------------------------------------------------------

    def open(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen.left() + _MARGIN, screen.bottom() - _BOTTOM - _H,
                         screen.width() - 2 * _MARGIN, _H)
        self._word = ""
        self._relabel()
        self._focus.update()
        self._focus_timer.start()
        self.show()

    def close_keyboard(self) -> None:
        self._focus_timer.stop()
        self.hide()
        if self._on_close is not None:
            self._on_close()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        float_everywhere(self)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(theme.NIGHT))
        p.drawRoundedRect(QRectF(self.rect()), 28, 28)
        p.end()

    # -- typing ------------------------------------------------------------

    def _send_text(self, text: str) -> None:
        self._focus.restore()
        type_text(text)

    def _type(self, ch: str) -> None:
        self._send_text(ch)
        if ch.isalnum() or ch in "'’-":
            self._word += ch
        else:
            self._word = ""
        if self._shift:
            self._shift = False
        self._relabel()

    def _pick(self, word: str) -> None:
        if not word:
            return
        rest = word[len(self._word):] if word.lower().startswith(self._word.lower()) else word
        self._send_text(rest + " ")
        self._word = ""
        self._relabel()

    def _delete(self) -> None:
        self._focus.restore()
        tap_key(KEY_DELETE)
        self._word = self._word[:-1]
        self._relabel()

    def _send(self) -> None:
        self._focus.restore()
        tap_key(KEY_RETURN)
        self._word = ""
        self._relabel()

    def _toggle_shift(self) -> None:
        self._shift = not self._shift
        self._relabel()

    def _toggle_symbols(self) -> None:
        self._symbols = not self._symbols
        self._relabel()

    def _relabel(self) -> None:
        layout = SYMBOLS if self._symbols else LETTERS
        for keys, chars in zip(self._rows, layout, strict=True):
            for key, ch in zip(keys, chars, strict=True):
                key.setText(ch.upper() if self._shift else ch)
        self._mode_btn.setText("ABC" if self._symbols else "123")
        self._shift_btn.setProperty("on", "true" if self._shift else "false")
        self._shift_btn.style().unpolish(self._shift_btn)
        self._shift_btn.style().polish(self._shift_btn)
        for chip, word in zip(self._chips, suggest(self._word) + [""] * _N_SUGGESTIONS,
                              strict=False):
            chip.setText(word)
            chip.setVisible(bool(word))
