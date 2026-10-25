"""EyeMouse Settings: voice typing, open at login, the gesture guide, recalibrate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from opengaze.consumer import theme
from opengaze.consumer.setup_window import PAUSE_SHORTCUT, GestureList


@dataclass
class SettingsHooks:
    voice_on: Callable[[], bool]
    set_voice: Callable[[bool], None]
    login_on: Callable[[], bool]
    set_login: Callable[[bool], None]
    recalibrate: Callable[[], None]
    run_setup: Callable[[], None]


class SettingsWindow(QWidget):
    def __init__(self, hooks: SettingsHooks) -> None:
        super().__init__()
        self._hooks = hooks
        self.setWindowTitle(f"{theme.APP_NAME} Settings")
        self.setObjectName("page")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(theme.stylesheet())
        self.setFixedWidth(480)
        col = QVBoxLayout(self)
        col.setContentsMargins(40, 32, 40, 32)
        col.setSpacing(14)
        title = QLabel("Settings")
        title.setProperty("role", "title")
        title.setStyleSheet("font-size: 32px;")
        col.addWidget(title)

        self.voice = QCheckBox("Voice typing (speech is turned into text by Cartesia, online)")
        self.voice.toggled.connect(hooks.set_voice)
        self.login = QCheckBox(f"Open {theme.APP_NAME} when this Mac starts")
        self.login.toggled.connect(hooks.set_login)
        col.addWidget(self.voice)
        col.addWidget(self.login)

        col.addSpacing(8)
        h = QLabel("Eye gestures")
        h.setStyleSheet(f"font-family: '{theme.DISPLAY}'; font-size: 18px; font-weight: 600;")
        col.addWidget(h)
        col.addWidget(GestureList())
        pause = QLabel(f"Pause or resume from the keyboard: {PAUSE_SHORTCUT}")
        pause.setProperty("role", "fine")
        col.addWidget(pause)

        col.addSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(12)
        for text, kind, action in (("Recalibrate", "primary", hooks.recalibrate),
                                   ("Run setup again", "ghost", hooks.run_setup)):
            btn = QPushButton(text)
            btn.setProperty("kind", kind)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, a=action: (self.close(), a()))
            row.addWidget(btn)
        row.addStretch(1)
        col.addLayout(row)

    def open(self) -> None:
        for box, value in ((self.voice, self._hooks.voice_on()),
                           (self.login, self._hooks.login_on())):
            box.blockSignals(True)
            box.setChecked(value)
            box.blockSignals(False)
        self.adjustSize()
        self.show()
        self.raise_()
        self.activateWindow()
