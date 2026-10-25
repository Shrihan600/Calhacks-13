"""EyeMouse in the menu bar: the gaze switch, its status, and everything else.

    Gaze cursor              ✓
    Tracking well
    ⚠ Glasses not connected     (only while there's a problem)
    ─────────────
    Pause                  ⌃⌥G
    Stop tracking          ⌃⌥T
    Voice typing             ✓
    Gaze keyboard
    ─────────────
    Recalibrate
    Settings…
    Quit EyeMouse
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from opengaze.consumer import theme
from opengaze.consumer.problems import Problem
from opengaze.desktop.desktop_mode import GazeStatus

_REFRESH_MS = 1000

STATUS_TEXT: dict[GazeStatus, str] = {
    "tracking": "Tracking well",
    "no_face": "Looking for your face",
    "mouse": "Mouse in control",
    "paused": "Paused",
    "off": "Tracking off",
}


@dataclass
class MenuHooks:
    gaze_on: Callable[[], bool]
    status: Callable[[], GazeStatus]
    set_gaze: Callable[[bool], None]
    toggle_pause: Callable[[], None]
    toggle_tracking: Callable[[], None]
    voice_on: Callable[[], bool]
    set_voice: Callable[[bool], None]
    keyboard: Callable[[], None]
    recalibrate: Callable[[], None]
    settings: Callable[[], None]
    quit: Callable[[], None]
    problem: Callable[[], Problem | None] = lambda: None  # e.g. the glasses aren't plugged in
    tracking_allowed: Callable[[], bool] = lambda: True  # False: launched with --no-tracking


class MenuBar:
    def __init__(self, hooks: MenuHooks) -> None:
        self._hooks = hooks
        self.tray = QSystemTrayIcon(theme.menu_bar_icon())
        self.tray.setToolTip(theme.APP_NAME)
        menu = QMenu()
        self.gaze = QAction("Gaze cursor", menu, checkable=True)
        self.gaze.triggered.connect(hooks.set_gaze)
        self.status = QAction("", menu, enabled=False)
        self.note = QAction("", menu, enabled=False, visible=False)
        self.pause = QAction("Pause", menu)
        # Qt's "Meta" is the Control key on macOS, so this shows as ⌃⌥G. The shortcut itself
        # is the global hotkey; the menu only shows it.
        self.pause.setShortcut(QKeySequence("Meta+Alt+G"))
        self.pause.triggered.connect(hooks.toggle_pause)
        self.tracking = QAction("Stop tracking", menu)
        self.tracking.setShortcut(QKeySequence("Meta+Alt+T"))
        self.tracking.triggered.connect(hooks.toggle_tracking)
        self.voice = QAction("Voice typing", menu, checkable=True)
        self.voice.triggered.connect(hooks.set_voice)
        keyboard = QAction("Gaze keyboard", menu)
        keyboard.triggered.connect(hooks.keyboard)
        recal = QAction("Recalibrate", menu)
        recal.triggered.connect(hooks.recalibrate)
        settings = QAction("Settings…", menu)
        settings.triggered.connect(hooks.settings)
        quit_ = QAction(f"Quit {theme.APP_NAME}", menu)
        quit_.triggered.connect(hooks.quit)
        for item in (self.gaze, self.status, self.note, None, self.pause, self.tracking,
                     self.voice, keyboard, None, recal, settings, quit_):
            if item is None:
                menu.addSeparator()
            else:
                menu.addAction(item)
        menu.aboutToShow.connect(self.refresh)
        self.menu = menu
        self.tray.setContextMenu(menu)
        self._timer = QTimer()
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self.refresh)

    def show(self) -> None:
        self.refresh()
        self.tray.show()
        self._timer.start()

    def hide(self) -> None:
        self._timer.stop()
        self.tray.hide()

    def refresh(self) -> None:
        status = self._hooks.status()
        self.gaze.setChecked(self._hooks.gaze_on())
        self.status.setText(STATUS_TEXT[status])
        problem = self._hooks.problem()
        self.note.setText(f"⚠ {problem.title}" if problem else "")
        self.note.setVisible(problem is not None)
        self.pause.setText("Resume" if status == "paused" else "Pause")
        self.pause.setEnabled(status != "off")
        self.tracking.setText("Start tracking" if status == "off" else "Stop tracking")
        allowed = self._hooks.tracking_allowed()
        self.gaze.setEnabled(allowed)
        self.tracking.setEnabled(allowed)
        if not allowed:
            self.status.setText("Tracking off for this session")
        self.voice.setChecked(self._hooks.voice_on())
        self.tray.setToolTip(f"{theme.APP_NAME}: "
                             + (problem.title if problem else STATUS_TEXT[status]))

    def notify(self, problem: Problem) -> None:
        """A macOS notification, for a problem that just appeared."""
        icon = (QSystemTrayIcon.MessageIcon.Critical if problem.severity == "error"
                else QSystemTrayIcon.MessageIcon.Warning)
        self.tray.showMessage(problem.title, problem.detail, icon, 10_000)
