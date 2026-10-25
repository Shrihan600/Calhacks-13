"""EyeMouse: the consumer app (``opengaze app``).

First run, a helper sets it up: permissions, calibration, a short practice. After that it
lives in the menu bar: the gaze cursor drives the real pointer in every app, a mic follows
the text caret for voice typing, and the gaze keyboard is a menu click (or gesture) away.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from opengaze.cameras.permissions import (
    CameraPermission,
    camera_permission_status,
    request_camera_permission,
)
from opengaze.cameras.wired import GlassesScan
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.consumer import theme
from opengaze.consumer.calibration import EyeMouseCalibration, Outcome, run_calibration_once
from opengaze.consumer.engine import GazeEngine
from opengaze.consumer.keyboard import GazeKeyboard
from opengaze.consumer.menubar import MenuBar, MenuHooks
from opengaze.consumer.problems import ProblemMonitor
from opengaze.consumer.settings import AppSettings, LoginItem
from opengaze.consumer.settings_window import SettingsHooks, SettingsWindow
from opengaze.consumer.setup_window import SetupHooks, SetupWindow
from opengaze.consumer.voice_typing import VoiceTyping
from opengaze.consumer.warning_banner import WarningBanner
from opengaze.desktop.macwin import KEY_D, KEY_G, CtrlOptHotkeys, become_accessory_app
from opengaze.desktop.run_ui import _launching_app
from opengaze.voice.cartesia import CartesiaConfig
from opengaze.voice.service import VoiceService

log = logging.getLogger("opengaze.consumer")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBLEM_CHECK_MS = 1000


def _permission_app() -> str:
    """The app macOS asks about: EyeMouse when bundled, else whatever launched Python."""
    return theme.APP_NAME if ".app/Contents/" in sys.executable else _launching_app()


def needs_setup(settings: AppSettings, report: ProfileReport, force: bool = False) -> bool:
    """Setup runs the first time, when asked, and on glasses it wasn't done with (an earlier
    webcam-only setup never showed the glasses page). The webcam dev fallback never asks
    again: it has no setup step of its own."""
    if force or not settings.setup_done:
        return True
    return _has_eye_cameras(report) and settings.setup_hardware != report.path.name


def _has_eye_cameras(report: ProfileReport) -> bool:
    assert report.profile is not None
    return any(cam.role.value.endswith("_eye") for cam in report.profile.enabled_cameras.values())


class EyeMouseApp:
    def __init__(
        self,
        report: ProfileReport,
        config: AppConfig,
        *,
        login_args: list[str] | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.qt = QApplication.instance() or QApplication(sys.argv[:1])
        theme.load_fonts()
        self.qt.setApplicationName(theme.APP_NAME)
        self.qt.setWindowIcon(theme.app_icon())
        self.qt.setQuitOnLastWindowClosed(False)

        self.report, self.config = report, config
        self.settings = settings or AppSettings.load()
        self.login_item = LoginItem()
        self._login_program = [sys.executable, "-m", "opengaze.cli", "app", *(login_args or [])]
        self._everyday = False

        later = self._later
        self.engine = GazeEngine(report, config, {
            "dictate": later(lambda: self.voice_typing.toggle()),
            "keyboard": later(self.toggle_keyboard),
            "recalibrate": later(self.recalibrate),
            "quit": later(self.quit),
        })
        self.voice_typing = VoiceTyping(self._make_voice())
        self.keyboard = GazeKeyboard()
        self.menu = MenuBar(MenuHooks(
            gaze_on=lambda: self.engine.enabled,
            status=lambda: self.engine.status,
            set_gaze=self.set_gaze,
            toggle_pause=self.toggle_pause,
            voice_on=lambda: self.settings.voice_typing,
            set_voice=self.set_voice,
            keyboard=self.toggle_keyboard,
            recalibrate=self.recalibrate,
            settings=self.open_settings,
            quit=self.quit,
            problem=lambda: self.problems.shown,
        ))
        self.hotkeys = CtrlOptHotkeys({
            KEY_G: self.toggle_pause,
            KEY_D: lambda: self.voice_typing.toggle(),
        })
        self.setup: SetupWindow | None = None
        self.settings_window: SettingsWindow | None = None
        self._calibration: EyeMouseCalibration | None = None

        # Hardware problems are shown, never only logged: a banner for errors, the menu bar
        # and a notification for both.
        self.problems = ProblemMonitor()
        self.banner = WarningBanner()
        self._problem_timer = QTimer()
        self._problem_timer.setInterval(_PROBLEM_CHECK_MS)
        self._problem_timer.timeout.connect(self.check_hardware)
        self._problem_timer.start()

    # -- hardware ------------------------------------------------------------

    def check_hardware(self) -> None:
        before = self.problems.shown
        denied = camera_permission_status() in (CameraPermission.DENIED,
                                                CameraPermission.RESTRICTED)
        now = self.problems.update(self.engine.problem(camera_denied=denied), time.monotonic())
        if now == before:
            return
        if now is None:
            log.info("hardware OK again")
        elif before is None or before.title != now.title:
            log.warning("%s: %s%s", now.title, now.detail,
                        f" [{now.technical}]" if now.technical else "")
            if self._everyday:
                self.menu.notify(now)
        self._sync_banner()
        self.menu.refresh()

    def _sync_banner(self) -> None:
        """The banner shows an error, except over calibration, which draws it itself."""
        problem = self.problems.shown
        if problem is not None and problem.severity == "error" and self._calibration is None:
            self.banner.show_problem(problem)
        else:
            self.banner.clear()

    def uses_glasses(self) -> bool:
        """False only for a developer's webcam profile (``--profile``)."""
        return _has_eye_cameras(self.report)

    def scan_glasses(self) -> GlassesScan:
        """A scan with the profile's serial settings."""
        assert self.report.profile is not None
        cams = self.report.profile.enabled_cameras
        options = cams["left_eye"].options if "left_eye" in cams else {}
        return GlassesScan(options.get("baud"), options.get("receiver_dir"))

    @staticmethod
    def _later(fn: Callable[[], None]) -> Callable[[], None]:
        """Gesture actions run inside the cursor's own tick; some of them stop it."""
        return lambda: QTimer.singleShot(0, fn)

    def _make_voice(self) -> VoiceService:
        return VoiceService(
            cloud_enabled=self.settings.voice_typing or self.config.privacy.cloud_features_enabled,
            cartesia=CartesiaConfig.from_env(),
        )

    # -- lifecycle ---------------------------------------------------------

    def run(self, force_setup: bool = False) -> int:
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C in the terminal quits
        if needs_setup(self.settings, self.report, force_setup):
            self.show_setup()
        else:
            self.start_everyday()
        try:
            return self.qt.exec()
        finally:
            self.shutdown()

    def quit(self) -> None:
        self.qt.quit()

    def shutdown(self) -> None:
        self._problem_timer.stop()
        self.banner.clear()
        self.hotkeys.remove()
        self.menu.hide()
        self.keyboard.close()
        self.voice_typing.shutdown()
        self.engine.shutdown()

    # -- setup -------------------------------------------------------------

    def show_setup(self) -> None:
        if self.setup is None:
            self.setup = SetupWindow(SetupHooks(
                calibrate=lambda: self.calibrate(self._setup_calibrated),
                finish=self._finish_setup,
                quit=self._setup_closed,
                voice_chosen=self.set_voice,
                has_voice_key=CartesiaConfig.from_env() is not None,
                permission_app=_permission_app(),
                scan_glasses=self.scan_glasses if self.uses_glasses() else None,
            ), self.settings.open_at_login)
            self.setup.go(SetupWindow.WELCOME)
        self.setup.show()
        self.setup.raise_()
        self.setup.activateWindow()

    def _setup_closed(self) -> None:
        self.setup = None
        if not self._everyday:
            self.quit()

    def _setup_calibrated(self, saved: bool) -> None:
        assert self.setup is not None
        if saved or self.engine.calibrated:
            self.engine.start_cursor()
            self.setup.go(SetupWindow.PRACTICE)
        else:
            self.setup.go(SetupWindow.MIC)
        self.setup.show()
        self.setup.raise_()
        self.setup.activateWindow()

    def _finish_setup(self, open_at_login: bool) -> None:
        self.setup = None
        self.settings.setup_done = True
        self.settings.setup_hardware = self.report.path.name
        self.set_login(open_at_login)
        self.start_everyday()

    # -- calibration -------------------------------------------------------

    def calibrate(self, then: Callable[[bool], None]) -> None:
        """Run calibration fullscreen (gaze cursor stopped); ``then(saved)`` afterwards."""
        if self._calibration is not None:
            return
        request_camera_permission()
        self.keyboard.hide()
        self.engine.stop_cursor()
        if self.setup is not None:
            self.setup.hide()
        runtime = self.engine.start_camera()

        def done(outcome: Outcome) -> None:
            self._calibration = None
            self._sync_banner()
            if outcome == "retry":
                QTimer.singleShot(0, lambda: self.calibrate(then))
            else:
                QTimer.singleShot(0, lambda: then(outcome == "saved"))

        self._calibration = run_calibration_once(runtime, self.report, self.config, done,
                                                 problem=lambda: self.problems.shown)
        self._sync_banner()

    def recalibrate(self) -> None:
        self.calibrate(lambda saved: self._resume_cursor())

    def _resume_cursor(self) -> None:
        if not self.engine.start_cursor() and self._everyday:
            log.warning("no calibration to run the gaze cursor with")

    # -- every day ---------------------------------------------------------

    def start_everyday(self) -> None:
        if not self.engine.calibrated:
            # New hardware or a deleted calibration: calibrate first, else back to setup.
            self.calibrate(lambda saved: self.start_everyday() if self.engine.calibrated
                           else self.show_setup())
            return
        self._everyday = True
        request_camera_permission()
        become_accessory_app()
        if not self.engine.running:
            self.engine.start_cursor()
        self.voice_typing.set_enabled(self.settings.voice_typing)
        self.menu.show()
        self.hotkeys.install()

    def set_gaze(self, on: bool) -> None:
        if on and not self.engine.running:
            self.engine.start_cursor()
        self.engine.set_enabled(on)
        self.menu.refresh()

    def toggle_pause(self) -> None:
        self.engine.toggle()
        self.menu.refresh()

    def set_voice(self, on: bool) -> None:
        if on != self.settings.voice_typing:
            self.settings.voice_typing = on
            self.settings.save()
            self.voice_typing.set_voice(self._make_voice())
        self.voice_typing.set_enabled(on and self._everyday)
        self.menu.refresh()

    def set_login(self, on: bool) -> None:
        self.settings.open_at_login = on
        self.settings.save()
        try:
            self.login_item.apply(on, self._login_program, _REPO_ROOT)
        except OSError as exc:
            log.warning("could not update the login item: %s", exc)

    def toggle_keyboard(self) -> None:
        if self.keyboard.isVisible():
            self.keyboard.close_keyboard()
        else:
            self.keyboard.open()

    def open_settings(self) -> None:
        if self.settings_window is None:
            self.settings_window = SettingsWindow(SettingsHooks(
                voice_on=lambda: self.settings.voice_typing,
                set_voice=self.set_voice,
                login_on=lambda: self.login_item.enabled,
                set_login=self.set_login,
                recalibrate=self.recalibrate,
                run_setup=self.show_setup,
            ))
        self.settings_window.open()


def run_app(
    report: ProfileReport,
    config: AppConfig,
    *,
    force_setup: bool = False,
    login_args: list[str] | None = None,
) -> int:
    return EyeMouseApp(report, config, login_args=login_args).run(force_setup)


def show_startup_error(title: str, detail: str) -> None:
    """A dialog for a problem that stops EyeMouse starting (a login item has no terminal)."""
    app = QApplication.instance() or QApplication(sys.argv[:1])
    theme.load_fonts()
    app.setApplicationName(theme.APP_NAME)
    box = QMessageBox(QMessageBox.Icon.Critical, title, title)
    box.setInformativeText(detail)
    box.exec()
