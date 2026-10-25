"""The gaze side of EyeMouse: camera → calibrated gaze → the real cursor, in every app.

One ``GazeEngine`` per process. It owns the camera runtime for the whole session (setup,
calibration and every day), and the system-wide cursor overlay from ``desktop_mode``,
restyled. Calibration needs the camera to itself, so the cursor stops while it runs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping

from PySide6.QtGui import QFont

from opengaze.calibration.storage import CalibrationStore
from opengaze.cameras.manager import CameraManager
from opengaze.cameras.runtime import CaptureRuntime
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.consumer import theme
from opengaze.consumer.problems import Problem, find_problem
from opengaze.control.cursor import QuartzCursorDriver
from opengaze.desktop.desktop_mode import GazeCursorOverlay, GazeStatus
from opengaze.gaze.loop import GazeLoop
from opengaze.gaze.tracker import GazeTracker
from opengaze.gestures.config import Binding
from opengaze.gestures.types import Gesture
from opengaze.modes import profile_schema

log = logging.getLogger("opengaze.consumer.engine")


def consumer_bindings(config: AppConfig) -> dict[Gesture, Binding]:
    """The config's gestures, with a triple blink starting voice typing (as the app teaches)."""
    bindings = dict(config.gestures.bindings)
    bindings[Gesture.TRIPLE_BLINK] = Binding(app="dictate")
    return bindings


class EyeMouseOverlay(GazeCursorOverlay):
    ring_color = theme.qcolor(theme.PURPLE)
    flash_color = theme.qcolor(theme.LIME, 0.9)
    warn_color = theme.qcolor(theme.PURPLE_DARK)
    notes = {"no_face": "Looking for your face", "mouse": "Mouse in control"}

    def note_font(self) -> QFont:
        return theme.text_font(14, 600)


class GazeEngine:
    def __init__(
        self,
        report: ProfileReport,
        config: AppConfig,
        app_handlers: Mapping[str, Callable[[], None]],
    ) -> None:
        assert report.profile is not None
        self.report = report
        self.config = config
        self.schema_id = profile_schema(report.profile)
        self.store = CalibrationStore()
        self._handlers = dict(app_handlers)
        self.runtime: CaptureRuntime | None = None
        self._camera_started = 0.0
        self._tracker: GazeTracker | None = None
        self._overlay: EyeMouseOverlay | None = None

    # -- camera ------------------------------------------------------------

    def start_camera(self) -> CaptureRuntime:
        if self.runtime is None:
            self.runtime = CaptureRuntime(
                CameraManager.from_profile(self.report.profile, self.config.capture)
            )
            self.runtime.start()
            self._camera_started = time.monotonic()
        return self.runtime

    def problem(self, camera_denied: bool = False) -> Problem | None:
        """What's wrong with the cameras or the IMU, if anything (None before they start)."""
        if self.runtime is None:
            return None
        imu = self._tracker.imu if self._tracker is not None else None
        return find_problem(
            self.runtime.manager.health(),
            imu.health() if imu is not None else None,
            time.monotonic() - self._camera_started,
            capture_running=self.runtime.running,
            camera_denied=camera_denied,
            app_name=theme.APP_NAME,
        )

    @property
    def calibrated(self) -> bool:
        return self.store.exists(self.schema_id, self.report.profile)  # type: ignore[arg-type]

    # -- cursor ------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._overlay is not None

    def start_cursor(self) -> bool:
        """Follow gaze with the real cursor, from the saved calibration. False if there's none."""
        self.stop_cursor()
        runtime = self.start_camera()
        try:
            bundle = self.store.load_bundle(self.schema_id, self.report.profile)  # type: ignore[arg-type]
        except (FileNotFoundError, ValueError) as exc:
            log.warning("no usable calibration: %s", exc)
            return False
        self._tracker = GazeTracker(self.report.profile, self.config)
        self._tracker.load_eye_states(bundle.eye_states)
        loop = GazeLoop(self._tracker, bundle.model, self.config.gestures)
        self._overlay = EyeMouseOverlay(
            runtime, loop, QuartzCursorDriver(), self.config,
            app_handlers=self._handlers, bindings=consumer_bindings(self.config),
        )
        self._overlay.show()
        return True

    def stop_cursor(self) -> None:
        if self._overlay is not None:
            self._overlay.close()
            self._overlay.deleteLater()
            self._overlay = None
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None

    @property
    def enabled(self) -> bool:
        return self._overlay is not None and self._overlay.enabled

    def set_enabled(self, on: bool) -> None:
        if self._overlay is not None and self._overlay.enabled != on:
            self._overlay.toggle()

    def toggle(self) -> None:
        if self._overlay is not None:
            self._overlay.toggle()

    @property
    def status(self) -> GazeStatus:
        return self._overlay.status if self._overlay is not None else "off"

    def shutdown(self) -> None:
        self.stop_cursor()
        if self.runtime is not None:
            self.runtime.stop()
            self.runtime = None
