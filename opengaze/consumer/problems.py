"""What's wrong with the hardware, in words for the person using EyeMouse.

``find_problem`` turns camera and IMU health into at most one ``Problem`` (the most serious);
``ProblemMonitor`` holds a problem back until it has lasted a moment, so a camera reconnecting
by itself doesn't flash a warning. The app shows what comes out: a banner on screen for an
error, the menu bar and a notification for both.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from opengaze.cameras.manager import CameraHealth
from opengaze.cameras.types import CameraRole, CameraState
from opengaze.imu.runtime import ImuHealth

# Finding the glasses and opening the cameras takes a few seconds after start.
STARTUP_GRACE_S = 10.0
# A problem must last this long before it's shown (reconnects usually take less).
SHOW_AFTER_S = 2.0

Severity = Literal["error", "warning"]  # error: gaze control can't work; warning: it works


@dataclass(frozen=True)
class Problem:
    title: str
    detail: str
    severity: Severity
    technical: str = ""  # the underlying error, for the log


def _side(health: CameraHealth) -> str:
    return health.role.value.removesuffix("_eye")


def _eye_problem(eyes: list[CameraHealth], settled: bool) -> Problem | None:
    failed = [h for h in eyes if h.state is CameraState.FAILED]
    if failed:  # bad configuration: retrying won't help
        return Problem("The glasses can’t start",
                       "Something in EyeMouse’s setup is wrong. Ask for help with this message: "
                       f"{failed[0].last_error or 'camera failed'}",
                       "error", failed[0].last_error or "")
    down = [h for h in eyes if h.state is not CameraState.STREAMING]
    errors = [h.last_error or "" for h in down]
    # Starting up, a camera without an error is still connecting; one with an error has
    # finished trying (the port search ends in a few seconds), so that counts at once.
    if not down or not (settled or all(errors)):
        return None
    technical = "; ".join(f"{h.camera_id}: {e}" for h, e in zip(down, errors, strict=True) if e)
    if any("receiver" in e for e in errors):
        return Problem("The glasses’ software is missing",
                       "EyeMouse can’t read the glasses without it. Ask for help with this "
                       f"message: {next(e for e in errors if 'receiver' in e)}",
                       "error", technical)
    off = " Gaze control is off until they’re back." if len(down) > 1 else \
        " Gaze control is off until it’s back."
    if all("isn't connected" in e for e in errors):  # the port search found nothing
        if len(down) == len(eyes):
            return Problem("Glasses not connected",
                           "Plug both glasses cables into this Mac." + off, "error", technical)
        side = _side(down[0])
        return Problem(f"{side.capitalize()} eye camera not connected",
                       f"Check the {side} glasses cable." + off, "error", technical)
    which = "The glasses aren’t" if len(down) == len(eyes) else \
        f"The {_side(down[0])} eye camera isn’t"
    return Problem(f"{which} sending video",
                   "Unplug the glasses and plug them back in." + off, "error", technical)


def find_problem(
    cameras: Mapping[str, CameraHealth],
    imu: ImuHealth | None,
    running_for_s: float,
    *,
    capture_running: bool = True,
    camera_denied: bool = False,
    app_name: str = "EyeMouse",
) -> Problem | None:
    """The most serious problem right now, or None. ``running_for_s``: since the cameras
    started (before ``STARTUP_GRACE_S``, a camera still connecting without an error isn't a
    problem yet)."""
    if not capture_running:
        return Problem("EyeMouse stopped reading the cameras",
                       f"Quit {app_name} and open it again.", "error", "capture loop ended")
    settled = running_for_s >= STARTUP_GRACE_S
    eyes = [h for h in cameras.values() if h.role in (CameraRole.LEFT_EYE, CameraRole.RIGHT_EYE)]
    problem = _eye_problem(eyes, settled)
    if problem is not None:
        return problem
    heads = [h for h in cameras.values()
             if h.role is CameraRole.HEAD and h.state is not CameraState.STREAMING]
    if heads and (settled or heads[0].state is CameraState.FAILED or heads[0].last_error):
        if camera_denied:
            return Problem("Camera access is off",
                           f"Turn on the camera for {app_name} in System Settings › Privacy & "
                           "Security › Camera, then quit and reopen it.",
                           "error", heads[0].last_error or "")
        return Problem("This Mac’s camera isn’t working",
                       f"{app_name} needs it to follow your head. Quit other apps that use the "
                       "camera, or restart this Mac.", "error", heads[0].last_error or "")
    if imu is not None and settled and imu.state != "streaming":
        return Problem("Head motion sensor not working",
                       "Gaze control still works, a little less steadily.", "warning",
                       imu.last_error or f"IMU {imu.state}")
    return None


class ProblemMonitor:
    """Shows a problem once it has lasted ``SHOW_AFTER_S``; clears it as soon as it's gone."""

    def __init__(self, show_after_s: float = SHOW_AFTER_S) -> None:
        self.show_after_s = show_after_s
        self.shown: Problem | None = None
        self._pending: Problem | None = None
        self._since = 0.0

    def update(self, problem: Problem | None, now: float) -> Problem | None:
        if problem is None:
            self.shown = self._pending = None
            return None
        key = (problem.title, problem.severity)
        if self.shown is not None and (self.shown.title, self.shown.severity) == key:
            self.shown = problem  # same problem, maybe new details
            return problem
        if self._pending is None or (self._pending.title, self._pending.severity) != key:
            self._pending, self._since = problem, now
        if now - self._since >= self.show_after_s:
            self.shown = problem
        return self.shown
