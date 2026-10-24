"""macOS cursor driver using Quartz CoreGraphics.

Requires pyobjc-framework-Quartz (already in the desktop extra).
Requires Accessibility permission — check with is_accessibility_granted()
and guide the user with open_accessibility_prefs() if missing.

Coordinates: macOS uses logical points (not physical pixels).
CGDisplayBounds returns the logical size; CGEventCreateMouseEvent
expects logical point coordinates.  On a Retina display the backing
scale factor is 2x but we never need to know it here.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenSize:
    width: int   # logical points
    height: int  # logical points


def get_screen_size() -> ScreenSize:
    """Return the main display size in logical points."""
    if sys.platform != "darwin":
        raise RuntimeError("cursor control is macOS-only")
    import Quartz
    display = Quartz.CGMainDisplayID()
    bounds = Quartz.CGDisplayBounds(display)
    return ScreenSize(int(bounds.size.width), int(bounds.size.height))


def is_accessibility_granted() -> bool:
    """True if this process has Accessibility (cursor control) permission."""
    if sys.platform != "darwin":
        return False
    try:
        import Quartz
        return bool(Quartz.AXIsProcessTrusted())
    except AttributeError:
        pass
    try:
        import ctypes
        lib = ctypes.CDLL(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        return bool(lib.AXIsProcessTrusted())
    except Exception:
        return False


def open_accessibility_prefs() -> None:
    """Open System Settings > Privacy & Security > Accessibility."""
    subprocess.run(
        ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"],
        check=False,
    )


class QuartzCursorDriver:
    """Moves and clicks the macOS cursor via Quartz CGEvents.

    All coordinates are in logical screen points (origin = top-left).
    """

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("QuartzCursorDriver is macOS-only")
        import Quartz as _Q
        self._Q = _Q
        self._screen = get_screen_size()

    @property
    def screen_size(self) -> ScreenSize:
        return self._screen

    def move(self, x: int, y: int) -> None:
        Q = self._Q
        event = Q.CGEventCreateMouseEvent(
            None, Q.kCGEventMouseMoved, (x, y), Q.kCGMouseButtonLeft
        )
        Q.CGEventPost(Q.kCGHIDEventTap, event)

    def click(self, x: int, y: int) -> None:
        Q = self._Q
        down = Q.CGEventCreateMouseEvent(
            None, Q.kCGEventLeftMouseDown, (x, y), Q.kCGMouseButtonLeft
        )
        up = Q.CGEventCreateMouseEvent(
            None, Q.kCGEventLeftMouseUp, (x, y), Q.kCGMouseButtonLeft
        )
        Q.CGEventPost(Q.kCGHIDEventTap, down)
        Q.CGEventPost(Q.kCGHIDEventTap, up)

    def norm_to_px(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        """Convert normalised [0,1] coordinates to screen pixels."""
        return (
            int(x_norm * self._screen.width),
            int(y_norm * self._screen.height),
        )
