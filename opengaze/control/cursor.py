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

from opengaze.gestures.keys import KeyCombo


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

    def _button(
        self, down_type: int, up_type: int, button: int, x: int, y: int, n: int = 1
    ) -> None:
        Q = self._Q
        for i in range(1, n + 1):
            for kind in (down_type, up_type):
                event = Q.CGEventCreateMouseEvent(None, kind, (x, y), button)
                # The click count is what makes macOS treat the second click as a double click.
                Q.CGEventSetIntegerValueField(event, Q.kCGMouseEventClickState, i)
                Q.CGEventPost(Q.kCGHIDEventTap, event)

    def scroll(self, dy_lines: int, dx_lines: int = 0) -> None:
        Q = self._Q
        event = Q.CGEventCreateScrollWheelEvent(
            None, Q.kCGScrollEventUnitLine, 2, dy_lines, dx_lines
        )
        Q.CGEventPost(Q.kCGHIDEventTap, event)

    # InputBackend protocol (opengaze/gestures/bindings.py) ------------------

    def mouse(self, action: str, x: int, y: int) -> None:
        Q = self._Q
        left = (Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp, Q.kCGMouseButtonLeft)
        right = (Q.kCGEventRightMouseDown, Q.kCGEventRightMouseUp, Q.kCGMouseButtonRight)
        middle = (Q.kCGEventOtherMouseDown, Q.kCGEventOtherMouseUp, Q.kCGMouseButtonCenter)
        if action == "click":
            self.click(x, y)
        elif action == "double_click":
            self._button(*left, x, y, 2)
        elif action == "right_click":
            self._button(*right, x, y)
        elif action == "middle_click":
            self._button(*middle, x, y)
        elif action in _SCROLL_LINES:
            dy, dx = _SCROLL_LINES[action]
            self.scroll(dy, dx)
        else:
            raise ValueError(f"unknown mouse action {action!r}")

    def key_combo(self, combo: KeyCombo) -> None:
        Q = self._Q
        flags = 0
        for m in combo.modifiers:
            flags |= getattr(Q, _MODIFIER_FLAGS[m])
        for down in (True, False):
            event = Q.CGEventCreateKeyboardEvent(None, combo.key_code, down)
            Q.CGEventSetFlags(event, flags)
            Q.CGEventPost(Q.kCGHIDEventTap, event)

    def norm_to_px(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        """Convert normalised [0,1] coordinates to screen pixels."""
        return (
            int(x_norm * self._screen.width),
            int(y_norm * self._screen.height),
        )


_SCROLL_LINES = {
    "scroll_up": (5, 0),
    "scroll_down": (-5, 0),
    "scroll_left": (0, 5),
    "scroll_right": (0, -5),
}

_MODIFIER_FLAGS = {
    "cmd": "kCGEventFlagMaskCommand",
    "shift": "kCGEventFlagMaskShift",
    "ctrl": "kCGEventFlagMaskControl",
    "alt": "kCGEventFlagMaskAlternate",
    "fn": "kCGEventFlagMaskSecondaryFn",
}
