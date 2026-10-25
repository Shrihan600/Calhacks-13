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
        self._origin = (0, 0)  # top-left of the display gaze maps onto, global points
        self._pressed = False  # left button held down by us (a hold/drag in progress)

    @property
    def screen_size(self) -> ScreenSize:
        return self._screen

    def set_display(self, x: int, y: int, width: int, height: int) -> None:
        """Map normalised gaze onto this display rectangle (global logical points).

        Gaze is calibrated against the fullscreen window, so pass that window's screen;
        the default is the main display, which is wrong when the window is on another one.
        """
        self._origin = (int(x), int(y))
        self._screen = ScreenSize(int(width), int(height))

    def position(self) -> tuple[int, int]:
        """Where macOS says the cursor is now (global logical points)."""
        Q = self._Q
        loc = Q.CGEventGetLocation(Q.CGEventCreate(None))
        return int(round(loc.x)), int(round(loc.y))

    @property
    def pressed(self) -> bool:
        """The left button is held down (a drag is in progress)."""
        return self._pressed

    def move(self, x: int, y: int) -> None:
        Q = self._Q
        # With the button down, apps only see a drag if the moves are "dragged" events.
        kind = Q.kCGEventLeftMouseDragged if self._pressed else Q.kCGEventMouseMoved
        event = Q.CGEventCreateMouseEvent(None, kind, (x, y), Q.kCGMouseButtonLeft)
        Q.CGEventPost(Q.kCGHIDEventTap, event)

    def click(self, x: int, y: int, count: int = 1) -> None:
        """Left click; ``count`` 2/3 makes it the second/third click of a double/triple."""
        Q = self._Q
        self._post_click(Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp, Q.kCGMouseButtonLeft,
                         x, y, count)

    def press(self, x: int, y: int) -> None:
        """Hold the left button down at (x, y) until ``release``."""
        if self._pressed:
            return
        Q = self._Q
        event = Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseDown, (x, y),
                                          Q.kCGMouseButtonLeft)
        Q.CGEventPost(Q.kCGHIDEventTap, event)
        self._pressed = True

    def release(self, x: int | None = None, y: int | None = None) -> None:
        """Let go of a held left button (where the cursor is, unless given). Safe to call
        any time: pause, cursor off and quit call it so the button can't stay stuck down."""
        if not self._pressed:
            return
        Q = self._Q
        if x is None or y is None or x < 0 or y < 0:
            x, y = self.position()
        event = Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseUp, (x, y),
                                          Q.kCGMouseButtonLeft)
        Q.CGEventPost(Q.kCGHIDEventTap, event)
        self._pressed = False

    def _post_click(
        self, down_type: int, up_type: int, button: int, x: int, y: int, count: int
    ) -> None:
        Q = self._Q
        for kind in (down_type, up_type):
            event = Q.CGEventCreateMouseEvent(None, kind, (x, y), button)
            # The click count is what makes macOS treat a second click as a double click.
            Q.CGEventSetIntegerValueField(event, Q.kCGMouseEventClickState, max(1, count))
            Q.CGEventPost(Q.kCGHIDEventTap, event)

    def _button(
        self, down_type: int, up_type: int, button: int, x: int, y: int, n: int = 1
    ) -> None:
        for i in range(1, n + 1):
            self._post_click(down_type, up_type, button, x, y, i)

    def scroll(self, dy_lines: int, dx_lines: int = 0) -> None:
        Q = self._Q
        event = Q.CGEventCreateScrollWheelEvent(
            None, Q.kCGScrollEventUnitLine, 2, dy_lines, dx_lines
        )
        Q.CGEventPost(Q.kCGHIDEventTap, event)

    # InputBackend protocol (opengaze/gestures/bindings.py) ------------------

    def mouse(self, action: str, x: int, y: int, count: int = 1) -> None:
        Q = self._Q
        left = (Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp, Q.kCGMouseButtonLeft)
        right = (Q.kCGEventRightMouseDown, Q.kCGEventRightMouseUp, Q.kCGMouseButtonRight)
        middle = (Q.kCGEventOtherMouseDown, Q.kCGEventOtherMouseUp, Q.kCGMouseButtonCenter)
        if action == "click":
            self.click(x, y, count)
        elif action == "press":
            self.press(x, y)
        elif action == "release":
            self.release(x, y)
        elif action == "double_click":
            self._button(*left, x, y, 2)
        elif action == "right_click":
            self._post_click(*right, x, y, count)
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
            self._origin[0] + int(x_norm * self._screen.width),
            self._origin[1] + int(y_norm * self._screen.height),
        )


def list_displays() -> list[tuple[int, int, int, int, bool]]:
    """Active displays as (x, y, width, height, is_main) in global logical points."""
    if sys.platform != "darwin":
        return []
    import Quartz
    err, ids, count = Quartz.CGGetActiveDisplayList(16, None, None)
    if err:
        return []
    main = Quartz.CGMainDisplayID()
    out = []
    for d in ids[:count]:
        b = Quartz.CGDisplayBounds(d)
        out.append((
            int(b.origin.x), int(b.origin.y), int(b.size.width), int(b.size.height), d == main
        ))
    return out


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
