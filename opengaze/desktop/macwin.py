"""macOS window tweaks for floating, system-wide OpenGaze windows."""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

log = logging.getLogger("opengaze.macwin")

# NSWindowCollectionBehavior: canJoinAllSpaces | stationary | fullScreenAuxiliary
_ALL_SPACES = (1 << 0) | (1 << 4) | (1 << 8)
_NONACTIVATING_PANEL = 1 << 7
_STATUS_LEVEL = 25  # NSStatusWindowLevel: above normal and floating windows

# NSEvent modifier flags and key codes (US layout positions).
_CTRL, _OPT, _CMD = 1 << 18, 1 << 19, 1 << 20
KEY_S, KEY_D, KEY_G = 1, 2, 5


def float_everywhere(widget: QWidget, *, click_through: bool = False) -> None:
    """Show ``widget`` over every Space and full-screen app without taking focus.

    Call after the widget is shown (its native window must exist).
    """
    if QGuiApplication.platformName() != "cocoa":
        return  # e.g. the offscreen platform in tests: winId() isn't an NSView
    try:
        import objc

        win = objc.objc_object(c_void_p=int(widget.winId())).window()
        win.setCollectionBehavior_(_ALL_SPACES)
        win.setLevel_(_STATUS_LEVEL)
        win.setHidesOnDeactivate_(False)
        if win.isKindOfClass_(objc.lookUpClass("NSPanel")):
            win.setStyleMask_(win.styleMask() | _NONACTIVATING_PANEL)
            win.setBecomesKeyOnlyIfNeeded_(True)
            # Adding the non-activating mask after the panel exists doesn't reach the window
            # server, so a click would still activate this app (and steal the text field's
            # focus). This private setter applies it.
            if win.respondsToSelector_(b"_setPreventsActivation:"):
                win._setPreventsActivation_(True)
        if click_through:
            win.setIgnoresMouseEvents_(True)
    except Exception as exc:
        log.info("native window tweaks unavailable: %s", exc)


def become_accessory_app() -> None:
    """No Dock icon or menu bar takeover; windows float above other apps."""
    try:
        import AppKit

        AppKit.NSApplication.sharedApplication().setActivationPolicy_(
            AppKit.NSApplicationActivationPolicyAccessory
        )
    except Exception:
        pass


class CtrlOptHotkeys:
    """System-wide ⌃⌥<key> shortcuts, delivered on the main thread.

    ``actions`` maps a key code (``KEY_D`` …) to a callback. Seeing keys typed in other apps
    needs Accessibility permission; without it only keys pressed in our own windows arrive.
    """

    def __init__(self, actions: dict[int, Callable[[], None]]) -> None:
        self._actions = dict(actions)
        self._monitors: list[object] = []

    def install(self) -> None:
        if self._monitors:
            return
        try:
            import AppKit
        except ImportError:
            return

        def match(ev) -> Callable[[], None] | None:
            flags = int(ev.modifierFlags())
            if flags & _CTRL and flags & _OPT and not flags & _CMD:
                return self._actions.get(int(ev.keyCode()))
            return None

        def on_global(ev) -> None:
            action = match(ev)
            if action:
                action()

        def on_local(ev):
            action = match(ev)
            if action:
                action()
                return None
            return ev

        mask = AppKit.NSEventMaskKeyDown
        self._monitors = [
            AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, on_global),
            AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, on_local),
        ]

    def remove(self) -> None:
        try:
            import AppKit

            for m in self._monitors:
                AppKit.NSEvent.removeMonitor_(m)
        except Exception:
            pass
        self._monitors = []
