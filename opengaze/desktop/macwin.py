"""macOS window tweaks for floating, system-wide OpenGaze windows."""

from __future__ import annotations

import logging

from PySide6.QtWidgets import QWidget

log = logging.getLogger("opengaze.macwin")

# NSWindowCollectionBehavior: canJoinAllSpaces | stationary | fullScreenAuxiliary
_ALL_SPACES = (1 << 0) | (1 << 4) | (1 << 8)
_NONACTIVATING_PANEL = 1 << 7
_STATUS_LEVEL = 25  # NSStatusWindowLevel: above normal and floating windows


def float_everywhere(widget: QWidget, *, click_through: bool = False) -> None:
    """Show ``widget`` over every Space and full-screen app without taking focus.

    Call after the widget is shown (its native window must exist).
    """
    try:
        import objc

        win = objc.objc_object(c_void_p=int(widget.winId())).window()
        win.setCollectionBehavior_(_ALL_SPACES)
        win.setLevel_(_STATUS_LEVEL)
        win.setHidesOnDeactivate_(False)
        if win.isKindOfClass_(objc.lookUpClass("NSPanel")):
            win.setStyleMask_(win.styleMask() | _NONACTIVATING_PANEL)
            win.setBecomesKeyOnlyIfNeeded_(True)
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
