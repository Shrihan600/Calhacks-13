"""System-wide text injection and selection capture for macOS.

Types into whatever app has keyboard focus by posting Quartz keyboard events
that carry a Unicode payload, so it works with any keyboard layout and with
punctuation/accents from the transcript. Requires Accessibility permission,
like the cursor driver.
"""

from __future__ import annotations

import os
import sys
import time

# CGEventKeyboardSetUnicodeString is truncated beyond ~20 UTF-16 units per event.
_CHUNK = 20
_KEY_C = 8
# Virtual key codes for the keys the gaze keyboard sends as real key presses.
KEY_DELETE = 51
KEY_RETURN = 36


def _utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def chunk_text(text: str, size: int = _CHUNK) -> list[str]:
    """Split so no chunk exceeds ``size`` UTF-16 units (never splits a code point)."""
    chunks: list[str] = []
    cur = ""
    for ch in text:
        if cur and _utf16_len(cur + ch) > size:
            chunks.append(cur)
            cur = ""
        cur += ch
    if cur:
        chunks.append(cur)
    return chunks


def type_text(text: str) -> None:
    """Type ``text`` into the focused app."""
    if sys.platform != "darwin":
        raise RuntimeError("system-wide typing is macOS-only")
    import Quartz as Q

    src = Q.CGEventSourceCreate(Q.kCGEventSourceStateHIDSystemState)
    for chunk in chunk_text(text):
        for down in (True, False):
            ev = Q.CGEventCreateKeyboardEvent(src, 0, down)
            Q.CGEventSetFlags(ev, 0)  # ignore modifiers still held from the hotkey
            Q.CGEventKeyboardSetUnicodeString(ev, _utf16_len(chunk), chunk)
            Q.CGEventPost(Q.kCGHIDEventTap, ev)
        time.sleep(0.008)


def _press(keycode: int, flags: int = 0) -> None:
    import Quartz as Q

    src = Q.CGEventSourceCreate(Q.kCGEventSourceStateHIDSystemState)
    for down in (True, False):
        ev = Q.CGEventCreateKeyboardEvent(src, keycode, down)
        Q.CGEventSetFlags(ev, flags)
        Q.CGEventPost(Q.kCGHIDEventTap, ev)


def tap_key(keycode: int) -> None:
    """Press and release one key (e.g. ``KEY_DELETE``) in the focused app."""
    if sys.platform != "darwin":
        raise RuntimeError("system-wide typing is macOS-only")
    _press(keycode)


def copy_selection(timeout_s: float = 0.5) -> str:
    """Return the focused app's selected text via ⌘C, restoring the clipboard after."""
    import AppKit
    import Quartz as Q

    pb = AppKit.NSPasteboard.generalPasteboard()
    saved = pb.stringForType_(AppKit.NSPasteboardTypeString)
    before = pb.changeCount()
    _press(_KEY_C, Q.kCGEventFlagMaskCommand)
    end = time.monotonic() + timeout_s
    while pb.changeCount() == before and time.monotonic() < end:
        time.sleep(0.02)
    text = ""
    if pb.changeCount() != before:
        text = str(pb.stringForType_(AppKit.NSPasteboardTypeString) or "")
        pb.clearContents()
        if saved is not None:
            pb.setString_forType_(saved, AppKit.NSPasteboardTypeString)
    return text


def set_clipboard(text: str) -> None:
    import AppKit

    pb = AppKit.NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, AppKit.NSPasteboardTypeString)


class FocusTracker:
    """Remembers the last app with focus that isn't us, so clicking our floating
    bar (which activates this process) doesn't redirect typing into ourselves."""

    def __init__(self) -> None:
        import AppKit

        self._ws = AppKit.NSWorkspace.sharedWorkspace()
        self._own_pid = os.getpid()
        self.last = None

    def update(self) -> None:
        app = self._ws.frontmostApplication()
        if app is not None and app.processIdentifier() != self._own_pid:
            self.last = app

    @property
    def last_name(self) -> str:
        return str(self.last.localizedName()) if self.last is not None else "the focused app"

    def restore(self) -> bool:
        """Bring the remembered app back to the front. True if focus had to move."""
        import AppKit

        # NSApp.isActive, not frontmostApplication: the latter only catches up once the run
        # loop turns, so right after a click activated us it still names the other app.
        if self.last is None or not AppKit.NSApplication.sharedApplication().isActive():
            return False
        self.last.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            front = self._ws.frontmostApplication()
            if front is not None and front.processIdentifier() == self.last.processIdentifier():
                break
            time.sleep(0.02)
        time.sleep(0.08)  # let the target's key window settle before typing
        return True
