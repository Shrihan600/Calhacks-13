"""macOS Accessibility API through ctypes: ask for permission, find the text caret.

PyObjC's ApplicationServices bindings aren't a dependency, so the few AX calls the app
needs go straight to the framework. Everything here returns None / False on failure
(not macOS, permission missing, the focused app not answering) instead of raising.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from dataclasses import dataclass

log = logging.getLogger("opengaze.ax")

_UTF8 = 0x08000100       # kCFStringEncodingUTF8
_RECT_TYPE = 3           # kAXValueCGRectType
_POINT_TYPE, _SIZE_TYPE = 1, 2
_TIMEOUT_S = 0.25        # a hung app must not stall the UI thread
# Roles that take typed text. Web text inputs report AXTextField / AXTextArea too.
_TEXT_ROLES = frozenset({"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"})


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class _Size(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class _Rect(ctypes.Structure):
    _fields_ = [("origin", _Point), ("size", _Size)]


@dataclass(frozen=True, slots=True)
class TextFocus:
    """Where typing goes, in global top-left screen points."""

    x: float
    y: float
    width: float
    height: float
    caret: bool  # True: the caret's own bounds; False: the whole text field


class _Lib:
    def __init__(self) -> None:
        vp, i32 = ctypes.c_void_p, ctypes.c_int32
        ax = ctypes.CDLL(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        self.ax, self.cf = ax, cf

        ax.AXIsProcessTrusted.restype = ctypes.c_bool
        ax.AXIsProcessTrustedWithOptions.restype = ctypes.c_bool
        ax.AXIsProcessTrustedWithOptions.argtypes = [vp]
        ax.AXUIElementCreateSystemWide.restype = vp
        ax.AXUIElementCopyAttributeValue.restype = i32
        ax.AXUIElementCopyAttributeValue.argtypes = [vp, vp, ctypes.POINTER(vp)]
        ax.AXUIElementCopyParameterizedAttributeValue.restype = i32
        ax.AXUIElementCopyParameterizedAttributeValue.argtypes = [vp, vp, vp, ctypes.POINTER(vp)]
        ax.AXUIElementSetMessagingTimeout.restype = i32
        ax.AXUIElementSetMessagingTimeout.argtypes = [vp, ctypes.c_float]
        ax.AXUIElementGetPid.restype = i32
        ax.AXUIElementGetPid.argtypes = [vp, ctypes.POINTER(i32)]
        ax.AXValueGetValue.restype = ctypes.c_bool
        ax.AXValueGetValue.argtypes = [vp, ctypes.c_uint32, vp]
        ax.AXValueGetTypeID.restype = ctypes.c_ulong

        cf.CFRelease.argtypes = [vp]
        cf.CFGetTypeID.restype = ctypes.c_ulong
        cf.CFGetTypeID.argtypes = [vp]
        cf.CFStringGetTypeID.restype = ctypes.c_ulong
        cf.CFStringCreateWithCString.restype = vp
        cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        cf.CFDictionaryCreate.restype = vp
        cf.CFDictionaryCreate.argtypes = [
            vp, ctypes.POINTER(vp), ctypes.POINTER(vp), ctypes.c_long, vp, vp
        ]
        self._names: dict[str, int] = {}

    def name(self, s: str) -> int:
        """A CFString for an attribute name, created once and kept for the process."""
        ref = self._names.get(s)
        if ref is None:
            ref = self.cf.CFStringCreateWithCString(None, s.encode(), _UTF8)
            self._names[s] = ref
        return ref


_lib: _Lib | None = None


def _get_lib() -> _Lib | None:
    global _lib
    if _lib is None and sys.platform == "darwin":
        try:
            _lib = _Lib()
        except (OSError, AttributeError) as exc:
            log.info("Accessibility API unavailable: %s", exc)
    return _lib


def prompt_for_accessibility() -> bool:
    """Ask macOS to list this app under Accessibility (showing its prompt if it isn't yet).

    Returns whether the permission is already granted.
    """
    lib = _get_lib()
    if lib is None:
        return False
    key = ctypes.c_void_p.in_dll(lib.ax, "kAXTrustedCheckOptionPrompt")
    true = ctypes.c_void_p.in_dll(lib.cf, "kCFBooleanTrue")
    keys = (ctypes.c_void_p * 1)(key.value)
    values = (ctypes.c_void_p * 1)(true.value)
    options = lib.cf.CFDictionaryCreate(None, keys, values, 1, None, None)
    try:
        return bool(lib.ax.AXIsProcessTrustedWithOptions(options))
    finally:
        lib.cf.CFRelease(options)


class FocusProbe:
    """Finds the focused text field in other apps, and the caret inside it."""

    def __init__(self) -> None:
        self._lib = _get_lib()
        self._system = None
        if self._lib is not None:
            self._system = self._lib.ax.AXUIElementCreateSystemWide()
            self._lib.ax.AXUIElementSetMessagingTimeout(self._system, _TIMEOUT_S)
        self._own_pid = os.getpid()

    def text_focus(self) -> TextFocus | None:
        """The focused text input of another app, or None (no text focus, or no permission)."""
        if self._lib is None or not self._system:
            return None
        owned: list[int] = []
        try:
            return self._probe(owned)
        except Exception as exc:  # AX misbehaving must never take the UI down
            log.debug("focus probe failed: %s", exc)
            return None
        finally:
            for ref in owned:
                self._lib.cf.CFRelease(ref)

    def _copy(self, element: int, attr: str, owned: list[int]) -> int | None:
        out = ctypes.c_void_p()
        err = self._lib.ax.AXUIElementCopyAttributeValue(
            element, self._lib.name(attr), ctypes.byref(out)
        )
        if err or not out.value:
            return None
        owned.append(out.value)
        return out.value

    def _probe(self, owned: list[int]) -> TextFocus | None:
        lib = self._lib
        focused = self._copy(self._system, "AXFocusedUIElement", owned)
        if focused is None:
            return None
        pid = ctypes.c_int32()
        if lib.ax.AXUIElementGetPid(focused, ctypes.byref(pid)) == 0 and pid.value == self._own_pid:
            return None
        if self._string(self._copy(focused, "AXRole", owned)) not in _TEXT_ROLES:
            return None

        selection = self._copy(focused, "AXSelectedTextRange", owned)
        if selection is not None:
            out = ctypes.c_void_p()
            err = lib.ax.AXUIElementCopyParameterizedAttributeValue(
                focused, lib.name("AXBoundsForRange"), selection, ctypes.byref(out)
            )
            if not err and out.value:
                owned.append(out.value)
                rect = _Rect()
                if self._value(out.value, _RECT_TYPE, rect) and rect.size.height > 0:
                    return TextFocus(rect.origin.x, rect.origin.y,
                                     rect.size.width, rect.size.height, caret=True)

        pos, size = _Point(), _Size()
        if self._value(self._copy(focused, "AXPosition", owned), _POINT_TYPE, pos) and \
                self._value(self._copy(focused, "AXSize", owned), _SIZE_TYPE, size):
            return TextFocus(pos.x, pos.y, size.width, size.height, caret=False)
        return None

    def _value(self, ref: int | None, kind: int, out: ctypes.Structure) -> bool:
        if ref is None or self._lib.cf.CFGetTypeID(ref) != self._lib.ax.AXValueGetTypeID():
            return False
        return bool(self._lib.ax.AXValueGetValue(ref, kind, ctypes.byref(out)))

    def _string(self, ref: int | None) -> str | None:
        if ref is None or self._lib.cf.CFGetTypeID(ref) != self._lib.cf.CFStringGetTypeID():
            return None
        buf = ctypes.create_string_buffer(256)
        if not self._lib.cf.CFStringGetCString(ref, buf, len(buf), _UTF8):
            return None
        return buf.value.decode()
