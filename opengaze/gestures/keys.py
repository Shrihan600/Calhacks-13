"""Key-combo parsing and macOS virtual key codes for gesture keybinds.

``parse_key_combo("cmd+shift+tab")`` → ``KeyCombo(modifiers=("cmd", "shift"), key="tab")``.
This module is pure data so the config validator can use it without Quartz; posting the
events lives in ``control/cursor.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Physical key codes (kVK_* in Carbon's Events.h), ANSI layout.
KEY_CODES: dict[str, int] = {
    "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05, "z": 0x06, "x": 0x07,
    "c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C, "w": 0x0D, "e": 0x0E, "r": 0x0F, "y": 0x10,
    "t": 0x11, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17, "=": 0x18,
    "9": 0x19, "7": 0x1A, "-": 0x1B, "8": 0x1C, "0": 0x1D, "]": 0x1E, "o": 0x1F, "u": 0x20,
    "[": 0x21, "i": 0x22, "p": 0x23, "l": 0x25, "j": 0x26, "'": 0x27, "k": 0x28, ";": 0x29,
    "\\": 0x2A, ",": 0x2B, "/": 0x2C, "n": 0x2D, "m": 0x2E, ".": 0x2F, "`": 0x32,
    "return": 0x24, "enter": 0x24, "tab": 0x30, "space": 0x31, "delete": 0x33,
    "backspace": 0x33, "escape": 0x35, "esc": 0x35, "forward_delete": 0x75,
    "home": 0x73, "end": 0x77, "page_up": 0x74, "page_down": 0x79,
    "left": 0x7B, "right": 0x7C, "down": 0x7D, "up": 0x7E,
    "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76, "f5": 0x60, "f6": 0x61,
    "f7": 0x62, "f8": 0x64, "f9": 0x65, "f10": 0x6D, "f11": 0x67, "f12": 0x6F,
}

MODIFIER_ALIASES: dict[str, str] = {
    "cmd": "cmd", "command": "cmd", "⌘": "cmd",
    "shift": "shift", "⇧": "shift",
    "ctrl": "ctrl", "control": "ctrl", "⌃": "ctrl",
    "alt": "alt", "option": "alt", "opt": "alt", "⌥": "alt",
    "fn": "fn",
}


@dataclass(frozen=True, slots=True)
class KeyCombo:
    modifiers: tuple[str, ...]  # canonical names, in the order given
    key: str

    @property
    def key_code(self) -> int:
        return KEY_CODES[self.key]


def parse_key_combo(text: str) -> KeyCombo:
    parts = [p.strip().lower() for p in text.split("+")]
    # "cmd++" means cmd and the "+"/"=" key; keep it simple and refuse it explicitly.
    if not parts or any(not p for p in parts):
        raise ValueError(f"bad key combo {text!r}: use names joined by '+', e.g. 'cmd+shift+tab'")
    *mods, key = parts
    canonical: list[str] = []
    for m in mods:
        if m not in MODIFIER_ALIASES:
            raise ValueError(f"bad key combo {text!r}: unknown modifier {m!r}")
        canonical.append(MODIFIER_ALIASES[m])
    if key not in KEY_CODES:
        raise ValueError(f"bad key combo {text!r}: unknown key {key!r}")
    return KeyCombo(tuple(canonical), key)
