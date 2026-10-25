"""Check the cursor output stage on its own, with no cameras or gaze model involved.

If this passes and the cursor still doesn't follow your gaze, the problem is upstream: run
``opengaze run`` with the debug overlay (D) or a trace. If it fails, nothing upstream matters.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class CursorCheck:
    name: str
    ok: bool
    detail: str


@dataclass
class CursorTestResult:
    checks: list[CursorCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(CursorCheck(name, ok, detail))

    def format(self) -> str:
        lines = [f"[{'OK  ' if c.ok else 'FAIL'}] {c.name:14} {c.detail}" for c in self.checks]
        lines.append("")
        lines.append(
            "Cursor output works." if self.ok else "Cursor output is broken — fix this first."
        )
        return "\n".join(lines)


def run_cursor_test(
    move: bool = True, settle_s: float = 0.05, say: Callable[[str], None] | None = None
) -> CursorTestResult:
    """Accessibility, displays, and a move/read-back round trip at five points per display."""
    say = say or (lambda _m: None)
    res = CursorTestResult()
    if sys.platform != "darwin":
        res.add("platform", False, "cursor control is macOS-only")
        return res
    try:
        from opengaze.control.cursor import (
            QuartzCursorDriver,
            is_accessibility_granted,
            list_displays,
        )
    except ImportError as exc:
        res.add("quartz", False, f"pyobjc Quartz missing ({exc}); pip install -e '.[desktop]'")
        return res

    granted = is_accessibility_granted()
    res.add("accessibility", granted,
            "granted" if granted else
            "NOT granted — System Settings › Privacy & Security › Accessibility: enable the app "
            "you launch opengaze from (Terminal, iTerm, VS Code…), then restart it")

    displays = list_displays()
    desc = "; ".join(
        f"{w}x{h} at ({x},{y}){' main' if main else ''}" for x, y, w, h, main in displays
    )
    res.add("displays", bool(displays), desc or "none found")
    if len(displays) > 1:
        res.add("multi-display", True,
                "gaze maps onto the display the run window is on (it tells the driver on show)")

    if not (granted and move):
        return res
    driver = QuartzCursorDriver()
    home = driver.position()
    worst = 0.0
    try:
        for x, y, w, h, _main in displays:
            driver.set_display(x, y, w, h)
            for nx, ny in ((0.5, 0.5), (0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95)):
                target = driver.norm_to_px(nx, ny)
                driver.move(*target)
                time.sleep(settle_s)
                got = driver.position()
                err = math.dist(target, got)
                worst = max(worst, err)
                say(f"  sent {target} → cursor at {got}")
    finally:
        driver.move(*home)
    res.add("move/readback", worst <= 2.0,
            f"worst miss {worst:.0f} px over {5 * len(displays)} moves"
            + ("" if worst <= 2.0 else " — events are posted but the cursor doesn't land there "
                                       "(secure input? another app moving it?)"))
    return res
