"""EyeMouse's own preferences (setup done, voice typing, open at login), as a small JSON file.

Gaze tuning stays in the app config YAML; this is only what the person chooses in the app.
"""

from __future__ import annotations

import json
import logging
import plistlib
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger("opengaze.consumer.settings")

_CONFIG_DIR = Path.home() / ".config" / "opengaze"
_LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
LOGIN_ITEM_LABEL = "com.opengaze.eyemouse"


@dataclass
class AppSettings:
    setup_done: bool = False
    # Voice typing sends audio to Cartesia while the mic is on; the person turns it on in
    # setup ("Allow microphone") or in Settings.
    voice_typing: bool = False
    open_at_login: bool = True
    path: Path = field(default=_CONFIG_DIR / "eyemouse.json", repr=False, compare=False)

    @classmethod
    def load(cls, path: Path | None = None) -> AppSettings:
        path = path or cls.path
        settings = cls(path=path)
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return settings
        except (OSError, ValueError) as exc:
            log.warning("ignoring unreadable settings %s: %s", path, exc)
            return settings
        for f in fields(cls):
            if f.name != "path" and isinstance(data.get(f.name), type(getattr(settings, f.name))):
                setattr(settings, f.name, data[f.name])
        return settings

    def save(self) -> None:
        data = {k: v for k, v in asdict(self).items() if k != "path"}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(self.path)


class LoginItem:
    """Starts EyeMouse at login through a per-user LaunchAgent."""

    def __init__(self, agents_dir: Path | None = None, label: str = LOGIN_ITEM_LABEL) -> None:
        self.label = label
        self.path = (agents_dir or _LAUNCH_AGENTS) / f"{label}.plist"

    @property
    def enabled(self) -> bool:
        return self.path.exists()

    def enable(self, program: list[str], working_dir: Path) -> None:
        """Write the agent; launchd picks it up at the next login (nothing starts now)."""
        plist = {
            "Label": self.label,
            "ProgramArguments": program,
            "WorkingDirectory": str(working_dir),
            "RunAtLoad": True,
            "ProcessType": "Interactive",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(plistlib.dumps(plist))

    def disable(self) -> None:
        if not self.path.exists():
            return
        if sys.platform == "darwin":
            # Harmless if it isn't loaded; stops launchd keeping a stale copy around.
            subprocess.run(["launchctl", "unload", str(self.path)],
                           check=False, capture_output=True)
        self.path.unlink(missing_ok=True)

    def apply(self, on: bool, program: list[str], working_dir: Path) -> None:
        if on:
            self.enable(program, working_dir)
        else:
            self.disable()
