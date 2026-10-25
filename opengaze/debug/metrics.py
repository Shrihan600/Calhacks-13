"""Hardware link metrics: which inputs are up, and a JSONL log of their health over time.

Used by ``opengaze diagnostics`` and ``opengaze headless`` (``--metrics``). Only numbers are
written, never images, so a metrics file is safe to share when reporting a Wi-Fi problem.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opengaze.cameras.manager import CameraHealth
from opengaze.cameras.types import CameraState
from opengaze.imu.runtime import ImuHealth

DEFAULT_METRICS_DIR = Path("logs") / "metrics"


@dataclass(frozen=True, slots=True)
class InputStatus:
    name: str
    level: str  # "ok" | "warn" | "down"
    detail: str


def input_statuses(
    cameras: dict[str, CameraHealth], imu: ImuHealth | None
) -> list[InputStatus]:
    """One line per input (each camera, then the IMU): is it delivering fresh data?"""
    out: list[InputStatus] = []
    for cid, h in cameras.items():
        if h.state is CameraState.STREAMING:
            fps = f"{h.measured_fps:.1f} fps" if h.measured_fps else "streaming"
            level = "ok"
            if h.measured_fps is not None and h.measured_fps < 10.0:
                level, fps = "warn", f"{fps} (low)"
            out.append(InputStatus(cid, level, fps))
        elif h.state is CameraState.STALLED:
            age = f"{h.last_frame_age_ms / 1000:.1f} s" if h.last_frame_age_ms else "?"
            out.append(InputStatus(cid, "warn", f"stalled, last frame {age} ago"))
        else:
            out.append(InputStatus(cid, "down", h.state.value + _error(h.last_error)))
    if imu is not None:
        if imu.state == "streaming":
            rate = f"{imu.rate_hz:.0f} Hz" if imu.rate_hz else "streaming"
            level = "warn" if imu.units_warning else "ok"
            out.append(InputStatus("imu", level, rate))
        elif imu.state == "stalled":
            detail = "no telemetry" if imu.samples == 0 else "stalled"
            if not imu.transport.get("packets") and imu.transport.get("subscribes_sent"):
                detail += " (subscribed, nothing received: IP, port or firewall?)"
            out.append(InputStatus("imu", "warn" if imu.samples else "down", detail))
        else:
            out.append(InputStatus("imu", "down", imu.state + _error(imu.last_error)))
    return out


def _error(message: str | None) -> str:
    return f": {message}" if message else ""


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 3)
    return value


class MetricsRecorder:
    """Appends one JSON line per input per ``interval_s`` and keeps a running summary."""

    def __init__(self, path: Path, interval_s: float = 1.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_s = interval_s
        self._fh = self.path.open("a", encoding="utf-8")
        self._next = 0.0
        self._start = time.monotonic()
        self._fps: dict[str, list[float]] = defaultdict(list)
        self._non_streaming_s: dict[str, float] = defaultdict(float)
        self._max_age_ms: dict[str, float] = defaultdict(float)
        self._last: dict[str, dict[str, Any]] = {}

    @staticmethod
    def default_path() -> Path:
        return DEFAULT_METRICS_DIR / time.strftime("hardware-%Y%m%d-%H%M%S.jsonl")

    def record(self, cameras: dict[str, CameraHealth], imu: ImuHealth | None) -> bool:
        """Write a snapshot if the interval elapsed. Returns True if it wrote one."""
        now = time.monotonic()
        if now < self._next:
            return False
        first = self._next == 0.0
        self._next = now + self.interval_s
        t = round(now - self._start, 3)
        for cid, h in cameras.items():
            row = {
                "t": t,
                "input": cid,
                "state": h.state.value,
                "fps": _clean(h.measured_fps),
                "jitter_ms": _clean(h.interval_jitter_ms),
                "age_ms": _clean(h.last_frame_age_ms),
                "delay_ms": _clean(h.transport_delay_ms),
                "frames": h.frames_received,
                "skipped": h.frames_skipped_by_source,
                "overwritten": h.frames_overwritten,
                "partial": h.partial_frames,
                "reconnects": h.reconnects,
                "error": h.last_error,
            }
            self._write(row)
            if h.measured_fps is not None and h.state is CameraState.STREAMING:
                self._fps[cid].append(h.measured_fps)
            if not first and h.state is not CameraState.STREAMING:
                self._non_streaming_s[cid] += self.interval_s
            if h.last_frame_age_ms is not None:
                self._max_age_ms[cid] = max(self._max_age_ms[cid], h.last_frame_age_ms)
            self._last[cid] = row
        if imu is not None:
            row = {
                "t": t,
                "input": "imu",
                "state": imu.state,
                "rate_hz": _clean(imu.rate_hz),
                "age_ms": _clean(imu.last_sample_age_ms),
                "samples": imu.samples,
                **{k: _clean(v) for k, v in imu.transport.items()},
                "units_warning": imu.units_warning,
                "error": imu.last_error,
            }
            self._write(row)
            if imu.rate_hz is not None and imu.state == "streaming":
                self._fps["imu"].append(imu.rate_hz)
            if not first and imu.state != "streaming":
                self._non_streaming_s["imu"] += self.interval_s
            if imu.last_sample_age_ms is not None:
                self._max_age_ms["imu"] = max(self._max_age_ms["imu"], imu.last_sample_age_ms)
            self._last["imu"] = row
        self._fh.flush()
        return True

    def _write(self, row: dict[str, Any]) -> None:
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    def summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, last in self._last.items():
            rates = self._fps.get(name, [])
            entry: dict[str, Any] = {
                "duration_s": round(time.monotonic() - self._start, 1),
                "mean_rate": round(sum(rates) / len(rates), 1) if rates else None,
                "min_rate": round(min(rates), 1) if rates else None,
                "not_streaming_s": round(self._non_streaming_s.get(name, 0.0), 1),
                "max_age_ms": round(self._max_age_ms.get(name, 0.0)),
            }
            for key in ("frames", "skipped", "partial", "reconnects", "samples", "packets",
                        "dropped", "invalid", "out_of_order", "parse_errors"):
                if key in last:
                    entry[key] = last[key]
            out[name] = entry
        return out

    def format_summary(self) -> str:
        lines = [f"metrics: {self.path}"]
        for name, entry in self.summary().items():
            fields = "  ".join(f"{k}={v}" for k, v in entry.items() if v is not None)
            lines.append(f"  {name:10} {fields}")
        return "\n".join(lines)

    def close(self) -> None:
        if not self._fh.closed:
            self._write({"t": round(time.monotonic() - self._start, 3), "summary": self.summary()})
            self._fh.close()
