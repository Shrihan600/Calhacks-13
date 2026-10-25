"""Per-tick pipeline trace: what every stage did, from camera frame to cursor position.

One JSON object per line (JSONL), flushed as it goes so a crash keeps everything up to it:

    {"kind": "header", ...}            run metadata: profile, schema, screen, loop settings
    {"kind": "tick", "t": ..., ...}    one per gaze-loop tick (see ``tick_record``)
    {"kind": "event", "t": ..., ...}   things that happened between ticks (cursor toggled, …)

A trace holds eye *features* and gaze positions but no images. ``analyze.py`` turns a trace
into a stage-by-stage report.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from opengaze.config.fingerprint import profile_fingerprint
from opengaze.config.schema import HardwareProfile
from opengaze.gaze.loop import GazeLoop, GazeState
from opengaze.gaze.tracker import GazeTracker, TrackerFrame

TRACE_DIR = Path.home() / ".config" / "opengaze" / "traces"
TRACE_SUFFIX = ".trace.jsonl"


@dataclass
class CursorTick:
    """What the cursor stage did on one tick."""

    enabled: bool                         # cursor control switched on (Tab)
    backend: bool                         # a cursor driver exists (Accessibility granted)
    commanded: tuple[int, int] | None     # where the cursor was told to go this tick
    actual: tuple[int, int] | None        # where macOS says the cursor is, read back
    blocked: str | None = None            # why it wasn't moved, if it wasn't


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def _cameras(frame: TrackerFrame, now_ns: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cid, f in frame.camera_frames.items():
        entry: dict[str, Any] = {
            "seq": f.sequence,
            "fresh": frame.fresh.get(cid, False),
            "age_ms": round((now_ns - f.host_receive_ns) / 1e6, 1),  # since it arrived
            "status": f.status.value,
            "kind": f.source_kind.value,
        }
        if f.capture_ns is not None:  # since the sensor captured it (device timestamps)
            entry["lat_ms"] = round((now_ns - f.capture_ns) / 1e6, 1)
        truth = f.ground_truth or {}
        if "gaze_x" in truth:
            entry["truth"] = [_r(truth["gaze_x"]), _r(truth["gaze_y"])]
        out[cid] = entry
    return out


def _eyes(frame: TrackerFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for side, obs, closed in (
        ("left", frame.left, frame.left_closed),
        ("right", frame.right, frame.right_closed),
    ):
        result = frame.eye_results.get(side)
        if obs is None and result is None and closed is None:
            continue  # this eye isn't observed at all
        entry: dict[str, Any] = {"detected": obs is not None, "closed": closed}
        if obs is not None:
            entry.update(x=_r(obs.pupil_x_norm), y=_r(obs.pupil_y_norm), conf=_r(obs.confidence, 3))
        if result is not None:
            entry["feature_conf"] = _r(result.confidence, 3)
            if result.gaze is not None:
                entry["gaze"] = [_r(v) for v in result.gaze.direction]
                entry["model_conf"] = _r(result.gaze.model_confidence, 3)
        out[side] = entry
    return out


def _head(frame: TrackerFrame) -> dict[str, Any] | None:
    h = frame.head
    if h is None:
        return None
    return {
        "yaw": _r(h.yaw_deg, 2),
        "pitch": _r(h.pitch_deg, 2),
        "roll": _r(h.roll_deg, 2),
        "pos_cm": [_r(v, 2) for v in h.position_cm] if h.position_cm else None,
        "source": h.source,
        "conf": _r(h.confidence, 3),
    }


def truth_of(frame: TrackerFrame) -> tuple[float, float] | None:
    """Where a synthetic eye was really looking this tick (first eye camera that knows)."""
    for f in frame.camera_frames.values():
        truth = f.ground_truth or {}
        if "gaze_x" in truth:
            return float(truth["gaze_x"]), float(truth["gaze_y"])
    return None


def tick_record(
    state: GazeState,
    t: float,
    *,
    cursor: CursorTick | None = None,
    imu: Any = None,
    now_ns: int | None = None,
) -> dict[str, Any]:
    """JSON-ready record of one gaze-loop tick."""
    frame = state.frame
    fv = frame.features
    rec: dict[str, Any] = {
        "kind": "tick",
        "t": round(t, 4),
        "cams": _cameras(frame, now_ns if now_ns is not None else time.monotonic_ns()),
        "eyes": _eyes(frame),
        "head": _head(frame),
        "feat": None,
        "raw": [_r(v) for v in state.raw] if state.raw else None,
        "conf": _r(state.confidence, 3),
        "drop": state.drop_reason,
        "gaze": [_r(state.gaze_x), _r(state.gaze_y)],
        "updated": state.updated,
        "gestures": [e.gesture.value for e in state.gesture.events],
    }
    if fv is not None:
        rec["feat"] = {
            "schema": fv.schema_id,
            "values": [_r(v, 5) for v in fv.values],
            "valid": [bool(v) for v in fv.valid],
            "skew_ms": round(fv.max_skew_ns / 1e6, 1),
        }
    truth = truth_of(frame)
    if truth is not None:
        rec["truth"] = [_r(truth[0]), _r(truth[1])]
    if imu is not None:
        rec["imu"] = {"state": imu.state, "rate_hz": _r(imu.rate_hz, 1)}
    if cursor is not None:
        rec["cursor"] = {
            "enabled": cursor.enabled,
            "backend": cursor.backend,
            "cmd": list(cursor.commanded) if cursor.commanded else None,
            "actual": list(cursor.actual) if cursor.actual else None,
            "blocked": cursor.blocked,
        }
    return rec


def run_header(
    profile: HardwareProfile, tracker: GazeTracker, loop: GazeLoop, **extra: Any
) -> dict[str, Any]:
    """Trace header: which rig, schema and loop settings produced the ticks."""
    return {
        "profile_id": profile.id,
        "fingerprint": profile_fingerprint(profile),
        "schema": tracker.schema_id,
        "mode": tracker.mode.value if tracker.mode else "A",
        "estimation": profile.calibration.geometry_estimation,
        "cameras": sorted(profile.enabled_cameras),
        "min_confidence": loop.min_confidence,
        "needs_head": profile.calibration.geometry_estimation == "geometric",
        **extra,
    }


class TraceWriter:
    """Appends records to a ``.trace.jsonl`` file."""

    def __init__(self, path: Path, header: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f: IO[str] | None = path.open("w", buffering=1)  # line-buffered
        self.n_ticks = 0
        self._write({"kind": "header", "created": time.time(), **header})

    @classmethod
    def new(cls, header: dict[str, Any], trace_dir: Path | None = None, name: str = "run"):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return cls((trace_dir or TRACE_DIR) / f"{name}-{stamp}{TRACE_SUFFIX}", header)

    def _write(self, rec: dict[str, Any]) -> None:
        if self._f is not None:
            self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def tick(self, rec: dict[str, Any]) -> None:
        self._write(rec)
        self.n_ticks += 1

    def event(self, t: float, message: str, **data: Any) -> None:
        self._write({"kind": "event", "t": round(t, 4), "msg": message, **data})

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


@dataclass
class Trace:
    header: dict[str, Any]
    ticks: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


def load_trace(path: Path) -> Trace:
    """Read a trace; a truncated last line (crash mid-write) is skipped."""
    return parse_trace(path.read_text().splitlines())


def parse_trace(lines: Iterable[str]) -> Trace:
    trace = Trace(header={})
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = rec.get("kind")
        if kind == "header":
            trace.header = rec
        elif kind == "tick":
            trace.ticks.append(rec)
        elif kind == "event":
            trace.events.append(rec)
    return trace


def list_traces(trace_dir: Path | None = None) -> list[Path]:
    """Saved traces, oldest first (searches session folders too)."""
    d = trace_dir or TRACE_DIR
    if not d.exists():
        return []
    return sorted(d.rglob(f"*{TRACE_SUFFIX}"), key=lambda p: p.stat().st_mtime)
