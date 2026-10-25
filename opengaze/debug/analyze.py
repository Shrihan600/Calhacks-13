"""Turn trace ticks into a stage-by-stage report: where does the pipeline break?

Stages, in pipeline order: cameras → eyes → head → features → mapping → gaze → cursor.
Each gets a status (ok / warn / fail / n/a), a few metrics, and findings that say what to
check. ``TraceReport.first_problem`` is the earliest failing stage: fix that first, because
everything after it inherits the damage.

Works on a saved trace or on the last couple of seconds of a live one (the run window's
debug overlay).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Status = Literal["ok", "warn", "fail", "n/a"]
_RANK = {"n/a": 0, "ok": 1, "warn": 2, "fail": 3}

# Thresholds. Fractions are of ticks unless said otherwise.
MIN_CAMERA_FPS = 15.0
MAX_FRAME_AGE_MS = 150.0  # since arrival: the tracker isn't keeping up
MAX_LATENCY_MS = 120.0    # since capture: transport/driver delay
MAX_SKEW_MS = 50.0        # between the observations combined into one feature vector
MIN_DETECTION = 0.8       # of ticks with the eye open
MAX_CLOSED = 0.3
MIN_GAZE_UPDATED = 0.6
MAX_OUT_OF_BOUNDS = 0.15  # raw prediction more than 5% outside the screen
MAX_JITTER_PX = 40.0
MAX_ERROR_NORM = 0.08     # vs ground truth (simulated rigs)
MAX_CURSOR_LAG_PX = 30.0

_DROP_HINTS = {
    "no_model": "no calibration loaded — press C (or A) to calibrate",
    "no_frames": "no camera has delivered a frame",
    "eye_closed": "an eye is reported closed — gaze freezes while either eye is closed",
    "predict_error": "the model rejected the features (schema mismatch?) — recalibrate",
    "no_prediction": "too few valid features for any calibrated sub-model "
                     "(an eye or head pose missing that calibration had)",
    "low_confidence": "prediction confidence below the loop's min_confidence",
}


@dataclass
class Stage:
    name: str
    status: Status = "n/a"
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)

    def flag(self, status: Status, finding: str) -> None:
        if _RANK[status] > _RANK[self.status]:
            self.status = status
        self.findings.append(finding)

    def ok(self) -> None:
        if self.status == "n/a":
            self.status = "ok"


@dataclass
class TraceReport:
    n_ticks: int
    duration_s: float
    tick_hz: float | None
    stages: list[Stage]

    def stage(self, name: str) -> Stage:
        return next(s for s in self.stages if s.name == name)

    @property
    def first_problem(self) -> Stage | None:
        for status in ("fail", "warn"):
            for s in self.stages:
                if s.status == status:
                    return s
        return None

    def format(self) -> str:
        hz = f"{self.tick_hz:.1f} Hz" if self.tick_hz else "—"
        lines = [f"{self.n_ticks} ticks over {self.duration_s:.1f} s ({hz})", ""]
        for s in self.stages:
            metrics = "  ".join(f"{k}={_fmt(k, v)}" for k, v in s.metrics.items())
            lines.append(f"[{s.status.upper():4}] {s.name:9} {metrics}")
            for f in s.findings:
                lines.append(f"         - {f}")
        first = self.first_problem
        lines.append("")
        lines.append(
            f"First problem: {first.name} — {first.findings[0]}" if first and first.findings
            else "No problems found."
        )
        return "\n".join(lines)


# Metrics that are fractions of ticks/frames (printed as percentages).
_FRACTIONS = frozenset({
    "detect", "closed", "3d", "present", "valid", "all_invalid", "updated", "drops",
    "off_screen", "enabled", "moved", "blocked",
})


def _fmt(key: str, v: Any) -> str:
    if isinstance(v, dict):
        return ",".join(f"{k}:{_fmt(key if key in _FRACTIONS else k, x)}" for k, x in v.items())
    if isinstance(v, float):
        return f"{v:.0%}" if key in _FRACTIONS else f"{v:g}"
    return str(v)


def _frac(n: int, d: int) -> float:
    return n / d if d else 0.0


# ---------------------------------------------------------------------------


def analyze(
    ticks: Sequence[dict[str, Any]], header: dict[str, Any] | None = None
) -> TraceReport:
    header = header or {}
    n = len(ticks)
    ts = np.array([t["t"] for t in ticks], dtype=float)
    duration = float(ts[-1] - ts[0]) if n > 1 else 0.0
    tick_hz = (n - 1) / duration if duration > 0 else None
    screen = header.get("screen_px")
    stages = [
        _cameras(ticks, duration, header),
        _eyes(ticks, "eye3d" in str(header.get("schema", ""))),
        _head(ticks, header),
        _features(ticks),
        _mapping(ticks),
        _gaze(ticks, screen, tick_hz),
        _cursor(ticks),
    ]
    return TraceReport(n, duration, tick_hz, stages)


def _cameras(ticks: Sequence[dict], duration: float, header: dict) -> Stage:
    st = Stage("cameras")
    if not ticks:
        st.flag("fail", "no ticks recorded")
        return st
    expected = set(header.get("cameras", ())) | {c for t in ticks for c in t["cams"]}
    if not expected:
        st.flag("fail", "no camera delivered any frame — check device_id (`opengaze "
                        "probe-cameras`) and camera permission")
        return st
    for cid in sorted(expected):
        seen = [t["cams"][cid] for t in ticks if cid in t["cams"]]
        if not seen:
            st.flag("fail", f"{cid}: no frames")
            continue
        fresh = sum(1 for c in seen if c["fresh"])
        fps = fresh / duration if duration > 0 else None
        age = float(np.percentile([c["age_ms"] for c in seen], 95))
        st.metrics[cid] = {"fps": round(fps, 1) if fps else 0.0, "age95": round(age)}
        lats = [c["lat_ms"] for c in seen if c.get("lat_ms") is not None]
        if lats:
            lat = float(np.percentile(lats, 95))
            st.metrics[cid]["lat95"] = round(lat)
            if lat > MAX_LATENCY_MS:
                st.flag("warn", f"{cid}: frames reach the tracker {lat:.0f} ms after capture "
                                "(p95) — USB bandwidth, a slow hub or driver buffering")
        if fps is not None and fps < MIN_CAMERA_FPS:
            st.flag("warn", f"{cid}: only {fps:.0f} new frames/s reach the tracker")
        if age > MAX_FRAME_AGE_MS:
            st.flag("warn", f"{cid}: frames are {age:.0f} ms old (p95) — capture is lagging")
        bad = sum(1 for c in seen if c["status"] != "ok")
        if bad > 0.05 * len(seen):
            st.flag("warn", f"{cid}: {_frac(bad, len(seen)):.0%} partial/corrupt frames")
    st.ok()
    return st


def _eyes(ticks: Sequence[dict], eye_3d: bool) -> Stage:
    st = Stage("eyes")
    for side in ("left", "right"):
        entries = [t["eyes"][side] for t in ticks if side in t["eyes"]]
        if not entries:
            continue
        closed = sum(1 for e in entries if e["closed"])
        open_ = [e for e in entries if not e["closed"]]
        detected = sum(1 for e in open_ if e["detected"])
        det = _frac(detected, len(open_))
        m: dict[str, Any] = {"detect": det, "closed": _frac(closed, len(entries))}
        confs = [e["conf"] for e in entries if e.get("conf") is not None]
        if confs:
            m["conf"] = round(float(np.mean(confs)), 2)
        if eye_3d:
            m["3d"] = _frac(sum(1 for e in open_ if "gaze" in e), len(open_))
        st.metrics[side] = m
        if open_ and det < MIN_DETECTION:
            st.flag("fail" if det < 0.5 else "warn",
                    f"{side} pupil found in {det:.0%} of open-eye frames — check focus, IR "
                    "lighting and orientation (`opengaze diagnostics`, normalized view)")
        if _frac(closed, len(entries)) > MAX_CLOSED:
            st.flag("warn", f"{side} eye reported closed {_frac(closed, len(entries)):.0%} of "
                            "the time — a weak pupil detection counts as closed "
                            "(gaze.closed_confidence)")
        if "3d" in m and m["3d"] < 0.5 and det >= 0.5:
            st.flag("warn", f"{side} 3D eye model gave a gaze in {m['3d']:.0%} of frames — "
                            "it needs varied eye movement to fit (look around before calibrating)")
    if not st.metrics:
        st.flag("fail", "no eye observed — face/eye not found in any frame")
    st.ok()
    return st


def _head(ticks: Sequence[dict], header: dict) -> Stage:
    st = Stage("head")
    heads = [t["head"] for t in ticks if t.get("head")]
    imus = [t["imu"] for t in ticks if t.get("imu")]
    if imus:
        states = Counter(i["state"] for i in imus)
        st.metrics["imu"] = dict(states)
        if states.get("streaming", 0) < 0.9 * len(imus):
            st.flag("warn", f"IMU streaming only {_frac(states.get('streaming', 0), len(imus)):.0%}"
                            " of the time — check the serial/UDP link")
    if not heads:
        if header.get("needs_head"):
            st.flag("warn", "no head pose at all — head movement won't be compensated")
        return st
    st.metrics["present"] = _frac(len(heads), len(ticks))
    st.metrics["source"] = dict(Counter(h["source"] for h in heads))
    yaws = [h["yaw"] for h in heads]
    st.metrics["yaw_range"] = round(max(yaws) - min(yaws), 1)
    st.ok()
    return st


def _features(ticks: Sequence[dict]) -> Stage:
    st = Stage("features")
    feats = [t["feat"] for t in ticks if t.get("feat")]
    if not feats:
        st.flag("fail", "no feature vectors — nothing upstream produced data")
        return st
    valid = [float(np.mean(f["valid"])) for f in feats]
    st.metrics["valid"] = round(float(np.mean(valid)), 2)
    st.metrics["all_invalid"] = _frac(sum(1 for v in valid if v == 0.0), len(feats))
    skews = [f["skew_ms"] for f in feats if f.get("skew_ms")]
    if skews:
        st.metrics["skew95_ms"] = round(float(np.percentile(skews, 95)), 1)
    if st.metrics["all_invalid"] > 0.5:
        st.flag("fail", "most feature vectors are entirely invalid")
    if skews and st.metrics["skew95_ms"] > MAX_SKEW_MS:
        st.flag("warn", f"observations combined into one sample are {st.metrics['skew95_ms']:.0f}"
                        " ms apart (p95) — one camera lags the other; the eyes are seen at "
                        "different moments")
    st.ok()
    return st


def _mapping(ticks: Sequence[dict]) -> Stage:
    st = Stage("mapping")
    n = len(ticks)
    if not n:
        return st
    updated = sum(1 for t in ticks if t["updated"])
    drops = Counter(t["drop"] for t in ticks if t.get("drop"))
    st.metrics["updated"] = _frac(updated, n)
    if drops:
        st.metrics["drops"] = {k: _frac(v, n) for k, v in drops.most_common()}
    if _frac(updated, n) < MIN_GAZE_UPDATED:
        reason, count = drops.most_common(1)[0] if drops else ("?", 0)
        st.flag("fail" if _frac(updated, n) < 0.2 else "warn",
                f"gaze moved on only {_frac(updated, n):.0%} of ticks; mostly "
                f"{reason} ({_frac(count, n):.0%}): {_DROP_HINTS.get(reason, reason)}")
    st.ok()
    return st


def _gaze(ticks: Sequence[dict], screen: Sequence[int] | None, tick_hz: float | None) -> Stage:
    st = Stage("gaze")
    upd = [t for t in ticks if t["updated"] and t.get("raw")]
    if len(upd) < 3:
        return st
    raw = np.array([t["raw"] for t in upd], dtype=float)
    out = np.any((raw < -0.05) | (raw > 1.05), axis=1).mean()
    st.metrics["off_screen"] = float(out)
    if out > MAX_OUT_OF_BOUNDS:
        st.flag("warn", f"{out:.0%} of predictions land off screen — calibration doesn't fit "
                        "(eye camera or headset moved since calibrating?) — recalibrate")
    scale = np.array(screen, dtype=float) if screen else None

    jit = _jitter(upd, scale)
    if jit is not None:
        unit = "px" if scale is not None else "norm"
        st.metrics[f"jitter_{unit}"] = round(jit, 3 if scale is None else 1)
        if scale is not None and jit > MAX_JITTER_PX:
            st.flag("warn", f"raw gaze jitters {jit:.0f} px while fixating — noisy pupil "
                            "detection; the filter will lag if you smooth harder")

    with_truth = [t for t in upd if t.get("truth")]
    if len(with_truth) >= 3:
        truth = np.array([t["truth"] for t in with_truth], dtype=float)
        rawt = np.array([t["raw"] for t in with_truth], dtype=float)
        err = np.linalg.norm(np.clip(rawt, 0, 1) - truth, axis=1)
        st.metrics["err_median"] = round(float(np.median(err)), 3)
        st.metrics["err_p90"] = round(float(np.percentile(err, 90)), 3)
        if scale is not None:
            err_px = np.linalg.norm((np.clip(rawt, 0, 1) - truth) * scale, axis=1)
            st.metrics["err_median_px"] = round(float(np.median(err_px)), 1)
        if np.median(err) > MAX_ERROR_NORM:
            st.flag("fail", f"median error vs ground truth {np.median(err):.3f} of the screen "
                            "— calibration or mapping is wrong")
        lag = _lag_ms(ticks, tick_hz)
        if lag is not None:
            st.metrics["lag_ms"] = round(lag)
    st.ok()
    return st


def _jitter(upd: Sequence[dict], scale: np.ndarray | None) -> float | None:
    """Frame-to-frame noise of the raw gaze: RMS 2D spread a fixation would show.

    Estimated from the median step between consecutive ticks, which saccades (rare, large
    steps) barely move. With ground truth, the steps of (raw − truth) are used, so smooth
    pursuit doesn't count as noise. For Gaussian noise of RMS spread s, median step = 1.18 s.
    """
    pts = np.array([u["raw"] for u in upd], dtype=float)
    if all(u.get("truth") for u in upd):
        pts = pts - np.array([u["truth"] for u in upd], dtype=float)
    if scale is not None:
        pts = pts * scale
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(np.median(steps) / 1.1774) if len(steps) >= 3 else None


def _lag_ms(ticks: Sequence[dict], tick_hz: float | None, max_lag_s: float = 0.6) -> float | None:
    """Delay between where the eye looked (truth) and the smoothed gaze, by best alignment."""
    if not tick_hz:
        return None
    pairs = [(t["truth"], t["gaze"]) for t in ticks if t.get("truth")]
    if len(pairs) < 20:
        return None
    truth = np.array([p[0] for p in pairs], dtype=float)
    gaze = np.array([p[1] for p in pairs], dtype=float)
    if truth.std(axis=0).max() < 0.05:
        return None  # truth barely moves: lag isn't measurable
    best, best_err = 0, math.inf
    for k in range(int(max_lag_s * tick_hz) + 1):
        e = float(np.linalg.norm(gaze[k:] - truth[: len(truth) - k], axis=1).mean())
        if e < best_err:
            best, best_err = k, e
    return best / tick_hz * 1000.0


def _cursor(ticks: Sequence[dict]) -> Stage:
    st = Stage("cursor")
    cur = [t.get("cursor") for t in ticks]
    if not any(cur):
        return st
    n = len(ticks)
    enabled = [c for c in cur if c and c["enabled"]]
    st.metrics["enabled"] = _frac(len(enabled), n)
    if not any(c["backend"] for c in cur if c):
        st.flag("fail", "no cursor driver — grant Accessibility permission to the app you run "
                        "from (`opengaze cursor-test`)")
        return st
    if not enabled:
        st.flag("warn", "cursor control was off the whole time — press Tab to turn it on")
        return st
    moved = sum(1 for c in enabled if c["cmd"])
    st.metrics["moved"] = _frac(moved, len(enabled))
    blocked = Counter(c["blocked"] for c in enabled if c["blocked"])
    if blocked:
        st.metrics["blocked"] = {k: _frac(v, len(enabled)) for k, v in blocked.most_common()}
    # Commands are posted asynchronously: compare each command with the next tick's readback.
    lags = []
    for a, b in zip(cur, cur[1:], strict=False):
        if a and b and a["enabled"] and a["cmd"] and b["actual"]:
            lags.append(math.dist(a["cmd"], b["actual"]))
    if lags:
        lag = float(np.median(lags))
        st.metrics["follow_px"] = round(lag, 1)
        if lag > MAX_CURSOR_LAG_PX:
            st.flag("fail", f"the cursor ends up {lag:.0f} px from where it was sent — another "
                            "input is moving it, or the gaze window is on a different display "
                            "than the one being driven (`opengaze cursor-test`)")
    st.ok()
    return st
