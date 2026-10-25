"""Headless, deterministic runs of the full gaze pipeline: simulate a user, or replay a session.

``SteppedCameras`` drives real camera sources (synthetic or recorded) on a virtual clock and
serves their newest frames like ``CameraManager.latest``, so the tracker, calibration, gaze
loop and cursor stage run exactly as in the app, only faster than real time and repeatable.

``simulate``: a simulated user calibrates (9-dot grid or adaptive), then follows a scripted
gaze path — random fixations, then a smooth pursuit — while a fake cursor records where it
was sent. The trace carries ground truth, so the report measures accuracy and lag.

``replay``: re-runs a session recorded with ``opengaze run --record-video`` through the
current code, tick for tick, with the calibration that session used.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from opengaze.calibration.adaptive import AdaptiveCalibrationSession
from opengaze.calibration.mappers import make_mapper
from opengaze.calibration.model import ScreenMapper
from opengaze.calibration.session import GRID_9, CalibrationPoint, CalibrationSession
from opengaze.calibration.storage import load_bundle_file
from opengaze.cameras.registry import create_source
from opengaze.cameras.types import CameraFrame, EndOfStream
from opengaze.config.schema import AppConfig, HardwareProfile
from opengaze.debug import puppet
from opengaze.debug.analyze import TraceReport, analyze
from opengaze.debug.trace import CursorTick, TraceWriter, load_trace, run_header, tick_record
from opengaze.gaze.loop import GazeLoop
from opengaze.gaze.tracker import GazeTracker, TrackerFrame

TICK_HZ = 30.0
SIM_SCREEN = (1440, 900)


# ---------------------------------------------------------------------------
# Cameras on a virtual clock
# ---------------------------------------------------------------------------


class SteppedCameras:
    """Camera sources read on demand: ``advance(t)`` delivers every frame that has arrived by
    virtual time ``t``.

    Each source runs on its own virtual clock; frame timestamps are rebased onto one shared
    clock (``start_ns``), so injected latency delays a frame's arrival and dropped frames
    leave gaps, as they would live.
    """

    def __init__(self, profile: HardwareProfile) -> None:
        self._loop = asyncio.new_event_loop()
        self.sources = {}
        for cid, cam in profile.enabled_cameras.items():
            if cam.source_type not in ("synthetic", "recorded"):
                raise ValueError(f"{cid}: headless runs need synthetic or recorded cameras, "
                                 f"not {cam.source_type!r}")
            options = {**cam.options, "realtime": False}  # virtual clock, as fast as possible
            self.sources[cid] = create_source(cam.model_copy(update={"options": options}))
        self._latest: dict[str, CameraFrame | None] = dict.fromkeys(self.sources)
        self._pending: dict[str, CameraFrame | None] = dict.fromkeys(self.sources)
        self._origin: dict[str, int] = {}  # source clock value at its frame 0
        self._period: dict[str, float] = {}
        self.ended: set[str] = set()
        for cid, src in self.sources.items():
            self._run(src.connect())
            fps = src.get_capabilities().reported_fps or profile.cameras[cid].fps or TICK_HZ
            self._period[cid] = 1e9 / fps
        self.start_ns = time.monotonic_ns()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def _read(self, cid: str) -> CameraFrame | None:
        """Next frame from the source, rebased onto the shared clock."""
        try:
            frame = self._run(self.sources[cid].read_frame())
        except EndOfStream:
            self.ended.add(cid)
            return None
        if cid not in self._origin:
            first = frame.capture_ns if frame.capture_ns is not None else frame.host_receive_ns
            self._origin[cid] = first - int(frame.sequence * self._period[cid])
        shift = self.start_ns - self._origin[cid]
        return replace(
            frame,
            host_receive_ns=frame.host_receive_ns + shift,
            capture_ns=frame.capture_ns + shift if frame.capture_ns is not None else None,
        )

    def read_one(self, cid: str) -> CameraFrame | None:
        """Deliver the next frame of one camera now (replay: one recorded frame per tick)."""
        if cid not in self.ended:
            frame = self._pending[cid] or self._read(cid)
            self._pending[cid] = None
            if frame is not None:
                self._latest[cid] = frame
        return self._latest[cid]

    def advance(self, t: float) -> None:
        now = self.now_ns(t)
        for cid in self.sources:
            while cid not in self.ended:
                frame = self._pending[cid] or self._read(cid)
                if frame is None:
                    break
                if frame.host_receive_ns > now:
                    self._pending[cid] = frame  # still in transit
                    break
                self._pending[cid] = None
                self._latest[cid] = frame

    def latest(self, cid: str, *, consume: bool = True) -> CameraFrame | None:
        return self._latest.get(cid)

    def now_ns(self, t: float) -> int:
        return self.start_ns + int(t * 1e9)

    def close(self) -> None:
        for src in self.sources.values():
            self._run(src.disconnect())
        self._loop.close()


# ---------------------------------------------------------------------------
# Simulated user
# ---------------------------------------------------------------------------


@dataclass
class SimulatedUser:
    """Where the simulated eyes look, given where they're asked to look.

    The eyes reach a new target after ``reaction_s`` (saccade latency) and then hold it with
    Gaussian fixation jitter (``jitter_norm``, in screen fractions; 0.005 ≈ 7 px on 1440).
    """

    reaction_s: float = 0.2
    jitter_norm: float = 0.004
    seed: int = 0
    _rng: np.random.Generator = field(init=False)
    _asked: tuple[float, float] = field(init=False, default=(0.5, 0.5))
    _asked_at: float = field(init=False, default=0.0)
    _prev: tuple[float, float] = field(init=False, default=(0.5, 0.5))

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def ask(self, target: tuple[float, float], t: float, *, pursuit: bool = False) -> None:
        """Ask the eyes to look at ``target`` from time ``t``; pursuit = follow without delay."""
        if target != self._asked:
            if not pursuit:
                self._prev = self.eye(t)
                self._asked_at = t
            self._asked = target

    def eye(self, t: float) -> tuple[float, float]:
        base = self._prev if t - self._asked_at < self.reaction_s else self._asked
        jx, jy = self._rng.normal(0.0, self.jitter_norm, 2) if self.jitter_norm else (0.0, 0.0)
        return float(np.clip(base[0] + jx, 0, 1)), float(np.clip(base[1] + jy, 0, 1))

    def update(self, t: float) -> tuple[float, float]:
        x, y = self.eye(t)
        puppet.look_at(x, y)
        return x, y


class FakeCursor:
    """Stands in for ``QuartzCursorDriver``: remembers where it was sent."""

    def __init__(self, screen: tuple[int, int] = SIM_SCREEN) -> None:
        self.screen = screen
        self.position: tuple[int, int] | None = None
        self.moves = 0

    def norm_to_px(self, x: float, y: float) -> tuple[int, int]:
        return int(x * self.screen[0]), int(y * self.screen[1])

    def move(self, x: int, y: int) -> None:
        self.position = (x, y)
        self.moves += 1


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    report: TraceReport
    trace_path: Path
    calibration_points: int
    model: ScreenMapper


def sim_profile(profile: HardwareProfile) -> HardwareProfile:
    """The profile with every synthetic eye camera looking where the puppet says.

    The IMU is left out: its own real-time thread can't follow the virtual clock.
    """
    cams = {}
    for cid, cam in profile.cameras.items():
        if cam.source_type == "synthetic" and cam.options.get("pattern", "eye") == "eye":
            cam = cam.model_copy(update={"options": {**cam.options, "motion": "puppet"}})
        cams[cid] = cam
    imu = profile.imu.model_copy(update={"enabled": False}) if profile.imu else None
    return profile.model_copy(update={"cameras": cams, "imu": imu})


def _sim_config(config: AppConfig) -> AppConfig:
    # Synthetic head cameras show a test card: running MediaPipe on it only costs time.
    vision = config.vision.model_copy(update={"enable_face_tracking": False})
    return config.model_copy(update={"vision": vision})


class _Clock:
    def __init__(self, cams: SteppedCameras, tracker: GazeTracker, user: SimulatedUser) -> None:
        self.cams, self.tracker, self.user = cams, tracker, user
        self.t = 0.0

    def step(self) -> TrackerFrame:
        self.t += 1.0 / TICK_HZ
        self.user.update(self.t)
        self.cams.advance(self.t)
        return self.tracker.step(self.cams)


def _collect_point(clock: _Clock, target, settle_s: float, collect_s: float):
    clock.user.ask(target, clock.t)
    point, frames = CalibrationPoint(*target), []
    for _ in range(round(settle_s * TICK_HZ)):
        clock.step()
    for _ in range(round(collect_s * TICK_HZ)):
        frame = clock.step()
        if frame.features is not None:
            point.samples.append(frame.features)
            frames.append(frame)
    return point, frames


def _calibrate(clock: _Clock, profile: HardwareProfile, adaptive: bool, rng) -> tuple:
    tracker = clock.tracker
    estimation = profile.calibration.geometry_estimation
    factory = lambda: make_mapper(tracker.schema_id, estimation)  # noqa: E731
    tracker.freeze(False)
    # Intro screen: the eyes wander, which a 3D eye model needs to fit.
    for _ in range(round(3.0 * TICK_HZ)):
        if clock.t % 0.4 < 1 / TICK_HZ:
            clock.user.ask(tuple(rng.uniform(0.05, 0.95, 2)), clock.t)
        clock.step()

    def refeature(frames):
        return [fv for r in frames if (fv := tracker.refeature(r)) is not None]

    if adaptive:
        session = AdaptiveCalibrationSession(
            tracker.schema_id, mapper_factory=factory, geometry_estimation=estimation,
            tolerance_norm=0.15 if tracker.is_mode_a else 0.08, screen_px=SIM_SCREEN,
        )
        frames_by_point: dict[int, list] = {}
        while not session.done:
            had_model = session.model is not None
            point, frames = _collect_point(clock, session.next_target(), 0.35, 1.0)
            frames_by_point[id(point)] = frames
            session.submit(point)
            if not had_model and session.model is not None and not tracker.is_mode_a:
                tracker.freeze(True)
                session.refeature(lambda p: refeature(frames_by_point.get(id(p), [])))
        tracker.freeze(True)
        return session.result().model, len(session.training)

    session = CalibrationSession(
        tracker.schema_id, targets=GRID_9, collect_duration_s=1e9, mapper_factory=factory
    )
    records = []
    for i, target in enumerate(session.targets):
        point, frames = _collect_point(clock, target, 0.35, 1.5)
        session.points[i].samples = point.samples
        session.points[i].accepted = point.representative() is not None
        records.append(frames)
    tracker.freeze(True)
    for pt, frames in zip(session.points, records, strict=True):
        pt.samples = refeature(frames)
    return session.fit(*SIM_SCREEN).model, session.n_accepted


def simulate(
    profile: HardwareProfile,
    config: AppConfig,
    *,
    adaptive: bool = False,
    fixations: int = 16,
    fixation_s: float = 1.2,
    pursuit_s: float = 8.0,
    user: SimulatedUser | None = None,
    seed: int = 0,
    trace_dir: Path | None = None,
    record: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> SimResult:
    """Calibrate a simulated user, then drive the cursor along a scripted gaze path.

    ``record``: also save the test phase as a session in that folder (for ``replay``).
    """
    say = progress or (lambda _msg: None)
    profile = sim_profile(profile)
    rng = np.random.default_rng(seed)
    user = user or SimulatedUser(seed=seed)
    cams = SteppedCameras(profile)
    tracker = GazeTracker(profile, _sim_config(config))
    clock = _Clock(cams, tracker, user)
    try:
        say("calibrating" + (" (adaptive)" if adaptive else " (9-point grid)"))
        model, n_points = _calibrate(clock, profile, adaptive, rng)
        loop = GazeLoop(tracker, model, config.gestures)
        cursor = FakeCursor()
        recorder = SessionRecorder(record, profile, TICK_HZ) if record is not None else None
        writer = TraceWriter.new(
            {**run_header(profile, tracker, loop, screen_px=list(SIM_SCREEN), simulated=True,
                       calibration_points=n_points),
             "needs_head": False},  # the IMU isn't simulated, so no head pose is expected
            recorder.dir if recorder is not None else trace_dir,
            name="run" if recorder is not None else "sim",
        )
        say(f"calibrated on {n_points} points; following {fixations} fixations + pursuit")

        def tick(target, pursuit=False):
            user.ask(target, clock.t, pursuit=pursuit)
            clock.t += 1.0 / TICK_HZ
            user.update(clock.t)
            cams.advance(clock.t)
            state = loop.tick(cams, clock.t)
            actual = cursor.position  # read back before moving, as the run window does
            cmd = None
            if state.updated:
                cmd = cursor.norm_to_px(state.gaze_x, state.gaze_y)
                cursor.move(*cmd)
            cur = CursorTick(True, True, cmd, actual,
                             None if cmd else f"gaze:{state.drop_reason}")
            writer.tick(tick_record(state, clock.t, cursor=cur, now_ns=cams.now_ns(clock.t)))
            if recorder is not None:
                recorder.add(state.frame)

        writer.event(clock.t, "test: fixations")
        for _ in range(fixations):
            target = tuple(float(v) for v in rng.uniform(0.08, 0.92, 2))
            for _ in range(round(fixation_s * TICK_HZ)):
                tick(target)
        writer.event(clock.t, "test: pursuit")
        t0 = clock.t
        for _ in range(round(pursuit_s * TICK_HZ)):
            a = 2 * math.pi * (clock.t - t0) / 4.0
            tick((0.5 + 0.3 * math.cos(a), 0.5 + 0.3 * math.sin(a)), pursuit=True)
        writer.close()
        if recorder is not None:
            from opengaze.calibration.storage import CalibrationStore

            calib = CalibrationStore(recorder.dir).save(
                model, profile, eye_states=tracker.eye_states()
            )
            recorder.close(calib)
            calib.unlink()
        trace = load_trace(writer.path)
        return SimResult(analyze(trace.ticks, trace.header), writer.path, n_points, model)
    finally:
        puppet.release()
        tracker.close()
        cams.close()


# ---------------------------------------------------------------------------
# Session recording (for replay)
# ---------------------------------------------------------------------------

SESSION_DIR = Path.home() / ".config" / "opengaze" / "sessions"


class SessionRecorder:
    """Writes what every camera gave the tracker, tick by tick, plus what replay needs.

    One frame per camera per tick, repeats included, so tick i of the replay sees exactly
    what tick i of the session saw (the trace's ``fresh`` flags say which were repeats).
    Eye video is biometric data: only created when the user asks and recording is allowed.
    """

    def __init__(self, directory: Path, profile: HardwareProfile, tick_hz: float) -> None:
        import cv2

        self._cv2 = cv2
        directory.mkdir(parents=True, exist_ok=True)
        self.dir = directory
        self.profile = profile
        self.tick_hz = tick_hz
        self._writers: dict[str, Any] = {}
        self.frames = 0

    @classmethod
    def new(cls, profile: HardwareProfile, tick_hz: float, root: Path | None = None):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return cls((root or SESSION_DIR) / f"{profile.id}-{stamp}", profile, tick_hz)

    def _writer(self, cid: str, image: np.ndarray):
        if cid not in self._writers:
            cv2 = self._cv2
            h, w = image.shape[:2]
            path = str(self.dir / f"{cid}.avi")
            for codec in ("FFV1", "MJPG"):  # lossless if this OpenCV build has it
                vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), self.tick_hz, (w, h))
                if vw.isOpened():
                    break
            self._writers[cid] = vw
        return self._writers[cid]

    def add(self, frame: TrackerFrame) -> None:
        cv2 = self._cv2
        for cid, f in frame.camera_frames.items():
            image = f.image
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif f.pixel_format.value == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self._writer(cid, image).write(np.ascontiguousarray(image))
        self.frames += 1

    def close(self, calibration: Path | None = None) -> Path:
        for vw in self._writers.values():
            vw.release()
        data = self.profile.model_dump(mode="json", exclude_none=True)
        data["id"] = f"{self.profile.id}-replay"
        data.pop("imu", None)  # IMU readings aren't recorded; replay uses what the trace has
        cams = {}
        for cid, cam in data["cameras"].items():
            if cid not in self._writers:
                continue
            for key in ("base_dir", "device_id", "resolution", "fps", "controls"):
                cam.pop(key, None)
            cam["source_type"] = "recorded"
            cam["options"] = {"path": f"{cid}.avi", "fps_override": self.tick_hz}
            cams[cid] = cam
        data["cameras"] = cams
        (self.dir / "profile.yaml").write_text(yaml.safe_dump({"hardware_profile": data}))
        if calibration is not None and calibration.exists():
            shutil.copy(calibration, self.dir / "calibration.npz")
        (self.dir / "session.json").write_text(json.dumps(
            {"source_profile": self.profile.id, "frames": self.frames, "tick_hz": self.tick_hz}
        ))
        return self.dir


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    report: TraceReport
    original: TraceReport | None
    trace_path: Path


def load_session_model(session: Path) -> tuple[ScreenMapper | None, dict]:
    """The calibration a session ran with (no fingerprint check: same rig by construction)."""
    path = session / "calibration.npz"
    if not path.exists():
        return None, {}
    bundle = load_bundle_file(path)
    return bundle.model, bundle.eye_states


def replay(session: Path, config: AppConfig, trace_dir: Path | None = None) -> ReplayResult:
    from opengaze.config.loader import check_hardware_profile

    report = check_hardware_profile(session / "profile.yaml")
    if not report.ok or report.profile is None:
        raise ValueError("session profile is invalid:\n" + "\n".join(map(str, report.errors)))
    profile = report.profile
    originals = sorted(session.glob("run-*.trace.jsonl"))
    original = load_trace(originals[0]) if originals else None

    model, eye_states = load_session_model(session)
    cams = SteppedCameras(profile)
    tracker = GazeTracker(profile, config)
    tracker.load_eye_states(eye_states)
    loop = GazeLoop(tracker, model, config.gestures)
    header = run_header(profile, tracker, loop, replay_of=str(session))
    if original:
        for key in ("screen_px", "min_confidence", "needs_head", "simulated"):
            if key in original.header:
                header[key] = original.header[key]
        loop.min_confidence = header["min_confidence"]
    writer = TraceWriter.new(header, trace_dir or session, name="replay")
    try:
        ticks = original.ticks if original else []
        n = len(ticks) or 10**9
        for i in range(n):
            orig = ticks[i] if ticks else None
            for cid in cams.sources:
                fresh = orig is None or orig["cams"].get(cid, {}).get("fresh", True)
                if fresh or cams.latest(cid) is None:
                    cams.read_one(cid)
            if len(cams.ended) == len(cams.sources):
                break
            t = orig["t"] if orig else i / TICK_HZ
            state = loop.tick(cams, t)
            rec = tick_record(state, t)
            if orig is not None:
                # Frame ages are about the live capture, not the replay; keep the originals.
                for cid, cam in rec["cams"].items():
                    for key in ("age_ms", "lat_ms"):
                        if key in orig["cams"].get(cid, {}):
                            cam[key] = orig["cams"][cid][key]
                        else:
                            cam.pop(key, None)
                if "truth" in orig:
                    rec["truth"] = orig["truth"]
            writer.tick(rec)
    finally:
        writer.close()
        tracker.close()
        cams.close()
    new = load_trace(writer.path)
    return ReplayResult(
        analyze(new.ticks, new.header),
        analyze(original.ticks, original.header) if original else None,
        writer.path,
    )
