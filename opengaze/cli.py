"""Command-line entry point: ``opengaze <command>``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from opengaze.config.fingerprint import profile_fingerprint
from opengaze.config.loader import ProfileReport, check_hardware_profile, load_app_config
from opengaze.config.schema import AppConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

log = logging.getLogger("opengaze")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _with_cloud_flag(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """``--cloud`` opts into cloud features for this session only."""
    if not getattr(args, "cloud", False):
        return config
    privacy = config.privacy.model_copy(update={"cloud_features_enabled": True})
    return config.model_copy(update={"privacy": privacy})


def _load(args: argparse.Namespace) -> tuple[AppConfig, ProfileReport]:
    config = _with_cloud_flag(load_app_config(args.config), args)
    configure_logging(args.log_level or config.app.log_level)
    profile_path = Path(args.profile) if args.profile else config.app.hardware_profile
    report = check_hardware_profile(profile_path, [Path(p) for p in args.overlay])
    return config, report


def _print_report(report: ProfileReport) -> None:
    print(f"profile:     {report.path}")
    if report.profile is not None:
        p = report.profile
        print(f"id:          {p.id} (version {p.version}, frame design '{p.frame_design}')")
        print(f"mode:        {report.mode.label if report.mode else '—'}")
        if report.mode is not None and report.mode.value != "A":
            g = p.gaze
            stages = g.detector_2d.type
            if g.pipeline == "eye_3d":
                stages += f" → {g.eye_model_3d.type}"
            mapping = p.calibration.geometry_estimation
            print(f"gaze:        {g.pipeline} ({stages}), {mapping} mapping")
        print(f"fingerprint: {profile_fingerprint(p)}")
        for cid, cam in p.cameras.items():
            state = "" if cam.enabled else "  [disabled]"
            print(f"  {cid:10} role={cam.role.value:9} source={cam.source_type}{state}")
    for issue in report.issues:
        print(f"  {issue}")
    print("OK" if report.ok else f"INVALID: {len(report.errors)} error(s)")


def ensure_camera_permission(report: ProfileReport) -> None:
    """Ask for macOS camera access on the main thread if any live camera is configured."""
    from opengaze.cameras.permissions import GUIDANCE, request_camera_permission
    from opengaze.cameras.registry import get_source_class
    from opengaze.cameras.types import SourceKind

    assert report.profile is not None
    cameras = report.profile.enabled_cameras.values()
    if any(get_source_class(c.source_type).source_kind is SourceKind.LIVE for c in cameras):
        status = request_camera_permission()
        if status in GUIDANCE:
            print(f"camera permission: {status.value}. {GUIDANCE[status]}")


def cmd_validate(args: argparse.Namespace) -> int:
    configure_logging(args.log_level or "WARNING")
    report = check_hardware_profile(Path(args.profile_path), [Path(p) for p in args.overlay])
    _print_report(report)
    return 0 if report.ok else 1


def cmd_diagnostics(args: argparse.Namespace) -> int:
    config, report = _load(args)
    if not report.ok:
        _print_report(report)
        return 1
    try:
        from opengaze.desktop.diagnostics import run_diagnostics
    except ImportError as exc:
        print(f"desktop UI unavailable ({exc}); install with: pip install -e '.[desktop]'")
        return 1
    ensure_camera_permission(report)
    return run_diagnostics(report, config)


def cmd_headless(args: argparse.Namespace) -> int:
    """Run capture without a UI and print camera health — handy over SSH or in CI."""
    from opengaze.cameras.manager import CameraManager

    config, report = _load(args)
    if not report.ok:
        _print_report(report)
        return 1
    assert report.profile is not None
    ensure_camera_permission(report)
    manager = CameraManager.from_profile(report.profile, config.capture)

    async def monitor() -> None:
        task = asyncio.create_task(manager.run())
        deadline = time.monotonic() + args.seconds
        header = (
            f"{'camera':10} {'kind':9} {'state':12} {'fps':>6} {'age ms':>7} {'frames':>7} "
            f"{'skip':>5} {'over':>5} {'part':>5} {'recon':>5}  last error"
        )
        while time.monotonic() < deadline:
            await asyncio.sleep(args.interval)
            print(header)
            for h in manager.health().values():
                fps = f"{h.measured_fps:6.1f}" if h.measured_fps else "     —"
                age = (
                    f"{h.last_frame_age_ms:7.0f}" if h.last_frame_age_ms is not None else "      —"
                )
                print(
                    f"{h.camera_id:10} {h.source_kind.value:9} {h.state.value:12} {fps} {age} "
                    f"{h.frames_received:7} {h.frames_skipped_by_source:5} "
                    f"{h.frames_overwritten:5} {h.partial_frames:5} {h.reconnects:5}  "
                    f"{h.last_error or ''}"
                )
            print()
        await manager.stop()
        await task

    asyncio.run(monitor())
    return 0


def cmd_probe_cameras(args: argparse.Namespace) -> int:
    """Open OpenCV device indices one by one and report what each delivers."""
    import cv2

    from opengaze.cameras.opencv_source import default_api_preference
    from opengaze.cameras.permissions import GUIDANCE, request_camera_permission

    configure_logging(args.log_level or "WARNING")
    status = request_camera_permission()
    if status in GUIDANCE:
        print(f"camera permission: {status.value}. {GUIDANCE[status]}")
        return 1
    print("Probing camera indices (each camera's light may turn on briefly)…")
    found = 0
    for index in range(args.max_index + 1):
        cap = cv2.VideoCapture(index, default_api_preference())
        if not cap.isOpened():
            cap.release()
            continue
        ok, image = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        backend = cap.getBackendName()
        cap.release()
        if not ok or image is None:
            print(f"  index {index}: opened but delivered no frame ({backend})")
            continue
        found += 1
        fps_text = f"{fps:.0f} fps reported" if fps > 0 else "fps not reported"
        print(f"  index {index}: {image.shape[1]}x{image.shape[0]}, {fps_text} ({backend})")
    if not found:
        print("No cameras delivered frames. On macOS, check camera permission for this terminal.")
    print("Set device_id in your hardware profile to the index you want.")
    print("OpenCV can't report camera names; confirm the assignment in `opengaze diagnostics`.")
    return 0


def cmd_download_models(args: argparse.Namespace) -> int:
    """Download the MediaPipe FaceLandmarker model to the local cache."""
    from opengaze.vision.mediapipe_detector import _MODEL_FILENAME, default_model_path, ensure_model

    configure_logging(args.log_level or "INFO")
    path = Path(args.output) if args.output else default_model_path()
    if path.exists() and not args.force:
        print(f"already present: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
        return 0
    print(f"downloading {_MODEL_FILENAME} …")
    try:
        ensure_model(path)
        print(f"saved to {path}  ({path.stat().st_size / 1e6:.1f} MB)")
        return 0
    except Exception as exc:
        print(f"download failed: {exc}")
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Open the gaze test environment (auto-calibrates if needed)."""
    config, report = _load(args)
    if not report.ok:
        _print_report(report)
        return 1
    try:
        from opengaze.desktop.run_ui import run_gaze_test
    except ImportError as exc:
        print(f"desktop UI unavailable ({exc}); install with: pip install -e '.[desktop]'")
        return 1
    if args.record_video and not config.privacy.allow_recording:
        print("--record-video saves eye-camera video (biometric data). Set "
              "privacy.allow_recording: true in the app config to allow it.")
        return 1
    ensure_camera_permission(report)
    return run_gaze_test(
        report, config, adaptive=args.adaptive, trace=args.trace, record_video=args.record_video
    )


def cmd_voice_test(args: argparse.Namespace) -> int:
    """Check Cartesia TTS and STT end to end, with no camera or calibration needed."""
    from opengaze.voice.cartesia import CartesiaConfig
    from opengaze.voice.service import VoiceService, VoiceState

    config = _with_cloud_flag(load_app_config(args.config), args)
    configure_logging(args.log_level or "WARNING")
    cartesia = CartesiaConfig.from_env()
    if cartesia is None:
        print("CARTESIA_API_KEY not set. Add it to .env (see .env.example).")
        return 1
    if not config.privacy.cloud_features_enabled:
        print("Cloud features are off. Re-run with --cloud, or set "
              "privacy.cloud_features_enabled: true in configs/default.yaml.")
        return 1
    print(f"key: ...{cartesia.api_key[-4:]}  voice: {cartesia.voice_id}  "
          f"language: {cartesia.language}")

    svc = VoiceService(cloud_enabled=True, cartesia=cartesia)

    def wait_idle(timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            transcripts.extend(svc.poll())
            if svc.state is VoiceState.IDLE:
                return True
            time.sleep(0.05)
        return False

    transcripts: list[str] = []
    print("\n[1/2] Text-to-speech (Sonic): you should hear a sentence…")
    svc.speak(args.text)
    wait_idle(30)
    print(f"      {svc.status}")
    if "unavailable" in svc.status or "failed" in svc.status:
        return 1

    if args.tts_only:
        return 0
    print("\n[2/2] Speech-to-text (Ink-Whisper): say something now; "
          "it stops after a 1.5 s pause…")
    svc.start_dictation()
    if svc.state is not VoiceState.LISTENING:
        print(f"      {svc.status}")
        return 1
    while svc.state is VoiceState.LISTENING:
        bar = "#" * int(svc.mic_level * 30)
        print(f"\r      mic [{bar:<30}]", end="", flush=True)
        transcripts.extend(svc.poll())
        time.sleep(0.05)
    print("\r      transcribing…" + " " * 30)
    wait_idle(30)
    print(f"      {svc.status}")
    if not transcripts:
        return 1
    heard = " ".join(transcripts)
    print(f'\n      heard: "{heard}"\n      speaking it back…')
    svc.speak(heard)
    wait_idle(30)
    return 0


def cmd_dictate(args: argparse.Namespace) -> int:
    """System-wide voice bar: dictate into any app, read selections aloud."""
    from opengaze.voice.cartesia import CartesiaConfig
    from opengaze.voice.service import VoiceService

    config = _with_cloud_flag(load_app_config(args.config), args)
    configure_logging(args.log_level or "WARNING")
    if sys.platform != "darwin":
        print("system-wide dictation is macOS-only")
        return 1
    try:
        from opengaze.desktop.dictation_ui import run_dictation_bar
    except ImportError as exc:
        print(f"desktop UI unavailable ({exc}); install with: pip install -e '.[desktop]'")
        return 1
    from opengaze.control.cursor import is_accessibility_granted

    voice = VoiceService(
        cloud_enabled=config.privacy.cloud_features_enabled,
        cartesia=CartesiaConfig.from_env(),
    )
    print(voice.status)
    if not voice.can_dictate:
        print("Dictation is off; run with --cloud and set CARTESIA_API_KEY in .env.")
    if not is_accessibility_granted():
        print("Accessibility permission is missing for this terminal app, so hotkeys and\n"
              "typing into other apps won't work (transcripts go to the clipboard instead).\n"
              "Grant it in System Settings › Privacy & Security › Accessibility, then rerun.")
    print("Voice bar is up. ⌃⌥D = dictate, ⌃⌥S = read selection, Ctrl+C here to quit.")
    return run_dictation_bar(voice)


def cmd_desktop(args: argparse.Namespace) -> int:
    """System-wide gaze cursor + voice bar over every app."""
    from opengaze.voice.cartesia import CartesiaConfig
    from opengaze.voice.service import VoiceService

    config, report = _load(args)
    if not report.ok:
        _print_report(report)
        return 1
    if sys.platform != "darwin":
        print("system-wide gaze control is macOS-only")
        return 1
    try:
        from opengaze.desktop.desktop_mode import run_desktop_mode
    except ImportError as exc:
        print(f"desktop UI unavailable ({exc}); install with: pip install -e '.[all]'")
        return 1
    from opengaze.control.cursor import is_accessibility_granted, open_accessibility_prefs

    if not is_accessibility_granted():
        print("Accessibility permission is required to move the cursor in other apps.\n"
              "Grant it to this terminal app in System Settings › Privacy & Security ›\n"
              "Accessibility (opening it now), then rerun.")
        open_accessibility_prefs()
        return 1
    ensure_camera_permission(report)
    voice = VoiceService(
        cloud_enabled=config.privacy.cloud_features_enabled,
        cartesia=CartesiaConfig.from_env(),
    )
    print(voice.status)
    print("Gaze cursor is ON across all apps. Long blink = click.\n"
          "⌃⌥G pause/resume gaze · ⌃⌥D dictate · ⌃⌥S read selection · "
          "moving the mouse takes over for 2 s · Ctrl+C here to quit.")
    return run_desktop_mode(report, config, voice)


def cmd_calibrate(args: argparse.Namespace) -> int:
    config, report = _load(args)
    if not report.ok:
        _print_report(report)
        return 1
    try:
        if args.adaptive:
            from opengaze.desktop.adaptive_calibration_ui import (
                run_adaptive_calibration as run_calibration,
            )
        else:
            from opengaze.desktop.calibration_ui import run_calibration
    except ImportError as exc:
        print(f"desktop UI unavailable ({exc}); install with: pip install -e '.[desktop]'")
        return 1
    ensure_camera_permission(report)
    return run_calibration(report, config)


def cmd_calibration_log(args: argparse.Namespace) -> int:
    """Summarise an adaptive calibration log and replay how many points it needed."""
    from opengaze.calibration.adaptive import AdaptiveLog, list_logs, points_needed_curve

    configure_logging(args.log_level or "WARNING")
    if args.path:
        path = Path(args.path)
    else:
        logs = list_logs()
        if not logs:
            print("No adaptive calibration logs yet — run `opengaze calibrate --adaptive`.")
            return 1
        path = logs[-1]
    run = AdaptiveLog.load(path)
    print(f"log:      {path}")
    print(f"schema:   {run.schema_id} ({run.geometry_estimation}), settings {run.settings}")
    print(f"summary:  {run.summary()}")
    print()
    print(f"{'#':>3} {'target':>13} {'phase':5} {'action':6} {'error':>7} {'px':>6} "
          f"{'samples':>7} {'conf':>5} {'train':>5}")
    for r in run.records:
        err = f"{r.error_norm:7.3f}" if r.error_norm is not None else "      —"
        px = f"{r.error_px:6.0f}" if r.error_px is not None else "     —"
        print(f"{r.index:3} ({r.target_x:.2f}, {r.target_y:.2f}) {r.phase:5} {r.action:6} "
              f"{err} {px} {r.n_samples:7} {r.mean_confidence:5.2f} {r.n_training_after:5}")
    curve = points_needed_curve(run)
    if curve:
        print()
        print("Replay: fit on the first k points shown, test on the rest")
        print(f"{'k':>3} {'mean':>7} {'max':>7} {'mean px':>8} {'tested':>6}")
        for c in curve:
            px = f"{c.mean_error_px:8.0f}" if c.mean_error_px is not None else "       —"
            print(f"{c.n_training:3} {c.mean_error_norm:7.3f} {c.max_error_norm:7.3f} "
                  f"{px} {c.n_eval:6}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Calibrate a simulated user and drive the cursor headlessly; report every stage."""
    from opengaze.debug.harness import SimulatedUser, simulate

    if not args.profile:
        args.profile = str(REPO_ROOT / "configs" / "hardware" / "synthetic-headset.yaml")
    config, report = _load(args)
    if not report.ok:
        _print_report(report)
        return 1
    assert report.profile is not None
    user = SimulatedUser(reaction_s=args.reaction, jitter_norm=args.jitter, seed=args.seed)
    result = simulate(
        report.profile, config, adaptive=args.adaptive, fixations=args.fixations,
        user=user, seed=args.seed, progress=lambda m: print(f"… {m}"),
        record=Path(args.record).expanduser() if args.record else None,
    )
    print()
    print(result.report.format())
    print(f"\ntrace: {result.trace_path}")
    if args.record:
        print(f"session: {args.record}  (opengaze replay {args.record})")
    return 1 if any(s.status == "fail" for s in result.report.stages) else 0


def _latest_trace() -> Path | None:
    from opengaze.debug.harness import SESSION_DIR
    from opengaze.debug.trace import list_traces

    found = list_traces() + list_traces(SESSION_DIR)
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def cmd_trace_report(args: argparse.Namespace) -> int:
    """Stage-by-stage report of a trace: where does the path to the cursor break?"""
    from opengaze.debug.analyze import analyze
    from opengaze.debug.trace import load_trace

    configure_logging(args.log_level or "WARNING")
    path = Path(args.path) if args.path else _latest_trace()
    if path is None:
        print("No traces yet — record one with `opengaze run --trace` (or T in the run window).")
        return 1
    if path.is_dir():  # a recorded session
        runs = sorted(path.glob("run-*.trace.jsonl"))
        if not runs:
            print(f"no run trace in {path}")
            return 1
        path = runs[0]
    trace = load_trace(path)
    h = trace.header
    print(f"trace:   {path}")
    print(f"profile: {h.get('profile_id')} ({h.get('schema')}, {h.get('estimation')})"
          f"{'  SIMULATED' if h.get('simulated') else ''}")
    if trace.events:
        t0 = trace.ticks[0]["t"] if trace.ticks else 0.0
        print("events:  " + ", ".join(f"{e['t'] - t0:.1f}s {e['msg']}" for e in trace.events))
    print()
    if not trace.ticks:
        print("No ticks recorded.")
        return 1
    print(analyze(trace.ticks, h).format())
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Re-run a recorded session through the current code and compare with the original."""
    from opengaze.debug.harness import replay

    config = load_app_config(args.config)
    configure_logging(args.log_level or "WARNING")
    session = Path(args.session).expanduser()
    if not (session / "profile.yaml").exists():
        print(f"{session} is not a recorded session (no profile.yaml)")
        return 1
    result = replay(session, config)
    if result.original is not None:
        print("ORIGINAL (live)")
        print(result.original.format())
        print()
    print("REPLAY (current code)")
    print(result.report.format())
    print(f"\ntrace: {result.trace_path}")
    return 0


def cmd_cursor_test(args: argparse.Namespace) -> int:
    """Check Accessibility, the displays, and that posted cursor moves land."""
    from opengaze.debug.cursor_test import run_cursor_test

    configure_logging(args.log_level or "WARNING")
    if not args.no_move:
        print("Moving the cursor to the centre and corners of each display…")
    result = run_cursor_test(move=not args.no_move, say=print if args.verbose else None)
    print(result.format())
    return 0 if result.ok else 1


def cmd_calibration_info(args: argparse.Namespace) -> int:
    """Print info about the stored calibration for the current hardware profile."""
    from opengaze.calibration.storage import CalibrationStore
    from opengaze.modes import profile_schema

    config, report = _load(args)
    if not report.ok:
        _print_report(report)
        return 1
    assert report.profile is not None
    schema_id = profile_schema(report.profile)
    store = CalibrationStore()
    compat = store.compatibility(schema_id, report.profile)  # type: ignore[arg-type]
    if compat is None:
        print(f"No calibration found for schema={schema_id!r} with the current hardware profile.")
        return 1
    print(compat.explain())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opengaze", description=__doc__)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_profile_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="app config YAML")
        p.add_argument("--profile", help="hardware profile YAML (overrides the config's)")
        p.add_argument(
            "--overlay",
            action="append",
            default=[],
            help="per-user overlay YAML merged over the profile (repeatable)",
        )

    p = sub.add_parser("run", help="open gaze test environment (auto-calibrates if needed)")
    add_profile_args(p)
    p.add_argument("--adaptive", action="store_true", help="auto-calibrate adaptively")
    p.add_argument("--trace", action="store_true",
                   help="record a per-tick pipeline trace (`opengaze trace-report`)")
    p.add_argument("--record-video", action="store_true",
                   help="also save camera video for `opengaze replay` "
                        "(needs privacy.allow_recording)")
    p.add_argument("--cloud", action="store_true",
                   help="opt into Cartesia voice for this session")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("simulate",
                       help="calibrate a simulated user and drive the cursor headlessly")
    add_profile_args(p)
    p.add_argument("--adaptive", action="store_true", help="calibrate adaptively")
    p.add_argument("--fixations", type=int, default=16, help="test fixations after calibrating")
    p.add_argument("--jitter", type=float, default=0.004,
                   help="fixation jitter, fraction of the screen (default 0.004)")
    p.add_argument("--reaction", type=float, default=0.2, help="saccade latency, s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record", metavar="DIR",
                   help="also save the test phase as a session in DIR (try `opengaze replay`)")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("trace-report", help="stage-by-stage report of a pipeline trace")
    p.add_argument("path", nargs="?", help="trace or session folder (default: the most recent)")
    p.set_defaults(func=cmd_trace_report)

    p = sub.add_parser("replay", help="re-run a recorded session through the current code")
    p.add_argument("session", help="session folder from `opengaze run --record-video`")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="app config YAML")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("cursor-test", help="check that the macOS cursor can be driven")
    p.add_argument("--no-move", action="store_true", help="only check permission and displays")
    p.add_argument("-v", "--verbose", action="store_true", help="print every move")
    p.set_defaults(func=cmd_cursor_test)

    p = sub.add_parser("diagnostics", help="live camera previews and health (PySide6)")
    add_profile_args(p)
    p.set_defaults(func=cmd_diagnostics)

    p = sub.add_parser("headless", help="run capture without UI, print health")
    add_profile_args(p)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--interval", type=float, default=1.0)
    p.set_defaults(func=cmd_headless)

    p = sub.add_parser("validate", help="validate a hardware profile")
    p.add_argument("profile_path")
    p.add_argument("--overlay", action="append", default=[])
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("probe-cameras", help="list OpenCV camera indices that deliver frames")
    p.add_argument("--max-index", type=int, default=5)
    p.set_defaults(func=cmd_probe_cameras)

    p = sub.add_parser("download-models", help="download the MediaPipe FaceLandmarker model")
    p.add_argument("--output", help="save to this path instead of the default cache location")
    p.add_argument("--force", action="store_true", help="re-download even if already present")
    p.set_defaults(func=cmd_download_models)

    p = sub.add_parser("voice-test", help="check Cartesia speech + dictation (no camera)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="app config YAML")
    p.add_argument("--cloud", action="store_true",
                   help="opt into Cartesia voice for this session")
    p.add_argument("--tts-only", action="store_true", help="skip the microphone step")
    p.add_argument("--text", default="Hi, I'm using OpenGaze to talk with my eyes.",
                   help="sentence to speak in the TTS step")
    p.set_defaults(func=cmd_voice_test)

    p = sub.add_parser("desktop", help="gaze cursor + voice bar across every app (macOS)")
    add_profile_args(p)
    p.add_argument("--cloud", action="store_true",
                   help="opt into Cartesia voice for this session")
    p.set_defaults(func=cmd_desktop)

    p = sub.add_parser("dictate", help="system-wide voice bar: dictate into any app (Cartesia)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="app config YAML")
    p.add_argument("--cloud", action="store_true",
                   help="opt into Cartesia voice for this session")
    p.set_defaults(func=cmd_dictate)

    p = sub.add_parser("calibrate", help="run the 9-point gaze calibration wizard (PySide6)")
    add_profile_args(p)
    p.add_argument(
        "--adaptive",
        action="store_true",
        help="stream dots that test the model and train it where it's wrong, until accurate",
    )
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser(
        "calibration-log", help="show an adaptive calibration log and its points-needed replay"
    )
    p.add_argument("path", nargs="?", help="log file (default: the most recent)")
    p.set_defaults(func=cmd_calibration_log)

    p = sub.add_parser("calibration-info", help="show stored calibration status for this profile")
    add_profile_args(p)
    p.set_defaults(func=cmd_calibration_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
