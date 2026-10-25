"""Synthetic camera: rendered eye images with known ground truth, plus fault injection.

Lets every layer above the camera be developed and tested before hardware exists. Frames are
labeled ``SourceKind.SYNTHETIC`` and carry ``ground_truth``, so they can never be passed off as
live eye tracking.

The eye is rendered upright (the canonical orientation), then transformed by the simulated
physical mount (``mount_rotation_degrees`` / ``mount_mirror_horizontal``). A profile whose
rotation/mirror settings undo that mount produces normalized frames that match the canonical
render, and the ground-truth pupil coordinates refer to that canonical image.
"""

from __future__ import annotations

import asyncio
import math
import time
import zlib
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from opengaze.cameras.base import CameraSource, ConfigIssue
from opengaze.cameras.types import (
    CameraCapabilities,
    CameraConnectionError,
    CameraDisconnectedError,
    CameraFrame,
    CameraRole,
    CameraState,
    ControlSupport,
    FrameStatus,
    Orientation,
    PixelFormat,
    SettingResult,
    SourceKind,
    TimestampSource,
)
from opengaze.config.schema import CameraConfig
from opengaze.imaging import apply_orientation


class SyntheticOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern: Literal["eye", "test_card"] = "eye"
    # puppet: look wherever opengaze.debug.puppet says (calibration dots, the simulator);
    # saccades while nothing is set.
    motion: Literal["lissajous", "saccades", "static", "puppet"] = "lissajous"
    # Shared by both eyes of a rig so they look at the same target; noise is per camera.
    motion_seed: int = 0
    noise_seed: int | None = None
    noise_std: float = Field(default=4.0, ge=0.0)
    illumination_gradient: float = Field(default=0.25, ge=0.0, le=1.0)
    pupil_radius: float = Field(default=0.08, gt=0.0, lt=0.5, description="fraction of short side")
    # planar: the pupil slides as a circle (cheap, Phase 1 behaviour).
    # spherical: the pupil sits on a rotating eyeball and foreshortens into an ellipse, which a
    # 3D eye model needs. Ground truth then includes the gaze vector and eyeball geometry.
    eye_model: Literal["planar", "spherical"] = "planar"
    blink_interval_s: float | None = Field(default=4.0, gt=0.0)
    blink_duration_s: float = Field(default=0.15, gt=0.0)
    saccade_interval_s: float = Field(default=1.2, gt=0.0)
    mount_rotation_degrees: Literal[0, 90, 180, 270] = 0
    mount_mirror_horizontal: bool = False
    # false = render as fast as possible on a virtual clock (tests, offline rendering).
    realtime: bool = True
    # Fault injection.
    drop_probability: float = Field(default=0.0, ge=0.0, lt=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    jitter_ms: float = Field(default=0.0, ge=0.0)
    corrupt_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    disconnect_after_frames: int | None = Field(default=None, ge=1)
    fail_connect_attempts: int = Field(default=0, ge=0)


DEFAULT_RESOLUTION = {"eye": (320, 240), "test_card": (640, 360)}
DEFAULT_FPS = 30.0
SACCADE_TRANSITION_S = 0.05


class GazeTrajectory:
    """Deterministic gaze target in normalized screen coordinates as a function of time."""

    def __init__(self, motion: str, seed: int, saccade_interval_s: float) -> None:
        self.motion = motion
        self.seed = seed
        self.saccade_interval_s = saccade_interval_s
        phases = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi, 2)
        self._phase_x, self._phase_y = float(phases[0]), float(phases[1])

    def _fixation(self, k: int) -> tuple[float, float]:
        if k < 0:
            return 0.5, 0.5
        x, y = np.random.default_rng([self.seed, k]).uniform(0.05, 0.95, 2)
        return float(x), float(y)

    def __call__(self, t: float) -> tuple[float, float]:
        if self.motion == "puppet":
            from opengaze.debug import puppet

            target = puppet.target()
            if target is not None:
                return target
        if self.motion == "static":
            return 0.5, 0.5
        if self.motion == "lissajous":
            return (
                0.5 + 0.4 * math.sin(2.0 * math.pi * 0.13 * t + self._phase_x),
                0.5 + 0.4 * math.sin(2.0 * math.pi * 0.17 * t + self._phase_y),
            )
        k = int(t // self.saccade_interval_s)
        x1, y1 = self._fixation(k)
        alpha = min(1.0, (t - k * self.saccade_interval_s) / SACCADE_TRANSITION_S)
        if alpha >= 1.0:
            return x1, y1
        x0, y0 = self._fixation(k - 1)
        return x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)


def eyelid_closure(t: float, interval_s: float | None, duration_s: float) -> float:
    """0 = open, 1 = fully closed. Blinks start half an interval in, then every interval."""
    if interval_s is None:
        return 0.0
    since_first = t - 0.5 * interval_s
    if since_first < 0.0:
        return 0.0
    phase = (since_first % interval_s) / duration_s
    if phase >= 1.0:
        return 0.0
    return 1.0 - abs(2.0 * phase - 1.0)


def skin_background(width: int, height: int, gradient: float) -> np.ndarray:
    ramp = 150.0 * (1.0 - gradient / 2.0 + gradient * np.linspace(0.0, 1.0, width))
    return np.tile(ramp.astype(np.uint8), (height, 1))


# Spherical eye: eyeball radius as a fraction of image width, and the eye rotation that
# looking at each screen edge takes.
SPHERE_RADIUS_FRACTION = 0.32
SPHERE_MAX_YAW_DEG = 28.0
SPHERE_MAX_PITCH_DEG = 22.0


def spherical_gaze_vector(gaze: tuple[float, float]) -> np.ndarray:
    """Unit gaze direction in eye-camera coordinates (x right, y down, z into the scene) for a
    normalised screen target; the eye looks back toward the camera, so z < 0.

    Geometrically consistent with a flat screen centred in front of the eye whose edges are
    SPHERE_MAX_YAW_DEG / SPHERE_MAX_PITCH_DEG away (so a ray-plane model can fit it exactly).
    """
    tx = 2.0 * (gaze[0] - 0.5) * math.tan(math.radians(SPHERE_MAX_YAW_DEG))
    ty = 2.0 * (gaze[1] - 0.5) * math.tan(math.radians(SPHERE_MAX_PITCH_DEG))
    v = np.array([tx, ty, -1.0])
    return v / np.linalg.norm(v)


def render_eye(
    background: np.ndarray,
    gaze: tuple[float, float],
    closure: float,
    pupil_radius_fraction: float,
    spherical: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render an upright close-up eye (grayscale) and return it with its ground truth."""
    height, width = background.shape
    image = background.copy()
    cx, cy = width / 2.0, height / 2.0
    ax, ay = 0.40 * width, 0.30 * height
    short = min(width, height)
    iris_r = 0.20 * short
    pupil_r = pupil_radius_fraction * short

    mask = np.zeros_like(background)
    cv2.ellipse(mask, (round(cx), round(cy)), (round(ax), round(ay)), 0, 0, 360, 255, -1)
    eye = np.full_like(background, 215)
    extra: dict[str, float] = {}
    if spherical:
        # Orthographic view of a pupil disc on a sphere: it moves by R·(gx, gy) and shrinks
        # by |gz| along the radial direction.
        g = spherical_gaze_vector(gaze)
        radius = SPHERE_RADIUS_FRACTION * width
        px, py = cx + radius * g[0], cy + radius * g[1]
        squash = abs(g[2])
        radial_deg = math.degrees(math.atan2(g[1], g[0]))
        center = (round(px), round(py))
        for r, level in ((iris_r, 95), (pupil_r, 25)):
            axes = (max(1, round(r * squash)), round(r))
            cv2.ellipse(eye, center, axes, radial_deg, 0, 360, level, -1, cv2.LINE_AA)
        extra = {
            "gaze_vec_x": float(g[0]),
            "gaze_vec_y": float(g[1]),
            "gaze_vec_z": float(g[2]),
            "eye_center_x": cx,
            "eye_center_y": cy,
            "eye_radius": radius,
        }
    else:
        # Mildly nonlinear and cross-coupled, like the projection of a real eyeball, so a
        # calibration model has something to learn beyond a linear map.
        u, v = 2.0 * (gaze[0] - 0.5), 2.0 * (gaze[1] - 0.5)
        px = cx + 0.30 * width * u * (1.0 - 0.15 * u * u)
        py = cy + 0.20 * height * v * (1.0 - 0.15 * v * v) + 0.03 * height * u * u
        cv2.circle(eye, (round(px), round(py)), round(iris_r), 95, -1, cv2.LINE_AA)
        cv2.circle(eye, (round(px), round(py)), round(pupil_r), 25, -1, cv2.LINE_AA)
    # Corneal reflection of an illuminator: moves less than the pupil does.
    gx = cx + 0.3 * (px - cx) + 0.15 * iris_r
    gy = cy + 0.3 * (py - cy) - 0.20 * iris_r
    cv2.circle(eye, (round(gx), round(gy)), max(2, round(0.012 * short)), 250, -1, cv2.LINE_AA)
    np.copyto(image, eye, where=mask > 0)

    # Upper eyelid: always covers the top of the opening a little, all of it at full closure.
    top = cy - ay
    lid_y = top + (0.12 + 0.88 * closure) * 2.0 * ay
    lid_row = int(round(lid_y))
    if lid_row > 0:
        image[: min(lid_row, height)] = background[: min(lid_row, height)]
        dy = (lid_y - cy) / ay
        if abs(dy) < 1.0:
            half = ax * math.sqrt(1.0 - dy * dy)
            cv2.line(image, (round(cx - half), lid_row), (round(cx + half), lid_row), 60, 2)

    truth = {
        "gaze_x": gaze[0],
        "gaze_y": gaze[1],
        "pupil_x": px,
        "pupil_y": py,
        "pupil_radius": pupil_r,
        "eyelid_closure": closure,
        "pupil_visible": float(py - 0.3 * pupil_r > lid_y and closure < 0.6),
        **extra,
    }
    return image, truth


def render_test_card(width: int, height: int, t: float, sequence: int) -> tuple[np.ndarray, dict]:
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = (x * 0.6 + 40).astype(np.uint8)
    image[..., 1] = (y * 0.6 + 40).astype(np.uint8)
    image[..., 2] = 90
    bar_x = (t * 0.25 % 1.0) * width
    cv2.rectangle(
        image, (int(bar_x), 0), (int(bar_x) + max(4, width // 40), height), (255,) * 3, -1
    )
    scale = height / 360.0
    for i, text in enumerate(("SYNTHETIC", f"frame {sequence}", f"t = {t:7.2f} s")):
        org = (int(20 * scale), int((50 + 45 * i) * scale))
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale, (0, 0, 0), 4)
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale, (255, 255, 255), 2)
    return image, {"bar_x": bar_x}


class SyntheticCameraSource(CameraSource):
    source_type = "synthetic"
    source_kind = SourceKind.SYNTHETIC
    Options = SyntheticOptions
    non_geometric_options = frozenset(
        {
            "realtime",
            "drop_probability",
            "latency_ms",
            "jitter_ms",
            "corrupt_probability",
            "disconnect_after_frames",
            "fail_connect_attempts",
        }
    )

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        opts: SyntheticOptions = self.options
        self.width, self.height = config.resolution or DEFAULT_RESOLUTION[opts.pattern]
        self.fps = config.fps or DEFAULT_FPS
        self._period_ns = 1e9 / self.fps
        self._mount = Orientation(opts.mount_rotation_degrees, opts.mount_mirror_horizontal)
        noise_seed = opts.noise_seed
        if noise_seed is None:
            noise_seed = zlib.crc32(config.camera_id.encode())
        self._noise_rng = np.random.default_rng(noise_seed)
        self._fault_rng = np.random.default_rng([noise_seed, 1])
        self._trajectory = GazeTrajectory(opts.motion, opts.motion_seed, opts.saccade_interval_s)
        self._background: np.ndarray | None = None
        self._connected = False
        self._connect_attempts = 0
        self._index = 0
        self._start_ns = 0
        self._emitted_since_connect = 0

    @classmethod
    def check_config(cls, config: CameraConfig) -> list[ConfigIssue]:
        issues = super().check_config(config)
        if not issues:
            issues.append(
                ConfigIssue("info", f"{config.camera_id}: SYNTHETIC source — not a real camera")
            )
            pattern = config.options.get("pattern", "eye")
            eye_role = config.role in (CameraRole.LEFT_EYE, CameraRole.RIGHT_EYE)
            if eye_role and pattern != "eye":
                issues.append(
                    ConfigIssue("warning", f"{config.camera_id}: eye camera with non-eye pattern")
                )
        return issues

    async def connect(self) -> None:
        if self._connected:
            return
        self._set_state(CameraState.CONNECTING)
        self._connect_attempts += 1
        opts: SyntheticOptions = self.options
        if self._connect_attempts <= opts.fail_connect_attempts:
            message = (
                f"injected connect failure ({self._connect_attempts}/{opts.fail_connect_attempts})"
            )
            self._set_state(CameraState.IDLE, message)
            raise CameraConnectionError(message)
        if opts.pattern == "eye":
            self._background = skin_background(self.width, self.height, opts.illumination_gradient)
        # Resume the virtual timeline where it left off, anchored to "now".
        self._start_ns = time.monotonic_ns() - int(self._index * self._period_ns)
        self._emitted_since_connect = 0
        self._connected = True
        self._reset_timing()
        self._set_state(CameraState.STREAMING)

    async def disconnect(self) -> None:
        self._connected = False
        if self._state is not CameraState.FAILED:
            self._set_state(CameraState.STOPPED)

    def get_capabilities(self) -> CameraCapabilities:
        results = [
            SettingResult(
                "resolution",
                self.config.resolution,
                (self.width, self.height),
                ControlSupport.SUPPORTED,
            ),
            SettingResult("fps", self.config.fps, self.fps, ControlSupport.SUPPORTED),
        ]
        for name, value in self.config.controls.requested().items():
            results.append(
                SettingResult(
                    name,
                    value,
                    None,
                    ControlSupport.UNSUPPORTED,
                    "synthetic source has no sensor controls",
                )
            )
        return CameraCapabilities(
            source_type=self.source_type,
            source_kind=self.source_kind,
            backend="synthetic",
            device_name=f"synthetic {self.options.pattern}",
            resolution=(self.width, self.height),
            supported_resolutions=None,
            reported_fps=self.fps,
            pixel_formats=(PixelFormat.BGR8,),
            controls={r.name: r.support for r in results},
            provides_device_timestamps=True,
            intrinsics=self._intrinsics,
            distortion=self._distortion,
            setting_results=tuple(results),
            notes=("SYNTHETIC: generated images, not a camera",),
        )

    async def read_frame(self) -> CameraFrame:
        if not self._connected:
            raise CameraDisconnectedError("not connected")
        opts: SyntheticOptions = self.options
        while True:
            index = self._index
            self._index += 1
            capture_ns = self._start_ns + int(index * self._period_ns)
            delay_ns = int(opts.latency_ms * 1e6)
            if opts.jitter_ms > 0:
                delay_ns += int(abs(self._fault_rng.normal(0.0, opts.jitter_ms)) * 1e6)
            if opts.drop_probability and self._fault_rng.random() < opts.drop_probability:
                continue  # this frame is "lost in transit": a gap in the sequence numbers
            break

        if opts.realtime:
            wait_s = (capture_ns + delay_ns - time.monotonic_ns()) / 1e9
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            received_ns = time.monotonic_ns()
        else:
            received_ns = capture_ns + delay_ns  # virtual clock

        if (
            opts.disconnect_after_frames is not None
            and self._emitted_since_connect >= opts.disconnect_after_frames
        ):
            self._connected = False
            self._set_state(CameraState.IDLE, "injected disconnect")
            raise CameraDisconnectedError("injected disconnect")

        t = index / self.fps
        if opts.pattern == "eye":
            assert self._background is not None
            closure = eyelid_closure(t, opts.blink_interval_s, opts.blink_duration_s)
            gray, truth = render_eye(
                self._background,
                self._trajectory(t),
                closure,
                opts.pupil_radius,
                spherical=opts.eye_model == "spherical",
            )
            if opts.noise_std > 0:
                noise = self._noise_rng.standard_normal(gray.shape, dtype=np.float32)
                gray = np.clip(gray + noise * opts.noise_std, 0, 255).astype(np.uint8)
            canonical = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            canonical, truth = render_test_card(self.width, self.height, t, index)

        image = apply_orientation(canonical, self._mount)
        status, warnings = FrameStatus.OK, ()
        if opts.corrupt_probability and self._fault_rng.random() < opts.corrupt_probability:
            image = image.copy()
            rows = image.shape[0]
            y0 = int(self._fault_rng.integers(0, rows))
            y1 = min(rows, y0 + max(1, rows // 6))
            image[y0:y1] = self._fault_rng.integers(0, 256, image[y0:y1].shape, dtype=np.uint8)
            status, warnings = FrameStatus.PARTIAL, ("injected corruption",)

        self._emitted_since_connect += 1
        return self._make_frame(
            image,
            host_receive_ns=received_ns,
            capture_ns=capture_ns,
            timestamp_source=TimestampSource.SYNTHETIC,
            timestamp_uncertainty_ns=0,
            sequence=index,
            status=status,
            warnings=warnings,
            ground_truth=truth,
        )
