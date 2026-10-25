"""Hardware diagnostics window: live previews and health for every camera and the IMU.

Capture runs on a background thread (``CaptureRuntime``) and the IMU on its own
(``ImuRuntime``); this window only polls their newest data on a timer, so a slow or dead
device can never freeze the UI.
"""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from opengaze.cameras.manager import CameraHealth, CameraManager, is_emulated
from opengaze.cameras.runtime import CaptureRuntime
from opengaze.cameras.types import CameraFrame, CameraRole, CameraState, ControlSupport, SourceKind
from opengaze.config.fingerprint import profile_fingerprint
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.debug.metrics import MetricsRecorder, input_statuses
from opengaze.imu.runtime import ImuHealth, ImuRuntime, start_profile_imu
from opengaze.imu.types import GRAVITY
from opengaze.vision.head_pose import angles_from_rotation
from opengaze.vision.observations import EyeObservation, HeadObservation
from opengaze.vision.preprocessing import FrameNormalizer

log = logging.getLogger(__name__)

STATE_COLORS = {
    CameraState.STREAMING: "#2e9e5b",
    CameraState.STALLED: "#d08a00",
    CameraState.CONNECTING: "#3a7bd5",
    CameraState.RECONNECTING: "#3a7bd5",
    CameraState.FAILED: "#d64541",
    CameraState.STOPPED: "#7a7a7a",
    CameraState.IDLE: "#7a7a7a",
}
KIND_COLORS = {
    SourceKind.LIVE: "#2e9e5b",
    SourceKind.RECORDED: "#3a7bd5",
    SourceKind.SYNTHETIC: "#d9822b",
}

STYLE = """
QWidget { background: #15171a; color: #e8e8e8; font-size: 14px; }
QFrame#tile { border: 1px solid #33373d; border-radius: 8px; }
QLabel#title { font-size: 17px; font-weight: 600; }
QLabel#badge { border-radius: 4px; padding: 2px 8px; font-weight: 600; color: white; }
QLabel#banner { padding: 8px 12px; border-radius: 6px; font-weight: 600; }
QLabel#preview { background: #000; border-radius: 4px; }
QLabel#notes { color: #f0b429; }
QLabel#input { border-radius: 4px; padding: 4px 10px; font-weight: 600; color: white; }
"""

LEVEL_COLORS = {"ok": "#2e9e5b", "warn": "#d08a00", "down": "#d64541"}


def to_qimage(image: np.ndarray) -> QImage:
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    fmt = QImage.Format.Format_Grayscale8 if image.ndim == 2 else QImage.Format.Format_BGR888
    return QImage(image.data, w, h, image.strides[0], fmt).copy()


def _fmt(value: float | None, spec: str, unit: str = "") -> str:
    return "—" if value is None else f"{value:{spec}}{unit}"


def _mono(size: int) -> QFont:
    font = QFont("Menlo")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSize(size)
    return font


class TracePlot(QWidget):
    """Rolling plot of three channels (x, y, z) over the last few seconds."""

    COLORS = ("#ff6b6b", "#51cf66", "#4dabf7")

    def __init__(self, label: str, window_s: float = 5.0, min_range: float = 0.5) -> None:
        super().__init__()
        self.label = label
        self.window_s = window_s
        self.min_range = min_range
        self._t: np.ndarray = np.zeros(0)
        self._v: np.ndarray = np.zeros((0, 3))
        self.setMinimumHeight(90)

    def set_data(self, t_s: np.ndarray, values: np.ndarray) -> None:
        self._t, self._v = t_s, values
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#0d0f11"))
        p.setPen(QPen(QColor("#33373d"), 1))
        p.drawLine(0, h // 2, w, h // 2)
        span = self.min_range
        if len(self._t) > 1:
            span = max(span, float(np.abs(self._v).max()) * 1.1)
            t_end = float(self._t[-1])
            xs = (self._t - (t_end - self.window_s)) / self.window_s * w
            for ch, color in enumerate(self.COLORS):
                p.setPen(QPen(QColor(color), 1.5))
                ys = h / 2 - self._v[:, ch] / span * (h / 2 - 4)
                for i in range(1, len(xs)):
                    if xs[i - 1] >= 0:
                        p.drawLine(int(xs[i - 1]), int(ys[i - 1]), int(xs[i]), int(ys[i]))
        p.setPen(QColor("#9aa0a6"))
        p.drawText(6, 14, f"{self.label}  ±{span:.2f}   x y z")
        p.end()


class ImuTile(QFrame):
    def __init__(self, imu: ImuRuntime) -> None:
        super().__init__()
        self.setObjectName("tile")
        self.imu = imu
        self.title = QLabel(f"imu  ·  {imu.source.source_type}")
        self.title.setObjectName("title")
        self.kind = QLabel()
        self.kind.setObjectName("badge")
        self.state = QLabel()
        self.state.setObjectName("badge")
        header = QHBoxLayout()
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.kind)
        header.addWidget(self.state)

        self.values = QLabel()
        self.values.setFont(_mono(13))
        self.gyro_plot = TracePlot("gyro rad/s", min_range=0.2)
        self.accel_plot = TracePlot("accel m/s²", min_range=GRAVITY * 1.2)
        self.stats = QLabel()
        self.stats.setFont(_mono(12))
        self.notes = QLabel()
        self.notes.setObjectName("notes")
        self.notes.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.values)
        layout.addWidget(self.gyro_plot, 1)
        layout.addWidget(self.accel_plot, 1)
        layout.addWidget(self.stats)
        layout.addWidget(self.notes)

    def show_health(self, health: ImuHealth) -> None:
        emulated = bool(health.transport.get("emulated"))
        kind = "EMULATED" if emulated else self.imu.source.source_kind.upper()
        color = "#d9822b" if emulated or kind == "SYNTHETIC" else KIND_COLORS[SourceKind.LIVE]
        self.kind.setText(kind)
        self.kind.setStyleSheet(f"background: {color};")
        state = {
            "streaming": CameraState.STREAMING,
            "stalled": CameraState.STALLED,
            "connecting": CameraState.CONNECTING,
        }.get(health.state, CameraState.STOPPED)
        self.state.setText(health.state.upper())
        self.state.setStyleSheet(f"background: {STATE_COLORS[state]};")

        recent = self.imu.recent(400)
        sample = recent[-1] if recent else None
        if sample is not None:
            a, g = sample.accel, sample.gyro
            temp = f"   temp {sample.temperature_c:5.1f} °C" if sample.temperature_c else ""
            orient = ""
            pose = self.imu.pose_at()
            if pose is not None:
                yaw, pitch, roll = angles_from_rotation(pose)
                orient = f"\nfilter yaw {yaw:+6.1f}° (relative)  pitch {pitch:+6.1f}°  " \
                         f"roll {roll:+6.1f}°"
            self.values.setText(
                f"accel m/s²  x {a[0]:+7.3f}  y {a[1]:+7.3f}  z {a[2]:+7.3f}"
                f"   |a| {np.linalg.norm(a):6.3f}\n"
                f"gyro rad/s  x {g[0]:+7.4f}  y {g[1]:+7.4f}  z {g[2]:+7.4f}{temp}{orient}"
            )
            t = np.array([s.host_receive_ns for s in recent], dtype=np.float64) / 1e9
            self.gyro_plot.set_data(t, np.stack([s.gyro for s in recent]))
            self.accel_plot.set_data(t, np.stack([s.accel for s in recent]))
        else:
            self.values.setText("waiting for telemetry…")

        tr = health.transport
        counters = "  ".join(
            f"{k} {tr[k]}"
            for k in ("packets", "dropped", "invalid", "out_of_order", "parse_errors",
                      "wrong_device", "device_restarts")
            if k in tr
        )
        seq = ""
        if sample is not None and sample.sequence is not None:
            seq = f"  seq {sample.sequence}"
        self.stats.setText(
            f"rate {_fmt(health.rate_hz, '6.1f', ' Hz')}"
            f"  age {_fmt(health.last_sample_age_ms, '.0f', ' ms')}"
            f"  delay {_fmt(tr.get('clock_excess_delay_ms'), '.1f', ' ms')}"
            f"  samples {health.samples}{seq}\n"
            f"{counters or 'no transport counters'}  from {tr.get('sender', '—')}"
        )
        notes = []
        if health.units_warning:
            notes.append(f"⚠ units: {health.units_warning}")
        if health.last_error and health.state != "streaming":
            notes.append(f"⚠ {health.last_error}")
        if emulated:
            notes.append("emulated device (opengaze fake-esp32), not real hardware")
        self.notes.setText("\n".join(notes))
        self.notes.setVisible(bool(notes))


class CameraTile(QFrame):
    def __init__(self, camera_id: str) -> None:
        super().__init__()
        self.setObjectName("tile")
        self.camera_id = camera_id
        self.title = QLabel(camera_id)
        self.title.setObjectName("title")
        self.kind = QLabel()
        self.kind.setObjectName("badge")
        self.state = QLabel()
        self.state.setObjectName("badge")
        header = QHBoxLayout()
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.kind)
        header.addWidget(self.state)

        self.preview = QLabel("waiting for frames…")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 200)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.stats = QLabel()
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(12)
        self.stats.setFont(mono)
        self.notes = QLabel()
        self.notes.setObjectName("notes")
        self.notes.setWordWrap(True)

        self.pose = QLabel()
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(11)
        self.pose.setFont(mono)
        self.pose.setStyleSheet("color: #60c8ff;")
        self.pose.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.stats)
        layout.addWidget(self.pose)
        layout.addWidget(self.notes)

    def show_health(self, health: CameraHealth, frame: CameraFrame | None) -> None:
        self.title.setText(f"{health.camera_id}  ·  {health.role.value}  ·  {health.source_type}")
        if is_emulated(health):
            self.kind.setText("EMULATED")
            self.kind.setStyleSheet(f"background: {KIND_COLORS[SourceKind.SYNTHETIC]};")
        else:
            self.kind.setText(health.source_kind.value.upper())
            self.kind.setStyleSheet(f"background: {KIND_COLORS[health.source_kind]};")
        self.state.setText(health.state.value.upper())
        self.state.setStyleSheet(f"background: {STATE_COLORS[health.state]};")

        size = f"{frame.width}x{frame.height} {frame.pixel_format.value}" if frame else "—"
        ts = "—"
        if frame is not None:
            unc = frame.timestamp_uncertainty_ns
            ts = frame.timestamp_source.value + ("" if unc is None else f" ±{unc / 1e6:.0f} ms")
        backend = health.capabilities.backend if health.capabilities else None
        self.stats.setText(
            f"fps {_fmt(health.measured_fps, '5.1f')}"
            f"  jitter {_fmt(health.interval_jitter_ms, '.1f', ' ms')}"
            f"  age {_fmt(health.last_frame_age_ms, '.0f', ' ms')}"
            f"  delay {_fmt(health.transport_delay_ms, '.0f', ' ms')}\n"
            f"{size}  ts {ts}  backend {backend or '—'}\n"
            f"frames {health.frames_received}  skipped {health.frames_skipped_by_source}"
            f"  overwritten {health.frames_overwritten}  partial {health.partial_frames}"
            f"  reconnects {health.reconnects}"
        )

        notes = []
        if health.last_error and health.state is not CameraState.STREAMING:
            notes.append(f"⚠ {health.last_error}")
        if health.capabilities:
            for result in health.capabilities.setting_results:
                if result.support in (ControlSupport.IGNORED, ControlSupport.UNSUPPORTED):
                    detail = f" ({result.message})" if result.message else ""
                    notes.append(f"{result.name}: {result.support.value}{detail}")
        if frame is not None and frame.warnings:
            notes.append("frame: " + ", ".join(frame.warnings))
        self.notes.setText("\n".join(notes))
        self.notes.setVisible(bool(notes))

    def show_pose(self, head: HeadObservation | None, eyes: list[EyeObservation]) -> None:
        if head is None:
            self.pose.setVisible(False)
            return
        blinks = "".join(
            ("L" if any(e.eye == "left" and e.blink for e in eyes) else "·")
            + ("R" if any(e.eye == "right" and e.blink for e in eyes) else "·")
        )
        pos = ""
        if head.position_cm is not None:
            x, y, z = head.position_cm
            pos = f"  pos x {x:+5.1f} y {y:+5.1f} z {z:+6.1f} cm"
        # Check the signs: turn right → yaw +, look up → pitch +, tilt right → roll +,
        # move right → x +, move closer → z toward 0.
        self.pose.setText(
            f"yaw {head.yaw_deg:+6.1f}°  pitch {head.pitch_deg:+6.1f}°  "
            f"roll {head.roll_deg:+6.1f}°{pos}  blink {blinks}"
        )
        self.pose.setVisible(True)

    def show_frame(self, image: np.ndarray) -> None:
        pixmap = QPixmap.fromImage(to_qimage(image))
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


def draw_detections(
    image: np.ndarray, eyes: list[EyeObservation]
) -> np.ndarray:
    """Overlay detected iris positions on the image (returns a copy)."""
    if not eyes:
        return image
    image = image.copy()
    h, w = image.shape[:2]
    for eye in eyes:
        if eye.blink:
            continue
        cx = round(eye.pupil_x_norm * w)
        cy = round(eye.pupil_y_norm * h)
        r = max(4, round(eye.pupil_radius_norm * min(w, h)))
        cv2.circle(image, (cx, cy), r, (80, 200, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(image, (cx, cy), (80, 200, 255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
    return image


def draw_ground_truth(frame: CameraFrame) -> np.ndarray:
    """Mark the synthetic pupil position so orientation handling can be checked by eye."""
    truth = frame.ground_truth or {}
    if "pupil_x" not in truth:
        return frame.image
    image = frame.image.copy()
    point = (round(truth["pupil_x"]), round(truth["pupil_y"]))
    cv2.drawMarker(image, point, (60, 220, 60), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(
        image,
        "GT",
        (point[0] + 8, point[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (60, 220, 60),
        1,
        cv2.LINE_AA,
    )
    return image


class DiagnosticsWindow(QMainWindow):
    def __init__(
        self,
        runtime: CaptureRuntime,
        report: ProfileReport,
        config: AppConfig,
        *,
        imu: ImuRuntime | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        super().__init__()
        assert report.profile is not None
        self.runtime = runtime
        self.manager: CameraManager = runtime.manager
        self.imu = imu
        self.metrics = metrics
        self.profile = report.profile
        self.normalizer = FrameNormalizer()
        self._detector = _try_create_detector(config, report.profile)
        self._head_camera_ids = {
            cid
            for cid, cam in report.profile.enabled_cameras.items()
            if cam.role is CameraRole.HEAD
        }
        self.setWindowTitle(f"OpenGaze diagnostics — {self.profile.id}")
        self.setStyleSheet(STYLE)

        mode = report.mode.label if report.mode else "no tracking mode"
        info = QLabel(
            f"Profile <b>{self.profile.id}</b> v{self.profile.version} · frame "
            f"<b>{self.profile.frame_design}</b> · {mode} · geometry fingerprint "
            f"<code>{profile_fingerprint(self.profile)}</code>"
        )
        privacy = QLabel(
            "Processed locally · nothing is uploaded"
            + (f" · metrics (no images) → {metrics.path}" if metrics else " · nothing recorded")
        )
        privacy.setStyleSheet("color: #9aa0a6;")

        self.normalized = QCheckBox("Normalized view (apply profile rotation / mirror / crop)")
        self.normalized.setChecked(config.diagnostics.show_normalized)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(info)
        kinds = {self.manager.source(cid).source_kind for cid in self.manager.camera_ids}
        if kinds - {SourceKind.LIVE}:
            labels = " / ".join(k.value.upper() for k in sorted(kinds - {SourceKind.LIVE}))
            banner = QLabel(f"Showing {labels} data — this is not live eye tracking")
            banner.setObjectName("banner")
            banner.setStyleSheet("background: #5c3a12; color: #ffd8a8;")
            layout.addWidget(banner)
        controls = QHBoxLayout()
        controls.addWidget(self.normalized)
        controls.addStretch(1)
        controls.addWidget(privacy)
        layout.addLayout(controls)

        # One badge per input: is it delivering fresh data right now?
        self.inputs_row = QHBoxLayout()
        self.input_badges: dict[str, QLabel] = {}
        self.inputs_row.addWidget(QLabel("Inputs:"))
        self.inputs_row.addStretch(1)
        layout.addLayout(self.inputs_row)

        grid = QGridLayout()
        self.tiles: dict[str, CameraTile] = {}
        # Left eye, then right eye, side by side; everything else after.
        order = {CameraRole.LEFT_EYE: 0, CameraRole.RIGHT_EYE: 1}
        camera_ids = sorted(
            self.manager.camera_ids,
            key=lambda c: order.get(self.manager.source(c).role, 2),
        )
        for i, cid in enumerate(camera_ids):
            tile = CameraTile(cid)
            self.tiles[cid] = tile
            grid.addWidget(tile, i // 2, i % 2)
        self.imu_tile: ImuTile | None = None
        if imu is not None:
            self.imu_tile = ImuTile(imu)
            n = len(camera_ids)
            grid.addWidget(self.imu_tile, n // 2, n % 2)
        layout.addLayout(grid, 1)
        self.setCentralWidget(root)
        self.resize(1200, 800)

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / config.diagnostics.preview_fps))
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def _show_inputs(self, health: dict[str, CameraHealth], imu: ImuHealth | None) -> None:
        for status in input_statuses(health, imu):
            badge = self.input_badges.get(status.name)
            if badge is None:
                badge = QLabel()
                badge.setObjectName("input")
                self.input_badges[status.name] = badge
                self.inputs_row.insertWidget(self.inputs_row.count() - 1, badge)
            badge.setText(f"{status.name}: {status.level.upper()}")
            badge.setToolTip(status.detail)
            badge.setStyleSheet(f"background: {LEVEL_COLORS[status.level]};")

    def refresh(self) -> None:
        health = self.manager.health()
        imu_health = self.imu.health() if self.imu is not None else None
        self._show_inputs(health, imu_health)
        if self.metrics is not None:
            self.metrics.record(health, imu_health)
        if self.imu_tile is not None and imu_health is not None:
            self.imu_tile.show_health(imu_health)
        for cid, tile in self.tiles.items():
            frame = self.manager.latest(cid)
            tile.show_health(health[cid], frame)
            if frame is None:
                continue
            # Ground truth refers to the oriented, uncropped image, so only mark it there.
            show_truth = frame.source_kind is SourceKind.SYNTHETIC and frame.crop is None
            if self.normalized.isChecked():
                frame = self.normalizer(frame)
            show_truth = show_truth and frame.is_normalized

            head: HeadObservation | None = None
            eyes: list[EyeObservation] = []
            if self._detector is not None and cid in self._head_camera_ids:
                try:
                    head, eyes = self._detector.detect(frame)
                except Exception:
                    pass

            image = draw_ground_truth(frame) if show_truth else frame.image
            image = draw_detections(image, eyes)
            tile.show_frame(image)
            tile.show_pose(head, eyes)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt naming)
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        self.runtime.stop()
        if self.imu is not None:
            self.imu.stop()
        if self.metrics is not None:
            self.metrics.close()
            print(self.metrics.format_summary())
            self.metrics = None
        if self._detector is not None:
            self._detector.close()
        super().closeEvent(event)


def _try_create_detector(config: AppConfig, profile) -> object | None:
    """Create a FaceLandmarkDetector for the head camera if tracking is enabled and available."""
    from opengaze.gaze.tracker import DEFAULT_HEAD_HFOV_DEG, make_face_detector

    head = next(
        (c for c in profile.enabled_cameras.values() if c.role is CameraRole.HEAD), None
    )
    if head is None:
        return None
    return make_face_detector(
        config,
        mirrored=head.mirror_horizontal,
        hfov_deg=head.optics.horizontal_fov_deg or DEFAULT_HEAD_HFOV_DEG,
    )


def run_diagnostics(
    report: ProfileReport, config: AppConfig, metrics: MetricsRecorder | None = None
) -> int:
    assert report.profile is not None
    app = QApplication.instance() or QApplication(sys.argv[:1])
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()
    imu = start_profile_imu(report.profile)
    window = DiagnosticsWindow(runtime, report, config, imu=imu, metrics=metrics)
    window.show()
    try:
        return app.exec()
    finally:
        runtime.stop()
        if imu is not None:
            imu.stop()
