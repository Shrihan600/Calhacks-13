"""Camera diagnostics window: live previews and health for every configured camera.

Capture runs on a background thread (``CaptureRuntime``); this window only polls the newest
frame per camera on a timer, so a slow or dead camera can never freeze the UI.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QKeyEvent, QPixmap
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

from opengaze.cameras.manager import CameraHealth, CameraManager
from opengaze.cameras.runtime import CaptureRuntime
from opengaze.cameras.types import CameraFrame, CameraState, ControlSupport, SourceKind
from opengaze.config.fingerprint import profile_fingerprint
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.vision.preprocessing import FrameNormalizer

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
"""


def to_qimage(image: np.ndarray) -> QImage:
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    fmt = QImage.Format.Format_Grayscale8 if image.ndim == 2 else QImage.Format.Format_BGR888
    return QImage(image.data, w, h, image.strides[0], fmt).copy()


def _fmt(value: float | None, spec: str, unit: str = "") -> str:
    return "—" if value is None else f"{value:{spec}}{unit}"


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

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.stats)
        layout.addWidget(self.notes)

    def show_health(self, health: CameraHealth, frame: CameraFrame | None) -> None:
        self.title.setText(f"{health.camera_id}  ·  {health.role.value}  ·  {health.source_type}")
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

    def show_frame(self, image: np.ndarray) -> None:
        pixmap = QPixmap.fromImage(to_qimage(image))
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


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
    def __init__(self, runtime: CaptureRuntime, report: ProfileReport, config: AppConfig) -> None:
        super().__init__()
        assert report.profile is not None
        self.runtime = runtime
        self.manager: CameraManager = runtime.manager
        self.profile = report.profile
        self.normalizer = FrameNormalizer()
        self.setWindowTitle(f"OpenGaze diagnostics — {self.profile.id}")
        self.setStyleSheet(STYLE)

        mode = report.mode.label if report.mode else "no tracking mode"
        info = QLabel(
            f"Profile <b>{self.profile.id}</b> v{self.profile.version} · frame "
            f"<b>{self.profile.frame_design}</b> · {mode} · geometry fingerprint "
            f"<code>{profile_fingerprint(self.profile)}</code>"
        )
        privacy = QLabel("Processed locally · nothing is recorded or uploaded")
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

        grid = QGridLayout()
        self.tiles: dict[str, CameraTile] = {}
        for i, cid in enumerate(self.manager.camera_ids):
            tile = CameraTile(cid)
            self.tiles[cid] = tile
            grid.addWidget(tile, i // 2, i % 2)
        layout.addLayout(grid, 1)
        self.setCentralWidget(root)
        self.resize(1200, 800)

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / config.diagnostics.preview_fps))
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def refresh(self) -> None:
        health = self.manager.health()
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
            tile.show_frame(draw_ground_truth(frame) if show_truth else frame.image)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt naming)
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        self.runtime.stop()
        super().closeEvent(event)


def run_diagnostics(report: ProfileReport, config: AppConfig) -> int:
    assert report.profile is not None
    app = QApplication.instance() or QApplication(sys.argv[:1])
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()
    window = DiagnosticsWindow(runtime, report, config)
    window.show()
    try:
        return app.exec()
    finally:
        runtime.stop()
