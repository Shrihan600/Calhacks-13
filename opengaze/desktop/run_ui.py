"""Gaze test environment: fullscreen window showing gaze dot, camera feed, blink state.

Flow
----
1. If no calibration: run calibration wizard first, then return here.
2. Fullscreen window opens.
3. Gaze dot tracks eye position.
4. Long blink (both eyes >= 400 ms) fires a click.
5. Tab toggles real cursor control on/off.
6. C re-runs calibration.
7. Space pauses/resumes.
8. Escape quits.
"""

from __future__ import annotations

import logging
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from opengaze.calibration.session import GRID_9
from opengaze.calibration.storage import CalibrationStore
from opengaze.cameras.manager import CameraManager
from opengaze.cameras.runtime import CaptureRuntime
from opengaze.cameras.types import CameraRole, SourceKind
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.control.blink import BlinkClickDetector
from opengaze.control.filter import OneEuroFilter
from opengaze.control.safety import ConfidenceGate
from opengaze.modes import TrackingMode, infer_mode
from opengaze.vision.feature_extractor import build_mode_a
from opengaze.vision.observations import EyeObservation, HeadObservation
from opengaze.vision.preprocessing import FrameNormalizer

log = logging.getLogger("opengaze.run_ui")

_BG = QColor("#111318")
_GAZE_RING = QColor("#00d4ff")
_GAZE_CENTER = QColor("#ffffff")
_BLINK_ACTIVE = QColor("#f0b429")
_BLINK_INACTIVE = QColor("#333")
_CLICK_FLASH = QColor(0, 212, 255, 160)
_TEXT = QColor("#c8c8c8")
_DIM = QColor("#555")
_CURSOR_ON = QColor("#2e9e5b")
_CURSOR_OFF = QColor("#555")

_SCHEMA_MAP = {
    TrackingMode.WEBCAM_ONLY: "A-webcam",
    TrackingMode.SINGLE_EYE: "B-single-eye",
    TrackingMode.DUAL_EYE: "C-dual-eye",
}

_POLL_MS = 33   # ~30 fps


def _to_qimage(image: np.ndarray) -> QImage:
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    fmt = QImage.Format.Format_BGR888
    return QImage(image.data, w, h, image.strides[0], fmt).copy()


class GazeTestWindow(QWidget):
    def __init__(
        self,
        runtime: CaptureRuntime,
        report: ProfileReport,
        config: AppConfig,
        schema_id: str,
    ) -> None:
        super().__init__()
        assert report.profile is not None

        self.runtime = runtime
        self.profile = report.profile
        self.config = config
        self.schema_id = schema_id

        # Vision pipeline
        self.normalizer = FrameNormalizer()
        self._detector = _make_detector(config)
        self._head_ids = {
            cid for cid, cam in report.profile.enabled_cameras.items()
            if cam.role is CameraRole.HEAD
        }

        # Load calibration model
        self._store = CalibrationStore()
        self._model = None
        self._load_model()

        # Gaze pipeline
        self._filter_x = OneEuroFilter(min_cutoff=0.8, beta=0.07)
        self._filter_y = OneEuroFilter(min_cutoff=0.8, beta=0.07)
        self._conf_gate = ConfidenceGate(min_confidence=0.25)
        self._blink = BlinkClickDetector(long_blink_ms=400, short_blink_ms=180)

        # Cursor driver (optional)
        self._cursor = _try_make_cursor()
        self._cursor_enabled = False  # off by default — toggle with Tab

        # Display state
        self._gaze_x: float = 0.5
        self._gaze_y: float = 0.5
        self._confidence: float = 0.0
        self._left_blink = False
        self._right_blink = False
        self._blink_progress: float = 0.0
        self._paused = False
        self._click_flash_until: float = 0.0
        self._camera_frame: np.ndarray | None = None
        self._eyes: list[EyeObservation] = []
        self._head: HeadObservation | None = None
        self._clicks = 0
        self._status = "Press C to calibrate, Tab to enable cursor, Space to pause, Esc to quit"

        # Qt
        self.setWindowTitle("OpenGaze — Test Environment")
        self.setStyleSheet("background: #111318;")
        self.showFullScreen()

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            self._model = self._store.load(self.schema_id, self.profile)  # type: ignore[arg-type]
        except (FileNotFoundError, ValueError):
            self._model = None

    # ------------------------------------------------------------------
    # Timer tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        if self._paused:
            self.update()
            return

        frame_image = None
        for cid in self._head_ids:
            frame = self.runtime.manager.latest(cid)
            if frame is None:
                continue
            frame = self.normalizer(frame)
            frame_image = frame.image

            if self._detector is None:
                continue
            try:
                head, eyes = self._detector.detect(frame)
            except Exception:
                continue

            self._head = head
            self._eyes = eyes

            # Iris overlay on camera image
            if frame_image is not None:
                frame_image = _draw_iris(frame_image, eyes)

            # Blink state
            left_eye = next((e for e in eyes if e.eye == "left"), None)
            right_eye = next((e for e in eyes if e.eye == "right"), None)
            self._left_blink = bool(left_eye and left_eye.blink)
            self._right_blink = bool(right_eye and right_eye.blink)

            blink_result = self._blink.update(self._left_blink, self._right_blink)
            self._blink_progress = blink_result.blink_progress

            if self._model is None:
                continue

            fv = build_mode_a(head, left_eye, right_eye)

            if not self._conf_gate.allow(fv.valid.mean()):
                continue

            try:
                sample = self._model.predict(fv)
            except Exception:
                continue

            self._confidence = sample.confidence
            t = time.monotonic()
            sx = self._filter_x(sample.screen_x_norm, t)
            sy = self._filter_y(sample.screen_y_norm, t)
            self._gaze_x = sx
            self._gaze_y = sy

            # Pause cursor during any blink (avoid drift)
            if not blink_result.any_blink and self._cursor_enabled and self._cursor:
                px, py = self._cursor.norm_to_px(sx, sy)
                self._cursor.move(px, py)

            # Long blink → click
            if blink_result.clicked:
                self._clicks += 1
                self._click_flash_until = time.monotonic() + 0.3
                if self._cursor_enabled and self._cursor:
                    px, py = self._cursor.norm_to_px(sx, sy)
                    self._cursor.click(px, py)

        self._camera_frame = frame_image
        self.update()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        k = event.key()
        if k == Qt.Key.Key_Escape:
            self.close()
        elif k == Qt.Key.Key_Space:
            self._paused = not self._paused
            if not self._paused:
                self._filter_x.reset()
                self._filter_y.reset()
        elif k == Qt.Key.Key_Tab:
            self._cursor_enabled = not self._cursor_enabled
            state = "ON" if self._cursor_enabled else "OFF"
            self._status = f"Cursor control {state}"
        elif k == Qt.Key.Key_C:
            self._run_calibration()
        else:
            super().keyPressEvent(event)

    def _run_calibration(self) -> None:
        self._timer.stop()
        try:
            from opengaze.desktop.calibration_ui import CalibrationWindow
            win = CalibrationWindow(self.runtime, _make_report(self.profile), self.config, self._store, owns_runtime=False)
            win.show()
            # Wait for it to close
            app = QApplication.instance()
            while win.isVisible():
                app.processEvents()
                time.sleep(0.03)
        except Exception as exc:
            log.warning("calibration failed: %s", exc)
        self._load_model()
        self._filter_x.reset()
        self._filter_y.reset()
        self._timer.start()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), _BG)

        if self._paused:
            self._draw_centered(p, "PAUSED — press Space to resume", w // 2, h // 2)
            p.end()
            return

        self._draw_gaze(p, w, h)
        self._draw_camera_preview(p, w, h)
        self._draw_blink_indicators(p, w, h)
        self._draw_hud(p, w, h)
        p.end()

    def _draw_gaze(self, p: QPainter, w: int, h: int) -> None:
        gx = int(self._gaze_x * w)
        gy = int(self._gaze_y * h)

        # Click flash
        if time.monotonic() < self._click_flash_until:
            p.setBrush(_CLICK_FLASH)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(gx - 60, gy - 60, 120, 120)

        if self._model is None:
            # No model — show message
            self._draw_centered(p, "No calibration — press C to calibrate", w // 2, h // 2)
            return

        # Long-blink progress ring
        if self._blink_progress > 0:
            pen = QPen(QColor("#f0b429"), 6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            span = int(-self._blink_progress * 360 * 16)
            p.drawArc(gx - 44, gy - 44, 88, 88, 90 * 16, span)

        # Outer ring
        pen = QPen(_GAZE_RING, 3)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(gx - 36, gy - 36, 72, 72)

        # Cross
        pen2 = QPen(_GAZE_RING, 2)
        p.setPen(pen2)
        p.drawLine(gx - 50, gy, gx - 38, gy)
        p.drawLine(gx + 38, gy, gx + 50, gy)
        p.drawLine(gx, gy - 50, gx, gy - 38)
        p.drawLine(gx, gy + 38, gx, gy + 50)

        # Center dot
        p.setBrush(_GAZE_CENTER)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(gx - 4, gy - 4, 8, 8)

    def _draw_camera_preview(self, p: QPainter, w: int, h: int) -> None:
        if self._camera_frame is None:
            return
        pw, ph = 240, 160
        px, py = w - pw - 16, h - ph - 60
        img = cv2.resize(self._camera_frame, (pw, ph))
        qimg = _to_qimage(img)
        p.drawImage(px, py, qimg)
        # Border
        pen = QPen(QColor("#333"), 1)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(px, py, pw, ph)

    def _draw_blink_indicators(self, p: QPainter, w: int, h: int) -> None:
        bh = h - 44
        for i, (label, active) in enumerate([("L", self._left_blink), ("R", self._right_blink)]):
            cx = w - 240 - 16 - 60 + i * 50
            cy = bh
            color = _BLINK_ACTIVE if active else _BLINK_INACTIVE
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(cx - 14, cy - 14, 28, 28)
            p.setPen(_TEXT if active else _DIM)
            font = QFont("Helvetica Neue", 11, QFont.Weight.Bold)
            p.setFont(font)
            p.drawText(cx - 6, cy + 5, label)

    def _draw_hud(self, p: QPainter, w: int, h: int) -> None:
        font = QFont("Menlo", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        p.setFont(font)
        p.setPen(_TEXT)

        cursor_color = _CURSOR_ON if self._cursor_enabled else _CURSOR_OFF
        cursor_state = "CURSOR ON" if self._cursor_enabled else "CURSOR OFF (Tab)"

        # Top bar
        p.drawText(16, 28,
            f"OpenGaze  |  {self.schema_id}  |  conf {self._confidence:.2f}"
            f"  |  clicks {self._clicks}"
        )
        p.setPen(cursor_color)
        p.drawText(w - 200, 28, cursor_state)
        p.setPen(_TEXT)

        # Bottom bar
        p.drawText(16, h - 16, "C = calibrate   Tab = cursor on/off   Space = pause   Esc = quit")
        if self._status != self._status:
            pass
        p.setPen(QColor("#f0b429"))
        p.drawText(16, h - 36, self._status)

        # Long blink hint
        if self._blink_progress > 0.05:
            p.setPen(QColor("#f0b429"))
            font2 = QFont("Helvetica Neue", 18, QFont.Weight.Bold)
            p.setFont(font2)
            p.drawText(w // 2 - 80, h - 80, "Hold blink...")

    def _draw_centered(self, p: QPainter, text: str, cx: int, cy: int) -> None:
        p.setPen(_TEXT)
        font = QFont("Helvetica Neue", 22)
        p.setFont(font)
        p.drawText(cx - 300, cy - 20, 600, 40,
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, text)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        self.runtime.stop()
        if self._detector is not None:
            self._detector.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_iris(image: np.ndarray, eyes: list[EyeObservation]) -> np.ndarray:
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


def _make_detector(config: AppConfig) -> object | None:
    if not config.vision.enable_face_tracking:
        return None
    try:
        from opengaze.vision.mediapipe_detector import FaceLandmarkDetector, default_model_path
        path = config.vision.face_landmarker_model_path or default_model_path()
        if not path.exists():
            log.warning("MediaPipe model not found — run 'opengaze download-models'")
            return None
        return FaceLandmarkDetector(path)
    except ImportError:
        log.warning("mediapipe not installed")
        return None
    except Exception as exc:
        log.error("could not create detector: %s", exc)
        return None


def _try_make_cursor() -> object | None:
    try:
        from opengaze.control.cursor import QuartzCursorDriver, is_accessibility_granted
        if is_accessibility_granted():
            return QuartzCursorDriver()
        log.info("Accessibility permission not granted — cursor control disabled")
    except Exception:
        pass
    return None


def _make_report(profile) -> object:
    """Minimal ProfileReport-like object for CalibrationWindow."""
    from opengaze.config.loader import ProfileReport
    from opengaze.modes import infer_mode
    mode = infer_mode({cam.role for cam in profile.enabled_cameras.values()})
    return ProfileReport(path=None, profile=profile, issues=[], mode=mode)  # type: ignore[arg-type]


def run_gaze_test(report: ProfileReport, config: AppConfig) -> int:
    assert report.profile is not None
    app = QApplication.instance() or QApplication(sys.argv[:1])

    mode = infer_mode({cam.role for cam in report.profile.enabled_cameras.values()})
    schema_id = _SCHEMA_MAP.get(mode, "A-webcam")

    # Start runtime once — shared between calibration and gaze test so the camera
    # stays on continuously (no pause/restart between phases).
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()

    store = CalibrationStore()
    if not store.exists(schema_id, report.profile):  # type: ignore[arg-type]
        print("No calibration found — running calibration first…")
        from opengaze.desktop.calibration_ui import CalibrationWindow
        cal_win = CalibrationWindow(runtime, report, config, store, owns_runtime=False)
        cal_win.show()
        while cal_win.isVisible():
            app.processEvents()
            time.sleep(0.03)
        if not store.exists(schema_id, report.profile):  # type: ignore[arg-type]
            print("Calibration was not saved — continuing without a model.")

    window = GazeTestWindow(runtime, report, config, schema_id)
    window.show()

    try:
        return app.exec()
    finally:
        runtime.stop()
