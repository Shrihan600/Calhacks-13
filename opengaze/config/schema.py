"""Typed schema for application settings and hardware profiles.

Hardware profiles describe cameras without assuming any particular module, lens, frame or
connection. Every physical quantity is optional and ``null`` means "unknown" — calibration is
data-driven, so the system works before any of it is measured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

from opengaze.cameras.types import CameraIntrinsics, CameraRole, CropRegion, Orientation

ROLE_KEYS = {role.value for role in CameraRole}

MountingPosition = Literal[
    "unknown",
    "below_eye",
    "above_eye",
    "nasal_side",
    "temporal_side",
    "front",
    "display_top",
    "custom",
]


class _Model(BaseModel):
    # extra="forbid" turns YAML typos into load errors instead of silently ignored settings.
    model_config = ConfigDict(extra="forbid", frozen=True)


class CropConfig(_Model):
    """Region of interest as fractions of the oriented image."""

    x: float = Field(ge=0.0, lt=1.0)
    y: float = Field(ge=0.0, lt=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _inside_image(self) -> CropConfig:
        if self.x + self.width > 1.0 + 1e-9 or self.y + self.height > 1.0 + 1e-9:
            raise ValueError(
                "crop region must lie inside the image (x + width <= 1, y + height <= 1)"
            )
        return self

    def to_region(self) -> CropRegion:
        return CropRegion(self.x, self.y, self.width, self.height)


class IntrinsicsConfig(_Model):
    fx: float = Field(gt=0.0)
    fy: float = Field(gt=0.0)
    cx: float
    cy: float
    width: PositiveInt = Field(description="capture width the intrinsics were measured at")
    height: PositiveInt = Field(description="capture height the intrinsics were measured at")

    def to_intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(self.fx, self.fy, self.cx, self.cy, self.width, self.height)


class OpticsConfig(_Model):
    focal_length_mm: float | None = Field(default=None, gt=0.0)
    horizontal_fov_deg: float | None = Field(default=None, gt=0.0, lt=180.0)
    intrinsics: IntrinsicsConfig | None = None
    # OpenCV distortion model: (k1, k2, p1, p2[, k3[, k4, k5, k6[, s1..s4[, tx, ty]]]])
    distortion: tuple[float, ...] | None = None

    @field_validator("distortion")
    @classmethod
    def _distortion_length(cls, value: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if value is not None and len(value) not in (4, 5, 8, 12, 14):
            raise ValueError("distortion must have 4, 5, 8, 12 or 14 coefficients (OpenCV model)")
        return value


class GeometryConfig(_Model):
    """Physical placement of a camera relative to the eye or frame. All optional."""

    position_mm: tuple[float, float, float] | None = None
    orientation_deg: tuple[float, float, float] | None = Field(
        default=None, description="yaw, pitch, roll of the camera in the frame's coordinates"
    )
    camera_to_eye_mm: float | None = Field(default=None, gt=0.0)
    angle_to_pupil_deg: float | None = Field(default=None, ge=-90.0, le=90.0)
    adjustable: bool | None = None


class ControlsConfig(_Model):
    """Sensor controls passed to the driver as-is (units are backend-specific).

    Drivers often ignore these; the source reads each one back and reports what happened.
    """

    exposure: float | None = None
    gain: float | None = None
    focus: float | None = None
    auto_exposure: bool | None = None
    autofocus: bool | None = None

    def requested(self) -> dict[str, float | bool]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class CameraConfig(_Model):
    camera_id: str
    role: CameraRole
    source_type: str
    enabled: bool = True
    device_id: int | str | None = None
    resolution: tuple[PositiveInt, PositiveInt] | None = None
    fps: float | None = Field(default=None, gt=0.0, le=1000.0)
    rotation_degrees: int = 0
    mirror_horizontal: bool = False
    mirror_vertical: bool = False
    crop: CropConfig | None = None
    mounting_position: MountingPosition = "unknown"
    geometry: GeometryConfig = GeometryConfig()
    optics: OpticsConfig = OpticsConfig()
    controls: ControlsConfig = ControlsConfig()
    connect_timeout_s: float = Field(default=10.0, gt=0.0)
    read_timeout_s: float = Field(default=2.0, gt=0.0)
    # Adapter-specific settings, validated by the adapter's own Options model.
    options: dict[str, Any] = Field(default_factory=dict)
    # Set by the loader: directory relative option paths resolve against.
    base_dir: Path | None = None

    @field_validator("rotation_degrees")
    @classmethod
    def _right_angle(cls, value: int) -> int:
        if value not in (0, 90, 180, 270):
            raise ValueError(
                "must be 0, 90, 180 or 270 — image rotation is right-angle only; describe a "
                "tilted mount with geometry.orientation_deg instead"
            )
        return value

    def orientation(self) -> Orientation:
        return Orientation(self.rotation_degrees, self.mirror_horizontal, self.mirror_vertical)


class CalibrationPolicy(_Model):
    screen_mapping: Literal["per_user"] = "per_user"
    geometry_estimation: Literal["data_driven", "geometric"] = "data_driven"


class HardwareProfile(_Model):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    version: PositiveInt = 1
    frame_design: str = "unfinalized"
    description: str = ""
    cameras: dict[str, CameraConfig] = Field(min_length=1)
    calibration: CalibrationPolicy = CalibrationPolicy()

    @model_validator(mode="before")
    @classmethod
    def _camera_ids_from_keys(cls, data: Any) -> Any:
        """Each camera's YAML key is its id; well-known keys (left_eye, head, …) imply the role."""
        if not isinstance(data, dict) or not isinstance(data.get("cameras"), dict):
            return data
        cameras: dict[str, Any] = {}
        for key, camera in data["cameras"].items():
            if isinstance(camera, dict):
                camera = dict(camera)
                if camera.setdefault("camera_id", key) != key:
                    raise ValueError(f"camera '{key}' declares a different camera_id")
                if "role" not in camera and key in ROLE_KEYS:
                    camera["role"] = key
            cameras[key] = camera
        return {**data, "cameras": cameras}

    @property
    def enabled_cameras(self) -> dict[str, CameraConfig]:
        return {cid: cam for cid, cam in self.cameras.items() if cam.enabled}


class AppSection(_Model):
    hardware_profile: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class CaptureSettings(_Model):
    stale_frame_timeout_s: float = Field(default=1.0, gt=0.0)
    reconnect_initial_delay_s: float = Field(default=0.5, gt=0.0)
    reconnect_max_delay_s: float = Field(default=5.0, gt=0.0)


class DiagnosticsSettings(_Model):
    preview_fps: int = Field(default=30, ge=1, le=120)
    show_normalized: bool = True


class PrivacySettings(_Model):
    allow_recording: bool = False
    cloud_features_enabled: bool = False


class AppConfig(_Model):
    app: AppSection
    capture: CaptureSettings = CaptureSettings()
    diagnostics: DiagnosticsSettings = DiagnosticsSettings()
    privacy: PrivacySettings = PrivacySettings()
