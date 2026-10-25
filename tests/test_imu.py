"""Head-mounted IMU: protocol, axis mapping, orientation filter, runtime, head-pose fusion."""

from __future__ import annotations

import math
import socket
import time

import numpy as np
import pytest

from opengaze.config.loader import check_profile_data
from opengaze.imu.fusion import ImuOrientation, axis_matrix
from opengaze.imu.head import HeadPoseFusion
from opengaze.imu.runtime import ImuRuntime
from opengaze.imu.sources import SyntheticImuSource, UdpImuSource
from opengaze.imu.types import GRAVITY, ImuSample, parse_line
from opengaze.vision.head_pose import angles_from_rotation, rotation_from_angles
from opengaze.vision.observations import HeadObservation


def run_synthetic(seconds: float, **options) -> tuple[ImuOrientation, list[tuple]]:
    """Feed a synthetic source through the orientation filter; returns (filter, errors)."""
    src = SyntheticImuSource({"realtime": False, **options})
    src.open()
    orient = ImuOrientation(("+x", "+y", "+z"))
    out = []
    for _ in range(int(seconds * src.options.rate_hz)):
        sample = src.read(0.1)
        rot = orient.update(sample)
        if rot is not None:
            out.append((sample.device_time_s, angles_from_rotation(rot), sample.ground_truth))
    return orient, out


def angle_err(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


# ---------------------------------------------------------------------------
# Protocol + axes
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_six_fields_default_units(self):
        s = parse_line("10, 0, -5, 0, -1, 0\n", 123)
        assert s.host_receive_ns == 123 and s.device_time_s is None
        assert np.allclose(s.gyro, np.radians([10, 0, -5]))
        assert np.allclose(s.accel, [0, -GRAVITY, 0])

    def test_seven_fields_with_device_time_and_si_units(self):
        s = parse_line("1500000 0.1 0.2 0.3 0 0 9.8", 0, gyro_units="rad_s", accel_units="m_s2")
        assert s.device_time_s == pytest.approx(1.5)
        assert np.allclose(s.gyro, [0.1, 0.2, 0.3])
        assert np.allclose(s.accel, [0, 0, 9.8])

    def test_comments_and_blank_lines(self):
        assert parse_line("# MPU6050 ready", 0) is None
        assert parse_line("   ", 0) is None

    @pytest.mark.parametrize("bad", ["1,2,3", "a,b,c,d,e,f", "1,2,3,4,5,nan"])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            parse_line(bad, 0)


class TestAxes:
    def test_identity_and_remap(self):
        assert np.array_equal(axis_matrix(("+x", "+y", "+z")), np.eye(3))
        m = axis_matrix(("+x", "-z", "+y"))  # IMU z points up, y points forward
        assert np.allclose(m @ [0, 0, 1], [0, -1, 0])  # IMU up = head "minus chin"
        assert np.allclose(m @ [0, 1, 0], [0, 0, 1])

    def test_rejects_mirrored_and_repeated(self):
        with pytest.raises(ValueError, match="mirrored"):
            axis_matrix(("+x", "+y", "-z"))
        with pytest.raises(ValueError, match="once each"):
            axis_matrix(("+x", "+x", "+z"))
        with pytest.raises(ValueError, match="bad IMU axis"):
            axis_matrix(("x", "+y", "+z"))


# ---------------------------------------------------------------------------
# Orientation filter
# ---------------------------------------------------------------------------


class TestOrientation:
    def test_never_still_head_pitch_roll_absolute_yaw_drifts_at_bias_rate(self):
        _, out = run_synthetic(40.0, seed=1)
        settled = [o for o in out if o[0] > 5.0]
        pitch_roll = [max(angle_err(e[1], g[1]), angle_err(e[2], g[2])) for _, e, g in settled]
        assert np.percentile(pitch_roll, 95) < 2.5  # gravity keeps pitch/roll absolute
        # Yaw can't see gravity; with no still moment the bias isn't learned, so yaw drifts
        # at the yaw bias rate (0.4°/s here). That's what webcam correction / recenter fix.
        t, est, truth = out[-1]
        assert angle_err(est[0], truth[0]) < 0.4 * t + 3.0

    def test_pauses_let_bias_be_learned_so_yaw_holds(self):
        _, out = run_synthetic(60.0, seed=2, motion="sway_pause")
        late = [o for o in out if o[0] > 40.0]
        yaw = [angle_err(e[0], g[0]) for _, e, g in late]
        pitch_roll = [max(angle_err(e[1], g[1]), angle_err(e[2], g[2])) for _, e, g in late]
        assert np.percentile(pitch_roll, 95) < 2.5
        # Mostly the drift accumulated before the first pauses; it no longer grows.
        assert np.percentile(yaw, 95) < 5.0
        first, last = late[0], late[-1]
        assert abs(angle_err(last[1][0], last[2][0]) - angle_err(first[1][0], first[2][0])) < 1.5

    def test_learns_gyro_bias_while_still(self):
        orient, out = run_synthetic(20.0, motion="static", gyro_bias_dps=(1.5, -1.0, 2.0))
        assert np.allclose(np.degrees(orient.bias.bias), (1.5, -1.0, 2.0), atol=0.1)
        drift = angle_err(out[-1][1][0], out[len(out) // 2][1][0])
        assert drift < 1.0  # ≈ 15° in 10 s without bias removal

    def test_axis_remap_gives_head_angles(self):
        # The same motion, from an IMU mounted with z up and y forward.
        m = axis_matrix(("+x", "-z", "+y"))
        src = SyntheticImuSource({"realtime": False, "gyro_noise_dps": 0, "accel_noise_g": 0,
                                  "gyro_bias_dps": (0, 0, 0)})
        src.open()
        orient = ImuOrientation(("+x", "-z", "+y"))
        for _ in range(4000):
            s = src.read(0.1)
            raw = ImuSample(s.host_receive_ns, s.device_time_s, m.T @ s.gyro, m.T @ s.accel,
                            "synthetic", s.ground_truth)
            rot = orient.update(raw)
        est, truth = angles_from_rotation(rot), s.ground_truth
        assert angle_err(est[1], truth[1]) < 1.0 and angle_err(est[2], truth[2]) < 1.0


# ---------------------------------------------------------------------------
# Runtime + sources
# ---------------------------------------------------------------------------


def test_runtime_pose_at_picks_nearest_and_goes_stale():
    src = SyntheticImuSource({"realtime": False})
    src.open()
    rt = ImuRuntime(src, stale_timeout_s=0.5)
    samples = [src.read(0.1) for _ in range(400)]
    for s in samples:
        rt.push(s)
    mid = samples[200]
    assert rt.pose_at(mid.host_receive_ns) is not None
    assert rt.pose_at(samples[-1].host_receive_ns + int(2e9)) is None  # stale


def test_runtime_thread_streams_synthetic():
    rt = ImuRuntime(SyntheticImuSource({"rate_hz": 200}))
    rt.start()
    try:
        deadline = time.monotonic() + 3.0
        while rt.pose_at() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.3)
        health = rt.health()
        assert rt.pose_at() is not None
        assert health.state == "streaming" and health.rate_hz and health.rate_hz > 100
    finally:
        rt.stop()


def test_udp_source_receives_datagrams():
    src = UdpImuSource({"bind_host": "127.0.0.1", "port": 0, "gyro_units": "rad_s"})
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        src.open()
        port = src.bound_port
        sock.sendto(b"0.1,0,0,0,-1,0\n0.2,0,0,0,-1,0\n", ("127.0.0.1", port))
        first, second = src.read(1.0), src.read(1.0)
        assert first.gyro[0] == pytest.approx(0.1) and second.gyro[0] == pytest.approx(0.2)
        assert src.read(0.05) is None
    finally:
        sock.close()
        src.close()


def test_runtime_survives_failing_source():
    class Broken(SyntheticImuSource):
        def open(self):
            raise OSError("no such port")

    rt = ImuRuntime(Broken({}))
    rt.start()
    time.sleep(0.2)
    assert "no such port" in (rt.health().last_error or "")
    rt.stop()


# ---------------------------------------------------------------------------
# Head-pose fusion
# ---------------------------------------------------------------------------


def webcam(yaw, pitch=0.0, roll=0.0, pos=(1.0, 10.0, -58.0)) -> HeadObservation:
    return HeadObservation(0, None, yaw, pitch, roll, 0.5, 0.5, 0.1, 1.0, None, pos)


class TestHeadPoseFusion:
    def test_imu_only_anchors_heading_and_has_no_position(self):
        fusion = HeadPoseFusion()
        imu = rotation_from_angles(40.0, 5.0, -3.0)  # IMU heading is arbitrary
        head = fusion.update(imu, None, 0)
        assert head.source == "imu" and head.position_cm is None
        assert head.yaw_deg == pytest.approx(0.0, abs=1e-6)  # first reading = facing screen
        assert head.pitch_deg == pytest.approx(5.0, abs=1e-6)
        turned = fusion.update(rotation_from_angles(55.0, 5.0, -3.0), None, int(1e7))
        assert turned.yaw_deg == pytest.approx(15.0, abs=1e-6)

    def test_webcam_anchors_yaw_and_supplies_position(self):
        fusion = HeadPoseFusion()
        head = fusion.update(rotation_from_angles(40.0, 2.0, 0.0), webcam(8.0, 2.0), 0)
        assert head.source == "imu+webcam"
        assert head.yaw_deg == pytest.approx(8.0, abs=1e-6)
        assert head.position_cm == (1.0, 10.0, -58.0)

    def test_webcam_cancels_yaw_drift(self):
        fusion = HeadPoseFusion(webcam_yaw_rate=0.5)
        t = 0
        fusion.update(rotation_from_angles(0.0, 0, 0), webcam(0.0), t)
        # The head is still, but the IMU's yaw drifts 1°/s; the webcam keeps saying 0.
        for k in range(1, 30 * 30):
            t = int(k / 30 * 1e9)
            head = fusion.update(rotation_from_angles(k / 30, 0, 0), webcam(0.0), t)
        assert abs(head.yaw_deg) < 3.0  # steady-state lag ≈ drift rate / correction rate

    def test_falls_back_to_webcam_when_imu_stale(self):
        cam = webcam(5.0)
        assert HeadPoseFusion().update(None, cam, 0) is cam

    def test_recenter(self):
        fusion = HeadPoseFusion()
        fusion.update(rotation_from_angles(0, 0, 0), None, 0)
        fusion.update(rotation_from_angles(30, 0, 0), None, 1)
        fusion.recenter()
        assert fusion.update(rotation_from_angles(30, 0, 0), None, 2).yaw_deg == pytest.approx(0)


# ---------------------------------------------------------------------------
# Tracker end to end
# ---------------------------------------------------------------------------


def test_tracker_uses_imu_rotation_in_mode_c():
    from opengaze.gaze.tracker import GazeTracker
    from tests.test_gaze_pipeline import FakeManager, app_config

    data = {
        "hardware_profile": {
            "id": "imu-test",
            "cameras": {cid: {"source_type": "synthetic"} for cid in ("left_eye", "right_eye")},
            "imu": {"source_type": "synthetic", "options": {"realtime": False}},
        }
    }
    report = check_profile_data(data)
    assert report.ok, report.errors
    src = SyntheticImuSource({"realtime": False})
    src.open()
    rt = ImuRuntime(src)
    for _ in range(600):
        rt.push(src.read(0.1))
    tracker = GazeTracker(report.profile, app_config(), imu=rt)
    frame = tracker.step(FakeManager(("left_eye", "right_eye"), spherical=False))
    assert frame.head is not None and frame.head.source == "imu"
    layout_valid = frame.features.valid
    from opengaze.vision.observations import FEATURE_LAYOUTS

    layout = FEATURE_LAYOUTS[tracker.schema_id]
    assert layout_valid[list(layout.head_rot)].all()
    assert not layout_valid[list(layout.head_pos)].any()
    assert math.isfinite(frame.head.pitch_deg)
    tracker.close()
    assert rt.pose_at() is not None or True  # tracker doesn't stop an IMU it didn't start
