"""MPU-6050 JSON telemetry over UDP (right ESP32): parsing, drops, handshake, units, metrics."""

from __future__ import annotations

import json
import socket
import time

import numpy as np
import pytest

from opengaze.cameras.manager import CameraHealth
from opengaze.cameras.types import CameraRole, CameraState, SourceKind
from opengaze.config.loader import parse_overrides
from opengaze.debug.fake_esp32 import FakeImu, Faults
from opengaze.debug.metrics import MetricsRecorder, input_statuses
from opengaze.imu.runtime import ImuHealth, ImuRuntime
from opengaze.imu.sources import UdpImuSource
from opengaze.imu.types import GRAVITY, ImuSample, check_units, iter_json_objects, parse_json_record


def reading(seq: int, t_us: int, **extra) -> dict:
    return {
        "device_id": "right_esp32",
        "sensor": "mpu6050",
        "sequence": seq,
        "timestamp_us": t_us,
        "accel_x": 0.0,
        "accel_y": -9.79,
        "accel_z": 0.1,
        "gyro_x": 0.12,
        "gyro_y": -0.08,
        "gyro_z": 0.03,
        "valid": True,
        **extra,
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestJsonRecord:
    def test_protocol_units_by_default(self):
        rec = parse_json_record(reading(5, 1_500_000, temp_c=31.5), 42)
        s = rec.sample
        assert rec.sequence == 5 and rec.device_id == "right_esp32"
        assert np.allclose(s.gyro, [0.12, -0.08, 0.03])  # rad/s untouched
        assert np.allclose(s.accel, [0.0, -9.79, 0.1])  # m/s² untouched
        assert s.device_time_s == pytest.approx(1.5) and s.temperature_c == 31.5

    def test_configured_units_convert(self):
        rec = parse_json_record(
            reading(0, 0, gyro_x=90.0, accel_y=-1.0), 0, gyro_units="deg_s", accel_units="g"
        )
        assert rec.sample.gyro[0] == pytest.approx(np.pi / 2)
        assert rec.sample.accel[1] == pytest.approx(-GRAVITY)

    def test_invalid_reading_keeps_sequence(self):
        rec = parse_json_record({"sequence": 9, "timestamp_us": 1, "valid": False}, 0)
        assert rec.sample is None and rec.sequence == 9

    @pytest.mark.parametrize(
        "bad",
        [
            {"sequence": 1, "accel_x": 0},  # fields missing
            {**reading(1, 1), "gyro_z": "fast"},
            {**reading(1, 1), "gyro_z": float("nan")},
            {**reading(1, 1), "sequence": "one"},
            [1, 2, 3],
        ],
    )
    def test_malformed_raises(self, bad):
        with pytest.raises(ValueError):
            parse_json_record(bad, 0)

    def test_datagram_shapes(self):
        one = json.dumps(reading(1, 1))
        assert len(list(iter_json_objects(one))) == 1
        assert len(list(iter_json_objects(json.dumps([reading(1, 1), reading(2, 2)])))) == 2
        assert len(list(iter_json_objects(one + "\n" + one + "\n"))) == 2


# ---------------------------------------------------------------------------
# UDP source
# ---------------------------------------------------------------------------


@pytest.fixture
def udp():
    sources: list[UdpImuSource] = []
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def make(**options) -> tuple[UdpImuSource, callable]:
        src = UdpImuSource({"bind_host": "127.0.0.1", "port": 0, **options})
        src.open()
        sources.append(src)

        def send(payload) -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            sender.sendto(data, ("127.0.0.1", src.bound_port))

        return src, send

    yield make
    sender.close()
    for src in sources:
        src.close()


def drain(src: UdpImuSource, timeout: float = 0.3) -> list[ImuSample]:
    out = []
    while (s := src.read(timeout)) is not None:
        out.append(s)
    return out


def test_counts_drops_reorders_and_invalid(udp):
    src, send = udp(format="json")
    for seq in (0, 1, 4, 3, 5):  # 2 and 3 lost; 3 then arrives late
        send(reading(seq, 1_000_000 + seq * 10_000))
    send({"sequence": 6, "timestamp_us": 1_060_000, "valid": False})
    samples = drain(src)
    assert [s.sequence for s in samples] == [0, 1, 4, 5]
    st = src.stats()
    assert st["dropped"] == 2 and st["out_of_order"] == 1 and st["invalid"] == 1
    assert st["packets"] == 6 and st["samples"] == 4


def test_maps_device_time_to_host(udp):
    src, send = udp()
    send([reading(i, 7_000_000 + i * 10_000) for i in range(10)])  # one batched datagram
    samples = drain(src)
    assert len(samples) == 10
    gaps = np.diff([s.capture_ns for s in samples])
    assert np.allclose(gaps, 10_000_000)  # device spacing kept, not host arrival spacing
    assert all(s.capture_ns <= s.host_receive_ns for s in samples)


def test_bad_packet_is_counted_not_fatal(udp):
    src, send = udp()
    send(b"{not json")
    send({"sequence": 1, "accel_x": 1})
    send(reading(2, 5))
    samples = drain(src)
    assert [s.sequence for s in samples] == [2]
    assert src.stats()["parse_errors"] == 2 and "bad packet" in src.last_error


def test_filters_other_devices(udp):
    src, send = udp(expected_device_id="right_esp32")
    send(reading(1, 1, device_id="left_esp32"))
    send(reading(1, 1))
    assert len(drain(src)) == 1 and src.stats()["wrong_device"] == 1


def test_device_restart_is_not_out_of_order(udp):
    src, send = udp()
    send(reading(50_000, 900_000_000))
    send(reading(0, 1_000))  # rebooted: sequence and clock start over
    assert len(drain(src)) == 2
    assert src.stats()["device_restarts"] == 1


def test_text_lines_still_work_in_auto_mode(udp):
    src, send = udp()
    send(b"1500000 10 0 0 0 -1 0\n")  # legacy text protocol: deg/s and g by default
    s = drain(src)[0]
    assert s.gyro[0] == pytest.approx(np.radians(10)) and s.accel[1] == pytest.approx(-GRAVITY)


def test_subscribe_handshake_with_emulated_right_esp32():
    imu = FakeImu(rate_hz=200.0, batch=2).start()
    src = UdpImuSource(
        {"bind_host": "127.0.0.1", "port": 0, "device_host": "127.0.0.1",
         "device_port": imu.port, "subscribe_interval_s": 0.2}
    )
    src.open()
    try:
        assert imu.destination() is None  # nothing streams until the Mac subscribes
        samples = []
        deadline = time.monotonic() + 3.0
        while len(samples) < 40 and time.monotonic() < deadline:
            if (s := src.read(0.2)) is not None:
                samples.append(s)
        assert len(samples) >= 40
        assert imu.destination() == ("127.0.0.1", src.bound_port)
        st = src.stats()
        assert st["subscribes_sent"] >= 1 and st.get("emulated") is True
        norms = [np.linalg.norm(s.accel) for s in samples]
        assert np.median(norms) == pytest.approx(GRAVITY, rel=0.1)
    finally:
        src.close()
        imu.stop()


def test_runtime_reports_transport_and_device_rate():
    imu = FakeImu(rate_hz=100.0, faults=Faults(drop_rate=0.1)).start()
    rt = ImuRuntime(UdpImuSource(
        {"bind_host": "127.0.0.1", "port": 0, "device_host": "127.0.0.1",
         "device_port": imu.port, "subscribe_interval_s": 0.2}
    ))
    rt.start()
    try:
        deadline = time.monotonic() + 5.0
        while rt.health().samples < 150 and time.monotonic() < deadline:
            time.sleep(0.05)
        h = rt.health()
        assert h.state == "streaming" and h.samples >= 150
        # 10% of readings are lost, so ~90 arrive per second of device time... but the rate
        # is measured over device time between received samples, which is the true 100 Hz
        # minus the gaps: anything well below 100 and above 80 is right.
        assert 80 < h.rate_hz < 101
        assert h.transport["dropped"] > 0
        assert h.units_warning is None
        assert rt.pose_at() is not None
    finally:
        rt.stop()
        imu.stop()


# ---------------------------------------------------------------------------
# Units check
# ---------------------------------------------------------------------------


def _samples(accel, gyro=(0.0, 0.0, 0.0), n=30):
    return [ImuSample(i, None, np.array(gyro), np.array(accel), "live") for i in range(n)]


@pytest.mark.parametrize(
    ("accel", "gyro", "needle"),
    [
        ((0.0, -1.0, 0.0), (0, 0, 0), "sends g but m_s2"),  # g sent, m/s² configured
        ((0.0, -96.0, 0.0), (0, 0, 0), "sends m/s² but g"),  # m/s² sent, g configured
        ((0.0, -9.8, 0.0), (90.0, 0, 0), "°/s"),  # degrees sent, rad/s configured
        ((0.0, -30.0, 0.0), (0, 0, 0), "far from 1 g"),
    ],
)
def test_units_warnings(accel, gyro, needle):
    assert needle in check_units(_samples(accel, gyro))


def test_units_ok_at_rest():
    assert check_units(_samples((0.1, -9.8, 0.2), (0.01, 0.0, -0.02))) is None
    assert check_units(_samples((0.0, -1.0, 0.0), n=5)) is None  # too few to judge


# ---------------------------------------------------------------------------
# Overrides, input status, metrics
# ---------------------------------------------------------------------------


def test_parse_overrides():
    out = parse_overrides(
        ["cameras.left_eye.options.host=192.168.1.42", "imu.options.port=6000",
         "hardware_profile.cameras.head.enabled=false"]
    )["hardware_profile"]
    assert out["cameras"]["left_eye"]["options"]["host"] == "192.168.1.42"
    assert out["imu"]["options"]["port"] == 6000
    assert out["cameras"]["head"]["enabled"] is False
    with pytest.raises(ValueError):
        parse_overrides(["no_equals_sign"])


def _cam(cid: str, state: CameraState, fps: float | None = 20.0) -> CameraHealth:
    return CameraHealth(
        camera_id=cid, role=CameraRole.LEFT_EYE, source_type="esp32", source_kind=SourceKind.LIVE,
        state=state, frames_received=100, frames_skipped_by_source=3, frames_overwritten=0,
        partial_frames=0, measured_fps=fps, interval_jitter_ms=2.0, last_frame_age_ms=1500.0,
        transport_delay_ms=4.0, reconnects=1, last_error="stream closed by the device",
        capabilities=None,
    )


def test_input_statuses():
    statuses = input_statuses(
        {"left_eye": _cam("left_eye", CameraState.STREAMING),
         "right_eye": _cam("right_eye", CameraState.STALLED),
         "head": _cam("head", CameraState.RECONNECTING)},
        ImuHealth("stalled", 0, None, None, transport={"subscribes_sent": 4}),
    )
    levels = {s.name: s.level for s in statuses}
    assert levels == {"left_eye": "ok", "right_eye": "warn", "head": "down", "imu": "down"}
    imu = next(s for s in statuses if s.name == "imu")
    assert "firewall" in imu.detail


def test_metrics_recorder_writes_jsonl_and_summary(tmp_path):
    path = tmp_path / "m.jsonl"
    rec = MetricsRecorder(path, interval_s=0.0)
    imu = ImuHealth("streaming", 500, 99.0, None, 3.0, {"packets": 500, "dropped": 4})
    for state in (CameraState.STREAMING, CameraState.STALLED, CameraState.STREAMING):
        assert rec.record({"left_eye": _cam("left_eye", state)}, imu)
    rec.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r.get("input") for r in rows[:-1]} == {"left_eye", "imu"}
    summary = rows[-1]["summary"]
    assert summary["left_eye"]["skipped"] == 3 and summary["left_eye"]["reconnects"] == 1
    assert summary["imu"]["dropped"] == 4 and summary["imu"]["mean_rate"] == 99.0
    assert "image" not in path.read_text()  # numbers only
