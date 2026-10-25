"""UDP receiver for OpenGaze IMU + status packets (protocol: firmware/include/telemetry_protocol.h)."""
from __future__ import annotations
import math
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from synchronization import ClockTracker

MAGIC, VERSION = 0x474F, 1
T_IMU, T_STATUS = 1, 2
HDR = struct.Struct("<HBBBBIQ")          # 18 bytes
IMU = struct.Struct("<7f")               # 28 bytes
STATUS = struct.Struct("<IIIIIHHHHbBBB") # 32 bytes
assert HDR.size == 18 and IMU.size == 28 and STATUS.size == 32

DEVICE_NAMES = {1: "left_cam", 2: "right_cam"}
RESET_REASONS = {0: "unknown", 1: "power-on", 3: "software", 4: "panic", 5: "int-wdt", 6: "task-wdt",
                 7: "wdt", 8: "deepsleep", 9: "brownout", 10: "sdio"}
MAX_ACCEL_G, MAX_GYRO_DPS = 16.0, 2000.0  # sanity limits (configured full-scale is +/-4 g, +/-500 dps)


@dataclass
class ImuSample:
    dev_us: int
    host_ts: float
    seq: int
    ax: float; ay: float; az: float
    gx: float; gy: float; gz: float
    temp_c: float
    valid: bool

    def as_dict(self, device="left_cam"):
        return {"device_id": device, "sequence": self.seq, "timestamp_us": self.dev_us,
                "ax": self.ax, "ay": self.ay, "az": self.az, "gx": self.gx, "gy": self.gy, "gz": self.gz,
                "temperature_c": self.temp_c, "valid": self.valid}


@dataclass
class DevState:
    packets: int = 0
    lost: int = 0
    reordered: int = 0
    range_errors: int = 0
    bad_packets: int = 0
    last_seq: Optional[int] = None
    last_host_ts: float = 0.0
    last_dev_us: int = 0
    arrivals: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    status: Optional[dict] = None
    status_host_ts: float = 0.0
    clock: ClockTracker = field(default_factory=ClockTracker)
    rel_delay_ms: Optional[float] = None


class TelemetryReceiver(threading.Thread):
    def __init__(self, port: int = 5005, bind: str = "0.0.0.0", buffer: int = 4000):
        super().__init__(daemon=True, name="udp-telemetry")
        self.port, self.bind = port, bind
        self._stop_evt = threading.Event()
        self.imu: Deque[ImuSample] = deque(maxlen=buffer)       # LEFT device samples
        self.last_imu: Optional[ImuSample] = None
        self.dev: Dict[int, DevState] = {1: DevState(), 2: DevState()}
        self.sock_error = ""
        self._seq_state_imu: Optional[int] = None
        self._ingest_lock = threading.Lock()

    def stop(self):
        self._stop_evt.set()

    def ingest(self, data: bytes, now: Optional[float] = None):
        """Feed one telemetry packet that arrived by some other transport (e.g. the wired serial link)."""
        with self._ingest_lock:
            self._handle_packet(data, time.monotonic() if now is None else now)

    # ---------------------------------------------------------------- derived stats
    def rate_hz(self, dev_id: int = 1) -> float:
        a = self.dev[dev_id].arrivals
        if len(a) < 5 or time.monotonic() - a[-1] > 1.5:
            return 0.0
        return (len(a) - 1) / (a[-1] - a[0]) if a[-1] > a[0] else 0.0

    def age_s(self, dev_id: int = 1) -> float:
        t = self.dev[dev_id].last_host_ts
        return time.monotonic() - t if t else float("inf")

    def connected(self, dev_id: int = 1) -> bool:
        d = self.dev[dev_id]
        last = max(d.last_host_ts, d.status_host_ts)
        return bool(last) and (time.monotonic() - last) < 2.0

    def loss_pct(self, dev_id: int = 1) -> float:
        d = self.dev[dev_id]
        tot = d.packets + d.lost
        return 100.0 * d.lost / tot if tot else 0.0

    # ---------------------------------------------------------------- thread
    def run(self):
        s = None
        while not self._stop_evt.is_set():
            try:
                if s is None:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
                    s.bind((self.bind, self.port))
                    s.settimeout(0.5)
                    self.sock_error = ""
                data, _addr = s.recvfrom(512)
            except socket.timeout:
                continue
            except OSError as e:
                self.sock_error = str(e)
                if s:
                    s.close(); s = None
                self._stop_evt.wait(1.0)
                continue
            self._handle_packet(data, time.monotonic())
        if s:
            s.close()

    def _handle_packet(self, data: bytes, now: float):
        if len(data) < HDR.size:
            self.dev[1].bad_packets += 1
            return
        magic, ver, typ, dev_id, flags, seq, dev_us = HDR.unpack_from(data, 0)
        if magic != MAGIC or ver != VERSION or dev_id not in self.dev:
            self.dev[1].bad_packets += 1
            return
        d = self.dev[dev_id]
        body = data[HDR.size:]
        if typ == T_IMU and len(body) == IMU.size:
            ax, ay, az, gx, gy, gz, t = IMU.unpack(body)
            valid = bool(flags & 1)
            if valid and (any(math.isnan(v) or math.isinf(v) for v in (ax, ay, az, gx, gy, gz, t))
                          or max(abs(ax), abs(ay), abs(az)) > MAX_ACCEL_G
                          or max(abs(gx), abs(gy), abs(gz)) > MAX_GYRO_DPS):
                d.range_errors += 1
                valid = False
            self._track_seq(d, seq)
            d.packets += 1
            d.last_host_ts, d.last_dev_us = now, dev_us
            d.arrivals.append(now)
            d.clock.update(dev_us, now)
            d.rel_delay_ms = d.clock.rel_delay_ms(dev_us, now)
            smp = ImuSample(dev_us, now, seq, ax, ay, az, gx, gy, gz, t, valid)
            self.last_imu = smp if dev_id == 1 else self.last_imu
            if dev_id == 1:
                self.imu.append(smp)
        elif typ == T_STATUS and len(body) == STATUS.size:
            (up, heap, psram, frames, capfail, fps10, imu10, reconn, imuerr,
             rssi, rr, bo, clients) = STATUS.unpack(body)
            d.status = {"uptime_ms": up, "free_heap": heap, "free_psram": psram, "frames_sent": frames,
                        "capture_fail": capfail, "fps": fps10 / 10.0, "imu_hz": imu10 / 10.0,
                        "wifi_reconnects": reconn, "imu_errors": imuerr, "rssi_dbm": rssi,
                        "reset_reason": RESET_REASONS.get(rr, str(rr)), "brownout_seen": bool(bo),
                        "stream_clients": clients}
            d.status_host_ts = now
            d.clock.update(dev_us, now)   # status packets also feed the clock estimate
        else:
            d.bad_packets += 1

    @staticmethod
    def _track_seq(d: DevState, seq: int):
        if d.last_seq is None:
            d.last_seq = seq
            return
        diff = (seq - d.last_seq) & 0xFFFFFFFF
        if diff == 0:
            return
        if diff < 0x80000000:
            d.lost += diff - 1
            d.last_seq = seq
        else:
            d.reordered += 1   # late/duplicate packet; already counted as lost, undo one
            d.lost = max(0, d.lost - 1)
