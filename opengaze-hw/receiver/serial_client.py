"""Wired transport client: reads OpenGaze frames from a USB-serial bridge.

Frame format (see firmware/include/serial_link.h):
    A5 5A | type u8 | len u32 LE | payload | crc16 LE   (CRC-16/CCITT-FALSE over type,len,payload)
    type 1 JPEG   payload = capture_us u64 | frame_id u32 | jpeg
    type 2 IMU    payload = 46-byte telemetry packet   -> TelemetryReceiver.ingest()
    type 3 STATUS payload = 50-byte telemetry packet   -> TelemetryReceiver.ingest()
Bytes that are not part of a valid frame (ESP32 boot text, noise) are skipped, so the receiver
re-synchronises on its own after unplugging, resets or corruption.
SerialCamera subclasses MJPEGClient so main.py / diagnostics treat it identically.
"""
from __future__ import annotations
import binascii
import struct
import time
from collections import deque

import serial

from camera_client import MJPEGClient

SYNC = b"\xA5\x5A"
SL_JPEG, SL_IMU, SL_STATUS = 1, 2, 3
MAX_PAYLOAD = 400_000


class SerialCamera(MJPEGClient):
    def __init__(self, name: str, port: str, baud: int = 921600, telemetry=None):
        super().__init__(name, host=port, port=0)
        self.port_name, self.baud, self.tel = port, baud, telemetry
        self.crc_errors = 0
        self.garbage_bytes = 0
        self.log_tail = deque(maxlen=30)       # recent non-frame text from the device (boot messages etc.)
        self._text = bytearray()
        self._last_good = 0.0

    # ---------------------------------------------------------------- thread
    def run(self):
        while not self._stop_evt.is_set():
            ser = None
            try:
                self.state = "connecting"
                ser = serial.Serial(self.port_name, self.baud, timeout=0.2)
                self.last_error = ""
                buf = bytearray()
                self._last_good = time.monotonic()
                while not self._stop_evt.is_set():
                    chunk = ser.read(max(1, ser.in_waiting))
                    if chunk:
                        buf += chunk
                        self._consume(buf)
                    if self.state == "streaming" and time.monotonic() - self._last_good > 3.0:
                        self.state = "waiting"
            except (serial.SerialException, OSError) as e:
                self.last_error = f"{type(e).__name__}: {e}"
            finally:
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass
            if self._stop_evt.is_set():
                break
            self.state = "reconnecting"
            self.failures += 1
            self._stop_evt.wait(1.0)
        self.state = "stopped"

    # ---------------------------------------------------------------- parser
    def _consume(self, buf: bytearray):
        i, n = 0, len(buf)
        while True:
            j = buf.find(SYNC, i)
            if j < 0:
                keep = 1 if buf.endswith(b"\xA5") else 0
                self._garbage(buf[i:n - keep])
                del buf[:n - keep]
                return
            if j > i:
                self._garbage(buf[i:j])
            if n - j < 7:
                del buf[:j]
                return
            typ = buf[j + 2]
            ln = int.from_bytes(buf[j + 3:j + 7], "little")
            if typ not in (SL_JPEG, SL_IMU, SL_STATUS) or ln > MAX_PAYLOAD:
                self._garbage(buf[j:j + 1])      # false sync pattern inside data
                i = j + 1
                continue
            total = 7 + ln + 2
            if n - j < total:
                del buf[:j]
                return
            body = bytes(buf[j + 2:j + 7 + ln])
            crc = int.from_bytes(buf[j + 7 + ln:j + total], "little")
            if binascii.crc_hqx(body, 0xFFFF) != crc:
                self.crc_errors += 1
                i = j + 1                        # rescan: the real next frame may start inside this one
                continue
            self._handle_packet(typ, body[5:], time.monotonic())
            i = j + total

    def _handle_packet(self, typ: int, payload: bytes, now: float):
        self._last_good = now
        if typ == SL_JPEG:
            if len(payload) <= 12:
                return
            ts, fid = struct.unpack_from("<QI", payload, 0)
            self.state = "streaming"
            self._on_jpeg(payload[12:], ts, fid, now)
        elif self.tel is not None:
            self.tel.ingest(payload, now)

    def _garbage(self, b):
        if not b:
            return
        self.garbage_bytes += len(b)
        self._text += bytes(b)
        while b"\n" in self._text:
            line, _, rest = self._text.partition(b"\n")
            self._text = bytearray(rest)
            good = sum(1 for c in line if 32 <= c < 127 or c in (9, 13))
            if len(line) >= 4 and good >= 0.95 * len(line):          # real text only, not binary noise
                self.log_tail.append(line.decode("ascii", "ignore").strip())
        if len(self._text) > 4096:
            self._text = bytearray()
