"""Dashboard rendering + CSV logging for the OpenGaze hardware receiver."""
from __future__ import annotations
import csv
import time
from typing import Optional

import cv2
import numpy as np

TILE_W, TILE_H = 480, 360
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, YELLOW, RED, WHITE, GRAY = (80, 220, 80), (0, 220, 255), (60, 60, 255), (240, 240, 240), (150, 150, 150)


def put(img, text, org, color=WHITE, scale=0.5, thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _fmt(v: Optional[float], spec="{:.1f}", none="--"):
    return none if v is None else spec.format(v)


def video_tile(label: str, client, ) -> np.ndarray:
    tile = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    f = client.get_latest()
    if f is not None and client.frame_age_s < 2.0:
        h, w = f.img.shape[:2]
        s = min(TILE_W / w, TILE_H / h)
        r = cv2.resize(f.img, (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST)
        y0, x0 = (TILE_H - r.shape[0]) // 2, (TILE_W - r.shape[1]) // 2
        tile[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    else:
        put(tile, "NO SIGNAL", (TILE_W // 2 - 60, TILE_H // 2), RED, 0.8, 2)
    col = GREEN if client.state == "streaming" and client.fps > 5 else (YELLOW if client.state != "stopped" else RED)
    put(tile, f"{label}  [{client.state}]", (8, 20), col, 0.55, 1)
    put(tile, f"{client.fps:5.1f} fps  {client.resolution[0]}x{client.resolution[1]}", (8, 40), WHITE, 0.5)
    return tile


def render(left, right, tel, loop_ms: float, aligned_note: str = "") -> np.ndarray:
    top = np.hstack([video_tile("LEFT eye", left), video_tile("RIGHT eye", right)])
    panel = np.zeros((250, top.shape[1], 3), np.uint8)
    d1, d2 = tel.dev[1], tel.dev[2]
    s = tel.last_imu

    # --- IMU column
    put(panel, "MPU-6050 (left_cam)", (10, 20), GREEN if tel.connected(1) else RED, 0.55)
    if s is not None:
        put(panel, f"accel [g]    x {s.ax:+7.3f}  y {s.ay:+7.3f}  z {s.az:+7.3f}", (10, 45))
        put(panel, f"gyro [deg/s] x {s.gx:+7.2f}  y {s.gy:+7.2f}  z {s.gz:+7.2f}", (10, 65))
        put(panel, f"temp {s.temp_c:5.1f} C   valid={s.valid}   seq={s.seq}", (10, 85))
    else:
        put(panel, "no IMU packets yet", (10, 45), YELLOW)
    put(panel, f"rate {tel.rate_hz(1):6.1f} Hz   loss {tel.loss_pct(1):5.2f}%  (lost {d1.lost}, reord {d1.reordered})", (10, 110))
    put(panel, f"packet age {_fmt(tel.age_s(1) * 1e3 if tel.age_s(1) != float('inf') else None, '{:.0f}')} ms   "
               f"rel.delay {_fmt(d1.rel_delay_ms)} ms   range_err {d1.range_errors}  bad {d1.bad_packets}", (10, 130))

    # --- per-device health (from UDP status packets) + video
    y = 160
    for label, d, cl, dev in (("LEFT ", d1, left, 1), ("RIGHT", d2, right, 2)):
        st = d.status
        line = f"{label} link {'OK ' if tel.connected(dev) else 'LOST'}"
        if st:
            line += (f" rssi {st['rssi_dbm']}dBm heap {st['free_heap'] // 1024}K psram {st['free_psram'] // 1024}K "
                     f"wifi_reconn {st['wifi_reconnects']} capfail {st['capture_fail']} rst={st['reset_reason']}"
                     f"{' BROWNOUT!' if st['brownout_seen'] else ''}")
        put(panel, line, (10, y), RED if (st and st['brownout_seen']) else WHITE, 0.45)
        y += 18
        vid = (f"      video {cl.fps:4.1f}fps  rel.delay {_fmt(cl.rel_delay_ms)}ms  decode {cl.decode_ms:4.1f}ms  "
               f"stalls {cl.stalls}  overwritten {cl.overwritten}  decode_fail {cl.decode_fail}  reconn {cl.reconnects} fail {cl.failures}"
               + (f"  crc_err {cl.crc_errors}" if hasattr(cl, "crc_errors") else ""))
        put(panel, vid, (10, y), WHITE, 0.45)
        y += 18
        if cl.last_error and cl.state != "streaming":
            put(panel, f"      {cl.last_error[:100]}", (10, y), YELLOW, 0.42)
        y += 18
    put(panel, f"UDP socket: {tel.sock_error or 'ok'}   main-loop {loop_ms:.1f} ms   {aligned_note}", (10, y + 4), GRAY, 0.45)
    return np.vstack([top, panel])


class CsvLogger:
    FIELDS = ["t_s", "left_fps", "right_fps", "left_rel_delay_ms", "right_rel_delay_ms", "imu_hz", "imu_loss_pct",
              "left_stalls", "right_stalls", "left_reconn", "right_reconn", "left_rssi", "right_rssi",
              "left_heap", "right_heap", "loop_ms"]

    def __init__(self, path: str):
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, self.FIELDS)
        self._w.writeheader()
        self._t0 = time.monotonic()

    def log(self, left, right, tel, loop_ms: float):
        s1, s2 = tel.dev[1].status or {}, tel.dev[2].status or {}
        self._w.writerow({
            "t_s": round(time.monotonic() - self._t0, 2), "left_fps": round(left.fps, 2), "right_fps": round(right.fps, 2),
            "left_rel_delay_ms": _fmt(left.rel_delay_ms, "{:.1f}", ""), "right_rel_delay_ms": _fmt(right.rel_delay_ms, "{:.1f}", ""),
            "imu_hz": round(tel.rate_hz(1), 2), "imu_loss_pct": round(tel.loss_pct(1), 3),
            "left_stalls": left.stalls, "right_stalls": right.stalls, "left_reconn": left.reconnects, "right_reconn": right.reconnects,
            "left_rssi": s1.get("rssi_dbm", ""), "right_rssi": s2.get("rssi_dbm", ""),
            "left_heap": s1.get("free_heap", ""), "right_heap": s2.get("free_heap", ""), "loop_ms": round(loop_ms, 2)})
        self._f.flush()

    def close(self):
        self._f.close()
