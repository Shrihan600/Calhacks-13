#!/usr/bin/env python3
"""OpenGaze hardware receiver: dual ESP32-CAM MJPEG streams + MPU-6050 UDP telemetry.

  python main.py --left opengaze-left.local --right opengaze-right.local
  python main.py --left 192.168.4.21 --right 192.168.4.22 --log-csv run1.csv
  python main.py --headless --duration 60 ...     # no window; prints a summary (used for soak tests)

Keys (window focused): q quit | s save snapshots of both eyes
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import cv2

from camera_client import MJPEGClient
from serial_client import SerialCamera
from diagnostics import CsvLogger, render
from imu_client import TelemetryReceiver
from synchronization import ImuAligner


def split_hostport(s: str, default_port: int):
    if ":" in s:
        h, p = s.rsplit(":", 1)
        return h, int(p)
    return s, default_port


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", default="opengaze-left.local", help="host or host:port of LEFT camera (stream port default 81)")
    ap.add_argument("--right", default="opengaze-right.local", help="host or host:port of RIGHT camera")
    ap.add_argument("--left-serial", default="", help="WIRED: serial port of the LEFT camera bridge, e.g. /dev/cu.usbmodem2101")
    ap.add_argument("--right-serial", default="", help="WIRED: serial port of the RIGHT camera bridge")
    ap.add_argument("--baud", type=int, default=921600, help="WIRED: must match SERIAL_BAUD in the firmware")
    ap.add_argument("--stream-port", type=int, default=81)
    ap.add_argument("--udp-port", type=int, default=5005)
    ap.add_argument("--headless", action="store_true", help="no GUI window; print periodic summaries")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = run until quit)")
    ap.add_argument("--log-csv", default="", help="write 1 Hz diagnostics to this CSV")
    ap.add_argument("--snap-dir", default="snapshots")
    a = ap.parse_args()

    lh, lp = split_hostport(a.left, a.stream_port)
    rh, rp = split_hostport(a.right, a.stream_port)
    tel = TelemetryReceiver(port=a.udp_port)
    left = SerialCamera("left", a.left_serial, a.baud, tel) if a.left_serial else MJPEGClient("left", lh, lp)
    right = SerialCamera("right", a.right_serial, a.baud, tel) if a.right_serial else MJPEGClient("right", rh, rp)
    aligner = ImuAligner(tel.imu, tel.dev[1].clock, tel.dev[2].clock)
    for t in (tel, left, right):
        t.start()
    logger = CsvLogger(a.log_csv) if a.log_csv else None

    t_start = time.monotonic()
    last_log = last_print = 0.0
    loop_ms = 0.0
    try:
        while True:
            t0 = time.perf_counter()
            now = time.monotonic()
            note = ""
            lf = left.get_latest()
            if lf is not None and lf.dev_ts_us:
                smp, dt = aligner.near_left_frame(lf.dev_ts_us)
                if smp is not None:
                    note = f"IMU<->L-frame dt {dt:+.1f} ms (same clock)"
            if not a.headless:
                canvas = render(left, right, tel, loop_ms, note)
                cv2.imshow("OpenGaze hardware receiver", canvas)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("s"):
                    os.makedirs(a.snap_dir, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    for nm, c in (("left", left), ("right", right)):
                        fr = c.get_latest()
                        if fr is not None:
                            cv2.imwrite(os.path.join(a.snap_dir, f"{stamp}_{nm}.png"), fr.img)
                    print("snapshots saved to", a.snap_dir)
                time.sleep(0.005)
            else:
                right.get_latest()      # consume like the GUI would, so 'overwritten' stays meaningful
                time.sleep(0.02)
            loop_ms = 0.9 * loop_ms + 0.1 * (time.perf_counter() - t0) * 1e3
            if logger and now - last_log >= 1.0:
                logger.log(left, right, tel, loop_ms)
                last_log = now
            if a.headless and now - last_print >= 2.0:
                last_print = now
                print(f"[{now - t_start:6.1f}s] L {left.state:<10} {left.fps:5.1f}fps relDelay {left.rel_delay_ms or 0:5.1f}ms | "
                      f"R {right.state:<10} {right.fps:5.1f}fps | IMU {tel.rate_hz(1):6.1f}Hz loss {tel.loss_pct(1):.2f}% "
                      f"| link L:{tel.connected(1)} R:{tel.connected(2)}", flush=True)
            if a.duration and now - t_start >= a.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for t in (left, right, tel):
            t.stop()
        if logger:
            logger.close()
        cv2.destroyAllWindows()

    d = tel.dev[1]
    print("\n===== summary =====")
    for c in (left, right):
        print(f"{c.label:5s}: frames={c.frames_ok} fps_now={c.fps:.1f} reconnects={c.reconnects} failures={c.failures} stalls={c.stalls} "
              f"overwritten={c.overwritten} decode_fail={c.decode_fail} bytes={c.bytes_total / 1e6:.2f}MB last_err='{c.last_error}'"
              + (f" crc_errors={c.crc_errors} garbage_bytes={c.garbage_bytes}" if hasattr(c, "crc_errors") else ""))
    print(f"IMU  : packets={d.packets} lost={d.lost} reordered={d.reordered} loss={tel.loss_pct(1):.3f}% "
          f"range_err={d.range_errors} bad={d.bad_packets} rate_now={tel.rate_hz(1):.1f}Hz")
    for c in (left, right):
        if hasattr(c, "log_tail") and c.log_tail:
            print(f"-- last device text from {c.label}:")
            for ln in list(c.log_tail)[-6:]:
                print("   ", ln)
    ok = left.frames_ok > 0 and right.frames_ok > 0 and d.packets > 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
