#!/usr/bin/env python3
"""Software stand-in for a WIRED OpenGaze camera (NOT hardware validation).
Creates a pseudo-terminal, prints its path, and writes the same framed protocol the *_serial
firmware emits, throttled to the chosen baud rate, with boot text, noise and corrupted frames.

  python tools/sim_serial_device.py --role left      # prints e.g. /dev/pts/5
  python main.py --left-serial /dev/pts/5 --right-serial /dev/pts/6
"""
import argparse, binascii, math, os, pty, random, struct, sys, threading, time, tty
import cv2, numpy as np

HDR = struct.Struct("<HBBBBIQ"); IMU = struct.Struct("<7f"); STATUS = struct.Struct("<IIIIIHHHHbBBB")
BOOT = time.monotonic() - random.uniform(5, 5000)
def dev_us(): return int((time.monotonic() - BOOT) * 1e6)

def frame(typ, payload):
    body = bytes([typ]) + struct.pack("<I", len(payload)) + payload
    return b"\xA5\x5A" + body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["left", "right"], required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--corrupt", type=float, default=0.02, help="probability a JPEG frame is corrupted")
    ap.add_argument("--noise", type=float, default=0.01, help="probability of random noise before a frame")
    ap.add_argument("--quality", type=int, default=50)
    A = ap.parse_args()
    master, slave = pty.openpty(); tty.setraw(slave)
    print(os.ttyname(slave), flush=True)
    lock = threading.Lock(); bps = A.baud / 10.0
    q = []   # IMU/status frames waiting for the single writer, like the firmware's queue
    def write(b):
        with lock:
            os.write(master, b)
    def throttled(b):
        t0 = time.monotonic()
        mv = memoryview(b)
        while mv:
            n = os.write(master, mv[:1024]); mv = mv[n:]
        time.sleep(max(0, len(b) / bps - (time.monotonic() - t0)))   # emulate wire time
    dev = 1 if A.role == "left" else 2
    write(b"ets Jul 29 2019 12:21:46\r\nrst:0x1 (POWERON_RESET)\r\n=== OpenGaze sim ===\r\n[cam] sensor PID=0x3660\r\n")
    def imu_status():
        seq_i = seq_s = 0; t = 0.0; nxt = time.monotonic(); last_s = 0
        while True:
            if dev == 1:
                pkt = HDR.pack(0x474F, 1, 1, 1, 1, seq_i, dev_us()) + IMU.pack(0.02*math.sin(t), 0.01, 1.0, 3*math.sin(2*t), 2*math.cos(t), 0.1, 31.5)
                q.append(frame(2, pkt))
                seq_i += 1; t += 0.01
            if time.monotonic() - last_s > 1:
                last_s = time.monotonic()
                st = STATUS.pack(int((last_s - BOOT) * 1000) & 0xFFFFFFFF, 150000, 3_900_000, 0, 0, int(A.fps*10), 1000 if dev == 1 else 0, 0, 0, 0, 1, 0, 1)
                q.append(frame(3, HDR.pack(0x474F, 1, 2, dev, 1, seq_s, dev_us()) + st)); seq_s += 1
            nxt += 0.01; time.sleep(max(0, nxt - time.monotonic()))
    threading.Thread(target=imu_status, daemon=True).start()
    fid, period, nxt = 0, 1.0 / A.fps, time.monotonic()
    while True:
        ts = dev_us()
        img = np.full((240, 320, 3), 40, np.uint8)
        cx = int(160 + 90*math.sin(fid/15)); cy = int(120 + 50*math.cos(fid/11))
        cv2.circle(img, (cx, cy), 25, (200, 200, 200), -1); cv2.circle(img, (cx, cy), 10, (0, 0, 0), -1)
        cv2.putText(img, f"WIRED SIM {A.role} #{fid}", (8, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, A.quality])[1].tobytes()
        data = bytearray(frame(1, struct.pack("<QI", ts, fid) + jpg))
        if random.random() < A.noise: write(bytes(random.getrandbits(8) for _ in range(random.randint(1, 40))))
        if random.random() < A.corrupt: data[random.randint(12, len(data) - 3)] ^= 0xFF
        while q: write(q.pop(0))
        throttled(bytes(data)); fid += 1; nxt += period
        while q: write(q.pop(0))
        time.sleep(max(0, nxt - time.monotonic()))

if __name__ == "__main__":
    main()
