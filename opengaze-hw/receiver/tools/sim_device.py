#!/usr/bin/env python3
"""Software stand-in for an OpenGaze ESP32-CAM (NOT hardware validation).

Reproduces the firmware's wire format: chunked HTTP multipart MJPEG with X-Timestamp-Us /
X-Frame-Id headers on /stream, and binary UDP IMU (left only) + status packets.

  python tools/sim_device.py --role left  --http-port 8081 &
  python tools/sim_device.py --role right --http-port 8082 &
  python main.py --left 127.0.0.1:8081 --right 127.0.0.1:8082
"""
import argparse, math, random, socket, struct, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

HDR = struct.Struct("<HBBBBIQ"); IMU = struct.Struct("<7f"); STATUS = struct.Struct("<IIIIIHHHHbBBB")
BOOT = time.monotonic() - random.uniform(5, 5000)          # each sim device has its own unrelated clock
def dev_us(): return int((time.monotonic() - BOOT) * 1e6)
A = None

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def chunk(self, b): self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")
    def do_GET(self):
        if self.path != "/stream":
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=ogframe")
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        fid, period, nxt = 0, 1.0 / A.fps, time.monotonic()
        try:
            while True:
                ts = dev_us()
                img = np.full((240, 320, 3), 40, np.uint8)
                cx = int(160 + 90 * math.sin(fid / 15)); cy = int(120 + 50 * math.cos(fid / 11))
                cv2.circle(img, (cx, cy), 25, (200, 200, 200), -1); cv2.circle(img, (cx, cy), 10, (0, 0, 0), -1)
                cv2.putText(img, f"SIM {A.role} #{fid}", (8, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                part = (f"\r\n--ogframe\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n"
                        f"X-Timestamp-Us: {ts}\r\nX-Frame-Id: {fid}\r\n\r\n").encode()
                self.chunk(part); self.chunk(jpg)           # two chunks per frame, like esp_http_server
                fid += 1; nxt += period
                time.sleep(max(0, nxt - time.monotonic() + random.uniform(0, A.jitter)))
        except (BrokenPipeError, ConnectionResetError):
            pass

def udp_loop():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); dev = 1 if A.role == "left" else 2
    seq_i = seq_s = 0; t = 0.0; nxt = time.monotonic(); last_status = 0
    while True:
        now = time.monotonic()
        if dev == 1:
            ax, ay, az = 0.02 * math.sin(t), 0.01 * math.cos(t * 1.3), 1.0 + 0.005 * math.sin(t * 5)
            gx, gy, gz = 3 * math.sin(t * 2), 2 * math.cos(t), 0.2 * math.sin(t * 7)
            pkt = HDR.pack(0x474F, 1, 1, 1, 1, seq_i, dev_us()) + IMU.pack(ax, ay, az, gx, gy, gz, 31.5)
            seq_i += 1
            if random.random() >= A.loss: s.sendto(pkt, (A.udp_host, A.udp_port))
            t += 0.01
        if now - last_status > 1.0:
            last_status = now
            st = STATUS.pack(int((now - BOOT) * 1000) & 0xFFFFFFFF, 150000, 3_900_000, 0, 0, int(A.fps * 10), 1000 if dev == 1 else 0, 0, 0, -58, 1, 0, 1)
            s.sendto(HDR.pack(0x474F, 1, 2, dev, 1, seq_s, dev_us()) + st, (A.udp_host, A.udp_port)); seq_s += 1
        nxt += 0.01; time.sleep(max(0, nxt - time.monotonic()))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["left", "right"], required=True)
    p.add_argument("--http-port", type=int, default=8081)
    p.add_argument("--udp-host", default="127.0.0.1"); p.add_argument("--udp-port", type=int, default=5005)
    p.add_argument("--fps", type=float, default=25); p.add_argument("--loss", type=float, default=0.0, help="IMU packet drop prob")
    p.add_argument("--jitter", type=float, default=0.004, help="max random extra delay per frame (s)")
    A = p.parse_args()
    threading.Thread(target=udp_loop, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", A.http_port), H).serve_forever()
