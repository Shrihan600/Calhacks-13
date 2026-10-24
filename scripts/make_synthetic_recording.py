"""Render a synthetic eye video (+ ground-truth CSV) for the `replay` hardware profile.

    python scripts/make_synthetic_recording.py [--seconds 20] [--out demo/synthetic_eye.avi]

The output contains generated images only, no real eyes, so it's safe to share.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

import cv2

from opengaze.cameras.simulated import SyntheticCameraSource
from opengaze.config.schema import CameraConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


async def render(out: Path, seconds: float, fps: float, motion: str) -> int:
    config = CameraConfig(
        camera_id="left_eye",
        role="left_eye",
        source_type="synthetic",
        resolution=(320, 240),
        fps=fps,
        options={"pattern": "eye", "motion": motion, "realtime": False, "noise_std": 5.0},
    )
    source = SyntheticCameraSource(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    if not writer.isOpened():
        raise SystemExit(f"OpenCV could not open a video writer for {out}")
    count = int(seconds * fps)
    with out.with_suffix(".csv").open("w", newline="") as fh:
        rows = None
        async with source:
            for _ in range(count):
                frame = await source.read_frame()
                writer.write(frame.image)
                truth = {"frame": frame.sequence, **(frame.ground_truth or {})}
                if rows is None:
                    rows = csv.DictWriter(fh, fieldnames=list(truth))
                    rows.writeheader()
                rows.writerow(truth)
    writer.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "demo" / "synthetic_eye.avi")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--motion", choices=["lissajous", "saccades", "static"], default="saccades")
    args = parser.parse_args()
    count = asyncio.run(render(args.out, args.seconds, args.fps, args.motion))
    print(f"wrote {count} frames to {args.out} (ground truth: {args.out.with_suffix('.csv')})")


if __name__ == "__main__":
    main()
