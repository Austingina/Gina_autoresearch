#!/usr/bin/env python3
"""Export separate RoboTwin camera videos from a collected HDF5 episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


CAMERA_OUTPUTS = {
    "head_camera": "top_camera",
    "left_camera": "left_wrist_camera",
    "right_camera": "right_wrist_camera",
}


def _decode_rgb(data: np.ndarray) -> np.ndarray:
    import cv2

    frames = []
    for buf in data.ravel():
        raw = np.frombuffer(bytes(buf).rstrip(b"\0"), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode RoboTwin RGB frame")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def _write_video(frames: np.ndarray, out_path: Path, fps: float) -> None:
    import imageio.v2 as imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[OK] {out_path} ({len(frames)} frames)")


def export_videos(hdf5_path: Path, *, output_dir: Path | None, fps: float) -> None:
    output_dir = output_dir or hdf5_path.parent.parent / "video"
    stem = hdf5_path.stem
    with h5py.File(hdf5_path, "r") as f:
        observations = f["observation"]
        for camera_name, output_name in CAMERA_OUTPUTS.items():
            if camera_name not in observations or "rgb" not in observations[camera_name]:
                print(f"[skip] {camera_name}/rgb not found")
                continue
            frames = _decode_rgb(np.asarray(observations[camera_name]["rgb"]))
            _write_video(frames, output_dir / f"{stem}_{output_name}.mp4", fps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export top and wrist camera videos from RoboTwin HDF5")
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    export_videos(args.hdf5, output_dir=args.output_dir, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
