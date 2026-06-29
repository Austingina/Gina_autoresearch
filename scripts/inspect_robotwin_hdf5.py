#!/usr/bin/env python3
"""Inspect RoboTwin trajectory HDF5 files before writing an importer.

RoboTwin stores images as encoded byte strings, while AutoResearch expects
decoded mask/depth arrays under its own demo schema. This script prints the
HDF5 tree and optionally decodes image-like datasets to verify their shapes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _is_encoded_image_dataset(name: str, dataset: h5py.Dataset) -> bool:
    return "rgb" in name.lower() and dataset.dtype.kind in {"S", "O"}


def _decode_images(data: np.ndarray) -> np.ndarray:
    import cv2

    frames = []
    for buf in data.ravel():
        if isinstance(buf, np.ndarray):
            raw = buf
        else:
            raw = np.frombuffer(bytes(buf).rstrip(b"\0"), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes")
        frames.append(image)
    return np.stack(frames, axis=0)


def _walk(node: h5py.Group, prefix: str = "") -> None:
    for name, item in node.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Group):
            print(f"{path}/")
            _walk(item, path)
        else:
            print(f"{path}: shape={item.shape} dtype={item.dtype}")


def inspect(path: Path, *, decode_images: bool) -> None:
    with h5py.File(path, "r") as f:
        print(f"file: {path}")
        print("attrs:", dict(f.attrs))
        _walk(f)
        if not decode_images:
            return
        print("\ndecoded image datasets:")
        def visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset) and _is_encoded_image_dataset(name, obj):
                images = _decode_images(obj[()])
                print(f"{name}: decoded_shape={images.shape} dtype={images.dtype}")

        f.visititems(visit)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect RoboTwin HDF5 trajectory structure")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--decode-images", action="store_true")
    args = parser.parse_args()

    for path in args.paths:
        inspect(path, decode_images=args.decode_images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
