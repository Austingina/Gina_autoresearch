#!/usr/bin/env python3
"""Inspect current RoboTwin/XPolicyLab trajectory HDF5 payloads.

The current writer stores encoded RGB under ``vision/*/colors``.  Decoding is
always delegated to RoboTwin's pinned ``decode_image_bit`` implementation so
legacy and marker-stamped JPEG streams both return RGB correctly.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np


def _is_encoded_image_dataset(name: str, dataset: h5py.Dataset) -> bool:
    parts = name.strip("/").split("/")
    return (
        len(parts) >= 3
        and parts[0] == "vision"
        and parts[-1] == "colors"
        and dataset.dtype.kind in {"S", "O", "U"}
    )


def _load_decoder(robotwin_root: Path) -> Callable[[object], np.ndarray]:
    codec_path = robotwin_root / "data/decode_image_bit.py"
    spec = importlib.util.spec_from_file_location("pinned_robotwin_image_codec", codec_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load RoboTwin codec: {codec_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_image_bit


def _walk(node: h5py.Group, prefix: str = "") -> None:
    for name, item in node.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Group):
            print(f"{path}/")
            _walk(item, path)
        else:
            print(f"{path}: shape={item.shape} dtype={item.dtype}")


def inspect(path: Path, *, decode_images: bool, robotwin_root: Path | None) -> None:
    decoder = None
    if decode_images:
        if robotwin_root is None:
            raise ValueError("--robotwin-root is required with --decode-images")
        decoder = _load_decoder(robotwin_root)

    with h5py.File(path, "r") as f:
        print(f"file: {path}")
        print("attrs:", dict(f.attrs))
        _walk(f)
        if not decode_images:
            return
        print("\ndecoded image datasets:")

        found = []

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and _is_encoded_image_dataset(name, obj):
                assert decoder is not None
                images = np.asarray(decoder(obj[()]))
                if images.ndim != 4 or images.shape[-1] != 3:
                    raise ValueError(f"unexpected decoded image shape for {name}: {images.shape}")
                found.append(name)
                print(
                    f"{name}: decoded_shape={images.shape} dtype={images.dtype} "
                    f"min={int(images.min())} max={int(images.max())} std={float(images.std()):.6f}"
                )

        f.visititems(visit)
        if not found:
            raise ValueError("no current-schema vision/*/colors dataset found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--decode-images", action="store_true")
    parser.add_argument("--robotwin-root", type=Path)
    args = parser.parse_args()

    for path in args.paths:
        inspect(
            path,
            decode_images=args.decode_images,
            robotwin_root=args.robotwin_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
