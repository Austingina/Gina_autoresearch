#!/usr/bin/env python3
"""Build a verified contiguous robot/scene/action cache from CALVIN episode NPZ files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np


FIELDS = ("robot_obs", "scene_obs", "rel_actions")
EPISODE_RE = re.compile(r"episode_(\d{7})\.npz$")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache(split_dir: Path, output: Path) -> dict[str, object]:
    split_dir = split_dir.resolve()
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"Output already exists; refusing overwrite: {output}")
    episode_rows: list[tuple[int, Path]] = []
    for path in split_dir.glob("episode_*.npz"):
        match = EPISODE_RE.fullmatch(path.name)
        if match:
            episode_rows.append((int(match.group(1)), path))
    episode_rows.sort()
    if not episode_rows:
        raise RuntimeError(f"No episode files found under {split_dir}")
    frame_indices = np.asarray([index for index, _ in episode_rows], dtype=np.int64)
    if len(np.unique(frame_indices)) != len(frame_indices):
        raise RuntimeError("Duplicate episode frame indices")

    rows: dict[str, list[np.ndarray]] = {field: [] for field in FIELDS}
    expected_shapes: dict[str, tuple[int, ...]] = {}
    for position, (frame_index, path) in enumerate(episode_rows, start=1):
        with np.load(path, allow_pickle=False) as episode:
            absent = [field for field in FIELDS if field not in episode.files]
            if absent:
                raise RuntimeError(f"{path} is missing fields: {absent}")
            for field in FIELDS:
                value = np.asarray(episode[field], dtype=np.float32)
                if not np.all(np.isfinite(value)):
                    raise RuntimeError(f"{path} field {field} contains non-finite values")
                if field not in expected_shapes:
                    expected_shapes[field] = value.shape
                elif value.shape != expected_shapes[field]:
                    raise RuntimeError(
                        f"{path} field {field} shape {value.shape} != {expected_shapes[field]}"
                    )
                rows[field].append(value)
        if position % 1000 == 0 or position == len(episode_rows):
            print(
                f"[packed-cache] {position}/{len(episode_rows)} frame={frame_index}",
                flush=True,
            )

    packed = {
        "frame_index": frame_indices,
        **{field: np.stack(rows[field]).astype(np.float32) for field in FIELDS},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **packed)
    os.replace(tmp, output)

    with np.load(output, allow_pickle=False) as check:
        if check.files != ["frame_index", *FIELDS]:
            raise RuntimeError(f"Unexpected packed fields: {check.files}")
        for field, expected in packed.items():
            if check[field].dtype != expected.dtype or check[field].shape != expected.shape:
                raise RuntimeError(f"Packed cache shape/dtype mismatch for {field}")
            if not np.array_equal(check[field], expected):
                raise RuntimeError(f"Packed cache value mismatch for {field}")

    receipt: dict[str, object] = {
        "format": "calvin_packed_state_cache_v1",
        "split_dir": split_dir.as_posix(),
        "output": output.as_posix(),
        "fields": ["frame_index", *FIELDS],
        "frames": int(len(frame_indices)),
        "frame_min": int(frame_indices.min()),
        "frame_max": int(frame_indices.max()),
        "field_shapes": {field: list(value.shape) for field, value in packed.items()},
        "bytes": int(output.stat().st_size),
        "cache_sha256": sha256_path(output),
        "complete": True,
    }
    manifest = output.with_name(f"{output.stem}_manifest.json")
    manifest_tmp = manifest.with_suffix(manifest.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_cache(args.split_dir, args.output)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
