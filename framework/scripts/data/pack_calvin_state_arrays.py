#!/usr/bin/env python3
"""Pack compact CALVIN frame records into one contiguous training cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np


DEFAULT_FIELDS = ("robot_obs", "scene_obs", "rel_actions", "minimal_g")


def load_episode(job: tuple[str, tuple[str, ...]]) -> tuple[int, tuple[np.ndarray, ...]]:
    path_name, fields = job
    path = Path(path_name)
    frame_index = int(path.stem.rsplit("_", 1)[1])
    with np.load(path, allow_pickle=False) as source:
        absent = [field for field in fields if field not in source.files]
        if absent:
            raise RuntimeError(f"{path} is missing fields: {absent}")
        arrays = tuple(np.asarray(source[field]) for field in fields)
    return frame_index, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--progress-interval", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    split_dir = data_root / "training"
    output_path = split_dir / "state_only_arrays.npz"
    manifest_path = data_root / "packed_state_manifest.json"
    if output_path.exists() and not args.force:
        print(f"[pack] already exists: {output_path}", flush=True)
        return

    paths = sorted(split_dir.glob("episode_*.npz"))
    if not paths:
        raise RuntimeError(f"No episode records found under {split_dir}")
    jobs = [(str(path), tuple(args.fields)) for path in paths]
    start_time = time.time()
    frame_indices: list[int] = []
    field_values: dict[str, list[np.ndarray]] = {field: [] for field in args.fields}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(load_episode, jobs, chunksize=128)
        for index, (frame_index, values) in enumerate(results, start=1):
            frame_indices.append(frame_index)
            for field, value in zip(args.fields, values):
                field_values[field].append(value)
            if index == 1 or index == len(jobs) or index % args.progress_interval == 0:
                print(
                    f"[pack] {index}/{len(jobs)} elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )

    packed = {
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
        **{
            field: np.stack(values).astype(np.float32)
            for field, values in field_values.items()
        },
    }
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **packed)
    os.replace(tmp, output_path)

    with np.load(output_path, allow_pickle=False) as check:
        for field, expected in packed.items():
            if check[field].shape != expected.shape or check[field].dtype != expected.dtype:
                raise RuntimeError(f"Packed cache verification failed for {field}")
        for position in sorted({0, len(paths) // 2, len(paths) - 1}):
            _, source_values = load_episode(jobs[position])
            for field, source_value in zip(args.fields, source_values):
                if not np.array_equal(check[field][position], source_value):
                    raise RuntimeError(
                        f"Packed cache value mismatch for {field} at position {position}"
                    )

    manifest = {
        "data_root": str(data_root),
        "packed_path": str(output_path),
        "frames": len(paths),
        "fields": args.fields,
        "field_shapes": {field: list(value.shape) for field, value in packed.items()},
        "bytes": output_path.stat().st_size,
        "workers": args.workers,
        "elapsed_seconds": time.time() - start_time,
        "complete": True,
    }
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
