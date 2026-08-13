#!/usr/bin/env python3
"""Build a resumable state/action/G-only copy of a compact CALVIN dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np


DEFAULT_FIELDS = ("robot_obs", "scene_obs", "rel_actions", "minimal_g")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def strip_episode(job: tuple[str, str, tuple[str, ...], bool]) -> tuple[str, int]:
    source_name, output_name, fields, compress = job
    output_path = Path(output_name)
    if output_path.exists():
        return "preexisting", output_path.stat().st_size

    with np.load(source_name, allow_pickle=False) as source:
        absent = [field for field in fields if field not in source]
        if absent:
            raise RuntimeError(f"{source_name} is missing fields: {absent}")
        arrays = {field: np.asarray(source[field]) for field in fields}

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        if compress:
            np.savez_compressed(handle, **arrays)
        else:
            np.savez(handle, **arrays)
    os.replace(tmp, output_path)
    return "written", output_path.stat().st_size


def copy_metadata(source_dir: Path, output_dir: Path) -> int:
    copied = 0
    for source_path in source_dir.rglob("*"):
        if not source_path.is_file() or source_path.match("episode_*.npz"):
            continue
        relative_path = source_path.relative_to(source_dir)
        if relative_path.name in {
            "state_only_manifest.json",
            "state_only_progress.json",
            ".state_only_complete",
        }:
            continue
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--compress", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if source_dir == output_dir or source_dir in output_dir.parents:
        parser.error("--output-dir must not be the source directory or one of its children")
    if not source_dir.is_dir():
        parser.error(f"Source directory does not exist: {source_dir}")
    if args.workers < 1:
        parser.error("--workers must be positive")

    source_paths = sorted(source_dir.glob("training/episode_*.npz"))
    source_file_count = len(source_paths)
    if args.max_files > 0:
        source_paths = source_paths[: args.max_files]
    if not source_paths:
        raise RuntimeError(f"No training/episode_*.npz files found under {source_dir}")

    output_training = output_dir / "training"
    output_training.mkdir(parents=True, exist_ok=True)
    metadata_files = copy_metadata(source_dir, output_dir)
    jobs = [
        (
            str(source_path),
            str(output_training / source_path.name),
            tuple(args.fields),
            args.compress,
        )
        for source_path in source_paths
    ]

    start_time = time.time()
    written = 0
    preexisting = 0
    output_bytes = 0
    progress_path = output_dir / "state_only_progress.json"
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(strip_episode, jobs, chunksize=64)
        for index, (status, file_bytes) in enumerate(results, start=1):
            written += int(status == "written")
            preexisting += int(status == "preexisting")
            output_bytes += file_bytes
            if index == 1 or index == len(jobs) or index % args.progress_interval == 0:
                progress = {
                    "source_dir": str(source_dir),
                    "output_dir": str(output_dir),
                    "fields": args.fields,
                    "compress": args.compress,
                    "workers": args.workers,
                    "source_episode_files": source_file_count,
                    "selected_episode_files": len(jobs),
                    "visited_episode_files": index,
                    "written_this_run": written,
                    "preexisting": preexisting,
                    "output_bytes_visited": output_bytes,
                    "metadata_files_copied": metadata_files,
                    "elapsed_seconds": time.time() - start_time,
                    "complete": False,
                }
                atomic_write_json(progress_path, progress)
                print(
                    f"[strip] {index}/{len(jobs)} written={written} "
                    f"preexisting={preexisting} elapsed={progress['elapsed_seconds']:.1f}s",
                    flush=True,
                )

    complete = len(jobs) == source_file_count
    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "fields": args.fields,
        "compress": args.compress,
        "workers": args.workers,
        "source_episode_files": source_file_count,
        "selected_episode_files": len(jobs),
        "written_this_run": written,
        "preexisting": preexisting,
        "output_bytes": output_bytes,
        "metadata_files_copied": metadata_files,
        "elapsed_seconds": time.time() - start_time,
        "complete": complete,
    }
    atomic_write_json(output_dir / "state_only_manifest.json", manifest)
    atomic_write_json(progress_path, manifest)
    if complete:
        (output_dir / ".state_only_complete").touch()
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
