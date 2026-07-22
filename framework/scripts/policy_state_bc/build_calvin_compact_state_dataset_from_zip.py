#!/usr/bin/env python3
"""Extract a resumable state/action-only CALVIN subset from an official zip."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import time
from typing import Any
from zipfile import ZipFile

import numpy as np


REQUIRED_EPISODE_KEYS = ("robot_obs", "scene_obs", "rel_actions")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def find_single_suffix(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one zip entry ending in {suffix!r}, found {len(matches)}")
    return matches[0]


def select_items(values: Any, positions: list[int]) -> Any:
    if isinstance(values, np.ndarray):
        return values[np.asarray(positions, dtype=np.int64)]
    if isinstance(values, tuple):
        return tuple(values[index] for index in positions)
    return [values[index] for index in positions]


def select_aligned_items(values: Any, positions: list[int], total: int) -> Any:
    try:
        value_count = len(values)
    except TypeError:
        return values
    if value_count != total:
        return values
    return select_items(values, positions)


def select_language_annotations(payload: dict[str, Any], positions: list[int]) -> dict[str, Any]:
    total = len(payload["language"]["task"])
    selected = {
        "language": {
            key: select_aligned_items(values, positions, total)
            for key, values in payload["language"].items()
        },
        "info": {
            key: select_aligned_items(values, positions, total)
            for key, values in payload["info"].items()
        },
    }
    for key, value in payload.items():
        if key not in selected:
            selected[key] = value
    return selected


def atomic_save_episode(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-tasks", nargs="+", required=True)
    parser.add_argument("--progress-interval", type=int, default=500)
    args = parser.parse_args()

    zip_path = args.zip.resolve()
    out_dir = args.out_dir.resolve()
    split_dir = out_dir / "training"
    lang_dir = split_dir / "lang_annotations"
    split_dir.mkdir(parents=True, exist_ok=True)
    lang_dir.mkdir(parents=True, exist_ok=True)

    include_tasks = set(args.include_tasks)
    start_time = time.time()
    print(f"[open] indexing {zip_path}", flush=True)
    with ZipFile(zip_path) as archive:
        names = archive.namelist()
        language_name = find_single_suffix(
            names, "/training/lang_annotations/auto_lang_ann.npy"
        )
        archive_root = language_name[: -len("training/lang_annotations/auto_lang_ann.npy")]
        with archive.open(language_name) as handle:
            language_payload = np.load(BytesIO(handle.read()), allow_pickle=True).item()

        tasks = [str(task) for task in language_payload["language"]["task"]]
        intervals = np.asarray(language_payload["info"]["indx"], dtype=np.int64)
        selected_positions = [
            index for index, task in enumerate(tasks) if task in include_tasks
        ]
        if not selected_positions:
            raise RuntimeError(f"No language segments matched tasks: {sorted(include_tasks)}")

        selected_payload = select_language_annotations(language_payload, selected_positions)
        np.save(lang_dir / "auto_lang_ann.npy", selected_payload, allow_pickle=True)

        episode_ids: set[int] = set()
        segments_by_task: dict[str, int] = {}
        for position in selected_positions:
            task = tasks[position]
            start, end = (int(value) for value in intervals[position])
            episode_ids.update(range(start, end + 1))
            segments_by_task[task] = segments_by_task.get(task, 0) + 1

        ordered_ids = sorted(episode_ids)
        progress_path = out_dir / "compact_extract_progress.json"
        completed = 0
        skipped = 0
        missing: list[int] = []
        for offset, episode_id in enumerate(ordered_ids, start=1):
            output_path = split_dir / f"episode_{episode_id:07d}.npz"
            if output_path.exists():
                skipped += 1
            else:
                archive_name = (
                    f"{archive_root}training/episode_{episode_id:07d}.npz"
                )
                try:
                    raw = archive.read(archive_name)
                except KeyError:
                    missing.append(episode_id)
                    continue
                with np.load(BytesIO(raw), allow_pickle=True) as source:
                    absent = [key for key in REQUIRED_EPISODE_KEYS if key not in source]
                    if absent:
                        raise RuntimeError(f"{archive_name} is missing keys: {absent}")
                    compact = {
                        key: np.asarray(source[key], dtype=np.float32)
                        for key in REQUIRED_EPISODE_KEYS
                    }
                atomic_save_episode(output_path, compact)
                completed += 1

            if offset == 1 or offset == len(ordered_ids) or offset % args.progress_interval == 0:
                progress = {
                    "zip": zip_path,
                    "archive_root": archive_root,
                    "out_dir": out_dir,
                    "include_tasks": sorted(include_tasks),
                    "segments": len(selected_positions),
                    "segments_by_task": segments_by_task,
                    "episode_files_total": len(ordered_ids),
                    "episode_files_visited": offset,
                    "episode_files_written_this_run": completed,
                    "episode_files_preexisting": skipped,
                    "missing_episode_ids": missing,
                    "elapsed_seconds": time.time() - start_time,
                    "complete": False,
                }
                atomic_write_json(progress_path, progress)
                print(
                    f"[extract] {offset}/{len(ordered_ids)} written={completed} "
                    f"preexisting={skipped} missing={len(missing)}",
                    flush=True,
                )

    manifest = {
        "zip": zip_path,
        "archive_root": archive_root,
        "out_dir": out_dir,
        "include_tasks": sorted(include_tasks),
        "segments": len(selected_positions),
        "segments_by_task": segments_by_task,
        "episode_files_total": len(ordered_ids),
        "episode_files_written_this_run": completed,
        "episode_files_preexisting": skipped,
        "missing_episode_ids": missing,
        "elapsed_seconds": time.time() - start_time,
        "complete": not missing,
    }
    atomic_write_json(out_dir / "compact_extract_manifest.json", manifest)
    atomic_write_json(progress_path, manifest)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} selected episode files")
    print(json.dumps(manifest, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
