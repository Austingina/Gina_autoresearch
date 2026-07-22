#!/usr/bin/env python3
"""Measure per-task state diversity in fixed CALVIN demonstration windows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402


DEFAULT_TASKS = [
    "open_drawer",
    "move_slider_left",
    "move_slider_right",
    "push_into_drawer",
    "push_blue_block_left",
    "push_blue_block_right",
    "push_pink_block_left",
    "push_pink_block_right",
    "push_red_block_left",
    "push_red_block_right",
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
]


SCENE_FIELDS: dict[str, slice | int] = {
    "slider_joint": 0,
    "drawer_joint": 1,
    "lightbulb_state": 4,
    "led_state": 5,
    "red_block_xyz": slice(6, 9),
    "blue_block_xyz": slice(12, 15),
    "pink_block_xyz": slice(18, 21),
}


def vector_stats(values: list[np.ndarray], round_decimals: int = 3) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.stack(values).astype(np.float64)
    rounded = np.round(array, decimals=round_decimals)
    return {
        "count": int(len(array)),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "span": np.ptp(array, axis=0).tolist(),
        "unique_rounded": int(len(np.unique(rounded, axis=0))),
        "round_decimals": int(round_decimals),
    }


def scalar_stats(values: list[float], round_decimals: int = 3) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    rounded = np.round(array, decimals=round_decimals)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "span": float(np.ptp(array)),
        "unique_rounded": int(len(np.unique(rounded))),
        "round_decimals": int(round_decimals),
    }


def init_store() -> dict[str, Any]:
    return {
        "eef_xyz": [],
        **{name: [] for name in SCENE_FIELDS},
    }


def append_state(store: dict[str, Any], robot: np.ndarray, scene: np.ndarray) -> None:
    store["eef_xyz"].append(np.asarray(robot[:3], dtype=np.float32))
    for name, index in SCENE_FIELDS.items():
        value = scene[index]
        if isinstance(index, slice):
            store[name].append(np.asarray(value, dtype=np.float32))
        else:
            store[name].append(float(value))


def summarize_store(store: dict[str, Any]) -> dict[str, Any]:
    summary = {"eef_xyz": vector_stats(store["eef_xyz"])}
    for name, index in SCENE_FIELDS.items():
        summary[name] = vector_stats(store[name]) if isinstance(index, slice) else scalar_stats(store[name])
    return summary


def compact_row(task: str, scope: str, summary: dict[str, Any], segments: int) -> dict[str, Any]:
    eef = summary["eef_xyz"]
    red = summary["red_block_xyz"]
    blue = summary["blue_block_xyz"]
    pink = summary["pink_block_xyz"]
    return {
        "task": task,
        "scope": scope,
        "segments": segments,
        "states": eef.get("count", 0),
        "eef_unique_1mm": eef.get("unique_rounded", 0),
        "eef_std_x": eef.get("std", [0, 0, 0])[0],
        "eef_std_y": eef.get("std", [0, 0, 0])[1],
        "eef_std_z": eef.get("std", [0, 0, 0])[2],
        "eef_span_x": eef.get("span", [0, 0, 0])[0],
        "eef_span_y": eef.get("span", [0, 0, 0])[1],
        "eef_span_z": eef.get("span", [0, 0, 0])[2],
        "red_xyz_unique_1mm": red.get("unique_rounded", 0),
        "blue_xyz_unique_1mm": blue.get("unique_rounded", 0),
        "pink_xyz_unique_1mm": pink.get("unique_rounded", 0),
        "slider_unique_1mm": summary["slider_joint"].get("unique_rounded", 0),
        "drawer_unique_1mm": summary["drawer_joint"].get("unique_rounded", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    split_dir = args.data_root.resolve() / "training"
    tasks = set(args.tasks)
    all_segments = base.load_lang_segments(split_dir)
    random.Random(args.data_seed).shuffle(all_segments)
    segments = [
        (segment_id, segment)
        for segment_id, segment in enumerate(all_segments)
        if str(segment["task"]) in tasks
        and int(segment["end"]) - int(segment["start"]) + 1 >= args.chunk_size
    ]
    unique_segment_ids = np.asarray([segment_id for segment_id, _ in segments], dtype=np.int64)
    split_rng = np.random.default_rng(args.split_seed)
    split_rng.shuffle(unique_segment_ids)
    val_count = max(1, int(round(len(unique_segment_ids) * args.val_fraction)))
    val_segment_ids = set(int(value) for value in unique_segment_ids[:val_count])

    total_segment_counts: dict[str, int] = defaultdict(int)
    train_segment_counts: dict[str, int] = defaultdict(int)
    available_starts: dict[str, dict[str, Any]] = defaultdict(init_store)
    available_windows: dict[str, dict[str, Any]] = defaultdict(init_store)
    train_starts: dict[str, dict[str, Any]] = defaultdict(init_store)
    train_windows: dict[str, dict[str, Any]] = defaultdict(init_store)

    for index, (segment_id, segment) in enumerate(segments, start=1):
        task = str(segment["task"])
        start = int(segment["start"])
        end = int(segment["end"])
        is_train = segment_id not in val_segment_ids
        total_segment_counts[task] += 1
        train_segment_counts[task] += int(is_train)

        with np.load(base.episode_file(split_dir, start), allow_pickle=True) as start_data:
            robot = np.asarray(start_data["robot_obs"], dtype=np.float32)
            scene = np.asarray(start_data["scene_obs"], dtype=np.float32)
            append_state(available_starts[task], robot, scene)
            if is_train:
                append_state(train_starts[task], robot, scene)

        for frame in range(start, end - args.chunk_size + 2, max(1, args.stride)):
            with np.load(base.episode_file(split_dir, frame), allow_pickle=True) as data:
                robot = np.asarray(data["robot_obs"], dtype=np.float32)
                scene = np.asarray(data["scene_obs"], dtype=np.float32)
                append_state(available_windows[task], robot, scene)
                if is_train:
                    append_state(train_windows[task], robot, scene)
        if index == 1 or index == len(segments) or index % 200 == 0:
            print(f"[scan] {index}/{len(segments)} task={task}", flush=True)

    full: dict[str, Any] = {
        "data_root": args.data_root.resolve().as_posix(),
        "chunk_size": args.chunk_size,
        "stride": args.stride,
        "data_seed": args.data_seed,
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "tasks": {},
    }
    rows: list[dict[str, Any]] = []
    for task in sorted(tasks):
        available_start_summary = summarize_store(available_starts[task])
        available_window_summary = summarize_store(available_windows[task])
        train_start_summary = summarize_store(train_starts[task])
        train_window_summary = summarize_store(train_windows[task])
        full["tasks"][task] = {
            "available_segments": int(total_segment_counts[task]),
            "supervised_train_segments": int(train_segment_counts[task]),
            "validation_segments": int(total_segment_counts[task] - train_segment_counts[task]),
            "available_segment_starts": available_start_summary,
            "available_windows": available_window_summary,
            "train_segment_starts": train_start_summary,
            "train_windows": train_window_summary,
        }
        rows.append(compact_row(task, "available_segment_starts", available_start_summary, int(total_segment_counts[task])))
        rows.append(compact_row(task, "available_windows", available_window_summary, int(total_segment_counts[task])))
        rows.append(compact_row(task, "train_segment_starts", train_start_summary, int(train_segment_counts[task])))
        rows.append(compact_row(task, "train_windows", train_window_summary, int(train_segment_counts[task])))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "task_state_diversity.json", "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)
    with open(args.out_dir / "task_state_diversity.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
