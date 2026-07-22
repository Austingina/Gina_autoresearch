#!/usr/bin/env python3
"""Survey endpoint motion directions across CALVIN language tasks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402
import train_eval_state_chunk_flow_policy as chunk  # noqa: E402


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)


def direction_summary(deltas: list[np.ndarray], minimum_motion: float) -> dict[str, Any]:
    if not deltas:
        return {"segments": 0, "moving_segments": 0}
    values = np.stack(deltas).astype(np.float32)
    norms = np.linalg.norm(values, axis=1)
    moving = norms >= minimum_motion
    if not np.any(moving):
        return {"segments": int(len(values)), "moving_segments": 0}
    directions = unit(values[moving])
    median = np.median(directions, axis=0)
    median /= max(float(np.linalg.norm(median)), 1e-6)
    return {
        "segments": int(len(values)),
        "moving_segments": int(np.sum(moving)),
        "median_displacement": float(np.median(norms[moving])),
        "median_unit_direction": median.tolist(),
        "positive_y_fraction": float(np.mean(directions[:, 1] > 0.25)),
        "negative_y_fraction": float(np.mean(directions[:, 1] < -0.25)),
        "positive_z_fraction": float(np.mean(directions[:, 2] > 0.25)),
        "negative_z_fraction": float(np.mean(directions[:, 2] < -0.25)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--minimum-motion", type=float, default=0.03)
    args = parser.parse_args()

    split_dir = args.data_root.resolve() / "training"
    eef_deltas: dict[str, list[np.ndarray]] = defaultdict(list)
    active_block_deltas: dict[str, list[np.ndarray]] = defaultdict(list)
    active_block_indices: dict[str, list[int]] = defaultdict(list)
    for segment in base.load_lang_segments(split_dir):
        task = str(segment["task"])
        try:
            with np.load(base.episode_file(split_dir, int(segment["start"])), allow_pickle=True) as data:
                start_robot = np.asarray(data["robot_obs"], dtype=np.float32)
                start_scene = np.asarray(data["scene_obs"], dtype=np.float32)
            with np.load(base.episode_file(split_dir, int(segment["end"])), allow_pickle=True) as data:
                end_robot = np.asarray(data["robot_obs"], dtype=np.float32)
                end_scene = np.asarray(data["scene_obs"], dtype=np.float32)
        except Exception:
            continue
        eef_deltas[task].append(end_robot[:3] - start_robot[:3])
        block_delta = chunk._scene_block_positions(end_scene) - chunk._scene_block_positions(start_scene)
        active_idx = int(np.argmax(np.linalg.norm(block_delta, axis=1)))
        active_block_deltas[task].append(block_delta[active_idx])
        active_block_indices[task].append(active_idx)

    report: dict[str, Any] = {"tasks": {}}
    for task in sorted(eef_deltas):
        report["tasks"][task] = {
            "eef": direction_summary(eef_deltas[task], args.minimum_motion),
            "active_block": direction_summary(active_block_deltas[task], args.minimum_motion),
            "active_block_counts": [
                int(sum(index == color_idx for index in active_block_indices[task]))
                for color_idx in range(3)
            ],
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "task_direction_support.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    markdown = [
        "# CALVIN Task Direction Support",
        "",
        "| Task | Moving blocks | Median block direction | +y | -y | +z | -z |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for task, rec in report["tasks"].items():
        block = rec["active_block"]
        if not block.get("moving_segments"):
            continue
        direction = ", ".join(f"{value:.3f}" for value in block["median_unit_direction"])
        markdown.append(
            f"| {task} | {block['moving_segments']} | {direction} | "
            f"{block['positive_y_fraction']:.3f} | {block['negative_y_fraction']:.3f} | "
            f"{block['positive_z_fraction']:.3f} | {block['negative_z_fraction']:.3f} |"
        )
    with open(args.out_dir / "task_direction_support.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown) + "\n")
    print("\n".join(markdown), flush=True)


if __name__ == "__main__":
    main()
