#!/usr/bin/env python3
"""Evaluate observation-only active-object heuristics for generic object tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402
import train_eval_state_chunk_flow_policy as chunk  # noqa: E402


def choose(blocks: np.ndarray, eef: np.ndarray, target: np.ndarray) -> dict[str, int]:
    block_target = np.linalg.norm(blocks - target[None, :], axis=1)
    block_target_xy = np.linalg.norm(blocks[:, :2] - target[None, :2], axis=1)
    eef_block = np.linalg.norm(blocks - eef[None, :], axis=1)
    eef_block_xy = np.linalg.norm(blocks[:, :2] - eef[None, :2], axis=1)
    return {
        "nearest_target_3d": int(np.argmin(block_target)),
        "nearest_target_xy": int(np.argmin(block_target_xy)),
        "nearest_eef_3d": int(np.argmin(eef_block)),
        "nearest_eef_xy": int(np.argmin(eef_block_xy)),
        "shortest_eef_block_target_3d": int(np.argmin(eef_block + block_target)),
        "shortest_eef_block_target_xy": int(np.argmin(eef_block_xy + block_target_xy)),
        "min_x": int(np.argmin(blocks[:, 0])),
        "max_x": int(np.argmax(blocks[:, 0])),
        "min_y": int(np.argmin(blocks[:, 1])),
        "max_y": int(np.argmax(blocks[:, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default="push_into_drawer")
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    split_dir = args.data_root.resolve() / "training"
    primitive = chunk.fit_primitive_query_geometry(args.data_root.resolve())
    if args.task not in primitive:
        raise ValueError(f"No fitted primitive geometry for task: {args.task}")
    target = np.asarray(primitive[args.task]["target_center"], dtype=np.float32)
    endpoint_correct: dict[str, list[int]] = {}
    frame_correct: dict[str, list[int]] = {}
    frame_correct_by_phase: dict[str, dict[str, list[int]]] = {}
    true_counts = np.zeros(3, dtype=np.int64)
    segment_records: list[dict[str, Any]] = []

    for segment in base.load_lang_segments(split_dir):
        if str(segment["task"]) != args.task:
            continue
        start_idx, end_idx = int(segment["start"]), int(segment["end"])
        try:
            with np.load(base.episode_file(split_dir, start_idx), allow_pickle=True) as data:
                start_scene = np.asarray(data["scene_obs"], dtype=np.float32)
                start_robot = np.asarray(data["robot_obs"], dtype=np.float32)
            with np.load(base.episode_file(split_dir, end_idx), allow_pickle=True) as data:
                end_scene = np.asarray(data["scene_obs"], dtype=np.float32)
        except Exception:
            continue
        start_blocks = chunk._scene_block_positions(start_scene)
        end_blocks = chunk._scene_block_positions(end_scene)
        displacement = np.linalg.norm(end_blocks - start_blocks, axis=1)
        true_idx = int(np.argmax(displacement))
        if float(displacement[true_idx]) < 0.03:
            continue
        true_counts[true_idx] += 1
        endpoint_choices = choose(start_blocks, start_robot[:3], target)
        for name, prediction in endpoint_choices.items():
            endpoint_correct.setdefault(name, []).append(int(prediction == true_idx))

        segment_frame_correct: dict[str, list[int]] = {}
        sampled = list(range(start_idx, end_idx + 1, max(1, args.stride)))
        if sampled[-1] != end_idx:
            sampled.append(end_idx)
        for frame_order, frame_idx in enumerate(sampled):
            try:
                with np.load(base.episode_file(split_dir, frame_idx), allow_pickle=True) as data:
                    robot = np.asarray(data["robot_obs"], dtype=np.float32)
                    scene = np.asarray(data["scene_obs"], dtype=np.float32)
            except Exception:
                continue
            blocks = chunk._scene_block_positions(scene)
            choices = choose(blocks, robot[:3], target)
            phase_fraction = frame_order / max(1, len(sampled) - 1)
            phase = "early" if phase_fraction < 1.0 / 3.0 else "middle" if phase_fraction < 2.0 / 3.0 else "late"
            for name, prediction in choices.items():
                correct = int(prediction == true_idx)
                frame_correct.setdefault(name, []).append(correct)
                frame_correct_by_phase.setdefault(name, {}).setdefault(phase, []).append(correct)
                segment_frame_correct.setdefault(name, []).append(correct)
        segment_records.append(
            {
                "start": start_idx,
                "end": end_idx,
                "true_active": true_idx,
                "endpoint_choices": endpoint_choices,
                "frame_accuracy": {
                    name: float(np.mean(values)) for name, values in segment_frame_correct.items()
                },
            }
        )

    report = {
        "task": args.task,
        "target_center": target.tolist(),
        "num_segments": int(len(segment_records)),
        "true_active_counts": true_counts.tolist(),
        "endpoint_accuracy": {
            name: float(np.mean(values)) for name, values in endpoint_correct.items()
        },
        "all_frame_accuracy": {
            name: float(np.mean(values)) for name, values in frame_correct.items()
        },
        "frame_accuracy_by_phase": {
            name: {phase: float(np.mean(values)) for phase, values in phases.items()}
            for name, phases in frame_correct_by_phase.items()
        },
        "segments": segment_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "segments"}, indent=2))


if __name__ == "__main__":
    main()
