#!/usr/bin/env python3
"""Diagnose CALVIN articulation G-v2 asymmetry.

This script compares demo-side state/action/G statistics for articulation tasks,
with special attention to move_slider_left vs move_slider_right.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

POLICY_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "policy_state_bc"
sys.path.insert(0, POLICY_SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as pol  # noqa: E402


ARTICULATION_TASKS = ("open_drawer", "close_drawer", "move_slider_left", "move_slider_right")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def summarize_array(x: np.ndarray) -> dict[str, Any]:
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(len(x)),
        "mean": np.mean(x, axis=0),
        "std": np.std(x, axis=0),
        "p10": np.percentile(x, 10, axis=0),
        "p50": np.percentile(x, 50, axis=0),
        "p90": np.percentile(x, 90, axis=0),
        "min": np.min(x, axis=0),
        "max": np.max(x, axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--geometry-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-segments-per-task", type=int, default=0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    split_dir = args.data_root / "training"
    segments = [s for s in pol.load_lang_segments(split_dir) if s["task"] in ARTICULATION_TASKS]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seg in segments:
        by_task[seg["task"]].append(seg)

    geometry = pol.load_articulation_geometry(args.geometry_json)
    records: dict[str, dict[str, list[Any]]] = {
        task: defaultdict(list) for task in ARTICULATION_TASKS
    }
    segment_rows = []

    for task in ARTICULATION_TASKS:
        task_segments = list(by_task[task])
        rng.shuffle(task_segments)
        if args.max_segments_per_task > 0:
            task_segments = task_segments[: args.max_segments_per_task]
        for seg in task_segments:
            start_npz = np.load(pol.episode_file(split_dir, seg["start"]), allow_pickle=True)
            end_npz = np.load(pol.episode_file(split_dir, seg["end"]), allow_pickle=True)
            start_scene = np.asarray(start_npz["scene_obs"], dtype=np.float32)
            end_scene = np.asarray(end_npz["scene_obs"], dtype=np.float32)
            target_controls = pol.articulation_target_controls(task, start_scene, end_scene)
            joint_name = pol.ARTICULATION_TASK_TO_JOINT[task]
            joint_idx = pol.ARTICULATION_JOINT_META[joint_name]["index"]
            joint_range = pol.ARTICULATION_JOINT_META[joint_name]["range"]
            start_q = float(start_scene[joint_idx])
            end_q = float(end_scene[joint_idx])
            target_q = float(target_controls[joint_idx])
            segment_rows.append(
                {
                    "task": task,
                    "joint": joint_name,
                    "start": seg["start"],
                    "end": seg["end"],
                    "start_q": start_q,
                    "end_q": end_q,
                    "target_q": target_q,
                    "demo_q_delta": end_q - start_q,
                    "target_residual_start": target_q - start_q,
                    "target_residual_end": target_q - end_q,
                    "normalized_residual_start": (target_q - start_q) / joint_range,
                    "normalized_residual_end": (target_q - end_q) / joint_range,
                }
            )
            for idx in range(seg["start"], seg["end"] + 1, max(1, args.stride)):
                f = pol.episode_file(split_dir, idx)
                if not f.exists():
                    continue
                try:
                    d = np.load(f, allow_pickle=True)
                    robot = np.asarray(d["robot_obs"], dtype=np.float32)
                    scene = np.asarray(d["scene_obs"], dtype=np.float32)
                    action = np.asarray(d["rel_actions"], dtype=np.float32)
                except Exception:
                    continue
                g = pol.make_articulation_compact_g(
                    robot[None, :],
                    scene[None, :],
                    np.asarray([task], dtype=object),
                    geometry,
                )[0]
                target_affordance = g[:3]
                residual_motion = g[3:6]
                stage = g[6:9]
                q = float(scene[joint_idx])
                target_q_frame = float(pol.articulation_target_controls(task, scene)[joint_idx])
                records[task]["q"].append(q)
                records[task]["target_q"].append(target_q_frame)
                records[task]["normalized_residual"].append((target_q_frame - q) / joint_range)
                records[task]["target_affordance"].append(target_affordance)
                records[task]["target_affordance_norm"].append(float(np.linalg.norm(target_affordance)))
                records[task]["residual_motion"].append(residual_motion)
                records[task]["stage_index"].append(int(np.argmax(stage)))
                records[task]["action_xyz"].append(action[:3])
                records[task]["action_gripper"].append(float(action[-1]))

    summary = {}
    for task in ARTICULATION_TASKS:
        task_records = records[task]
        stages = Counter(task_records["stage_index"])
        summary[task] = {
            "frames": len(task_records["q"]),
            "q": summarize_array(np.asarray(task_records["q"], dtype=np.float64)),
            "target_q": summarize_array(np.asarray(task_records["target_q"], dtype=np.float64)),
            "normalized_residual": summarize_array(np.asarray(task_records["normalized_residual"], dtype=np.float64)),
            "target_affordance": summarize_array(np.asarray(task_records["target_affordance"], dtype=np.float64)),
            "target_affordance_norm": summarize_array(np.asarray(task_records["target_affordance_norm"], dtype=np.float64)),
            "residual_motion": summarize_array(np.asarray(task_records["residual_motion"], dtype=np.float64)),
            "stage_counts": {str(k): int(v) for k, v in sorted(stages.items())},
            "stage_fractions": {str(k): float(v / max(1, len(task_records["q"]))) for k, v in sorted(stages.items())},
            "action_xyz": summarize_array(np.asarray(task_records["action_xyz"], dtype=np.float64)),
            "action_gripper": summarize_array(np.asarray(task_records["action_gripper"], dtype=np.float64)),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "articulation_g_v2_left_right_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)
    with open(args.out_dir / "articulation_g_v2_segment_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(segment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(segment_rows)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
