#!/usr/bin/env python3
"""Estimate contact-mechanics offsets for CALVIN articulation G-v3.

This augments the fitted articulation point model with task-conditioned contact
statistics. The key statistic is the EEF offset along the desired joint-motion
direction during demo frames where the joint actually moves.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
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


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def summarize_vectors(values: list[np.ndarray]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.stack(values).astype(np.float64)
    return {
        "count": int(arr.shape[0]),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
        "p10": np.percentile(arr, 10, axis=0),
        "p50": np.percentile(arr, 50, axis=0),
        "p90": np.percentile(arr, 90, axis=0),
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--geometry-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-segments-per-task", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--motion-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split_dir = args.data_root / "training"
    segments = [s for s in pol.load_lang_segments(split_dir) if s["task"] in ARTICULATION_TASKS]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seg in segments:
        by_task[seg["task"]].append(seg)

    rng = random.Random(args.seed)
    geometry = pol.load_articulation_geometry(args.geometry_json)
    if geometry is None:
        raise RuntimeError("Missing geometry")
    augmented = copy.deepcopy(geometry)
    augmented["contact_g_v3"] = {
        "description": "EEF offset along desired joint-motion direction during demo frames with actual joint progress.",
        "max_segments_per_task": args.max_segments_per_task,
        "stride": args.stride,
        "motion_eps": args.motion_eps,
        "seed": args.seed,
    }
    task_models: dict[str, Any] = {}

    for task in ARTICULATION_TASKS:
        task_segments = list(by_task[task])
        rng.shuffle(task_segments)
        if args.max_segments_per_task > 0:
            task_segments = task_segments[: args.max_segments_per_task]

        joint_name = pol.ARTICULATION_TASK_TO_JOINT[task]
        joint_model = geometry["joints"][joint_name]
        joint_idx = int(joint_model["joint_index"])
        joint_range = float(joint_model["joint_range"])
        axis = np.asarray(joint_model["axis_unit"], dtype=np.float64)
        intercept = np.asarray(joint_model["intercept"], dtype=np.float64)
        slope = np.asarray(joint_model["slope"], dtype=np.float64)

        offsets: list[float] = []
        lateral_dists: list[float] = []
        eef_offsets_from_part_point: list[np.ndarray] = []
        action_along_desired: list[float] = []
        q_progress: list[float] = []
        q_values: list[float] = []
        residual_values: list[float] = []

        for seg in task_segments:
            for idx in range(seg["start"], seg["end"], max(1, args.stride)):
                cur_file = pol.episode_file(split_dir, idx)
                next_file = pol.episode_file(split_dir, idx + 1)
                if not cur_file.exists() or not next_file.exists():
                    continue
                try:
                    cur = np.load(cur_file, allow_pickle=True)
                    nxt = np.load(next_file, allow_pickle=True)
                    robot = np.asarray(cur["robot_obs"], dtype=np.float64)
                    scene = np.asarray(cur["scene_obs"], dtype=np.float64)
                    next_scene = np.asarray(nxt["scene_obs"], dtype=np.float64)
                    action = np.asarray(cur["rel_actions"], dtype=np.float64)
                except Exception:
                    continue

                q = float(scene[joint_idx])
                target_q = float(pol.articulation_target_controls(task, scene.astype(np.float32))[joint_idx])
                residual = target_q - q
                residual_norm = residual / max(joint_range, 1e-6)
                if abs(residual_norm) < 0.08:
                    continue
                desired_sign = 1.0 if residual >= 0.0 else -1.0
                dq_desired = desired_sign * float(next_scene[joint_idx] - q)
                if dq_desired <= args.motion_eps:
                    continue

                desired_dir = desired_sign * axis
                part_point = intercept + slope * q
                eef = robot[:3]
                rel = part_point - eef
                offset = float(np.dot(rel, desired_dir))
                lateral = rel - offset * desired_dir

                offsets.append(offset)
                lateral_dists.append(float(np.linalg.norm(lateral)))
                eef_offsets_from_part_point.append((eef - part_point).astype(np.float64))
                action_along_desired.append(float(np.dot(action[:3], desired_dir)))
                q_progress.append(dq_desired)
                q_values.append(q)
                residual_values.append(residual_norm)

        offset_summary = summarize(offsets)
        if offset_summary["count"] == 0:
            contact_offset = 0.03
            eef_offset = np.zeros(3, dtype=np.float64)
        else:
            contact_offset = float(offset_summary["p50"])
            eef_offset = np.median(np.stack(eef_offsets_from_part_point), axis=0)
        task_models[task] = {
            "joint": joint_name,
            "contact_offset_along_desired_motion": contact_offset,
            "eef_offset_from_part_point": eef_offset,
            "offset_along_desired_motion": offset_summary,
            "eef_offset_from_part_point_summary": summarize_vectors(eef_offsets_from_part_point),
            "lateral_dist": summarize(lateral_dists),
            "action_along_desired_motion": summarize(action_along_desired),
            "q_progress_per_step": summarize(q_progress),
            "q": summarize(q_values),
            "normalized_residual": summarize(residual_values),
            "selected_segments": len(task_segments),
        }

    augmented["task_contact_models"] = task_models
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "oracle_g_v3_contact_mechanics_geometry.json", "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2, default=json_default)
    with open(args.out_dir / "oracle_g_v3_contact_mechanics_summary.json", "w", encoding="utf-8") as f:
        json.dump(task_models, f, indent=2, default=json_default)
    print(json.dumps(task_models, indent=2, default=json_default))


if __name__ == "__main__":
    main()
