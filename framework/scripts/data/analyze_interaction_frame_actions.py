#!/usr/bin/env python3
"""Audit whether demonstrations share an action distribution in the G frame."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


POLICY_DIR = Path(__file__).resolve().parents[1] / "policy_state_bc"
sys.path.insert(0, POLICY_DIR.as_posix())

import train_eval_state_chunk_flow_policy as policy  # noqa: E402


def interaction_frame(flow: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(flow))
    e_flow = flow / max(norm, 1e-6) if norm > 1e-6 else np.asarray([1.0, 0.0, 0.0])
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    up_raw = world_up - float(np.dot(world_up, e_flow)) * e_flow
    if np.linalg.norm(up_raw) <= 1e-4:
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        up_raw = fallback - float(np.dot(fallback, e_flow)) * e_flow
    e_up = up_raw / max(float(np.linalg.norm(up_raw)), 1e-6)
    e_side = np.cross(e_up, e_flow)
    return np.stack([e_flow, e_side, e_up], axis=-1).astype(np.float32)


def vector_summary(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "flow_positive_fraction": float(np.mean(values[:, 0] > 0)),
        "flow_negative_fraction": float(np.mean(values[:, 0] < 0)),
        "flow_abs_mean": float(np.mean(np.abs(values[:, 0]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--articulation-geometry-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()

    geometry = json.loads(args.articulation_geometry_json.read_text(encoding="utf-8"))
    text_to_task, _ = policy.base.load_eval_annotation_maps(args.calvin_code)
    segments = policy.base.load_lang_segments(args.data_root / "training")
    task_to_id = policy.base.build_task_index(segments, text_to_task)
    arrays, used_segments = policy.collect_chunk_records(
        args.data_root,
        task_to_id,
        chunk_size=1,
        max_windows=0,
        max_segments=0,
        stride=args.stride,
        seed=args.seed,
        include_tasks=set(args.tasks),
        articulation_geometry=geometry,
        minimal_g_scope="translational",
    )

    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    all_local: list[np.ndarray] = []
    all_task: list[str] = []
    all_phase: list[str] = []
    for task_value, action, g in zip(
        arrays["task_name"], arrays["action_chunks"][:, 0], arrays["articulation_minimal_g"]
    ):
        task = str(task_value)
        phase = "approach" if float(np.linalg.norm(g[:3])) > 1e-6 else "flow"
        frame = interaction_frame(g[3:6])
        local_xyz = frame.T @ action[:3]
        grouped[(task, phase)].append(local_xyz.astype(np.float32))
        all_local.append(local_xyz.astype(np.float32))
        all_task.append(task)
        all_phase.append(phase)

    task_phase = {
        task: {
            phase: vector_summary(np.stack(grouped[(task, phase)]))
            if grouped[(task, phase)]
            else vector_summary(np.empty((0, 3), dtype=np.float32))
            for phase in ("approach", "flow")
        }
        for task in args.tasks
    }
    direction_phase: dict[str, dict[str, Any]] = {}
    local_array = np.stack(all_local)
    task_array = np.asarray(all_task)
    phase_array = np.asarray(all_phase)
    for direction in ("left", "right"):
        direction_phase[direction] = {}
        direction_mask = np.asarray([task.endswith(f"_{direction}") for task in task_array])
        for phase in ("approach", "flow"):
            mask = direction_mask & (phase_array == phase)
            direction_phase[direction][phase] = vector_summary(local_array[mask])

    comparisons = {}
    for phase in ("approach", "flow"):
        left = local_array[
            np.asarray([task.endswith("_left") for task in task_array]) & (phase_array == phase)
        ]
        right = local_array[
            np.asarray([task.endswith("_right") for task in task_array]) & (phase_array == phase)
        ]
        pooled_std = np.sqrt((np.var(left, axis=0) + np.var(right, axis=0)) / 2.0)
        comparisons[phase] = {
            "left_right_mean_difference": (np.mean(left, axis=0) - np.mean(right, axis=0)).tolist(),
            "left_right_standardized_mean_gap": (
                np.abs(np.mean(left, axis=0) - np.mean(right, axis=0))
                / np.maximum(pooled_std, 1e-6)
            ).tolist(),
        }

    report = {
        "protocol": {
            "data_root": str(args.data_root),
            "tasks": args.tasks,
            "stride": args.stride,
            "seed": args.seed,
            "segments": len(used_segments),
            "windows": len(all_local),
            "frame_axes": ["desired_flow", "side", "world_up_orthogonalized"],
        },
        "task_phase": task_phase,
        "direction_phase": direction_phase,
        "left_right_comparison": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
