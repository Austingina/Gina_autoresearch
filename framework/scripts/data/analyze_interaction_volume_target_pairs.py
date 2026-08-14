#!/usr/bin/env python3
"""Audit same-flow, different-target supervision for interaction-volume G."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1] / "policy_state_bc"
sys.path.insert(0, POLICY_DIR.as_posix())

import train_eval_state_chunk_flow_policy as policy  # noqa: E402


DEFAULT_PAIRS = (
    "left:push_red_block_left:push_pink_block_left",
    "right:push_blue_block_right:push_pink_block_right",
)


def parse_pair(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("Pair must be NAME:TASK0:TASK1")
    return parts[0], parts[1], parts[2]


def percentiles(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    q = np.percentile(values, [10, 25, 50, 75, 90])
    return {name: float(value) for name, value in zip(("p10", "p25", "p50", "p75", "p90"), q)}


def nearest_pairs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    nearest_xy = policy.nearest_indices_batched(x, y)
    nearest_yx = policy.nearest_indices_batched(y, x)
    pairs = np.concatenate(
        [
            np.stack([np.arange(len(x)), nearest_xy], axis=1),
            np.stack([nearest_yx, np.arange(len(y))], axis=1),
        ],
        axis=0,
    )
    return np.unique(pairs, axis=0).astype(np.int64)


def summarize_family(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    family: str,
    task0: str,
    task1: str,
    *,
    max_relative_frame: int,
    min_attraction_norm: float,
    min_attraction_difference: float,
    min_action_difference: float,
    max_eef_distance: float,
    max_orientation_distance: float,
    own_alignment_threshold: float,
) -> dict[str, Any]:
    names = arrays["task_name"][train_idx].astype(str)
    robot = arrays["robot_obs"][train_idx].astype(np.float64)
    g = arrays["articulation_minimal_g"][train_idx].astype(np.float64)
    action = arrays["action_chunks"][train_idx, 0, :3].astype(np.float64)
    segment = arrays["segment_id"][train_idx]
    frame = arrays["frame_index"][train_idx]
    segment_start = {value: int(frame[segment == value].min()) for value in np.unique(segment)}
    relative_frame = np.asarray(
        [int(value) - segment_start[segment_id] for value, segment_id in zip(frame, segment)],
        dtype=np.int64,
    )

    attraction_norm = np.linalg.norm(g[:, :3], axis=1)
    position0 = np.flatnonzero(
        (names == task0)
        & (attraction_norm > min_attraction_norm)
        & (relative_frame <= max_relative_frame)
    )
    position1 = np.flatnonzero(
        (names == task1)
        & (attraction_norm > min_attraction_norm)
        & (relative_frame <= max_relative_frame)
    )
    if not len(position0) or not len(position1):
        return {
            "family": family,
            "tasks": [task0, task1],
            "candidate_windows": [int(len(position0)), int(len(position1))],
            "pairs": 0,
        }

    state0 = robot[position0, :14]
    state1 = robot[position1, :14]
    joined = np.concatenate([state0, state1], axis=0)
    mean = joined.mean(axis=0)
    std = joined.std(axis=0)
    std[std < 1e-5] = 1.0
    pair_positions = nearest_pairs((state0 - mean) / std, (state1 - mean) / std)
    index0 = position0[pair_positions[:, 0]]
    index1 = position1[pair_positions[:, 1]]

    eef_distance = np.linalg.norm(robot[index0, :3] - robot[index1, :3], axis=1)
    orientation_distance = np.linalg.norm(robot[index0, 3:6] - robot[index1, 3:6], axis=1)
    attraction_difference = g[index0, :3] - g[index1, :3]
    action_difference = action[index0] - action[index1]
    attraction_difference_norm = np.linalg.norm(attraction_difference, axis=1)
    action_difference_norm = np.linalg.norm(action_difference, axis=1)
    keep = (
        (eef_distance < max_eef_distance)
        & (orientation_distance < max_orientation_distance)
        & (attraction_difference_norm > min_attraction_difference)
        & (action_difference_norm > min_action_difference)
    )
    index0 = index0[keep]
    index1 = index1[keep]
    eef_distance = eef_distance[keep]
    orientation_distance = orientation_distance[keep]
    attraction_difference = attraction_difference[keep]
    action_difference = action_difference[keep]
    attraction_difference_norm = attraction_difference_norm[keep]
    action_difference_norm = action_difference_norm[keep]

    cosine = np.sum(attraction_difference * action_difference, axis=1) / (
        attraction_difference_norm * action_difference_norm + 1e-8
    )
    own0 = np.sum(g[index0, :3] * action[index0], axis=1) / (
        np.linalg.norm(g[index0, :3], axis=1) * np.linalg.norm(action[index0], axis=1) + 1e-8
    )
    own1 = np.sum(g[index1, :3] * action[index1], axis=1) / (
        np.linalg.norm(g[index1, :3], axis=1) * np.linalg.norm(action[index1], axis=1) + 1e-8
    )
    high_confidence = (own0 > own_alignment_threshold) & (own1 > own_alignment_threshold)
    return {
        "family": family,
        "tasks": [task0, task1],
        "candidate_windows": [int(len(position0)), int(len(position1))],
        "pairs": int(len(cosine)),
        "eef_distance_m": percentiles(eef_distance),
        "orientation_distance": percentiles(orientation_distance),
        "attraction_difference_norm": percentiles(attraction_difference_norm),
        "action_difference_norm": percentiles(action_difference_norm),
        "difference_cosine": percentiles(cosine),
        "difference_cosine_positive_fraction": float(np.mean(cosine > 0)) if len(cosine) else None,
        "high_confidence_pairs": int(high_confidence.sum()),
        "high_confidence_difference_cosine": percentiles(cosine[high_confidence]),
        "high_confidence_positive_fraction": (
            float(np.mean(cosine[high_confidence] > 0)) if np.any(high_confidence) else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pair", action="append", type=parse_pair)
    parser.add_argument("--max-relative-frame", type=int, default=32)
    parser.add_argument("--min-attraction-norm", type=float, default=0.015)
    parser.add_argument("--min-attraction-difference", type=float, default=0.02)
    parser.add_argument("--min-action-difference", type=float, default=0.03)
    parser.add_argument("--max-eef-distance", type=float, default=0.05)
    parser.add_argument("--max-orientation-distance", type=float, default=0.6)
    parser.add_argument("--own-alignment-threshold", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    meta = json.loads(args.run_meta.read_text())
    data_root = args.data_root or Path(meta["data_root"])
    geometry = json.loads(Path(meta["articulation_geometry_json"]).read_text())
    arrays, _ = policy.collect_chunk_records(
        data_root=data_root,
        task_to_id={str(key): int(value) for key, value in meta["tasks"].items()},
        chunk_size=int(meta["chunk_size"]),
        max_windows=0,
        max_segments=0,
        stride=int(meta["stride"]),
        seed=int(meta["loader_seed"]),
        include_tasks=set(meta["include_tasks"]),
        articulation_geometry=geometry,
        minimal_g_scope="translational",
        primitive_geometry={},
    )
    train_idx, val_idx = policy.segment_train_val_indices(
        arrays["segment_id"], int(meta["split_seed"])
    )
    pairs = args.pair or [parse_pair(value) for value in DEFAULT_PAIRS]
    settings = {
        key: value
        for key, value in vars(args).items()
        if key not in {"run_meta", "data_root", "pair", "output"}
    }

    def summarize_split(indices: np.ndarray) -> list[dict[str, Any]]:
        return [
            summarize_family(
                arrays,
                indices,
                family,
                task0,
                task1,
                max_relative_frame=args.max_relative_frame,
                min_attraction_norm=args.min_attraction_norm,
                min_attraction_difference=args.min_attraction_difference,
                min_action_difference=args.min_action_difference,
                max_eef_distance=args.max_eef_distance,
                max_orientation_distance=args.max_orientation_distance,
                own_alignment_threshold=args.own_alignment_threshold,
            )
            for family, task0, task1 in pairs
        ]

    report = {
        "run_meta": args.run_meta.as_posix(),
        "data_root": data_root.as_posix(),
        "train_windows": int(len(train_idx)),
        "validation_windows": int(len(val_idx)),
        "settings": settings,
        "families": summarize_split(train_idx),
        "validation_families": summarize_split(val_idx),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
