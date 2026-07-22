# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize offline lenient-teacher mask+depth HDF5 dataset statistics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

_OFFLINE_CQL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/teacher_student"
)
if str(_OFFLINE_CQL_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFFLINE_CQL_ROOT))

from offline_cql.rewards import compute_offline_reach_rewards, summarize_reward_array  # noqa: E402

ENV_ACTION_NAMES = [
    "arm_dx",
    "arm_dy",
    "arm_dz",
    "arm_rx",
    "arm_ry",
    "arm_rz",
    "gripper_l",
    "gripper_r",
]
STUDENT_STATE_NAMES = [
    "ee_x",
    "ee_y",
    "ee_z",
    "gripper_open",
    "cube_x",
    "cube_y",
    "cube_z",
    "goal_x",
    "goal_y",
    "goal_z",
]
SCORE_NAMES = ["reach_score", "grasp_score", "lift_score", "goal_score", "success_score"]


def _demo_sort_key(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def _read(group: h5py.Group, path: str) -> np.ndarray:
    node = group
    for part in path.split("/"):
        node = node[part]
    return np.asarray(node)


def _summarize_array(arr: np.ndarray) -> dict:
    return summarize_reward_array(arr)


def recommend_noise_adjustments(success_rate: float) -> list[str]:
    """Return human-readable recommendations for lenient-teacher noise tuning."""
    if success_rate > 0.9:
        return [
            "success_rate > 0.9: dataset is near-expert imitation. Increase noise_std, noise_hold_max, or noise_trigger_prob.",
        ]
    if success_rate > 0.8:
        return [
            "success_rate > 0.8: slightly imbalanced toward success. Consider increasing noise magnitude or perturbation frequency.",
        ]
    if success_rate < 0.1:
        return [
            "success_rate < 0.1: dataset is mostly failure. Decrease noise_std, noise_hold_max, or noise_trigger_prob.",
        ]
    if success_rate < 0.3:
        return [
            "success_rate < 0.3: too many failures for stable value learning. Consider decreasing noise magnitude or duration.",
        ]
    if 0.4 <= success_rate <= 0.6:
        return ["success_rate in [0.4, 0.6]: balanced mix for offline RL value ranking."]
    if 0.3 <= success_rate <= 0.8:
        return ["success_rate in [0.3, 0.8]: acceptable range for offline RL recovery learning."]
    return []


def analyze_dataset(dataset_path: str) -> dict:
    demo_rows: list[dict] = []
    all_reach_reward: list[np.ndarray] = []
    all_near_reward: list[np.ndarray] = []
    all_success_bonus: list[np.ndarray] = []
    all_reward_total: list[np.ndarray] = []
    all_progress_delta: list[np.ndarray] = []
    success_flags: list[bool] = []
    episode_lengths: list[int] = []
    first_success_steps: list[int] = []

    with h5py.File(dataset_path, "r") as h5:
        demo_names = sorted(h5["data"].keys(), key=_demo_sort_key)
        for demo_name in demo_names:
            g = h5["data"][demo_name]
            env_actions = _read(g, "env_actions").astype(np.float32)
            progress_delta = _read(g, "rewards/progress_delta").reshape(-1).astype(np.float32)
            success_score = _read(g, "rewards/success_score").reshape(-1).astype(np.float32)
            ee_pose = _read(g, "ee_pose").astype(np.float32)
            student_state = _read(g, "obs/student_state").astype(np.float32)
            episode_success = bool(g.attrs.get("success", False))

            t = min(len(env_actions), len(progress_delta), len(ee_pose), len(student_state), len(success_score))
            progress_delta = progress_delta[:t]
            success_score = success_score[:t]
            ee_pose = ee_pose[:t]
            student_state = student_state[:t]

            reward_parts = compute_offline_reach_rewards(
                ee_pose[:, :3],
                student_state[:, 4:7],
                success_score=success_score,
            )
            reach_reward = reward_parts["reach_reward"]
            near_reward = reward_parts["near_reward"]
            success_bonus = reward_parts["success_bonus"]
            reward_total = reward_parts["rewards"]

            success_idx = np.where(success_score >= 0.5)[0]
            first_success = int(success_idx[0]) if success_idx.size else -1

            demo_rows.append(
                {
                    "demo": demo_name,
                    "success": episode_success,
                    "length": t,
                    "first_success_step": first_success,
                    "reward_sum": float(np.sum(reward_total)),
                    "reward_mean": float(np.mean(reward_total)),
                    "reach_reward_mean": float(np.mean(reach_reward)),
                    "near_reward_mean": float(np.mean(near_reward)),
                }
            )

            success_flags.append(episode_success)
            episode_lengths.append(t)
            if first_success >= 0:
                first_success_steps.append(first_success)
            all_reach_reward.append(reach_reward)
            all_near_reward.append(near_reward)
            all_success_bonus.append(success_bonus)
            all_reward_total.append(reward_total)
            all_progress_delta.append(progress_delta)

    reach_reward = np.concatenate(all_reach_reward, axis=0) if all_reach_reward else np.array([])
    near_reward = np.concatenate(all_near_reward, axis=0) if all_near_reward else np.array([])
    success_bonus = np.concatenate(all_success_bonus, axis=0) if all_success_bonus else np.array([])
    reward_total = np.concatenate(all_reward_total, axis=0) if all_reward_total else np.array([])
    progress_delta = np.concatenate(all_progress_delta, axis=0) if all_progress_delta else np.array([])

    num_episodes = len(success_flags)
    success_count = int(np.sum(success_flags))
    failure_count = int(num_episodes - success_count)
    success_rate = float(success_count / max(num_episodes, 1))
    failure_rate = float(failure_count / max(num_episodes, 1))

    lengths_arr = np.asarray(episode_lengths, dtype=np.float64)
    first_success_arr = np.asarray(first_success_steps, dtype=np.float64) if first_success_steps else np.array([])

    reach_stats = _summarize_array(reach_reward)
    near_stats = _summarize_array(near_reward)
    success_bonus_stats = _summarize_array(success_bonus)
    reward_stats = _summarize_array(reward_total)
    delta_stats = _summarize_array(progress_delta)

    export_stats = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "reward_scheme": "reach_distance",
        "num_episodes": num_episodes,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "episode_length_mean": float(np.mean(lengths_arr)) if lengths_arr.size else 0.0,
        "episode_length_std": float(np.std(lengths_arr)) if lengths_arr.size else 0.0,
        "first_success_step_mean": float(np.mean(first_success_arr)) if first_success_arr.size else None,
        "first_success_step_std": float(np.std(first_success_arr)) if first_success_arr.size else None,
        "reach_reward_mean": reach_stats["mean"],
        "reach_reward_std": reach_stats["std"],
        "reach_reward_p50": reach_stats["p50"],
        "reach_reward_p95": reach_stats["p95"],
        "reach_reward_p99": reach_stats["p99"],
        "near_reward_mean": near_stats["mean"],
        "near_reward_std": near_stats["std"],
        "near_reward_p50": near_stats["p50"],
        "near_reward_p95": near_stats["p95"],
        "near_reward_p99": near_stats["p99"],
        "success_bonus_mean": success_bonus_stats["mean"],
        "success_bonus_std": success_bonus_stats["std"],
        "success_bonus_p50": success_bonus_stats["p50"],
        "success_bonus_p95": success_bonus_stats["p95"],
        "success_bonus_p99": success_bonus_stats["p99"],
        "reward_total_mean": reward_stats["mean"],
        "reward_total_std": reward_stats["std"],
        "reward_total_p50": reward_stats["p50"],
        "reward_total_p95": reward_stats["p95"],
        "reward_total_p99": reward_stats["p99"],
        "positive_reward_fraction": float(np.mean(reward_total > 0.0)) if reward_total.size else 0.0,
        "negative_reward_fraction": float(np.mean(reward_total < 0.0)) if reward_total.size else 0.0,
        "zero_reward_fraction": float(np.mean(reward_total == 0.0)) if reward_total.size else 0.0,
        "progress_delta_mean": delta_stats["mean"],
        "progress_delta_std": delta_stats["std"],
        "progress_delta_p50": delta_stats["p50"],
        "progress_delta_p95": delta_stats["p95"],
        "progress_delta_p99": delta_stats["p99"],
        "recommendations": recommend_noise_adjustments(success_rate),
    }

    return {
        "export_stats": export_stats,
        "reach_reward": reach_stats,
        "near_reward": near_stats,
        "success_bonus": success_bonus_stats,
        "reward_total": reward_stats,
        "progress_delta": delta_stats,
        "demo_summary": demo_rows,
    }


def print_export_report(report: dict) -> None:
    stats = report["export_stats"]
    print("=" * 72)
    print(f"Dataset export statistics: {stats['dataset_path']}")
    print(
        f"num_episodes={stats['num_episodes']} success_count={stats['success_count']} "
        f"failure_count={stats['failure_count']}"
    )
    print(f"success_rate={stats['success_rate']:.3f} failure_rate={stats['failure_rate']:.3f}")
    print(
        f"episode_length_mean={stats['episode_length_mean']:.1f} "
        f"episode_length_std={stats['episode_length_std']:.1f}"
    )
    if stats["first_success_step_mean"] is not None:
        print(
            f"first_success_step_mean={stats['first_success_step_mean']:.1f} "
            f"first_success_step_std={stats['first_success_step_std']:.1f}"
        )
    print(
        f"reach_reward: mean={stats['reach_reward_mean']:.5f} std={stats['reach_reward_std']:.5f} "
        f"p50={stats['reach_reward_p50']:.5f} p95={stats['reach_reward_p95']:.5f} p99={stats['reach_reward_p99']:.5f}"
    )
    print(
        f"near_reward: mean={stats['near_reward_mean']:.5f} std={stats['near_reward_std']:.5f} "
        f"p50={stats['near_reward_p50']:.5f} p95={stats['near_reward_p95']:.5f} p99={stats['near_reward_p99']:.5f}"
    )
    print(
        f"success_bonus: mean={stats['success_bonus_mean']:.5f} std={stats['success_bonus_std']:.5f} "
        f"p50={stats['success_bonus_p50']:.5f} p95={stats['success_bonus_p95']:.5f} p99={stats['success_bonus_p99']:.5f}"
    )
    print(
        f"reward_total: mean={stats['reward_total_mean']:.5f} std={stats['reward_total_std']:.5f} "
        f"p50={stats['reward_total_p50']:.5f} p95={stats['reward_total_p95']:.5f} p99={stats['reward_total_p99']:.5f}"
    )
    print(
        f"progress_delta (diagnostic): mean={stats['progress_delta_mean']:.6f} std={stats['progress_delta_std']:.6f} "
        f"p50={stats['progress_delta_p50']:.6f} p95={stats['progress_delta_p95']:.6f} p99={stats['progress_delta_p99']:.6f}"
    )
    print(
        "reward diagnostics: "
        f"positive={stats['positive_reward_fraction']:.3f} "
        f"negative={stats['negative_reward_fraction']:.3f} "
        f"zero={stats['zero_reward_fraction']:.3f}"
    )
    if stats["recommendations"]:
        print("Recommendations:")
        for line in stats["recommendations"]:
            print(f"  - {line}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze offline CQL mask+depth HDF5 dataset.")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="./datasets/lift_cube_lenient_teacher_maskdepth.hdf5",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Defaults to <dataset_stem>_export_stats.json beside the dataset.",
    )
    args = parser.parse_args()

    report = analyze_dataset(args.dataset_file)
    if args.output_json:
        out_path = Path(args.output_json)
    else:
        dataset_path = Path(args.dataset_file)
        out_path = dataset_path.with_name(f"{dataset_path.stem}_export_stats.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report["export_stats"], f, indent=2)
    print_export_report(report)
    print(f"[INFO] Wrote export statistics to {out_path}")


if __name__ == "__main__":
    main()
