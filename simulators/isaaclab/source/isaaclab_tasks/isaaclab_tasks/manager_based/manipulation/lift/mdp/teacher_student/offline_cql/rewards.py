# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reach-distance offline RL rewards recomputed from logged EE/cube positions."""

from __future__ import annotations

from typing import Any

import numpy as np

REACH_REWARD_SCALE = 20.0
NEAR_REWARD_SCALE = 0.1
NEAR_DISTANCE_DECAY = 30.0
SUCCESS_BONUS = 1.0

# ``obs/student_state`` layout from lenient-teacher mask+depth collection.
STUDENT_STATE_EE_SLICE = slice(0, 3)
STUDENT_STATE_CUBE_SLICE = slice(4, 7)


def ee_cube_distance(ee_pos: np.ndarray, cube_pos: np.ndarray) -> np.ndarray:
    """Return L2 distance between EE and cube positions, shape (T,)."""
    return np.linalg.norm(ee_pos.astype(np.float64) - cube_pos.astype(np.float64), axis=-1)


def compute_offline_reach_rewards(
    ee_pos: np.ndarray,
    cube_pos: np.ndarray,
    success_score: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute offline RL rewards for transitions ``obs[t] -> obs[t+1]``.

    Per transition:

        reach_reward = REACH_REWARD_SCALE * (d_ee_cube_t - d_ee_cube_{t+1})
        near_reward = NEAR_REWARD_SCALE * exp(-NEAR_DISTANCE_DECAY * d_ee_cube_{t+1})
        success_bonus = SUCCESS_BONUS if success at s_{t+1} else 0
        reward_total = reach_reward + near_reward + success_bonus

    Arrays are length ``T`` and aligned with ``env_actions[t]`` / CQL ``rewards[t]``.
    The final step uses ``d_{t+1} = d_t`` when no next observation exists.
    """
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    cube_pos = np.asarray(cube_pos, dtype=np.float64)
    if ee_pos.shape != cube_pos.shape:
        raise ValueError(f"ee_pos shape {ee_pos.shape} != cube_pos shape {cube_pos.shape}")
    if ee_pos.ndim != 2 or ee_pos.shape[-1] != 3:
        raise ValueError(f"Expected ee_pos (T, 3), got {ee_pos.shape}")

    num_steps = ee_pos.shape[0]
    distance = ee_cube_distance(ee_pos, cube_pos)

    distance_t = distance
    distance_tp1 = np.empty_like(distance)
    if num_steps > 1:
        distance_tp1[:-1] = distance[1:]
    distance_tp1[-1] = distance[-1]

    reach_delta = distance_t - distance_tp1
    reach_reward = (reach_delta * REACH_REWARD_SCALE).astype(np.float32)
    near_reward = (np.exp(-NEAR_DISTANCE_DECAY * distance_tp1) * NEAR_REWARD_SCALE).astype(np.float32)

    success_bonus = np.zeros(num_steps, dtype=np.float32)
    if success_score is not None:
        success = np.asarray(success_score, dtype=np.float32).reshape(-1)
        if success.shape[0] != num_steps:
            raise ValueError(f"success_score length {success.shape[0]} != num_steps {num_steps}")
        success_at_next = np.zeros(num_steps, dtype=np.float32)
        if num_steps > 1:
            success_at_next[:-1] = (success[1:] >= 0.5).astype(np.float32)
        success_at_next[-1] = float(success[-1] >= 0.5)
        success_bonus = success_at_next * SUCCESS_BONUS

    reward_total = (reach_reward + near_reward + success_bonus).astype(np.float32)

    return {
        "rewards": reward_total,
        "reach_reward": reach_reward,
        "near_reward": near_reward,
        "success_bonus": success_bonus,
        "ee_cube_distance": distance.astype(np.float32),
        "reach_delta": reach_delta.astype(np.float32),
    }


def summarize_reward_array(arr: np.ndarray) -> dict[str, float]:
    """Return mean/std/p50/p95/p99 for a 1D reward component."""
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
    }


def summarize_dataset_rewards(episodes: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """Aggregate reward component statistics across loaded episodes."""
    components = ("reach_reward", "near_reward", "success_bonus", "rewards")
    chunks = {name: [] for name in components}
    progress_delta_chunks: list[np.ndarray] = []

    for episode in episodes:
        for name in components:
            if name in episode:
                chunks[name].append(episode[name])
        if "progress_delta" in episode:
            progress_delta_chunks.append(episode["progress_delta"])

    summary: dict[str, Any] = {"reward_scheme": "reach_distance"}
    for name in components:
        if chunks[name]:
            summary[name] = summarize_reward_array(np.concatenate(chunks[name], axis=0))
        else:
            summary[name] = summarize_reward_array(np.array([]))

    if progress_delta_chunks:
        summary["progress_delta_diagnostic"] = summarize_reward_array(np.concatenate(progress_delta_chunks, axis=0))

    return summary


def format_reward_summary(summary: dict[str, Any]) -> str:
    """Human-readable multi-line summary for training logs."""
    lines = ["[INFO] Offline RL reward (reach-distance scheme):"]
    for key in ("reach_reward", "near_reward", "success_bonus", "rewards"):
        stats = summary.get(key, {})
        if not stats or stats.get("count", 0) == 0:
            continue
        lines.append(
            f"  {key}: mean={stats['mean']:.5f} std={stats['std']:.5f} "
            f"p50={stats['p50']:.5f} p95={stats['p95']:.5f} p99={stats['p99']:.5f}"
        )
    diag = summary.get("progress_delta_diagnostic")
    if diag and diag.get("count", 0) > 0:
        lines.append(
            f"  progress_delta (diagnostic only): mean={diag['mean']:.6f} std={diag['std']:.6f} "
            f"p50={diag['p50']:.6f} p95={diag['p95']:.6f} p99={diag['p99']:.6f}"
        )
    return "\n".join(lines)
