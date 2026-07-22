# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse rewards for teacher-student RL."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..data_collection import compute_grasp_success, compute_lift_success, get_oracle_episode_buffers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sparse_grasp_reward(env: ManagerBasedRLEnv) -> float:
    """Return 1.0 on the first step where grasp succeeds in an episode."""
    buffers = get_oracle_episode_buffers(env)
    grasp_now = compute_grasp_success(env)
    reward = (grasp_now & ~buffers.grasp_reward_fired).float()
    buffers.grasp_reward_fired |= grasp_now
    buffers.grasp_latched |= grasp_now
    return reward


def sparse_lift_reward(env: ManagerBasedRLEnv) -> float:
    """Return 1.0 on the first step where lift succeeds in an episode."""
    buffers = get_oracle_episode_buffers(env)
    lift_now = compute_lift_success(env)
    lift_now &= buffers.grasp_latched
    reward = (lift_now & ~buffers.lift_reward_fired).float()
    buffers.lift_reward_fired |= lift_now
    return reward
