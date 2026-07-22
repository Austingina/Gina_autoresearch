# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-simulation evaluation for offline CQL policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.manipulation.lift.mdp.data_collection import (
    EpisodeLiftSuccessTracker,
    compute_cube_to_goal_distance,
)

from .checkpoint import build_agent_from_checkpoint, load_checkpoint
from .obs_utils import stack_visual_from_policy_obs
from .policy import OfflineCQLPolicy


@dataclass
class SimEvalMetrics:
    success_rate: float
    grasp_rate: float
    mean_episode_length: float
    mean_final_cube_to_goal: float
    num_episodes: int


def load_policy_from_checkpoint(
    checkpoint_path: str,
    *,
    seq_len: int,
    device: torch.device,
) -> tuple[OfflineCQLPolicy, dict]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    agent, action_stats, ckpt_args = build_agent_from_checkpoint(checkpoint, device=device)
    policy = OfflineCQLPolicy(agent, action_stats, seq_len=seq_len, device=device)
    return policy, ckpt_args


@torch.no_grad()
def run_sim_eval(
    policy: OfflineCQLPolicy,
    env: ManagerBasedRLEnv,
    *,
    num_episodes: int,
    goal_threshold: float = 0.02,
    image_size: int | None = None,
    max_envs: int | None = None,
) -> SimEvalMetrics:
    """Roll out the offline CQL policy in simulation and measure lift success."""
    num_envs = env.num_envs
    eval_envs = num_envs if max_envs is None else min(max(max_envs, 1), num_envs)
    device = policy.device

    agent = policy.agent
    agent.encoder.eval()
    agent.actor.eval()
    agent.critic1.eval()
    agent.critic2.eval()
    policy.reset(num_envs)
    success_tracker = EpisodeLiftSuccessTracker(num_envs, device)
    ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    episodes_done = 0
    length_sum = 0.0
    final_dist_sum = 0.0
    grasp_success_total = 0
    lift_success_total = 0

    env.reset()
    max_total_steps = max(num_episodes * env.max_episode_length, env.max_episode_length)
    global_step = 0

    obs_mode = policy.obs_mode
    while episodes_done < num_episodes and global_step < max_total_steps:
        obs = env.obs_buf["policy"]
        visual = proprio = None
        if obs_mode in ("vision", "vision_proprio"):
            visual = stack_visual_from_policy_obs(obs, image_size=image_size)
        if obs_mode in ("proprio", "vision_proprio"):
            proprio = obs["student_state"].detach().cpu().numpy()

        pre_cube_to_goal = compute_cube_to_goal_distance(env)
        pre_cube_z_w = env.scene["object"].data.root_pos_w[:, 2].clone()

        env_actions_np = policy.act(visual=visual, proprio=proprio)
        env_actions = torch.as_tensor(env_actions_np, dtype=torch.float32, device=device)
        _, _, terminated, truncated, _ = env.step(env_actions)

        ep_steps += 1
        global_step += 1
        success_tracker.on_step(
            env,
            terminated=terminated,
            truncated=truncated,
            pre_cube_to_goal=pre_cube_to_goal,
            pre_cube_z_w=pre_cube_z_w,
            lift_threshold=goal_threshold,
        )

        done = terminated | truncated
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.ndim == 0:
                done_ids = done_ids.unsqueeze(0)
            for env_id in done_ids.tolist():
                if env_id >= eval_envs:
                    continue
                if episodes_done >= num_episodes:
                    break
                episodes_done += 1
                length_sum += float(ep_steps[env_id].item())
                final_dist_sum += float(pre_cube_to_goal[env_id].item())
                if success_tracker.ever_grasp_success[env_id].item():
                    grasp_success_total += 1
                if success_tracker.ever_lift_success[env_id].item():
                    lift_success_total += 1
            policy.reset_envs(done_ids)
            success_tracker.reset(done_ids)
            ep_steps[done_ids] = 0

    if episodes_done == 0:
        return SimEvalMetrics(0.0, 0.0, 0.0, 0.0, 0)

    return SimEvalMetrics(
        success_rate=lift_success_total / episodes_done,
        grasp_rate=grasp_success_total / episodes_done,
        mean_episode_length=length_sum / episodes_done,
        mean_final_cube_to_goal=final_dist_sum / episodes_done,
        num_episodes=episodes_done,
    )
