# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for oracle goal-conditioned data collection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_box_minus, subtract_frame_transforms

from isaaclab_tasks.manager_based.manipulation.lift.mdp.terminations import object_reached_goal

from . import observations as lift_obs


@dataclass
class OracleEpisodeBuffers:
    """Per-environment buffers tracked across an episode."""

    prev_ee_pos_b: torch.Tensor
    prev_ee_quat_b: torch.Tensor
    prev_gripper_opening: torch.Tensor
    grasp_latched: torch.Tensor
    grasp_reward_fired: torch.Tensor
    lift_reward_fired: torch.Tensor


@dataclass
class OracleStepData:
    """All custom oracle dataset signals for one simulation step."""

    ee_velocity: torch.Tensor  # (N, 7): lin vel (3,) + ang vel (3,) + gripper opening rate (1,)
    ee_pose: torch.Tensor
    goal_position: torch.Tensor
    goal_phase: torch.Tensor
    reward_grasp: torch.Tensor
    reward_lift: torch.Tensor


def get_oracle_episode_buffers(env: ManagerBasedRLEnv) -> OracleEpisodeBuffers:
    """Get or create episode buffers stored on the environment instance."""
    if not hasattr(env, "_oracle_episode_buffers"):
        env._oracle_episode_buffers = OracleEpisodeBuffers(
            prev_ee_pos_b=torch.zeros(env.num_envs, 3, device=env.device),
            prev_ee_quat_b=torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1),
            prev_gripper_opening=torch.zeros(env.num_envs, device=env.device),
            grasp_latched=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            grasp_reward_fired=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            lift_reward_fired=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        )
    return env._oracle_episode_buffers


def reset_oracle_episode_buffers(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset oracle episode buffers for selected environments."""
    buffers = get_oracle_episode_buffers(env)
    if env_ids is None:
        env_ids = slice(None)
    ee_pos_b, ee_quat_b = get_ee_pose_in_robot_root_frame(env)
    gripper_opening = get_mean_finger_opening(env)
    buffers.prev_ee_pos_b[env_ids] = ee_pos_b[env_ids]
    buffers.prev_ee_quat_b[env_ids] = ee_quat_b[env_ids]
    buffers.prev_gripper_opening[env_ids] = gripper_opening[env_ids]
    buffers.grasp_latched[env_ids] = False
    buffers.grasp_reward_fired[env_ids] = False
    buffers.lift_reward_fired[env_ids] = False
    if hasattr(env, "_oracle_step_cache"):
        del env._oracle_step_cache


def get_ee_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return end-effector pose in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w)
    return ee_pos_b, ee_quat_b


def get_mean_finger_opening(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Return mean finger joint position as a scalar gripper opening metric."""
    if finger_joint_names is None:
        finger_joint_names = ["panda_finger_joint.*"]
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_ids, _ = robot.find_joints(finger_joint_names)
    return torch.mean(robot.data.joint_pos[:, finger_joint_ids], dim=1)


def compute_grasp_success(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.04,
    ee_threshold: float = 0.07,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Detect whether the cube has been grasped and lifted off the table."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    lifted = object_pos_w[:, 2] > minimal_height
    return lifted & (ee_distance < ee_threshold)


def compute_cube_to_goal_distance(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return L2 distance from the cube to the command goal (world frame).

    Matches ``object_reached_goal`` / task terminations. Prefer reading before
    ``ManagerBasedRLEnv.step`` on terminal frames: ``step()`` auto-resets done envs,
    so post-step object poses reflect the reset spawn, not the end-of-episode pose.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    return torch.norm(des_pos_w - obj.data.root_pos_w[:, :3], dim=1)


def compute_lift_success_from_snapshot(
    cube_to_goal_distance: torch.Tensor,
    cube_z_w: torch.Tensor,
    *,
    threshold: float = 0.02,
    minimal_height: float = 0.04,
) -> torch.Tensor:
    """Lift success from pre-step snapshots (valid on done frames after ``env.step``)."""
    return (cube_to_goal_distance <= threshold) & (cube_z_w > minimal_height)


def compute_grasp_success_from_snapshot(
    cube_z_w: torch.Tensor,
    *,
    minimal_height: float = 0.04,
) -> torch.Tensor:
    """Grasp proxy from pre-step snapshot: cube lifted above the table."""
    return cube_z_w > minimal_height


def compute_lift_success(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    minimal_height: float = 0.04,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Detect whether the lifted cube reached the final command goal."""
    obj: RigidObject = env.scene[object_cfg.name]
    reached_goal = object_reached_goal(
        env,
        command_name=command_name,
        threshold=threshold,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    )
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    return reached_goal & lifted


class EpisodeLiftSuccessTracker:
    """Track per-env lift/grasp success across a rollout.

    ``ManagerBasedRLEnv.step`` auto-resets done envs before returning, so post-step
    success must be recorded on still-active envs during the loop. On terminal steps,
    merge the pre-step snapshot (valid end-of-episode pose) into the running flags.
    """

    def __init__(self, num_envs: int, device: str | torch.device):
        self.ever_lift_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ever_grasp_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.ever_lift_success[env_ids] = False
        self.ever_grasp_success[env_ids] = False

    def on_step(
        self,
        env: ManagerBasedRLEnv,
        *,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        pre_cube_to_goal: torch.Tensor,
        pre_cube_z_w: torch.Tensor,
        lift_threshold: float = 0.02,
    ) -> torch.Tensor:
        """Update running success and return envs that newly achieved lift success."""
        active = ~(terminated | truncated)
        if active.any():
            self.ever_grasp_success[active] |= compute_grasp_success(env)[active]
            self.ever_lift_success[active] |= compute_lift_success(env, threshold=lift_threshold)[active]

        done_now = terminated | truncated
        if done_now.any():
            terminal_lift = compute_lift_success_from_snapshot(
                pre_cube_to_goal, pre_cube_z_w, threshold=lift_threshold
            )
            terminal_grasp = compute_grasp_success_from_snapshot(pre_cube_z_w)
            self.ever_lift_success[done_now] |= terminal_lift[done_now]
            self.ever_grasp_success[done_now] |= terminal_grasp[done_now]

        return self.ever_lift_success & ~done_now


def get_lift_goal_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
) -> torch.Tensor:
    """Return the final lift target position in the robot root frame."""
    command = env.command_manager.get_command(command_name)
    return command[:, :3]


def compute_oracle_step_data(
    env: ManagerBasedRLEnv,
    *,
    control_dt: float,
    grasp_ee_threshold: float = 0.07,
    grasp_min_height: float = 0.04,
    lift_goal_threshold: float = 0.02,
) -> OracleStepData:
    """Compute custom oracle dataset signals for the current step."""
    step_count = env.common_step_counter
    cache = getattr(env, "_oracle_step_cache", None)
    if cache is not None and cache[0] == step_count:
        return cache[1]

    buffers = get_oracle_episode_buffers(env)
    ee_pos_b, ee_quat_b = get_ee_pose_in_robot_root_frame(env)
    ee_lin_vel_b = (ee_pos_b - buffers.prev_ee_pos_b) / control_dt
    ee_ang_vel_b = quat_box_minus(ee_quat_b, buffers.prev_ee_quat_b) / control_dt
    gripper_opening = get_mean_finger_opening(env)
    gripper_opening_vel = (gripper_opening - buffers.prev_gripper_opening) / control_dt
    ee_velocity = torch.cat([ee_lin_vel_b, ee_ang_vel_b, gripper_opening_vel.unsqueeze(-1)], dim=-1)
    buffers.prev_ee_pos_b[:] = ee_pos_b
    buffers.prev_ee_quat_b[:] = ee_quat_b
    buffers.prev_gripper_opening[:] = gripper_opening

    cube_pos_b = lift_obs.object_position_in_robot_root_frame(env)
    lift_goal_b = get_lift_goal_position_in_robot_root_frame(env)

    grasp_now = compute_grasp_success(
        env,
        minimal_height=grasp_min_height,
        ee_threshold=grasp_ee_threshold,
    )
    buffers.grasp_latched |= grasp_now

    lift_now = compute_lift_success(env, threshold=lift_goal_threshold, minimal_height=grasp_min_height)
    lift_now &= buffers.grasp_latched

    grasp_reward = (grasp_now & ~buffers.grasp_reward_fired).float()
    lift_reward = (lift_now & ~buffers.lift_reward_fired).float()
    buffers.grasp_reward_fired |= grasp_now
    buffers.lift_reward_fired |= lift_now

    goal_phase = buffers.grasp_latched.float().unsqueeze(-1)
    goal_position = torch.where(buffers.grasp_latched.unsqueeze(-1), lift_goal_b, cube_pos_b)

    ee_pose = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    data = OracleStepData(
        ee_velocity=ee_velocity,
        ee_pose=ee_pose,
        goal_position=goal_position,
        goal_phase=goal_phase,
        reward_grasp=grasp_reward.unsqueeze(-1),
        reward_lift=lift_reward.unsqueeze(-1),
    )
    env._oracle_step_cache = (step_count, data)
    return data


# ---------------------------------------------------------------------------
# Dense progress reward for offline RL (CQL)
# ---------------------------------------------------------------------------

REACH_DISTANCE_SCALE = 0.30
LIFT_HEIGHT_SCALE = 0.20
GOAL_DISTANCE_SCALE = 0.40
PROGRESS_WEIGHTS = {
    "reach": 0.20,
    "grasp": 0.20,
    "lift": 0.25,
    "goal": 0.30,
    "success": 0.05,
}
DEFAULT_TABLE_SURFACE_Z = 0.0
DEFAULT_PROGRESS_TERMINAL_BONUS = 0.0
PROGRESS_REWARD_SCALE = 50.0


@dataclass
class LiftProgressScores:
    """Decomposed progress scores in [0, 1]."""

    reach_score: torch.Tensor
    grasp_score: torch.Tensor
    lift_score: torch.Tensor
    goal_score: torch.Tensor
    success_score: torch.Tensor


@dataclass
class ProgressRewardStepData:
    """Progress-based offline RL reward for one transition (s_t, a_t) -> s_{t+1}."""

    progress: torch.Tensor
    progress_reward: torch.Tensor
    progress_delta: torch.Tensor
    terminal_bonus: torch.Tensor
    scores: LiftProgressScores


@dataclass
class ProgressEpisodeBuffers:
    """Per-environment progress state for reward differencing."""

    prev_progress: torch.Tensor


def get_progress_reward_config(env: ManagerBasedRLEnv) -> dict[str, float]:
    """Read progress-reward hyperparameters from ``env._progress_reward_cfg``."""
    defaults = {
        "goal_threshold": 0.02,
        "terminal_bonus": DEFAULT_PROGRESS_TERMINAL_BONUS,
        "table_z": DEFAULT_TABLE_SURFACE_Z,
        "grasp_min_height": 0.04,
        "grasp_ee_threshold": 0.07,
    }
    overrides = getattr(env, "_progress_reward_cfg", None)
    if overrides is None:
        return defaults
    return {**defaults, **overrides}


def get_progress_episode_buffers(env: ManagerBasedRLEnv) -> ProgressEpisodeBuffers:
    """Get or create per-env progress buffers on the environment instance."""
    if not hasattr(env, "_progress_episode_buffers"):
        env._progress_episode_buffers = ProgressEpisodeBuffers(
            prev_progress=torch.zeros(env.num_envs, device=env.device),
        )
    return env._progress_episode_buffers


def reset_progress_episode_buffers(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset progress buffers and seed ``prev_progress`` from the post-reset state."""
    buffers = get_progress_episode_buffers(env)
    if env_ids is None:
        env_ids = slice(None)
    progress = compute_lift_progress(env)[1]
    buffers.prev_progress[env_ids] = progress[env_ids]
    if hasattr(env, "_progress_reward_cache"):
        del env._progress_reward_cache


def compute_lift_progress_scores(
    env: ManagerBasedRLEnv,
    *,
    goal_threshold: float = 0.02,
    table_z: float = DEFAULT_TABLE_SURFACE_Z,
    grasp_min_height: float = 0.04,
    grasp_ee_threshold: float = 0.07,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> LiftProgressScores:
    """Compute decomposed progress scores for the current simulator state."""
    ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env, robot_cfg=robot_cfg)
    cube_pos_b = lift_obs.object_position_in_robot_root_frame(env, robot_cfg=robot_cfg, object_cfg=object_cfg)
    reach_distance = torch.norm(ee_pos_b - cube_pos_b, dim=-1)
    reach_score = 1.0 - torch.clamp(reach_distance / REACH_DISTANCE_SCALE, 0.0, 1.0)

    grasp_score = compute_grasp_success(
        env,
        minimal_height=grasp_min_height,
        ee_threshold=grasp_ee_threshold,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    ).float()

    obj: RigidObject = env.scene[object_cfg.name]
    cube_z_w = obj.data.root_pos_w[:, 2]
    lift_score = torch.clamp((cube_z_w - table_z) / LIFT_HEIGHT_SCALE, 0.0, 1.0)

    cube_to_goal = compute_cube_to_goal_distance(env, robot_cfg=robot_cfg, object_cfg=object_cfg)
    goal_score = 1.0 - torch.clamp(cube_to_goal / GOAL_DISTANCE_SCALE, 0.0, 1.0)

    success_score = object_reached_goal(
        env,
        threshold=goal_threshold,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    ).float()

    return LiftProgressScores(
        reach_score=reach_score,
        grasp_score=grasp_score,
        lift_score=lift_score,
        goal_score=goal_score,
        success_score=success_score,
    )


def compute_lift_progress(
    env: ManagerBasedRLEnv,
    *,
    goal_threshold: float = 0.02,
    table_z: float = DEFAULT_TABLE_SURFACE_Z,
    grasp_min_height: float = 0.04,
    grasp_ee_threshold: float = 0.07,
) -> tuple[LiftProgressScores, torch.Tensor]:
    """Return decomposed scores and scalar ``progress in [0, 1]``."""
    scores = compute_lift_progress_scores(
        env,
        goal_threshold=goal_threshold,
        table_z=table_z,
        grasp_min_height=grasp_min_height,
        grasp_ee_threshold=grasp_ee_threshold,
    )
    progress = (
        PROGRESS_WEIGHTS["reach"] * scores.reach_score
        + PROGRESS_WEIGHTS["grasp"] * scores.grasp_score
        + PROGRESS_WEIGHTS["lift"] * scores.lift_score
        + PROGRESS_WEIGHTS["goal"] * scores.goal_score
        + PROGRESS_WEIGHTS["success"] * scores.success_score
    )
    return scores, progress


def compute_progress_reward_step_data(env: ManagerBasedRLEnv) -> ProgressRewardStepData:
    """Compute scaled dense progress reward for offline RL.

    Learning reward:
        ``progress_reward = PROGRESS_REWARD_SCALE * (progress(s_{t+1}) - progress(s_t))``

    ``terminal_bonus`` is logged for debugging only and is not added to ``progress_reward``.
    """
    cfg = get_progress_reward_config(env)
    step_count = env.common_step_counter
    cache = getattr(env, "_progress_reward_cache", None)
    if cache is not None and cache[0] == step_count:
        return cache[1]

    scores, progress = compute_lift_progress(
        env,
        goal_threshold=cfg["goal_threshold"],
        table_z=cfg["table_z"],
        grasp_min_height=cfg["grasp_min_height"],
        grasp_ee_threshold=cfg["grasp_ee_threshold"],
    )
    buffers = get_progress_episode_buffers(env)
    progress_delta = progress - buffers.prev_progress
    terminal_bonus = scores.success_score * cfg["terminal_bonus"]
    progress_reward = PROGRESS_REWARD_SCALE * progress_delta
    buffers.prev_progress[:] = progress

    data = ProgressRewardStepData(
        progress=progress,
        progress_reward=progress_reward,
        progress_delta=progress_delta,
        terminal_bonus=terminal_bonus,
        scores=scores,
    )
    env._progress_reward_cache = (step_count, data)
    return data
