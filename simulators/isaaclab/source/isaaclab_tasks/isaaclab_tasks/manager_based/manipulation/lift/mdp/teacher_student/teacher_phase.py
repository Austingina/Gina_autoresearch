# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lenient teacher phase inference for student imitation (decoupled from strict oracle SM)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv

from ..data_collection import get_ee_pose_in_robot_root_frame
from ..observations import object_position_in_robot_root_frame
from ..oracle.lift_cube_oracle import LiftCubeOraclePolicy
from .gripper_schedule import SM_ABOVE, SM_APPROACH, SM_GRASP, SM_LIFT, SM_REST

# Match oracle PickSmWaitTime.GRASP_OBJECT (0.3 s) at default control_dt=0.02 s.
DEFAULT_GRASP_HOLD_STEPS = 15


def _get_teacher_phase_state(
    env: ManagerBasedRLEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return grasp-hold counters, at-grasp latch, close latch, lift latch, and latched EE–cube offset."""
    if not hasattr(env, "_teacher_grasp_hold_steps"):
        env._teacher_grasp_hold_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_teacher_at_grasp_latched"):
        env._teacher_at_grasp_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_teacher_gripper_close_latched"):
        env._teacher_gripper_close_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_teacher_lift_latched"):
        env._teacher_lift_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_teacher_lift_ee_offset_b"):
        env._teacher_lift_ee_offset_b = torch.zeros(env.num_envs, 3, device=env.device)
    return (
        env._teacher_grasp_hold_steps,
        env._teacher_at_grasp_latched,
        env._teacher_gripper_close_latched,
        env._teacher_lift_latched,
        env._teacher_lift_ee_offset_b,
    )


def is_teacher_gripper_close_latched(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return per-env binary gripper-close latch (commanded close, not finger joint opening)."""
    if not hasattr(env, "_teacher_gripper_close_latched"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env._teacher_gripper_close_latched


def _materialize_teacher_phase_state(env: ManagerBasedRLEnv) -> None:
    """Replace inference-mode tensors with writable clones (safe after ``inference_mode``)."""
    grasp_hold, at_grasp_latch, close_latch, lift_latch, lift_ee_offset = _get_teacher_phase_state(env)
    if grasp_hold.is_inference():
        env._teacher_grasp_hold_steps = grasp_hold.detach().clone()
        env._teacher_at_grasp_latched = at_grasp_latch.detach().clone()
        env._teacher_gripper_close_latched = close_latch.detach().clone()
        env._teacher_lift_latched = lift_latch.detach().clone()
        env._teacher_lift_ee_offset_b = lift_ee_offset.detach().clone()


def reset_teacher_phase_state(env: ManagerBasedRLEnv, env_ids: Sequence[int] | slice | None = None) -> None:
    """Reset lenient teacher phase latch state after env reset."""
    _materialize_teacher_phase_state(env)
    grasp_hold, at_grasp_latch, close_latch, lift_latch, lift_ee_offset = _get_teacher_phase_state(env)
    if env_ids is None:
        env_ids = slice(None)
    grasp_hold[env_ids] = 0
    at_grasp_latch[env_ids] = False
    close_latch[env_ids] = False
    lift_latch[env_ids] = False
    lift_ee_offset[env_ids] = 0.0


def infer_teacher_phase(
    env: ManagerBasedRLEnv,
    *,
    above_offset_z_m: float = 0.10,
    approach_height_m: float = 0.050,
    approach_start_xy_m: float = 0.060,
    approach_start_height_margin_m: float = 0.020,
    grasp_height_m: float = 0.015,
    grasp_xy_m: float = 0.050,
    grasp_hold_steps: int = DEFAULT_GRASP_HOLD_STEPS,
) -> torch.Tensor:
    """Infer a lenient teacher phase from EE–cube geometry.

    Unlike the oracle execution SM, this does not require the student to hit 1 cm
    waypoints or wait timers. When the EE is already near the cube in height/xy,
    the teacher advances to APPROACH / GRASP labels even if the strict oracle SM
    is still in ABOVE.

    After closing near the cube, the teacher latches into LIFT (oracle SM closes for
    0.3 s at the cube, then lifts). The EE frame sits above the cube center even
    when the fingers grasp, so LIFT must not require ``height_above <= grasp_height``.
    """
    env_overrides = getattr(env, "_teacher_phase_geometry_kwargs", None) or {}
    above_offset_z_m = float(env_overrides.get("above_offset_z_m", above_offset_z_m))
    grasp_hold_steps = int(env_overrides.get("grasp_hold_steps", grasp_hold_steps))
    ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
    cube_pos_b = object_position_in_robot_root_frame(env)

    height_above = torch.clamp(ee_pos_b[:, 2] - cube_pos_b[:, 2], min=0.0)
    xy_dist = torch.norm(ee_pos_b[:, :2] - cube_pos_b[:, :2], dim=-1)

    phase = torch.full((env.num_envs,), SM_ABOVE, dtype=torch.long, device=env.device)

    in_hover_band = height_above <= (above_offset_z_m + approach_start_height_margin_m)
    xy_aligned = xy_dist <= approach_start_xy_m
    at_approach_height = height_above <= approach_height_m
    should_descend = at_approach_height | (in_hover_band & xy_aligned)
    phase = torch.where(should_descend, torch.full_like(phase, SM_APPROACH), phase)

    at_grasp = (height_above <= grasp_height_m) & (xy_dist <= grasp_xy_m)

    grasp_hold, at_grasp_latch, gripper_close_latched, lift_latch, lift_ee_offset = _get_teacher_phase_state(env)
    at_grasp_latch = at_grasp_latch | at_grasp
    env._teacher_at_grasp_latched = at_grasp_latch

    # Binary close: once the EE reaches the grasp pose, latch close until episode reset.
    # Do not infer closure from finger joint opening — a grasped cube keeps a non-zero gap.
    gripper_close_latched = gripper_close_latched | at_grasp
    env._teacher_gripper_close_latched = gripper_close_latched

    # Hold closed briefly (oracle GRASP_OBJECT = 0.3 s), then latch LIFT.
    grasp_hold = torch.where(
        gripper_close_latched & ~lift_latch,
        grasp_hold + 1,
        torch.where(~gripper_close_latched, torch.zeros_like(grasp_hold), grasp_hold),
    )
    env._teacher_grasp_hold_steps = grasp_hold

    ready_to_lift = gripper_close_latched & (grasp_hold >= grasp_hold_steps) & ~lift_latch
    if ready_to_lift.any():
        lift_ee_offset[ready_to_lift] = ee_pos_b[ready_to_lift] - cube_pos_b[ready_to_lift]
        env._teacher_lift_ee_offset_b = lift_ee_offset
    lift_latch = lift_latch | ready_to_lift
    env._teacher_lift_latched = lift_latch
    phase = torch.where(lift_latch, torch.full_like(phase, SM_LIFT), phase)
    phase = torch.where(gripper_close_latched & ~lift_latch, torch.full_like(phase, SM_GRASP), phase)

    still_high = height_above > (above_offset_z_m + approach_height_m)
    phase = torch.where(still_high & (phase < SM_LIFT), torch.full_like(phase, SM_ABOVE), phase)

    return phase


def teacher_desired_ee_in_base_frame(
    env: ManagerBasedRLEnv,
    oracle: LiftCubeOraclePolicy,
    teacher_phase: torch.Tensor,
    *,
    above_offset_z_m: float = 0.10,
    command_name: str = "object_pose",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return teacher target EE pose in robot base frame for the lenient teacher phase."""
    _, object_pose, _ = oracle._get_task_poses()
    ee_pos_b, ee_quat_b = get_ee_pose_in_robot_root_frame(env)
    robot = env.scene["robot"]
    obj_pos_env = object_pose[:, :3]
    obj_quat_env = object_pose[:, 3:7]
    obj_pos_w = obj_pos_env + env.scene.env_origins
    cube_pos_b, des_quat_b_obj = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w, obj_quat_env
    )
    hover_pos_b = cube_pos_b.clone()
    hover_pos_b[:, 2] = hover_pos_b[:, 2] + above_offset_z_m

    # ``object_pose`` command is already in the robot root frame (see UniformPoseCommand).
    lift_goal_b = env.command_manager.get_command(command_name)[:, :3]
    _, _, _, lift_latch, lift_ee_offset = _get_teacher_phase_state(env)
    # Drive the EE so the grasped cube center can reach the command goal (not the EE origin).
    lift_ee_goal_b = lift_goal_b + lift_ee_offset

    des_pos_b = hover_pos_b.clone()
    des_pos_b = torch.where(
        teacher_phase.unsqueeze(-1) >= SM_APPROACH,
        cube_pos_b,
        des_pos_b,
    )
    des_pos_b = torch.where(
        lift_latch.unsqueeze(-1),
        lift_ee_goal_b,
        des_pos_b,
    )
    des_pos_b = torch.where(
        teacher_phase.unsqueeze(-1) <= SM_REST,
        ee_pos_b,
        des_pos_b,
    )

    des_quat_b = ee_quat_b.clone()
    des_quat_b = torch.where(
        (teacher_phase >= SM_APPROACH).unsqueeze(-1),
        des_quat_b_obj,
        des_quat_b,
    )
    return des_pos_b, des_quat_b


def teacher_desired_ee_pos_in_base_frame(
    env: ManagerBasedRLEnv,
    oracle: LiftCubeOraclePolicy,
    teacher_phase: torch.Tensor,
    *,
    above_offset_z_m: float = 0.10,
) -> torch.Tensor:
    """Teacher target position only (for early-done progress tracking)."""
    des_pos_b, _ = teacher_desired_ee_in_base_frame(
        env, oracle, teacher_phase, above_offset_z_m=above_offset_z_m
    )
    return des_pos_b
