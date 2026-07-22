# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Oracle teacher labels for 7D EE commands (6D twist velocity + gripper opening rate)."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv

from ..data_collection import get_ee_pose_in_robot_root_frame, get_mean_finger_opening
from ..oracle.lift_cube_oracle import LiftCubeOraclePolicy
from .gripper_schedule import SM_LIFT, compute_desired_gripper_opening
from .teacher_phase import infer_teacher_phase, is_teacher_gripper_close_latched, teacher_desired_ee_in_base_frame

GRIPPER_OPEN_MAX_M = 0.04
GRIPPER_CLOSE_SNAP_M = 0.006
TEACHER_GRIPPER_CLOSE_RATE_THRESHOLD = -0.01
TEACHER_MIN_LIN_SPEED_M_S = 0.10
TEACHER_MIN_ANG_SPEED_RAD_S = 0.20
TEACHER_VEL_EPS = 1e-6


def teacher_gripper_is_close(gripper_rate: torch.Tensor) -> torch.Tensor:
    """True where continuous teacher gripper rate requests closing."""
    return gripper_rate < TEACHER_GRIPPER_CLOSE_RATE_THRESHOLD


GRIPPER_BINARY_CLOSE_THRESHOLD_M = 0.015


def opening_to_binary_gripper_label(
    opening: torch.Tensor,
    *,
    open_max: float = GRIPPER_OPEN_MAX_M,
    close_snap_m: float = GRIPPER_CLOSE_SNAP_M,
    close_threshold_m: float = GRIPPER_BINARY_CLOSE_THRESHOLD_M,
) -> torch.Tensor:
    """Map finger opening (m) to binary teacher label (+1 open, -1 close)."""
    opening = torch.clamp(opening, 0.0, open_max)
    close = opening <= close_snap_m
    fully_open = opening >= open_max - 1e-3
    return torch.where(
        close,
        torch.full_like(opening, -1.0),
        torch.where(fully_open, torch.full_like(opening, 1.0), torch.where(opening < close_threshold_m, -1.0, 1.0)),
    )


def opening_to_gripper_raw(
    opening: torch.Tensor,
    *,
    open_max: float = GRIPPER_OPEN_MAX_M,
    close_snap_m: float = GRIPPER_CLOSE_SNAP_M,
) -> torch.Tensor:
    """Map finger opening (m) to JointPositionAction raw commands with endpoint snaps."""
    opening = torch.clamp(opening, 0.0, open_max)
    raw = (opening - open_max) / open_max
    raw = torch.where(opening <= close_snap_m, torch.full_like(raw, -1.0), raw)
    raw = torch.where(opening >= open_max - 1e-3, torch.zeros_like(raw), raw)
    return raw


def get_gripper_action_dim(env: ManagerBasedRLEnv, term_name: str = "gripper_action") -> int:
    """Return the gripper action dimension configured in the environment."""
    return int(env.action_manager.get_term(term_name).action_dim)


def twist_to_env_action(
    twist: torch.Tensor,
    *,
    control_dt: float,
    current_opening: torch.Tensor,
    arm_scale: float = 0.5,
    max_lin_vel: float = 0.5,
    max_ang_vel: float = 1.0,
    max_gripper_rate: float = 0.08,
    open_max: float = GRIPPER_OPEN_MAX_M,
    num_gripper_actions: int = 2,
    desired_opening: torch.Tensor | None = None,
    binary_gripper: bool = False,
) -> torch.Tensor:
    """Map a 7D EE command to IK-Rel + finger position actions.

    Gripper dim semantics:
    - ``binary_gripper=False``: dim 6 is continuous opening rate (m/s).
    - ``binary_gripper=True``: dim 6 is +1 open / -1 close.
    """
    twist = twist.clone()
    twist[:, :3] = torch.clamp(twist[:, :3], -max_lin_vel, max_lin_vel)
    twist[:, 3:6] = torch.clamp(twist[:, 3:6], -max_ang_vel, max_ang_vel)

    delta_pose = twist[:, :6] * control_dt
    raw_arm = delta_pose / arm_scale

    if current_opening.ndim == 1:
        current_opening = current_opening.unsqueeze(-1)

    if binary_gripper:
        gripper_cmd = twist[:, 6]
        gripper_raw = torch.where(gripper_cmd >= 0.0, 0.0, -1.0).unsqueeze(-1)
    elif desired_opening is not None:
        if desired_opening.ndim == 1:
            desired_opening = desired_opening.unsqueeze(-1)
        target_opening = torch.clamp(desired_opening, 0.0, open_max)
        gripper_raw = opening_to_gripper_raw(target_opening, open_max=open_max)
    else:
        gripper_rate = torch.clamp(twist[:, 6], -max_gripper_rate, max_gripper_rate)
        target_opening = torch.clamp(current_opening + gripper_rate.unsqueeze(-1) * control_dt, 0.0, open_max)
        gripper_raw = opening_to_gripper_raw(target_opening, open_max=open_max)
    if num_gripper_actions > 1:
        gripper_raw = gripper_raw.expand(-1, num_gripper_actions)
    return torch.cat([raw_arm, gripper_raw], dim=-1)


def _compute_teacher_arm_and_gripper_targets(
    env: ManagerBasedRLEnv,
    oracle: LiftCubeOraclePolicy,
    *,
    open_max: float = GRIPPER_OPEN_MAX_M,
    grasp_hold_steps: int | None = None,
    teacher_phase_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lenient teacher targets decoupled from strict oracle execution SM gating."""
    ee_pose, object_pose, des_object_pose = oracle._get_task_poses()
    # Tick strict SM for diagnostics only; teacher labels use lenient ``teacher_phase``.
    oracle._pick_sm.compute(ee_pose, object_pose, des_object_pose)

    phase_kwargs = dict(teacher_phase_kwargs or {})
    if grasp_hold_steps is not None:
        phase_kwargs["grasp_hold_steps"] = grasp_hold_steps
    teacher_phase = infer_teacher_phase(env, **phase_kwargs)
    desired_kwargs = {k: v for k, v in phase_kwargs.items() if k in ("above_offset_z_m", "command_name")}
    des_pos_b, des_quat_b = teacher_desired_ee_in_base_frame(env, oracle, teacher_phase, **desired_kwargs)
    desired_opening = compute_desired_gripper_opening(env, oracle, open_max=open_max, teacher_phase=teacher_phase)
    force_close_gripper = is_teacher_gripper_close_latched(env)
    return teacher_phase, des_pos_b, des_quat_b, desired_opening, force_close_gripper


def _apply_min_arm_twist_speed(
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    pos_error: torch.Tensor,
    *,
    min_lin_speed: float,
    min_ang_speed: float,
    teacher_phase: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raise arm twist speed norms to at least ``min_*``, preserving direction when possible."""
    lin_norm = torch.linalg.norm(lin_vel, dim=-1, keepdim=True)
    ang_norm = torch.linalg.norm(ang_vel, dim=-1, keepdim=True)
    pos_norm = torch.linalg.norm(pos_error, dim=-1, keepdim=True)

    lin_dir = lin_vel / lin_norm.clamp(min=TEACHER_VEL_EPS)
    pos_dir = pos_error / pos_norm.clamp(min=TEACHER_VEL_EPS)
    default_lin_dir_down = torch.zeros_like(lin_vel)
    default_lin_dir_down[:, 2] = -1.0
    default_lin_dir_up = torch.zeros_like(lin_vel)
    default_lin_dir_up[:, 2] = 1.0
    if teacher_phase is not None:
        default_lin_dir = torch.where(
            (teacher_phase >= SM_LIFT).unsqueeze(-1),
            default_lin_dir_up,
            default_lin_dir_down,
        )
    else:
        default_lin_dir = default_lin_dir_down

    lin_small = lin_norm.squeeze(-1) < min_lin_speed
    lin_dir = torch.where(
        lin_small.unsqueeze(-1) & (pos_norm > TEACHER_VEL_EPS),
        pos_dir,
        lin_dir,
    )
    lin_dir = torch.where(
        (lin_small & (pos_norm.squeeze(-1) <= TEACHER_VEL_EPS)).unsqueeze(-1),
        default_lin_dir,
        lin_dir,
    )
    lin_out_norm = torch.where(lin_small.unsqueeze(-1), torch.full_like(lin_norm, min_lin_speed), lin_norm)
    lin_vel = lin_dir * lin_out_norm

    ang_dir = ang_vel / ang_norm.clamp(min=TEACHER_VEL_EPS)
    default_ang_dir = torch.zeros_like(ang_vel)
    default_ang_dir[:, 2] = 1.0
    ang_small = ang_norm.squeeze(-1) < min_ang_speed
    ang_dir = torch.where(ang_small.unsqueeze(-1), default_ang_dir, ang_dir)
    ang_out_norm = torch.where(ang_small.unsqueeze(-1), torch.full_like(ang_norm, min_ang_speed), ang_norm)
    ang_vel = ang_dir * ang_out_norm
    return lin_vel, ang_vel


def _clamp_vector_magnitude(vel: torch.Tensor, max_mag: float) -> torch.Tensor:
    """Clamp a batch of vectors by L2 norm while preserving direction."""
    norm = torch.linalg.norm(vel, dim=-1, keepdim=True)
    scale = torch.clamp(norm, max=max_mag) / norm.clamp(min=TEACHER_VEL_EPS)
    return vel * scale


def _compute_teacher_arm_twist(
    ee_pos_b: torch.Tensor,
    ee_quat_b: torch.Tensor,
    des_pos_b: torch.Tensor,
    des_quat_b: torch.Tensor,
    *,
    control_dt: float,
    max_lin_vel: float,
    max_ang_vel: float,
    min_lin_speed: float,
    min_ang_speed: float,
    teacher_phase: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 6D arm twist with non-zero minimum linear/angular speed."""
    pos_error = des_pos_b - ee_pos_b
    lin_vel = pos_error / control_dt
    ang_vel = math_utils.quat_box_minus(des_quat_b, ee_quat_b) / control_dt
    lin_vel, ang_vel = _apply_min_arm_twist_speed(
        lin_vel,
        ang_vel,
        pos_error,
        min_lin_speed=min_lin_speed,
        min_ang_speed=min_ang_speed,
        teacher_phase=teacher_phase,
    )
    lin_vel = _clamp_vector_magnitude(lin_vel, max_lin_vel)
    ang_vel = _clamp_vector_magnitude(ang_vel, max_ang_vel)
    return lin_vel, ang_vel


def compute_teacher_twist(
    env: ManagerBasedRLEnv,
    oracle: LiftCubeOraclePolicy,
    *,
    control_dt: float,
    max_lin_vel: float = 0.5,
    max_ang_vel: float = 1.0,
    max_gripper_rate: float = 0.08,
    open_max: float = GRIPPER_OPEN_MAX_M,
    min_lin_speed: float = TEACHER_MIN_LIN_SPEED_M_S,
    min_ang_speed: float = TEACHER_MIN_ANG_SPEED_RAD_S,
) -> torch.Tensor:
    """Compute teacher 7D labels: 6D EE twist velocity + continuous gripper opening rate."""
    teacher_phase, des_pos_b, des_quat_b, desired_opening, force_close_gripper = _compute_teacher_arm_and_gripper_targets(
        env, oracle, open_max=open_max
    )

    ee_pos_b, ee_quat_b = get_ee_pose_in_robot_root_frame(env)
    lin_vel, ang_vel = _compute_teacher_arm_twist(
        ee_pos_b,
        ee_quat_b,
        des_pos_b,
        des_quat_b,
        control_dt=control_dt,
        max_lin_vel=max_lin_vel,
        max_ang_vel=max_ang_vel,
        min_lin_speed=min_lin_speed,
        min_ang_speed=min_ang_speed,
        teacher_phase=teacher_phase,
    )

    current_opening = get_mean_finger_opening(env)
    gripper_rate = (desired_opening - current_opening) / control_dt
    gripper_rate = torch.where(
        force_close_gripper & (current_opening > GRIPPER_CLOSE_SNAP_M),
        torch.full_like(gripper_rate, -max_gripper_rate),
        gripper_rate,
    )

    return torch.cat(
        [
            lin_vel,
            ang_vel,
            torch.clamp(gripper_rate.unsqueeze(-1), -max_gripper_rate, max_gripper_rate),
        ],
        dim=-1,
    )


def compute_teacher_twist_binary_gripper(
    env: ManagerBasedRLEnv,
    oracle: LiftCubeOraclePolicy,
    *,
    control_dt: float,
    max_lin_vel: float = 0.5,
    max_ang_vel: float = 1.0,
    open_max: float = GRIPPER_OPEN_MAX_M,
    min_lin_speed: float = TEACHER_MIN_LIN_SPEED_M_S,
    min_ang_speed: float = TEACHER_MIN_ANG_SPEED_RAD_S,
    grasp_hold_steps: int | None = None,
    teacher_phase_kwargs: dict | None = None,
) -> torch.Tensor:
    """Compute teacher 7D labels: 6D EE twist velocity + binary gripper (+1 open, -1 close)."""
    teacher_phase, des_pos_b, des_quat_b, desired_opening, _ = _compute_teacher_arm_and_gripper_targets(
        env,
        oracle,
        open_max=open_max,
        grasp_hold_steps=grasp_hold_steps,
        teacher_phase_kwargs=teacher_phase_kwargs,
    )

    ee_pos_b, ee_quat_b = get_ee_pose_in_robot_root_frame(env)
    lin_vel, ang_vel = _compute_teacher_arm_twist(
        ee_pos_b,
        ee_quat_b,
        des_pos_b,
        des_quat_b,
        control_dt=control_dt,
        max_lin_vel=max_lin_vel,
        max_ang_vel=max_ang_vel,
        min_lin_speed=min_lin_speed,
        min_ang_speed=min_ang_speed,
        teacher_phase=teacher_phase,
    )
    gripper_binary = torch.where(
        desired_opening <= 1e-3,
        torch.full_like(desired_opening, -1.0),
        torch.full_like(desired_opening, 1.0),
    )

    return torch.cat([lin_vel, ang_vel, gripper_binary.unsqueeze(-1)], dim=-1)


def teacher_binary_gripper_to_bce_target(gripper_label: torch.Tensor) -> torch.Tensor:
    """Map teacher gripper labels (+1 open, -1 close) to BCE targets in [0, 1]."""
    return (gripper_label + 1.0) * 0.5


def get_oracle_desired_ee_in_base_frame(
    env: ManagerBasedRLEnv,
    oracle: LiftCubeOraclePolicy,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return oracle SM desired EE pose in robot base frame and gripper command."""
    ee_pose, object_pose, des_object_pose = oracle._get_task_poses()
    task_cmd = oracle._pick_sm.compute(ee_pose, object_pose, des_object_pose)

    robot: Articulation = env.scene["robot"]
    des_pos_env = task_cmd[:, :3]
    des_quat_env = task_cmd[:, 3:7]
    des_pos_w = des_pos_env + env.scene.env_origins
    des_pos_b, des_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_w, des_quat_env
    )
    return des_pos_b, des_quat_b, task_cmd[:, 7:8]


def make_teacher_std(
    action_dim: int,
    *,
    lin_std: float = 0.05,
    ang_std: float = 0.10,
    gripper_std: float = 0.02,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a fixed diagonal teacher standard deviation for 7D EE commands."""
    std = torch.tensor(
        [lin_std, lin_std, lin_std, ang_std, ang_std, ang_std, gripper_std],
        device=device,
        dtype=torch.float32,
    )
    if std.shape[0] != action_dim:
        raise ValueError(f"Expected action_dim={action_dim}, got std with dim {std.shape[0]}")
    return std


def make_kl_dim_weights(
    action_dim: int = 7,
    *,
    arm_weight: float = 1.0,
    gripper_weight: float = 0.2,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Per-dimension KL weights: lower weight on gripper to avoid starving arm learning."""
    weights = torch.full((action_dim,), arm_weight, device=device, dtype=torch.float32)
    if action_dim >= 7:
        weights[6] = gripper_weight
    return weights


def make_teacher_arm_std(
    *,
    lin_std: float = 0.05,
    ang_std: float = 0.10,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a fixed diagonal teacher std for the 6D arm twist dims."""
    return torch.tensor(
        [lin_std, lin_std, lin_std, ang_std, ang_std, ang_std],
        device=device,
        dtype=torch.float32,
    )
