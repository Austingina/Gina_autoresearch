# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Interaction Geometry runtime helpers for lift (ψ, κ) — see AutoResearch docs/INTERACTION_GEOMETRY.md."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .observations import object_position_in_robot_root_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

DEFAULT_LIFT_HEIGHT_ABOVE_CUBE_M = 0.20


def lift_goal_above_cube_enabled(env: ManagerBasedRLEnv) -> bool:
    cfg = getattr(env, "cfg", None)
    return bool(getattr(cfg, "lift_goal_above_cube", False))


def compute_lift_actuate_goal_from_cube(
    cube_pos_b: torch.Tensor,
    *,
    height_above: float = DEFAULT_LIFT_HEIGHT_ABOVE_CUBE_M,
) -> torch.Tensor:
    """Attractor directly above cube: same x,y; z + height_above (robot base frame)."""
    goal = cube_pos_b.clone()
    goal[:, 2] = cube_pos_b[:, 2] + float(height_above)
    return goal


def sync_lift_goal_above_cube(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "object_pose",
    height_above: float = DEFAULT_LIFT_HEIGHT_ABOVE_CUBE_M,
) -> None:
    """On reset: set object_pose command to cube spawn xy + vertical lift height."""
    if not lift_goal_above_cube_enabled(env):
        return
    if isinstance(env_ids, slice):
        id_tensor = torch.arange(env.num_envs, device=env.device)
    else:
        id_tensor = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
    if id_tensor.numel() == 0:
        return

    term = env.command_manager.get_term(command_name)
    cube_pos_b = object_position_in_robot_root_frame(env)[id_tensor]
    goal_b = compute_lift_actuate_goal_from_cube(cube_pos_b, height_above=height_above)
    term.pose_command_b[id_tensor, :3] = goal_b


def compute_lift_subgoal_vectors_3d(
    ee_pos_b: torch.Tensor,
    cube_pos_b: torch.Tensor,
    teacher_phase: torch.Tensor,
    *,
    grasp_phase_min: int = 3,
) -> torch.Tensor:
    """Oracle / B3 G: REACH EE→cube; ACTUATE +z above cube (κ=vertical_free)."""
    grasped = teacher_phase >= grasp_phase_min
    vec = torch.zeros_like(ee_pos_b)
    if (~grasped).any():
        delta = cube_pos_b - ee_pos_b
        vec[~grasped] = torch.nn.functional.normalize(delta[~grasped], dim=-1, eps=1e-6)
    if grasped.any():
        vec[grasped] = torch.tensor([0.0, 0.0, 1.0], device=vec.device, dtype=vec.dtype)
    norm = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1e-6)
    return (vec / norm).to(dtype=torch.float32)
