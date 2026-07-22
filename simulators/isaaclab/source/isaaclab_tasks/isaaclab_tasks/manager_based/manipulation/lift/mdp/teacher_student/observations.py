# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student policy observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from ..data_collection import (
    get_ee_pose_in_robot_root_frame,
    get_lift_goal_position_in_robot_root_frame,
    get_mean_finger_opening,
)
from ..observations import object_position_in_robot_root_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def student_proprio_observation(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return student proprioception: EE pos, gripper opening, cube pos, lift goal (10,)."""
    ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg)
    gripper_opening = get_mean_finger_opening(env, robot_cfg=robot_cfg).unsqueeze(-1)
    cube_pos_b = object_position_in_robot_root_frame(env, robot_cfg=robot_cfg)
    lift_goal_b = get_lift_goal_position_in_robot_root_frame(env, command_name=command_name)
    return torch.cat([ee_pos_b, gripper_opening, cube_pos_b, lift_goal_b], dim=-1)


def student_vision_proprio_observation(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return vision student proprio: EE pos, gripper opening, lift goal (7,) — no cube GT."""
    ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg)
    gripper_opening = get_mean_finger_opening(env, robot_cfg=robot_cfg).unsqueeze(-1)
    lift_goal_b = get_lift_goal_position_in_robot_root_frame(env, command_name=command_name)
    return torch.cat([ee_pos_b, gripper_opening, lift_goal_b], dim=-1)
