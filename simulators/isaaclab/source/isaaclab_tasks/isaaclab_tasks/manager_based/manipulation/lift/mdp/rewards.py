# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_lifted_near_ee(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    ee_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward lifting only when the end-effector is still close enough to plausibly grasp the object."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    return ((object_pos_w[:, 2] > minimal_height) & (ee_distance < ee_threshold)).float()


def gripper_close_near_object(
    env: ManagerBasedRLEnv,
    ee_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing the binary gripper only when the hand is close to the object."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    gripper_close_command = torch.clamp(-env.action_manager.action[:, -1], min=0.0, max=1.0)

    return (ee_distance < ee_threshold).float() * gripper_close_command


def _franka_finger_opening(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the mean Franka finger joint opening."""
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_ids, _ = robot.find_joints(["panda_finger_joint.*"])
    return torch.mean(robot.data.joint_pos[:, finger_joint_ids], dim=1)


def gripper_open_before_grasp(
    env: ManagerBasedRLEnv,
    approach_threshold: float,
    close_threshold: float,
    open_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping the gripper open while approaching, before the final grasp window."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    finger_opening = _franka_finger_opening(env, robot_cfg)
    gripper_open_score = torch.clamp(finger_opening / open_threshold, min=0.0, max=1.0)
    in_pregrasp_window = (ee_distance < approach_threshold) & (ee_distance > close_threshold)

    return in_pregrasp_window.float() * gripper_open_score


def gripper_closed_before_grasp(
    env: ManagerBasedRLEnv,
    approach_threshold: float,
    close_threshold: float,
    closed_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize closing the gripper before reaching the final grasp window."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    finger_opening = _franka_finger_opening(env, robot_cfg)
    gripper_closed_score = torch.clamp((closed_threshold - finger_opening) / closed_threshold, min=0.0, max=1.0)
    in_pregrasp_window = (ee_distance < approach_threshold) & (ee_distance > close_threshold)

    return in_pregrasp_window.float() * gripper_closed_score


def object_grasped(
    env: ManagerBasedRLEnv,
    ee_threshold: float,
    force_threshold: float,
    min_finger_opening: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a plausible two-finger grasp using contact forces when available."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    gripper_is_closing = env.action_manager.action[:, -1] < 0.0
    finger_opening = _franka_finger_opening(env, robot_cfg)
    cube_between_fingers = finger_opening > min_finger_opening
    near_and_closing = (ee_distance < ee_threshold) & gripper_is_closing & cube_between_fingers

    if "contact_grasp" not in env.scene.keys() or env.scene["contact_grasp"] is None:
        return near_and_closing.float()

    contact_grasp: ContactSensor = env.scene["contact_grasp"]
    contact_force_norm = torch.linalg.vector_norm(contact_grasp.data.net_forces_w, dim=2)
    both_fingers_contact = torch.all(contact_force_norm > force_threshold, dim=1)
    return (near_and_closing & both_fingers_contact).float()


def object_lifted_when_grasped(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    ee_threshold: float,
    force_threshold: float,
    min_finger_opening: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward lifting after the cube is plausibly grasped."""
    object: RigidObject = env.scene[object_cfg.name]
    grasped = object_grasped(env, ee_threshold, force_threshold, min_finger_opening, object_cfg, ee_frame_cfg)
    return grasped * (object.data.root_pos_w[:, 2] > minimal_height)


def object_goal_tracking_sigmoid_near_ee(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    ee_threshold: float,
    command_name: str,
    gain: float = 8.0,
    center: float = 0.3,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """MotrixLab-style coarse target tracking gated by lift height and grasp proximity."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    goal_distance = torch.norm(des_pos_w - object_pos_w, dim=1)
    ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    tracking = torch.sigmoid(-gain * (goal_distance - center))
    return tracking * (object_pos_w[:, 2] > minimal_height) * (ee_distance < ee_threshold)


def object_goal_tracking_sigmoid_when_grasped(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    ee_threshold: float,
    force_threshold: float,
    min_finger_opening: float,
    command_name: str,
    gain: float = 8.0,
    center: float = 0.3,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Coarse target tracking after a plausible grasp."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    goal_distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    tracking = torch.sigmoid(-gain * (goal_distance - center))
    grasped = object_grasped(env, ee_threshold, force_threshold, min_finger_opening, object_cfg, ee_frame_cfg)
    return tracking * (object.data.root_pos_w[:, 2] > minimal_height) * grasped


def object_goal_tracking_fine_near_ee(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    ee_threshold: float,
    command_name: str,
    std: float = 0.4,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """MotrixLab-style fine target tracking gated by lift height and grasp proximity."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    goal_distance = torch.norm(des_pos_w - object_pos_w, dim=1)
    ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    tracking = 1 - torch.tanh(goal_distance / std)
    return tracking * (object_pos_w[:, 2] > minimal_height) * (ee_distance < ee_threshold)


def object_goal_tracking_fine_when_grasped(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    ee_threshold: float,
    force_threshold: float,
    min_finger_opening: float,
    command_name: str,
    std: float = 0.4,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Fine target tracking after a plausible grasp."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    goal_distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    tracking = 1 - torch.tanh(goal_distance / std)
    grasped = object_grasped(env, ee_threshold, force_threshold, min_finger_opening, object_cfg, ee_frame_cfg)
    return tracking * (object.data.root_pos_w[:, 2] > minimal_height) * grasped


def object_goal_tracking_close_near_ee(
    env: ManagerBasedRLEnv,
    ee_threshold: float,
    command_name: str,
    close_threshold: float = 0.2,
    std: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """MotrixLab-style close-goal reward that strongly pulls successful grasps into the target region."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    goal_distance = torch.norm(des_pos_w - object_pos_w, dim=1)
    ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    close_tracking = (1 - torch.tanh(goal_distance / std)) * (goal_distance < close_threshold) * (
        ee_distance < ee_threshold
    )
    return close_tracking


def object_goal_tracking_close_when_grasped(
    env: ManagerBasedRLEnv,
    ee_threshold: float,
    force_threshold: float,
    min_finger_opening: float,
    command_name: str,
    close_threshold: float = 0.2,
    std: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Close-goal reward once the cube is plausibly grasped."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    goal_distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    grasped = object_grasped(env, ee_threshold, force_threshold, min_finger_opening, object_cfg, ee_frame_cfg)
    return (1 - torch.tanh(goal_distance / std)) * (goal_distance < close_threshold) * grasped
