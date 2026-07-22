# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Oracle policy for Franka lift-cube data collection."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv

from .pick_and_lift_sm import PickAndLiftSm

_IK_ABS_ACTION_NAME = "DifferentialInverseKinematicsAction"
_IK_BODY_OFFSET = (0.0, 0.0, 0.107)


class LiftCubeOraclePolicy:
    """Pick-and-lift oracle that adapts to joint-position and IK action spaces.

    The pose conventions and action mapping follow Isaac Lab's reference script
    ``scripts/environments/state_machine/lift_cube_sm.py`` exactly.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        position_threshold: float = 0.01,
        ik_substeps: int = 8,
    ):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self._control_dt = env.cfg.sim.dt * env.cfg.decimation

        self._pick_sm = PickAndLiftSm(
            dt=self._control_dt,
            num_envs=self.num_envs,
            device=self.device,
            position_threshold=position_threshold,
        )

        # Match the reference state-machine script: fixed orientation for object/goal poses.
        self._desired_orientation = torch.zeros((self.num_envs, 4), device=self.device)
        self._desired_orientation[:, 1] = 1.0

        self._uses_ik_abs = self._detect_ik_abs_action_space()
        self._joint_mapper = None if self._uses_ik_abs else _JointPositionActionMapper(env, ik_substeps=ik_substeps)

    @property
    def uses_ik_abs(self) -> bool:
        """Whether the environment expects absolute IK pose actions."""
        return self._uses_ik_abs

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        """Reset the internal state machine."""
        self._pick_sm.reset_idx(env_ids)

    def compute(self) -> torch.Tensor:
        """Compute oracle actions for all environments."""
        ee_pose, object_pose, des_object_pose = self._get_task_poses()
        task_space_cmd = self._pick_sm.compute(ee_pose, object_pose, des_object_pose)

        # Reference script passes the state-machine output directly for IK-Abs control.
        if self._uses_ik_abs:
            return task_space_cmd

        return self._joint_mapper.compute(task_space_cmd)

    def _detect_ik_abs_action_space(self) -> bool:
        for term_name in self.env.action_manager.active_terms:
            term = self.env.action_manager.get_term(term_name)
            if term.__class__.__name__ == _IK_ABS_ACTION_NAME:
                return True
        return False

    def _get_task_poses(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build SM inputs using the same conventions as ``lift_cube_sm.py``."""
        ee_frame_sensor = self.env.scene["ee_frame"]
        tcp_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - self.env.scene.env_origins
        tcp_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

        object_data: RigidObjectData = self.env.scene["object"].data
        object_position = object_data.root_pos_w - self.env.scene.env_origins

        # Goal position is kept in the robot command frame, as in the reference script.
        desired_position = self.env.command_manager.get_command("object_pose")[..., :3]

        ee_pose = torch.cat([tcp_position, tcp_orientation], dim=-1)
        object_pose = torch.cat([object_position, self._desired_orientation], dim=-1)
        des_object_pose = torch.cat([desired_position, self._desired_orientation], dim=-1)
        return ee_pose, object_pose, des_object_pose


class _JointPositionActionMapper:
    """Map task-space oracle commands to joint-position actions."""

    def __init__(self, env: ManagerBasedRLEnv, ik_substeps: int = 8):
        self.env = env
        self.device = env.device
        self._ik_substeps = ik_substeps

        robot = env.scene["robot"]
        self._robot = robot
        self._joint_ids, _ = robot.find_joints(["panda_joint.*"])

        body_ids, _ = robot.find_bodies("panda_hand")
        if len(body_ids) != 1:
            raise ValueError("Expected exactly one panda_hand body for the lift-cube oracle.")
        self._body_idx = body_ids[0]

        if robot.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        self._offset_pos = torch.tensor(_IK_BODY_OFFSET, device=self.device).repeat(env.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(env.num_envs, 1)

        self._ik_controller = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=env.num_envs,
            device=self.device,
        )

        arm_action_term = env.action_manager.get_term("arm_action")
        self._arm_scale = float(arm_action_term.cfg.scale)
        self._default_joint_pos = robot.data.default_joint_pos[:, self._joint_ids].clone()

    def compute(self, task_space_cmd: torch.Tensor) -> torch.Tensor:
        """Convert task-space command to joint-position actions."""
        ik_command = task_space_cmd[:, :7]
        gripper_cmd = task_space_cmd[:, 7:8]

        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_pos_des = joint_pos.clone()
        jacobian = self._compute_frame_jacobian()

        for _ in range(self._ik_substeps):
            self._ik_controller.set_command(ik_command, ee_pos_b, ee_quat_b)
            if ee_quat_b.norm() != 0:
                joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_des)

        arm_actions = (joint_pos_des - self._default_joint_pos) / self._arm_scale
        return torch.cat([arm_actions, gripper_cmd], dim=-1)

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )
        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self) -> torch.Tensor:
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        base_rot = self._robot.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian
