# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Cube-relative vertical lift goal command (Interaction Geometry κ=vertical_free).

The goal position is resampled at episode reset to the cube spawn xy plus a fixed
height. Implemented as a command term so it runs inside ``command_manager.reset``
(after the object reset event), avoiding the resample-clobber of reset events.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class LiftGoalAboveCubeCommand(UniformPoseCommand):
    """Pose command whose position tracks cube spawn xy + ``height_above`` (base frame)."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._object = env.scene[getattr(cfg, "object_name", "object")]
        self._height_above = float(getattr(cfg, "height_above", 0.20))

    def _resample_command(self, env_ids: Sequence[int]):
        # Sample orientation (and a placeholder position) from the base implementation.
        super()._resample_command(env_ids)
        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_quat_w = self.robot.data.root_quat_w[env_ids]
        obj_pos_w = self._object.data.root_pos_w[env_ids, :3]
        obj_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, obj_pos_w)
        self.pose_command_b[env_ids, 0] = obj_pos_b[:, 0]
        self.pose_command_b[env_ids, 1] = obj_pos_b[:, 1]
        self.pose_command_b[env_ids, 2] = obj_pos_b[:, 2] + self._height_above
