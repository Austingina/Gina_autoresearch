# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-randomization helpers for teacher-student lift-cube env configs."""

from __future__ import annotations

from typing import Any

from ..lift_goal_command import LiftGoalAboveCubeCommand


# Match Isaac-Lift-Cube default command ranges (lift_env_cfg.CommandsCfg).
DEFAULT_LIFT_GOAL_RANGES = {
    "pos_x": (0.4, 0.6),
    "pos_y": (-0.25, 0.25),
    "pos_z": (0.25, 0.5),
}

# Match FrankaCubeLiftOracleTwistPlayRandomizedEnvCfg cube spawn offsets.
DEFAULT_CUBE_POSE_RANGE = {
    "x": (-0.1, 0.1),
    "y": (-0.25, 0.25),
    "z": (0.0, 0.0),
}

# Fixed x/y at default spawn; perturb cube height only (S1 simplified DR).
DEFAULT_CUBE_POSE_Z_ONLY_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
}


def cube_pose_range_z_only(*, z_min: float = 0.0, z_max: float = 0.0) -> dict[str, tuple[float, float]]:
    return {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (z_min, z_max)}


def cube_pose_range_xy(
    *,
    x_min: float = -0.1,
    x_max: float = 0.1,
    y_min: float = -0.25,
    y_max: float = 0.25,
    z_min: float = 0.0,
    z_max: float = 0.0,
) -> dict[str, tuple[float, float]]:
    return {"x": (x_min, x_max), "y": (y_min, y_max), "z": (z_min, z_max)}


def apply_lift_teacher_student_env_randomization(
    env_cfg: Any,
    *,
    randomize_cube: bool = False,
    randomize_lift_goal: bool = False,
    fixed_lift_goal: bool = True,
    lift_goal_above_cube: bool = True,
    lift_height_above_cube: float = 0.20,
    lift_goal_x: float = 0.5,
    lift_goal_y: float = 0.0,
    lift_goal_z: float = 0.375,
    cube_pose_range: dict[str, tuple[float, float]] | None = None,
    lift_goal_ranges: dict[str, tuple[float, float]] | None = None,
    debug_vis_goal: bool | None = None,
) -> None:
    """Configure cube spawn and lift-goal command ranges on a teacher-student env cfg."""
    env_cfg.lift_goal_above_cube = bool(lift_goal_above_cube)
    env_cfg.lift_height_above_cube = float(lift_height_above_cube)
    if randomize_cube:
        env_cfg.events.reset_object_position.params["pose_range"] = (
            cube_pose_range if cube_pose_range is not None else DEFAULT_CUBE_POSE_RANGE
        )
        if debug_vis_goal is None:
            debug_vis_goal = True

    if randomize_lift_goal:
        goal_ranges = lift_goal_ranges if lift_goal_ranges is not None else DEFAULT_LIFT_GOAL_RANGES
        env_cfg.commands.object_pose.ranges.pos_x = goal_ranges["pos_x"]
        env_cfg.commands.object_pose.ranges.pos_y = goal_ranges["pos_y"]
        env_cfg.commands.object_pose.ranges.pos_z = goal_ranges["pos_z"]
        if debug_vis_goal is None:
            debug_vis_goal = True
    elif lift_goal_above_cube:
        # Command overwritten on reset to cube_xy + lift_height; ranges are placeholders.
        env_cfg.commands.object_pose.ranges.pos_x = (0.0, 0.0)
        env_cfg.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        env_cfg.commands.object_pose.ranges.pos_z = (0.2, 0.5)
        if debug_vis_goal is None:
            debug_vis_goal = True
    elif fixed_lift_goal:
        env_cfg.commands.object_pose.ranges.pos_x = (lift_goal_x, lift_goal_x)
        env_cfg.commands.object_pose.ranges.pos_y = (lift_goal_y, lift_goal_y)
        env_cfg.commands.object_pose.ranges.pos_z = (lift_goal_z, lift_goal_z)

    if debug_vis_goal is not None:
        env_cfg.commands.object_pose.debug_vis = debug_vis_goal

    if lift_goal_above_cube and not randomize_lift_goal:
        # Resample goal = cube spawn xy + height inside command_manager.reset (after
        # the object reset event), so the vertical ACTUATE target tracks the cube.
        env_cfg.commands.object_pose.class_type = LiftGoalAboveCubeCommand
        env_cfg.commands.object_pose.object_name = "object"
        env_cfg.commands.object_pose.height_above = float(lift_height_above_cube)
