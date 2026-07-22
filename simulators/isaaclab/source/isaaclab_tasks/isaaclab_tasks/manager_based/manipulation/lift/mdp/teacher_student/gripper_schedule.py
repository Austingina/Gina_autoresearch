# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Continuous desired gripper opening schedule for oracle teacher labels."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from ..oracle.lift_cube_oracle import LiftCubeOraclePolicy

# PickAndLiftSm state indices (see pick_and_lift_sm.py).
SM_REST = 0
SM_ABOVE = 1
SM_APPROACH = 2
SM_GRASP = 3
SM_LIFT = 4


def compute_desired_gripper_opening(
    env: ManagerBasedRLEnv,
    oracle: LiftCubeOraclePolicy,
    *,
    open_max: float = 0.04,
    teacher_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return desired finger opening in meters for each environment.

    Schedule (no pre-close ramp):
    - REST/ABOVE/APPROACH: fully open.
    - After binary close is latched (EE reached grasp pose): fully closed until reset.
    """
    close_latched = getattr(env, "_teacher_gripper_close_latched", None)
    if close_latched is not None:
        desired = torch.full((env.num_envs,), open_max, device=env.device)
        desired = torch.where(close_latched, 0.0, desired)
        return desired.clamp(0.0, open_max)

    if teacher_phase is not None:
        phase = teacher_phase
    else:
        phase = oracle._pick_sm.sm_state.long()

    desired = torch.full((phase.shape[0],), open_max, device=phase.device)
    desired = torch.where(phase >= SM_GRASP, 0.0, desired)
    return desired.clamp(0.0, open_max)
