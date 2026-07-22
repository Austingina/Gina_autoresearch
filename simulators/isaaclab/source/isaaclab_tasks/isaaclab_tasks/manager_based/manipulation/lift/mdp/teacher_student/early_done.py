# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Early episode termination when the student stalls teacher phase progress."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from ..data_collection import get_ee_pose_in_robot_root_frame
from ..oracle.lift_cube_oracle import LiftCubeOraclePolicy
from .teacher_phase import infer_teacher_phase, teacher_desired_ee_pos_in_base_frame


def oracle_desired_ee_pos_in_base_frame(env: ManagerBasedRLEnv, oracle: LiftCubeOraclePolicy) -> torch.Tensor:
    """Strict oracle SM desired EE position (for legacy callers)."""
    teacher_phase = infer_teacher_phase(env)
    return teacher_desired_ee_pos_in_base_frame(env, oracle, teacher_phase)


class OraclePhaseStallTracker:
    """Terminate episodes when the student stops making phase progress (active_tracking-style early done).

    Progress uses the **lenient teacher phase** (geometry-based), not the strict oracle execution SM,
    so early done aligns with teacher supervision tolerances.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        max_stall_steps: int = 20,
        min_progress_m: float = 0.002,
        min_sm_state: int = 2,
    ):
        self.num_envs = num_envs
        self.device = device
        self.max_stall_steps = max_stall_steps
        self.min_progress_m = min_progress_m
        self.min_teacher_phase = min_sm_state

        self.stall_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.best_teacher_phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.best_phase_dist = torch.full((num_envs,), float("inf"), device=device)
        self._phase_des_pos_b = torch.zeros(num_envs, 3, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.stall_counter[env_ids] = 0
        self.best_teacher_phase[env_ids] = 0
        self.best_phase_dist[env_ids] = float("inf")

    def on_pre_step(self, oracle: LiftCubeOraclePolicy, env: ManagerBasedRLEnv) -> None:
        """Call after ``compute_teacher_twist`` so targets match the recorded label."""
        teacher_phase = infer_teacher_phase(env)
        des_pos_b = teacher_desired_ee_pos_in_base_frame(env, oracle, teacher_phase)
        self._phase_des_pos_b[:] = des_pos_b

        phase_advanced = teacher_phase > self.best_teacher_phase
        if phase_advanced.any():
            advanced_ids = phase_advanced.nonzero(as_tuple=False).squeeze(-1)
            if advanced_ids.ndim == 0:
                advanced_ids = advanced_ids.unsqueeze(0)
            self.stall_counter[advanced_ids] = 0
            self.best_teacher_phase[advanced_ids] = teacher_phase[advanced_ids]
            ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
            dist = torch.norm(des_pos_b - ee_pos_b, dim=-1)
            self.best_phase_dist[advanced_ids] = dist[advanced_ids]

    def on_post_step(self, oracle: LiftCubeOraclePolicy, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Call after ``env.step``. Returns a bool mask of envs that should end the episode early."""
        del oracle
        ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
        dist = torch.norm(self._phase_des_pos_b - ee_pos_b, dim=-1)
        dist_improved = dist < (self.best_phase_dist - self.min_progress_m)

        made_progress = dist_improved
        self.stall_counter = torch.where(made_progress, torch.zeros_like(self.stall_counter), self.stall_counter + 1)
        self.best_phase_dist = torch.where(dist_improved, dist, self.best_phase_dist)

        teacher_phase = infer_teacher_phase(env)
        stall_done = self.stall_counter >= self.max_stall_steps
        return stall_done & (teacher_phase < self.min_teacher_phase)
