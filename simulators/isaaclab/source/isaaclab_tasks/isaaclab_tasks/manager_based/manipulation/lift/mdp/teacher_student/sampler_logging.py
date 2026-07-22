# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostics for teacher-student sampler rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .gripper_schedule import SM_GRASP

ORACLE_SM_STATE_NAMES = ("REST", "ABOVE", "APPROACH", "GRASP", "LIFT")


@dataclass
class SamplerTeacherLabelStats:
    """Online teacher-label statistics collected during rollout."""

    env_steps: int = 0
    close_label_steps: int = 0
    grasp_phase_steps: int = 0

    def reset(self) -> None:
        self.env_steps = 0
        self.close_label_steps = 0
        self.grasp_phase_steps = 0

    def observe(self, teacher_gripper_rate: torch.Tensor, teacher_phase: torch.Tensor) -> None:
        n = int(teacher_gripper_rate.shape[0])
        self.env_steps += n
        self.close_label_steps += int((teacher_gripper_rate < -0.01).sum().item())
        self.grasp_phase_steps += int((teacher_phase >= SM_GRASP).sum().item())

    def close_rate(self) -> float:
        return self.close_label_steps / self.env_steps if self.env_steps > 0 else 0.0

    def grasp_phase_rate(self) -> float:
        return self.grasp_phase_steps / self.env_steps if self.env_steps > 0 else 0.0


@dataclass
class SamplerEpisodeStats:
    """Aggregated episode statistics for one training iteration."""

    episode_lengths: list[int] = field(default_factory=list)
    timeout_count: int = 0
    terminated_count: int = 0
    early_done_count: int = 0
    sm_states: list[int] = field(default_factory=list)
    grasp_success_count: int = 0
    lift_success_count: int = 0

    @property
    def num_episodes(self) -> int:
        return len(self.episode_lengths)

    def reset(self) -> None:
        self.episode_lengths.clear()
        self.timeout_count = 0
        self.terminated_count = 0
        self.early_done_count = 0
        self.sm_states.clear()
        self.grasp_success_count = 0
        self.lift_success_count = 0

    def record(
        self,
        *,
        length: int,
        timed_out: bool,
        terminated: bool,
        early_done: bool = False,
        sm_state: int,
        grasp_success: bool,
        lift_success: bool,
    ) -> None:
        if length <= 0:
            return
        self.episode_lengths.append(length)
        self.timeout_count += int(timed_out)
        self.terminated_count += int(terminated)
        self.early_done_count += int(early_done)
        self.sm_states.append(sm_state)
        self.grasp_success_count += int(grasp_success)
        self.lift_success_count += int(lift_success)

    def mean_episode_length(self) -> float:
        if not self.episode_lengths:
            return 0.0
        return sum(self.episode_lengths) / len(self.episode_lengths)

    def timeout_rate(self) -> float:
        n = self.num_episodes
        return self.timeout_count / n if n > 0 else 0.0

    def sm_rate_at_least(self, state_index: int) -> float:
        n = self.num_episodes
        if n == 0:
            return 0.0
        return sum(1 for state in self.sm_states if state >= state_index) / n

    def sm_state_histogram(self) -> dict[str, float]:
        n = self.num_episodes
        if n == 0:
            return {name: 0.0 for name in ORACLE_SM_STATE_NAMES}
        counts = [0] * len(ORACLE_SM_STATE_NAMES)
        for state in self.sm_states:
            idx = min(max(state, 0), len(counts) - 1)
            counts[idx] += 1
        return {name: counts[i] / n for i, name in enumerate(ORACLE_SM_STATE_NAMES)}

    def summary_line(self, max_episode_steps: int) -> str:
        del max_episode_steps
        if self.num_episodes == 0:
            return "rollout: no episodes"
        n = self.num_episodes
        return (
            f"rollout grasp={self.grasp_success_count / n:.2f} "
            f"lift={self.lift_success_count / n:.2f} "
            f"sm>=GRASP={self.sm_rate_at_least(3):.2f} "
            f"sm>=APPROACH={self.sm_rate_at_least(2):.2f}"
        )
