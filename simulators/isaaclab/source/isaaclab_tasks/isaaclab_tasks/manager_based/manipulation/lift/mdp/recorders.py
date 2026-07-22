# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder terms for lift-cube oracle data collection."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.managers.recorder_manager import RecorderTerm

from .data_collection import (
    compute_oracle_step_data,
    compute_progress_reward_step_data,
    reset_oracle_episode_buffers,
    reset_progress_episode_buffers,
)


class _OracleStepDataMixin:
    """Shared helper for oracle dataset recorders."""

    _env = None

    @property
    def control_dt(self) -> float:
        return self._env.cfg.sim.dt * self._env.cfg.decimation

    def _get_step_data(self):
        return compute_oracle_step_data(self._env, control_dt=self.control_dt)


class OracleEpisodeResetRecorder(_OracleStepDataMixin, RecorderTerm):
    """Reset oracle episode buffers after environment reset."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        reset_oracle_episode_buffers(self._env, env_ids)
        reset_progress_episode_buffers(self._env, env_ids)
        return None, None


class PostStepEEVelocityActionRecorder(_OracleStepDataMixin, RecorderTerm):
    """Record end-effector twist and gripper opening velocity in the robot root frame as the action."""

    def record_post_step(self):
        return "actions", self._get_step_data().ee_velocity


class PostStepEEPoseRecorder(_OracleStepDataMixin, RecorderTerm):
    """Record end-effector pose in the robot root frame."""

    def record_post_step(self):
        return "ee_pose", self._get_step_data().ee_pose


class PostStepPhasedGoalRecorder(_OracleStepDataMixin, RecorderTerm):
    """Record phase-dependent goal targets."""

    def record_post_step(self):
        data = self._get_step_data()
        return "goals", {"position": data.goal_position, "phase": data.goal_phase}


class PostStepSparseRewardRecorder(_OracleStepDataMixin, RecorderTerm):
    """Record sparse grasp/lift success indicators."""

    def record_post_step(self):
        data = self._get_step_data()
        return "rewards", {"grasp": data.reward_grasp, "lift": data.reward_lift}


class PostStepProgressRewardRecorder(RecorderTerm):
    """Record dense progress reward for offline RL: scaled Δprogress (terminal_bonus logged only)."""

    def record_post_step(self):
        data = compute_progress_reward_step_data(self._env)
        scores = data.scores
        return "rewards", {
            "progress": data.progress.unsqueeze(-1),
            "progress_delta": data.progress_delta.unsqueeze(-1),
            "progress_reward": data.progress_reward.unsqueeze(-1),
            "terminal_bonus": data.terminal_bonus.unsqueeze(-1),
            "reach_score": scores.reach_score.unsqueeze(-1),
            "grasp_score": scores.grasp_score.unsqueeze(-1),
            "lift_score": scores.lift_score.unsqueeze(-1),
            "goal_score": scores.goal_score.unsqueeze(-1),
            "success_score": scores.success_score.unsqueeze(-1),
        }


class PreStepTeacherTwistRecorder(RecorderTerm):
    """Record lenient teacher 7D twist labels set on ``env._lenient_teacher_record`` before ``step()``."""

    def record_pre_step(self):
        record = getattr(self._env, "_lenient_teacher_record", None)
        if record is None:
            return None, None
        return "actions", record["teacher_twist"]


class PreStepEnvActionsRecorder(RecorderTerm):
    """Record noisy IK-Rel env actions actually passed to ``step()``."""

    def record_pre_step(self):
        record = getattr(self._env, "_lenient_teacher_record", None)
        if record is None:
            return None, None
        return "env_actions", record["env_actions"]
