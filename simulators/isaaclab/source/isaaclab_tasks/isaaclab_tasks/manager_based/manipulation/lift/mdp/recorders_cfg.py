# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.mdp.recorders import (
    OracleEpisodeResetRecorder,
    PostStepEEPoseRecorder,
    PostStepEEVelocityActionRecorder,
    PostStepPhasedGoalRecorder,
    PostStepProgressRewardRecorder,
    PostStepSparseRewardRecorder,
    PreStepEnvActionsRecorder,
    PreStepTeacherTwistRecorder,
)


@configclass
class OracleEpisodeResetRecorderCfg(RecorderTermCfg):
    """Configuration for resetting oracle episode buffers."""

    class_type: type[RecorderTerm] = OracleEpisodeResetRecorder


@configclass
class PostStepEEVelocityActionRecorderCfg(RecorderTermCfg):
    """Configuration for end-effector velocity action recording."""

    class_type: type[RecorderTerm] = PostStepEEVelocityActionRecorder


@configclass
class PostStepEEPoseRecorderCfg(RecorderTermCfg):
    """Configuration for end-effector pose recording."""

    class_type: type[RecorderTerm] = PostStepEEPoseRecorder


@configclass
class PostStepPhasedGoalRecorderCfg(RecorderTermCfg):
    """Configuration for phased goal recording."""

    class_type: type[RecorderTerm] = PostStepPhasedGoalRecorder


@configclass
class PostStepSparseRewardRecorderCfg(RecorderTermCfg):
    """Configuration for sparse grasp/lift reward recording."""

    class_type: type[RecorderTerm] = PostStepSparseRewardRecorder


@configclass
class PostStepProgressRewardRecorderCfg(RecorderTermCfg):
    """Configuration for dense progress-based offline RL reward recording."""

    class_type: type[RecorderTerm] = PostStepProgressRewardRecorder


@configclass
class PreStepTeacherTwistRecorderCfg(RecorderTermCfg):
    """Configuration for lenient teacher twist label recording."""

    class_type: type[RecorderTerm] = PreStepTeacherTwistRecorder


@configclass
class PreStepEnvActionsRecorderCfg(RecorderTermCfg):
    """Configuration for executed (noisy) env action recording."""

    class_type: type[RecorderTerm] = PreStepEnvActionsRecorder


@configclass
class LiftOracleRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Recorder configuration for goal-conditioned oracle data collection."""

    record_pre_step_actions = None
    record_post_step_processed_actions = None

    record_oracle_episode_reset = OracleEpisodeResetRecorderCfg()
    record_post_step_ee_velocity = PostStepEEVelocityActionRecorderCfg()
    record_post_step_ee_pose = PostStepEEPoseRecorderCfg()
    record_post_step_phased_goals = PostStepPhasedGoalRecorderCfg()
    record_post_step_sparse_rewards = PostStepSparseRewardRecorderCfg()


@configclass
class LiftLenientTeacherMaskDepthRecorderManagerCfg(LiftOracleRecorderManagerCfg):
    """Recorder configuration for lenient-teacher mask+depth data collection."""

    record_pre_step_teacher_twist = PreStepTeacherTwistRecorderCfg()
    record_pre_step_env_actions = PreStepEnvActionsRecorderCfg()
    record_post_step_progress_reward = PostStepProgressRewardRecorderCfg()

    def __post_init__(self):
        self.record_pre_step_actions = None
        self.record_post_step_processed_actions = None
        self.record_post_step_ee_velocity = None
