# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .dagger_trainer import (
    DaggerAggregateBuffer,
    GraspGatedBetaState,
    compute_dagger_beta,
    compute_dagger_beta_adaptive,
    compute_dagger_beta_grasp_gated,
    mix_rollout_twist,
)
from .early_done import OraclePhaseStallTracker
from .env_randomization import (
    DEFAULT_CUBE_POSE_RANGE,
    DEFAULT_LIFT_GOAL_RANGES,
    apply_lift_teacher_student_env_randomization,
)
from .eval import (
    BinaryGripperEvalMetrics,
    EvalMetrics,
    run_student_eval,
    run_student_eval_binary_gripper,
    run_student_eval_binary_gripper_vision,
)
from .gripper_schedule import compute_desired_gripper_opening
from .kl_trainer import (
    TwistEpisodeReplayBuffer,
    TwistReplayBuffer,
    gaussian_kl_arm_gripper_unweighted,
    gaussian_kl_to_teacher,
    gripper_bce_to_teacher,
    optimize_kl_batch,
    optimize_kl_bce_batch,
    optimize_kl_bce_sequences,
    optimize_kl_sequences,
)
from .networks import RNNState, StudentTwistPolicy, StudentTwistPolicyBinaryGripper, reset_rnn_state
from .observations import student_proprio_observation, student_vision_proprio_observation
from .oracle_demo_loader import load_oracle_demos_into_buffer
from .sampler_logging import ORACLE_SM_STATE_NAMES, SamplerEpisodeStats, SamplerTeacherLabelStats
from .teacher import (
    GRIPPER_OPEN_MAX_M,
    TEACHER_MIN_ANG_SPEED_RAD_S,
    TEACHER_MIN_LIN_SPEED_M_S,
    compute_teacher_twist,
    compute_teacher_twist_binary_gripper,
    get_oracle_desired_ee_in_base_frame,
    get_gripper_action_dim,
    make_kl_dim_weights,
    make_teacher_arm_std,
    make_teacher_std,
    teacher_binary_gripper_to_bce_target,
    twist_to_env_action,
)
from .teacher_phase import (
    infer_teacher_phase,
    is_teacher_gripper_close_latched,
    reset_teacher_phase_state,
    teacher_desired_ee_in_base_frame,
    teacher_desired_ee_pos_in_base_frame,
)
from .vision_dagger_trainer import VisionDaggerAggregateBuffer
from .vision_kl_trainer import VisionEpisodeReplayBuffer, optimize_kl_bce_sequences_vision
from .vision_networks import StudentTwistPolicyBinaryGripperVision
from .vision_policy_obs import extract_proprio_from_policy_obs_torch, stack_visual_from_policy_obs_torch

__all__ = [
    "BinaryGripperEvalMetrics",
    "DaggerAggregateBuffer",
    "EvalMetrics",
    "GRIPPER_OPEN_MAX_M",
    "TEACHER_MIN_ANG_SPEED_RAD_S",
    "TEACHER_MIN_LIN_SPEED_M_S",
    "ORACLE_SM_STATE_NAMES",
    "OraclePhaseStallTracker",
    "DEFAULT_CUBE_POSE_RANGE",
    "DEFAULT_LIFT_GOAL_RANGES",
    "apply_lift_teacher_student_env_randomization",
    "SamplerEpisodeStats",
    "SamplerTeacherLabelStats",
    "RNNState",
    "reset_rnn_state",
    "StudentTwistPolicy",
    "StudentTwistPolicyBinaryGripper",
    "StudentTwistPolicyBinaryGripperVision",
    "TwistEpisodeReplayBuffer",
    "TwistReplayBuffer",
    "VisionDaggerAggregateBuffer",
    "VisionEpisodeReplayBuffer",
    "compute_desired_gripper_opening",
    "compute_dagger_beta",
    "compute_dagger_beta_adaptive",
    "compute_dagger_beta_grasp_gated",
    "GraspGatedBetaState",
    "compute_teacher_twist",
    "compute_teacher_twist_binary_gripper",
    "extract_proprio_from_policy_obs_torch",
    "gaussian_kl_to_teacher",
    "gaussian_kl_arm_gripper_unweighted",
    "get_oracle_desired_ee_in_base_frame",
    "get_gripper_action_dim",
    "gripper_bce_to_teacher",
    "load_oracle_demos_into_buffer",
    "mix_rollout_twist",
    "make_kl_dim_weights",
    "make_teacher_arm_std",
    "make_teacher_std",
    "optimize_kl_batch",
    "optimize_kl_bce_batch",
    "optimize_kl_bce_sequences",
    "optimize_kl_bce_sequences_vision",
    "optimize_kl_sequences",
    "run_student_eval",
    "run_student_eval_binary_gripper",
    "run_student_eval_binary_gripper_vision",
    "stack_visual_from_policy_obs_torch",
    "student_proprio_observation",
    "student_vision_proprio_observation",
    "teacher_binary_gripper_to_bce_target",
    "infer_teacher_phase",
    "is_teacher_gripper_close_latched",
    "reset_teacher_phase_state",
    "teacher_desired_ee_in_base_frame",
    "teacher_desired_ee_pos_in_base_frame",
    "twist_to_env_action",
]
