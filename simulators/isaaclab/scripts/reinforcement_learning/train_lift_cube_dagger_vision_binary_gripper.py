# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
DAgger training for lift-cube **vision** dual-head binary-gripper student.

Uses dual-camera GT mask + depth (8 channels, same stack as offline CQL) with
lenient teacher twist labels and aggregate-buffer DAgger.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Vision DAgger for lift-cube binary-gripper student.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-Vision-Teacher-Student-v0",
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument(
    "--obs_mode",
    type=str,
    default="vision",
    choices=("vision", "vision_proprio"),
    help="vision=mask+depth only; vision_proprio=mask+depth+7D proprio (no cube GT).",
)
parser.add_argument("--image_size", type=int, default=64, help="Stored/training visual resolution (native cam 84).")
parser.add_argument("--camera_height", type=int, default=0, help="Native camera height (0 = use image_size).")
parser.add_argument("--camera_width", type=int, default=0, help="Native camera width (0 = use image_size).")
parser.add_argument(
    "--topdown_close",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use close top-down third_person_camera offset (matches clean240 G_θ demos).",
)
parser.add_argument("--visual_channels", type=int, default=8)
parser.add_argument(
    "--visual_layout",
    type=str,
    default="channels",
    choices=("channels", "ee_channels", "mosaic"),
    help="channels=8ch ee/top; ee_channels=4ch EE mask RGB+depth; mosaic=4ch ee/top spatial concat.",
)
parser.add_argument(
    "--mosaic_history_len",
    type=int,
    default=1,
    help="Deprecated: mosaic uses a single frame; temporal context is via LSTM. Kept for checkpoint compat.",
)
parser.add_argument(
    "--debug_save_mosaic_dir",
    type=str,
    default="",
    help="If set, save (4,2H,W) mosaic tensors as PNG panels + .pt before policy.act (mosaic layout only).",
)
parser.add_argument(
    "--debug_save_mosaic_interval",
    type=int,
    default=10,
    help="Save every N global collect steps when --debug_save_mosaic_dir is set.",
)
parser.add_argument(
    "--debug_save_mosaic_max",
    type=int,
    default=200,
    help="Maximum number of mosaic dumps per training run.",
)
parser.add_argument(
    "--debug_save_mosaic_env_id",
    type=int,
    default=0,
    help="Which parallel env index to dump (default: env 0).",
)
parser.add_argument(
    "--mask_palette_mode",
    type=str,
    default="fixed",
    choices=("random", "fixed"),
    help="Semantic mask palette for non-cube classes. Cube remains white.",
)
parser.add_argument("--mask_robot_color", type=float, nargs=3, default=(0.10, 0.35, 0.95))
parser.add_argument("--mask_table_color", type=float, nargs=3, default=(0.55, 0.35, 0.18))
parser.add_argument("--mask_ground_color", type=float, nargs=3, default=(0.10, 0.10, 0.10))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--collect_steps", type=int, default=64, help="Env steps collected per DAgger iteration.")
parser.add_argument(
    "--max_aggregate_episodes",
    type=int,
    default=500,
    help="Maximum aggregated vision episodes kept (FIFO). Each ep stores (T,8,H,W) tensors.",
)
parser.add_argument(
    "--aggregate_backend",
    type=str,
    default="ram",
    choices=("ram", "disk"),
    help="Aggregate buffer backend: in-RAM tensors or per-episode disk files.",
)
parser.add_argument(
    "--aggregate_storage_dir",
    type=str,
    default="",
    help="Episode file directory for disk backend (default: <log_dir>/dagger_episodes).",
)
parser.add_argument(
    "--aggregate_cache_size",
    type=int,
    default=8,
    help="Disk backend LRU cache size (episodes kept in RAM after load).",
)
parser.add_argument(
    "--aggregate_visual_storage_dtype",
    type=str,
    default="float16",
    choices=("float16", "float32"),
    help="Visual dtype for on-disk episode files (disk backend only).",
)
parser.add_argument(
    "--min_buffer_episodes",
    type=int,
    default=32,
    help="Minimum aggregated episodes before optimization starts.",
)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=64)
parser.add_argument("--optim_steps", type=int, default=8, help="BPTT updates per DAgger iteration.")
parser.add_argument("--grad_interval", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--teacher_lin_std", type=float, default=0.05)
parser.add_argument("--teacher_ang_std", type=float, default=0.10)
parser.add_argument("--gripper_bce_weight", type=float, default=1.0)
parser.add_argument(
    "--aux_cube_ee_weight",
    type=float,
    default=0.4,
    help="Weight for train-only auxiliary Huber loss predicting normalized cube_pos_b - ee_pos_b.",
)
parser.add_argument(
    "--aux_cube_ee_scale",
    type=float,
    default=0.25,
    help="Meters used to normalize cube_pos_b - ee_pos_b before auxiliary Huber loss.",
)
parser.add_argument("--teacher_min_lin_speed", type=float, default=0.10)
parser.add_argument("--teacher_min_ang_speed", type=float, default=0.20)
parser.add_argument(
    "--teacher_max_lin_vel",
    type=float,
    default=0.5,
    help="Teacher arm linear velocity cap (m/s). Raise for faster EE motion.",
)
parser.add_argument(
    "--teacher_max_ang_vel",
    type=float,
    default=1.0,
    help="Teacher arm angular velocity cap (rad/s).",
)
parser.add_argument(
    "--teacher_grasp_hold_steps",
    type=int,
    default=15,
    help="Closed-gripper steps before LIFT phase (play fast teacher uses 2).",
)
parser.add_argument(
    "--teacher_above_offset_z",
    type=float,
    default=0.10,
    help="Hover height above cube during ABOVE phase (m).",
)
parser.add_argument("--arm_scale", type=float, default=0.5)
parser.add_argument("--log_dir", type=str, default="./runs/lift_cube_dagger_vision")
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument(
    "--buffer_save_interval",
    type=int,
    default=200,
    help="Save dagger_aggregate.pt every N iterations (0 = only with checkpoints).",
)
parser.add_argument("--position_threshold", type=float, default=0.01)
parser.add_argument(
    "--goal_threshold",
    type=float,
    default=0.02,
    help="Cube-to-goal distance for lift success / buffer latch (matches play_teacher).",
)
parser.add_argument(
    "--success_early_done",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="End rollout on lift success (same as play_teacher preview).",
)
parser.add_argument(
    "--buffer_success_only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Only add episodes to the DAgger buffer when lift succeeds (no timeout/failure segments).",
)
parser.add_argument("--eval_interval", type=int, default=50)
parser.add_argument("--eval_episodes", type=int, default=50)
parser.add_argument("--num_eval_envs", type=int, default=4)
parser.add_argument("--no_eval", action="store_true")
parser.add_argument(
    "--no_eval_diagnostics",
    action="store_true",
    help="Disable EE/cube failure-mode breakdown during periodic eval.",
)
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--early_done", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--phase_stall_steps", type=int, default=180)
parser.add_argument(
    "--eval_phase_stall_steps",
    type=int,
    default=60,
    help="Phase stall cutoff during eval only (keep lower than training to avoid 1250-step eval hangs).",
)
parser.add_argument(
    "--max_store_episode_steps",
    type=int,
    default=512,
    help="Cap replay episode length on disk/RAM (prevents multi-GB episodes from cross-iter partial rollouts).",
)
parser.add_argument(
    "--partial_rollout_cap",
    type=int,
    default=-1,
    help="Flush in-flight rollout segments at this length (-1=auto 2*collect_steps, 0=disable partial flush).",
)
parser.add_argument("--phase_min_progress_m", type=float, default=0.0005)
parser.add_argument("--early_done_min_sm_state", type=int, default=2)
parser.add_argument("--oracle_demo_file", type=str, default="", help="Unused for vision DAgger (proprio HDF5 incompatible).")
parser.add_argument("--oracle_demo_max_episodes", type=int, default=0)
parser.add_argument("--resume_checkpoint", type=str, default="")
parser.add_argument(
    "--resume_buffer",
    type=str,
    default="",
    help="Optional dagger_aggregate.pt. Defaults to <log_dir>/dagger_aggregate.pt when resuming.",
)
parser.add_argument(
    "--no_resume_buffer",
    action="store_true",
    help="Resume policy only; do not auto-load <log_dir>/dagger_aggregate.pt (avoids huge RAM spike).",
)
parser.add_argument("--grasp_sample_ratio", type=float, default=0.7)
parser.add_argument("--grasp_sample_min_phase", type=int, default=3)
parser.add_argument("--beta_start", type=float, default=1.0, help="Initial teacher execution probability.")
parser.add_argument("--beta_end", type=float, default=0.0, help="Final teacher execution probability.")
parser.add_argument(
    "--beta_warmup_iters",
    type=int,
    default=100,
    help="Keep β=beta_start for this many iterations (teacher-only rollout warm start).",
)
parser.add_argument(
    "--beta_decay_iters",
    type=int,
    default=2000,
    help="Linear β decay length after warmup (used when --no-beta_adaptive).",
)
parser.add_argument(
    "--beta_adaptive",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Adapt β from eval student success instead of fixed linear decay.",
)
parser.add_argument(
    "--beta_success_target",
    type=float,
    default=0.5,
    help="Eval success rate at which β reaches beta_end (adaptive schedule).",
)
parser.add_argument(
    "--beta_success_ema",
    type=float,
    default=0.7,
    help="EMA coefficient for eval success when updating adaptive β.",
)
parser.add_argument(
    "--beta_eval_metric",
    type=str,
    default="success",
    choices=("success", "success_5cm", "grasp"),
    help="Eval metric used as adaptive β input. Keep strict success for reporting; use success_5cm for bring-up.",
)
parser.add_argument(
    "--beta_grasp_gated",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Two-phase linear β: decay to beta_grasp_floor while collect grasp improves, then decay to 0 at beta_grasp_end_iter.",
)
parser.add_argument(
    "--beta_grasp_floor",
    type=float,
    default=0.6,
    help="Phase-1 β lower bound (min teacher mix) before grasp-stable phase-2 decay.",
)
parser.add_argument(
    "--beta_grasp_end_iter",
    type=int,
    default=2000,
    help="Absolute iteration when phase-2 β reaches beta_end (usually 0).",
)
parser.add_argument(
    "--beta_grasp_target",
    type=float,
    default=0.50,
    help="Collect rollout grasp EMA target that allows full phase-1 β decay speed.",
)
parser.add_argument(
    "--beta_grasp_collect_ema",
    type=float,
    default=0.7,
    help="EMA coefficient for collect rollout grasp rate (grasp-gated β).",
)
parser.add_argument(
    "--beta_grasp_stable_window",
    type=int,
    default=5,
    help="Collect-grasp window length to detect plateau (grasp-gated β).",
)
parser.add_argument(
    "--beta_grasp_stable_delta",
    type=float,
    default=0.03,
    help="Max tail delta in stable window to latch grasp-stable (grasp-gated β).",
)
parser.add_argument(
    "--beta_grasp_stable_min_rate",
    type=float,
    default=0.30,
    help="Minimum collect grasp EMA before latch to phase-2 (grasp-gated β).",
)
parser.add_argument(
    "--beta_grasp_stable_min_improvement",
    type=float,
    default=0.05,
    help="Minimum grasp EMA rise over stable window before latch (grasp-gated β).",
)
parser.add_argument(
    "--beta_phase1_decay_iters",
    type=int,
    default=2000,
    help="Nominal phase-1 linear decay length after warmup (scaled by grasp EMA).",
)
parser.add_argument(
    "--beta_grasp_reset_on_resume",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Reset grasp-gated β state when resuming (schedule counts from resume iteration).",
)
parser.add_argument(
    "--success_reserve_ratio",
    type=float,
    default=0.35,
    help="Fraction of aggregate buffer reserved for lift-success episodes.",
)
parser.add_argument(
    "--fixed_lift_goal",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use a fixed object_pose lift target instead of random resampling.",
)
parser.add_argument("--lift_goal_x", type=float, default=0.5)
parser.add_argument("--lift_goal_y", type=float, default=0.0)
parser.add_argument("--lift_goal_z", type=float, default=0.375)
parser.add_argument(
    "--lift_goal_above_cube",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Interaction-geometry lift: sync goal to cube xy + height (vertical ACTUATE) instead of fixed/random goal.",
)
parser.add_argument(
    "--lift_height_above_cube",
    type=float,
    default=0.20,
    help="Vertical ACTUATE goal height above cube spawn (with --lift_goal_above_cube).",
)
parser.add_argument(
    "--randomize_cube",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Randomize cube spawn offset on reset (Isaac default x/y ranges).",
)
parser.add_argument(
    "--robot_x_offset",
    type=float,
    default=0.0,
    help="Initial robot base x offset. Negative values move the EE camera farther back for cube visibility.",
)
parser.add_argument(
    "--panda_joint6",
    type=float,
    default=None,
    help="Optional initial panda_joint6 override for wrist-camera visibility.",
)
parser.add_argument(
    "--randomize_cube_z_only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Randomize cube spawn on z only (x/y fixed at default).",
)
parser.add_argument("--cube_z_min", type=float, default=0.0, help="Cube z pose range min (with --randomize_cube_z_only).")
parser.add_argument("--cube_z_max", type=float, default=0.0, help="Cube z pose range max (with --randomize_cube_z_only).")
parser.add_argument(
    "--cube_x_min",
    type=float,
    default=-0.1,
    help="Cube spawn x offset min from default pose (with --randomize_cube).",
)
parser.add_argument(
    "--cube_x_max",
    type=float,
    default=0.1,
    help="Cube spawn x offset max from default pose (with --randomize_cube).",
)
parser.add_argument(
    "--cube_y_min",
    type=float,
    default=-0.25,
    help="Cube spawn y offset min from default pose (with --randomize_cube).",
)
parser.add_argument(
    "--cube_y_max",
    type=float,
    default=0.25,
    help="Cube spawn y offset max from default pose (with --randomize_cube).",
)
parser.add_argument(
    "--randomize_lift_goal",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Sample lift goal from default Isaac-Lift-Cube command ranges each episode.",
)
parser.add_argument(
    "--guidance_mode",
    type=str,
    default="anchor_only",
    choices=("anchor_only", "anchor_field_demo", "anchor_field_rule"),
    help="anchor_only=8ch; anchor_field_demo=learned G_θ (+2ch); anchor_field_rule=GT G (+3ch, B3).",
)
parser.add_argument(
    "--field_checkpoint",
    type=str,
    default="",
    help="AutoResearch guidance field checkpoint (.pt) for anchor_field_demo.",
)
parser.add_argument(
    "--g_aug_noise_std",
    type=float,
    default=0.0,
    help="Training-only Gaussian noise std applied to appended G channels, followed by renormalization.",
)
parser.add_argument(
    "--g_aug_dropout_prob",
    type=float,
    default=0.0,
    help="Training-only probability of zeroing appended G channels per env/time step.",
)
parser.add_argument(
    "--g_aug_blend_gt_prob",
    type=float,
    default=0.0,
    help="Rollout-only probability of replacing learned G with runtime GT G before student policy inference.",
)
parser.add_argument(
    "--g_aug_reach_only",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply G augmentation only during REACH phase (teacher_phase < grasp_sample_min_phase).",
)
parser.add_argument(
    "--autoresearch_root",
    type=str,
    default=os.environ.get("AUTORESEARCH_ROOT", "/home/l/code/AutoResearch"),
    help="Path to AutoResearch repo (for loading MaskFieldCNN).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift.mdp.data_collection import (
    EpisodeLiftSuccessTracker,
    compute_cube_to_goal_distance,
    compute_grasp_success_from_snapshot,
    compute_lift_success_from_snapshot,
    get_ee_pose_in_robot_root_frame,
    get_mean_finger_opening,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.observations import object_position_in_robot_root_frame
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.vision_observations import (
    set_fixed_colored_mask_palette,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.oracle import LiftCubeOraclePolicy
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student import (
    OraclePhaseStallTracker,
    SamplerEpisodeStats,
    SamplerTeacherLabelStats,
    RNNState,
    apply_lift_teacher_student_env_randomization,
    reset_rnn_state,
    compute_teacher_twist_binary_gripper,
    get_gripper_action_dim,
    infer_teacher_phase,
    make_teacher_arm_std,
    reset_teacher_phase_state,
    run_student_eval_binary_gripper_vision,
    twist_to_env_action,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.eval_utils import early_reset_env_ids
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.dagger_trainer import (
    compute_dagger_beta,
    compute_dagger_beta_adaptive,
    compute_dagger_beta_grasp_gated,
    GraspGatedBetaState,
    mix_rollout_twist,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.vision_dagger_trainer import (
    VisionDaggerAggregateBuffer,
    create_vision_dagger_buffer,
    load_vision_dagger_buffer,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.vision_kl_trainer import (
    optimize_kl_bce_sequences_vision,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.vision_networks import (
    StudentTwistPolicyBinaryGripperVision,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.guidance_field_bridge import (
    LearnedGuidanceField,
    OracleRuleGuidanceField,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.vision_mosaic_debug import (
    MosaicPolicyInputDumper,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.vision_policy_obs import (
    VISUAL_LAYOUT_EE_CHANNELS,
    VISUAL_LAYOUT_MOSAIC,
    VisualMosaicHistory,
    extract_proprio_from_policy_obs_torch,
    mosaic_spatial_shape,
    policy_visual_channels,
    stack_policy_visual_torch,
    stack_visual_from_policy_obs_torch,
)
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.vision_teacher_student_env_cfg import (
    apply_vision_camera_settings,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


def _log_scalar(writer, tag: str, value: float, step: int) -> None:
    if writer is None:
        return
    writer.add_scalar(tag, value, step)
    writer.flush()


def _write_progress_json(log_dir: str, payload: dict) -> None:
    path = os.path.join(log_dir, "progress.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _log_progress_line(log_dir: str, line: str) -> None:
    print(line, flush=True)
    log_path = os.path.join(log_dir, "train.log")
    with open(log_path, "a") as f:
        f.write(line + "\n")
        f.flush()


def _log_sampler_stats(writer, stats: SamplerEpisodeStats, global_step: int, max_episode_steps: int) -> None:
    if stats.num_episodes == 0:
        return
    n = float(stats.num_episodes)
    _log_scalar(writer, "sampler/num_episodes", n, global_step)
    _log_scalar(writer, "sampler/mean_episode_length", stats.mean_episode_length(), global_step)
    _log_scalar(writer, "sampler/timeout_rate", stats.timeout_rate(), global_step)
    _log_scalar(writer, "sampler/grasp_success_rate", stats.grasp_success_count / n, global_step)
    _log_scalar(writer, "sampler/lift_success_rate", stats.lift_success_count / n, global_step)
    _log_scalar(writer, "train/max_episode_steps", float(max_episode_steps), global_step)


def _has_g_augmentation_enabled() -> bool:
    return (
        args_cli.guidance_mode == "anchor_field_demo"
        and (
            args_cli.g_aug_noise_std > 0.0
            or args_cli.g_aug_dropout_prob > 0.0
            or args_cli.g_aug_blend_gt_prob > 0.0
        )
    )


def _guidance_phase_mask(
    guidance: torch.Tensor,
    teacher_phase: torch.Tensor | None,
    *,
    reach_only: bool,
) -> torch.Tensor:
    mask_shape = guidance.shape[:-1] + (1,)
    if teacher_phase is None or not reach_only:
        return torch.ones(mask_shape, dtype=torch.bool, device=guidance.device)
    phase = teacher_phase.to(device=guidance.device)
    while phase.ndim < guidance.ndim - 1:
        phase = phase.unsqueeze(-1)
    return (phase < args_cli.grasp_sample_min_phase).reshape(mask_shape)


def _replace_guidance_channels(visual: torch.Tensor, guidance: torch.Tensor, guidance_channels: int) -> torch.Tensor:
    if visual.ndim == 4:
        return torch.cat(
            [
                visual[:, :-guidance_channels],
                guidance.reshape(guidance.shape[0], guidance_channels, 1, 1).expand_as(
                    visual[:, -guidance_channels:]
                ),
            ],
            dim=1,
        )
    if visual.ndim == 5:
        return torch.cat(
            [
                visual[:, :, :-guidance_channels],
                guidance.reshape(guidance.shape[0], guidance.shape[1], guidance_channels, 1, 1).expand_as(
                    visual[:, :, -guidance_channels:]
                ),
            ],
            dim=2,
        )
    raise ValueError(f"Expected visual ndim 4 or 5, got {visual.ndim}")


def _extract_guidance_channels(visual: torch.Tensor, guidance_channels: int) -> torch.Tensor:
    if visual.ndim == 4:
        return visual[:, -guidance_channels:, 0, 0]
    if visual.ndim == 5:
        return visual[:, :, -guidance_channels:, 0, 0]
    raise ValueError(f"Expected visual ndim 4 or 5, got {visual.ndim}")


def _augment_guidance_channels(
    visual: torch.Tensor,
    *,
    guidance_channels: int,
    teacher_phase: torch.Tensor | None = None,
    gt_guidance: torch.Tensor | None = None,
) -> torch.Tensor:
    if guidance_channels <= 0 or not _has_g_augmentation_enabled():
        return visual
    guidance = _extract_guidance_channels(visual, guidance_channels).to(dtype=torch.float32)
    apply_mask = _guidance_phase_mask(guidance, teacher_phase, reach_only=args_cli.g_aug_reach_only)
    out = guidance

    if gt_guidance is not None and args_cli.g_aug_blend_gt_prob > 0.0:
        gt = gt_guidance.to(device=out.device, dtype=out.dtype)
        if gt.shape[-1] != guidance_channels:
            gt = gt[..., :guidance_channels]
        blend_mask = (torch.rand(out.shape[:-1] + (1,), device=out.device) < args_cli.g_aug_blend_gt_prob) & apply_mask
        out = torch.where(blend_mask, gt.reshape_as(out), out)

    if args_cli.g_aug_noise_std > 0.0:
        noise = torch.randn_like(out) * args_cli.g_aug_noise_std
        out = torch.where(apply_mask, out + noise, out)
        out = torch.nn.functional.normalize(out, dim=-1, eps=1e-6)

    if args_cli.g_aug_dropout_prob > 0.0:
        drop_mask = (torch.rand(out.shape[:-1] + (1,), device=out.device) < args_cli.g_aug_dropout_prob) & apply_mask
        out = torch.where(drop_mask, torch.zeros_like(out), out)

    return _replace_guidance_channels(visual, out.to(dtype=visual.dtype), guidance_channels)


def _teacher_phase_kwargs() -> dict:
    return {
        "grasp_hold_steps": args_cli.teacher_grasp_hold_steps,
        "above_offset_z_m": args_cli.teacher_above_offset_z,
    }


def _cube_ee_delta_in_robot_root_frame(env) -> torch.Tensor:
    ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
    cube_pos_b = object_position_in_robot_root_frame(env)
    return cube_pos_b - ee_pos_b


def _apply_mask_palette_settings(env_cfg) -> None:
    if args_cli.mask_palette_mode != "fixed":
        return
    env_cfg.events.randomize_mask_palette.func = set_fixed_colored_mask_palette
    env_cfg.events.randomize_mask_palette.params = {
        "robot_color": tuple(float(x) for x in args_cli.mask_robot_color),
        "table_color": tuple(float(x) for x in args_cli.mask_table_color),
        "ground_color": tuple(float(x) for x in args_cli.mask_ground_color),
    }


def _partial_rollout_cap() -> int:
    cap = int(args_cli.partial_rollout_cap)
    if cap == 0:
        return 0
    if cap > 0:
        return cap
    return max(int(args_cli.collect_steps) * 2, 256)


def _reset_partial_envs(
    env,
    oracle,
    ep_step: torch.Tensor,
    rnn_hidden: RNNState,
    phase_tracker: OraclePhaseStallTracker | None,
    env_ids: torch.Tensor,
    mosaic_history: VisualMosaicHistory | None = None,
) -> None:
    if env_ids.numel() == 0:
        return
    done_list = env_ids.tolist()
    env.reset(env_ids=env_ids)
    oracle.reset(done_list)
    reset_teacher_phase_state(env, done_list)
    reset_rnn_state(rnn_hidden, env_ids)
    if mosaic_history is not None:
        mosaic_history.reset(env_ids)
    if phase_tracker is not None:
        phase_tracker.reset(env_ids)
    ep_step[env_ids] = 0


def _shift_episode_buffers_left(
    env_id: int,
    shift: int,
    *,
    ep_visual: torch.Tensor,
    ep_proprio: torch.Tensor | None,
    ep_cube_ee_delta: torch.Tensor | None,
    ep_teacher: torch.Tensor,
    ep_teacher_phase: torch.Tensor,
    ep_step: torch.Tensor,
) -> None:
    """Drop the first ``shift`` stored steps and compact in-flight episode tensors."""
    if shift <= 0:
        return
    total = int(ep_step[env_id].item())
    remaining = total - shift
    if remaining > 0:
        ep_visual[env_id, :remaining] = ep_visual[env_id, shift:total].clone()
        if ep_cube_ee_delta is not None:
            ep_cube_ee_delta[env_id, :remaining] = ep_cube_ee_delta[env_id, shift:total].clone()
        ep_teacher[env_id, :remaining] = ep_teacher[env_id, shift:total].clone()
        ep_teacher_phase[env_id, :remaining] = ep_teacher_phase[env_id, shift:total].clone()
        if ep_proprio is not None:
            ep_proprio[env_id, :remaining] = ep_proprio[env_id, shift:total].clone()
    ep_step[env_id] = max(remaining, 0)


def _flush_capped_partial_rollouts(
    env,
    oracle,
    buffer: VisionDaggerAggregateBuffer,
    ep_visual: torch.Tensor,
    ep_proprio: torch.Tensor | None,
    ep_cube_ee_delta: torch.Tensor | None,
    ep_teacher: torch.Tensor,
    ep_teacher_phase: torch.Tensor,
    ep_step: torch.Tensor,
    sampler_stats: SamplerEpisodeStats,
    rnn_hidden: RNNState,
    phase_tracker: OraclePhaseStallTracker | None,
    *,
    env_ids: torch.Tensor | None = None,
    reset_after_flush: bool = False,
    mosaic_history: VisualMosaicHistory | None = None,
) -> int:
    """Persist capped rollout segments to the buffer without ending the live episode.

    When an in-flight rollout reaches ``partial_cap``, write the leading segment to the
    aggregate buffer (subject to ``max_store_episode_steps``), shift the remaining stored
    tensors left, and keep the simulator env running so later segments can capture grasp /
    lift. Only optional ``reset_after_flush`` clears env/oracle/RNN state (e.g. eval envs).
    """
    partial_cap = _partial_rollout_cap()
    if partial_cap <= 0:
        return 0
    if env_ids is None:
        env_ids = torch.arange(ep_step.shape[0], device=ep_step.device)
    flush_ids: list[int] = []
    for env_id in env_ids.tolist():
        if int(ep_step[env_id].item()) >= partial_cap:
            flush_ids.append(env_id)
    if not flush_ids:
        return 0
    flushed = 0
    for env_id in flush_ids:
        while int(ep_step[env_id].item()) >= partial_cap:
            length = partial_cap
            prop = ep_proprio[env_id, :length] if ep_proprio is not None else None
            buffer.add_episode(
                ep_visual[env_id, :length],
                ep_teacher[env_id, :length],
                teacher_phase=ep_teacher_phase[env_id, :length],
                proprio=prop,
                cube_ee_delta=(
                    ep_cube_ee_delta[env_id, :length] if ep_cube_ee_delta is not None else None
                ),
                teacher_success=False,
            )
            sampler_stats.record(
                length=length,
                timed_out=False,
                terminated=False,
                early_done=False,
                sm_state=0,
                grasp_success=False,
                lift_success=False,
            )
            _shift_episode_buffers_left(
                env_id,
                length,
                ep_visual=ep_visual,
                ep_proprio=ep_proprio,
                ep_cube_ee_delta=ep_cube_ee_delta,
                ep_teacher=ep_teacher,
                ep_teacher_phase=ep_teacher_phase,
                ep_step=ep_step,
            )
            flushed += 1
            print(
                f"[INFO] Flushed rollout segment env={env_id} len={length} (cap={partial_cap}); "
                f"episode continues in-env with {int(ep_step[env_id].item())} buffered steps"
            )
    if reset_after_flush:
        ids = torch.tensor(flush_ids, device=ep_step.device, dtype=torch.long)
        _reset_partial_envs(env, oracle, ep_step, rnn_hidden, phase_tracker, ids, mosaic_history)
    return flushed


def _record_and_reset_episodes(
    done_ids: torch.Tensor,
    *,
    env,
    oracle,
    buffer: VisionDaggerAggregateBuffer,
    ep_visual: torch.Tensor,
    ep_proprio: torch.Tensor | None,
    ep_cube_ee_delta: torch.Tensor | None,
    ep_teacher: torch.Tensor,
    ep_teacher_phase: torch.Tensor,
    ep_step: torch.Tensor,
    sampler_stats: SamplerEpisodeStats,
    rnn_hidden: RNNState,
    phase_tracker: OraclePhaseStallTracker | None,
    truncated: torch.Tensor,
    terminated: torch.Tensor,
    early_done: torch.Tensor | None,
    goal_threshold: float,
    pre_teacher_phase: torch.Tensor,
    pre_cube_to_goal: torch.Tensor,
    pre_cube_z_w: torch.Tensor,
    manual_reset_ids: torch.Tensor | None = None,
    mosaic_history: VisualMosaicHistory | None = None,
    buffer_success_only: bool = False,
    success_early: torch.Tensor | None = None,
    ever_lift_success: torch.Tensor | None = None,
) -> None:
    if done_ids.numel() == 0:
        return

    grasp_ok = compute_grasp_success_from_snapshot(pre_cube_z_w)
    lift_ok = compute_lift_success_from_snapshot(
        pre_cube_to_goal, pre_cube_z_w, threshold=goal_threshold
    )
    if ever_lift_success is not None:
        lift_ok = lift_ok | ever_lift_success
    if success_early is not None:
        lift_ok = lift_ok | success_early
    early_mask = early_done[done_ids] if early_done is not None else None
    success_early_mask = success_early[done_ids] if success_early is not None else None

    for idx, env_id in enumerate(done_ids.tolist()):
        length = int(ep_step[env_id].item())
        if length > 0 and (not buffer_success_only or bool(lift_ok[env_id].item())):
            prop = ep_proprio[env_id, :length] if ep_proprio is not None else None
            buffer.add_episode(
                ep_visual[env_id, :length],
                ep_teacher[env_id, :length],
                teacher_phase=ep_teacher_phase[env_id, :length],
                proprio=prop,
                cube_ee_delta=(
                    ep_cube_ee_delta[env_id, :length] if ep_cube_ee_delta is not None else None
                ),
                teacher_success=bool(lift_ok[env_id].item()),
            )
        is_stall_early = bool(early_mask[idx].item()) if early_mask is not None else False
        is_success_early = bool(success_early_mask[idx].item()) if success_early_mask is not None else False
        sampler_stats.record(
            length=length,
            timed_out=bool(truncated[env_id].item()),
            terminated=bool(terminated[env_id].item()),
            early_done=is_stall_early or is_success_early,
            sm_state=int(pre_teacher_phase[env_id].item()),
            grasp_success=bool(grasp_ok[env_id].item()),
            lift_success=bool(lift_ok[env_id].item()),
        )
        ep_step[env_id] = 0

    done_list = done_ids.tolist()
    oracle.reset(done_list)
    reset_teacher_phase_state(env, done_list)
    reset_rnn_state(rnn_hidden, done_ids)
    if mosaic_history is not None:
        mosaic_history.reset(done_ids)
    if phase_tracker is not None:
        phase_tracker.reset(done_ids)
    if manual_reset_ids is not None and manual_reset_ids.numel() > 0:
        env.reset(env_ids=manual_reset_ids)


def _save_training_checkpoint(
    path: str,
    *,
    policy,
    optimizer,
    iteration: int,
    global_step: int,
    teacher_arm_std: torch.Tensor,
    best_success_rate: float,
    buffer: VisionDaggerAggregateBuffer | None = None,
    include_buffer: bool = False,
    proprio_dim: int = 0,
    image_size: int = 64,
    visual_channels: int | None = None,
    grasp_beta_state: GraspGatedBetaState | None = None,
) -> None:
    resolved_visual_channels = visual_channels if visual_channels is not None else args_cli.visual_channels
    payload = {
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "global_step": global_step,
        "teacher_arm_std": teacher_arm_std,
        "gripper_bce_weight": args_cli.gripper_bce_weight,
        "visual_channels": resolved_visual_channels,
        "proprio_dim": proprio_dim,
        "image_size": image_size,
        "obs_mode": args_cli.obs_mode,
        "policy_type": "binary_gripper_dual_head_vision_dagger",
        "guidance_mode": args_cli.guidance_mode,
        "field_checkpoint": args_cli.field_checkpoint or None,
        "g_aug_noise_std": args_cli.g_aug_noise_std,
        "g_aug_dropout_prob": args_cli.g_aug_dropout_prob,
        "g_aug_blend_gt_prob": args_cli.g_aug_blend_gt_prob,
        "g_aug_reach_only": args_cli.g_aug_reach_only,
        "aux_cube_ee_weight": args_cli.aux_cube_ee_weight,
        "aux_cube_ee_scale": args_cli.aux_cube_ee_scale,
        "aux_delta_head_enabled": policy.aux_delta_head is not None,
        "success_rate": best_success_rate,
        "beta_start": args_cli.beta_start,
        "beta_end": args_cli.beta_end,
        "beta_warmup_iters": args_cli.beta_warmup_iters,
        "beta_decay_iters": args_cli.beta_decay_iters,
        "beta_adaptive": args_cli.beta_adaptive,
        "beta_grasp_gated": args_cli.beta_grasp_gated,
        "beta_grasp_floor": args_cli.beta_grasp_floor,
        "beta_grasp_end_iter": args_cli.beta_grasp_end_iter,
        "beta_success_target": args_cli.beta_success_target,
        "success_reserve_ratio": args_cli.success_reserve_ratio,
        "visual_layout": args_cli.visual_layout,
        "mosaic_history_len": args_cli.mosaic_history_len,
        "mask_palette_mode": args_cli.mask_palette_mode,
        "mask_robot_color": tuple(args_cli.mask_robot_color),
        "mask_table_color": tuple(args_cli.mask_table_color),
        "mask_ground_color": tuple(args_cli.mask_ground_color),
        "cube_x_min": args_cli.cube_x_min,
        "cube_x_max": args_cli.cube_x_max,
        "cube_y_min": args_cli.cube_y_min,
        "cube_y_max": args_cli.cube_y_max,
    }
    if grasp_beta_state is not None:
        payload["grasp_beta_state"] = grasp_beta_state.to_dict()
    if include_buffer and buffer is not None:
        payload["aggregate_buffer"] = buffer.state_dict()
    torch.save(payload, path)


def _resolve_autoresearch_path(path: str) -> str:
    """Resolve user paths relative to AUTORESEARCH_ROOT when training cwd is IsaacLab."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    root = args_cli.autoresearch_root or os.environ.get("AUTORESEARCH_ROOT", "")
    if root:
        return os.path.abspath(os.path.join(root, path.lstrip("./")))
    return os.path.abspath(path)


def main():
    torch.manual_seed(args_cli.seed)
    device = args_cli.device
    args_cli.log_dir = _resolve_autoresearch_path(args_cli.log_dir)
    if args_cli.resume_checkpoint:
        args_cli.resume_checkpoint = _resolve_autoresearch_path(args_cli.resume_checkpoint)
    if args_cli.resume_buffer:
        args_cli.resume_buffer = _resolve_autoresearch_path(args_cli.resume_buffer)
    if args_cli.aggregate_storage_dir:
        args_cli.aggregate_storage_dir = _resolve_autoresearch_path(args_cli.aggregate_storage_dir)
    if args_cli.debug_save_mosaic_dir:
        args_cli.debug_save_mosaic_dir = _resolve_autoresearch_path(args_cli.debug_save_mosaic_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    _apply_mask_palette_settings(env_cfg)
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    if abs(args_cli.robot_x_offset) > 1.0e-9:
        env_cfg.scene.robot.init_state.pos = (args_cli.robot_x_offset, 0.0, 0.0)
        print(f"[INFO] robot init_state.pos=({args_cli.robot_x_offset}, 0.0, 0.0)")
    if args_cli.panda_joint6 is not None:
        joint_pos = dict(env_cfg.scene.robot.init_state.joint_pos)
        joint_pos["panda_joint6"] = args_cli.panda_joint6
        env_cfg.scene.robot.init_state.joint_pos = joint_pos
        print(f"[INFO] robot init panda_joint6={args_cli.panda_joint6}")
    if args_cli.randomize_lift_goal and args_cli.fixed_lift_goal:
        print("[WARN] --randomize_lift_goal overrides --fixed_lift_goal.")
    cube_pose_range = None
    if args_cli.randomize_cube_z_only:
        from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.env_randomization import (
            cube_pose_range_z_only,
        )

        cube_pose_range = cube_pose_range_z_only(z_min=args_cli.cube_z_min, z_max=args_cli.cube_z_max)
    elif args_cli.randomize_cube:
        from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.env_randomization import (
            cube_pose_range_xy,
        )

        cube_pose_range = cube_pose_range_xy(
            x_min=args_cli.cube_x_min,
            x_max=args_cli.cube_x_max,
            y_min=args_cli.cube_y_min,
            y_max=args_cli.cube_y_max,
        )
    apply_lift_teacher_student_env_randomization(
        env_cfg,
        randomize_cube=args_cli.randomize_cube or args_cli.randomize_cube_z_only,
        randomize_lift_goal=args_cli.randomize_lift_goal,
        fixed_lift_goal=args_cli.fixed_lift_goal and not args_cli.randomize_lift_goal,
        lift_goal_above_cube=args_cli.lift_goal_above_cube and not args_cli.randomize_lift_goal,
        lift_height_above_cube=args_cli.lift_height_above_cube,
        lift_goal_x=args_cli.lift_goal_x,
        lift_goal_y=args_cli.lift_goal_y,
        lift_goal_z=args_cli.lift_goal_z,
        cube_pose_range=cube_pose_range,
        debug_vis_goal=True if (args_cli.randomize_cube or args_cli.randomize_cube_z_only or args_cli.randomize_lift_goal) else None,
    )
    if args_cli.lift_goal_above_cube and not args_cli.randomize_lift_goal:
        print(
            f"[INFO] Interaction-geometry lift goal: vertical ACTUATE, cube xy + {args_cli.lift_height_above_cube} m above spawn"
        )
    cam_h = args_cli.camera_height or args_cli.image_size
    cam_w = args_cli.camera_width or args_cli.image_size
    apply_vision_camera_settings(
        env_cfg,
        camera_height=cam_h,
        camera_width=cam_w,
        topdown_close=args_cli.topdown_close,
    )
    env_cfg.actions.arm_action.scale = args_cli.arm_scale
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env._teacher_phase_geometry_kwargs = _teacher_phase_kwargs()

    control_dt = env_cfg.sim.dt * env_cfg.decimation
    gripper_action_dim = get_gripper_action_dim(env)
    if args_cli.obs_mode == "vision_proprio" and "Proprio" not in args_cli.task:
        print("[WARN] obs_mode=vision_proprio but task is not *-Proprio-v0; proprio may be missing.")

    proprio_dim = 7 if args_cli.obs_mode == "vision_proprio" else 0
    image_size = args_cli.image_size
    visual_layout = args_cli.visual_layout
    if visual_layout == VISUAL_LAYOUT_MOSAIC and args_cli.guidance_mode != "anchor_only":
        raise ValueError("--visual_layout=mosaic is only supported with --guidance_mode=anchor_only (B1) for now.")

    os.environ["AUTORESEARCH_ROOT"] = args_cli.autoresearch_root
    guidance_field: LearnedGuidanceField | OracleRuleGuidanceField | None = None
    if args_cli.guidance_mode == "anchor_field_demo":
        if not args_cli.field_checkpoint:
            raise ValueError("--field_checkpoint is required when --guidance_mode=anchor_field_demo")
        guidance_field = LearnedGuidanceField(args_cli.field_checkpoint, device=device)
        visual_channels = 8 + guidance_field.extra_visual_channels
    elif args_cli.guidance_mode == "anchor_field_rule":
        guidance_field = OracleRuleGuidanceField(device=device)
        visual_channels = 8 + guidance_field.extra_visual_channels
    elif visual_layout in (VISUAL_LAYOUT_EE_CHANNELS, VISUAL_LAYOUT_MOSAIC):
        visual_channels = policy_visual_channels(visual_layout=visual_layout)
    else:
        visual_channels = args_cli.visual_channels
    if visual_layout == VISUAL_LAYOUT_MOSAIC:
        policy_visual_h, policy_visual_w = mosaic_spatial_shape(image_size)
    else:
        policy_visual_h = policy_visual_w = image_size
    guidance_channels = 0 if guidance_field is None else int(guidance_field.extra_visual_channels)
    if visual_channels != args_cli.visual_channels and args_cli.guidance_mode == "anchor_only":
        pass
    elif args_cli.guidance_mode in ("anchor_field_demo", "anchor_field_rule") and args_cli.visual_channels != visual_channels:
        print(
            f"[INFO] visual_channels overridden {args_cli.visual_channels} -> {visual_channels} "
            f"(+{guidance_field.extra_visual_channels} guidance)"
        )

    policy = StudentTwistPolicyBinaryGripperVision(
        visual_channels=visual_channels,
        proprio_dim=proprio_dim,
        enable_aux_delta_head=args_cli.aux_cube_ee_weight > 0.0,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.learning_rate, amsgrad=True)
    oracle = LiftCubeOraclePolicy(env, position_threshold=args_cli.position_threshold)
    teacher_arm_std = make_teacher_arm_std(
        lin_std=args_cli.teacher_lin_std,
        ang_std=args_cli.teacher_ang_std,
        device=device,
    )

    start_iteration = 0
    global_step = 0
    best_success_rate = -1.0
    grasp_beta_state = GraspGatedBetaState()
    buffer_path = os.path.join(args_cli.log_dir, "dagger_aggregate.pt")
    resume_ckpt: dict | None = None
    embedded_buffer_state: dict | None = None

    if args_cli.resume_checkpoint:
        if not os.path.isfile(args_cli.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {args_cli.resume_checkpoint}")
        resume_ckpt = torch.load(args_cli.resume_checkpoint, map_location="cpu", weights_only=False)
        embedded_buffer_state = resume_ckpt.pop("aggregate_buffer", None)
        StudentTwistPolicyBinaryGripperVision.load_from_checkpoint(
            policy,
            resume_ckpt,
            visual_channels=visual_channels,
            proprio_dim=proprio_dim,
        )
        if "optimizer" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
            except ValueError as exc:
                print(
                    "[WARN] Skipping optimizer state from checkpoint because the policy parameter set changed "
                    f"(likely aux head added): {exc}"
                )
        start_iteration = int(resume_ckpt.get("iteration", 0))
        global_step = int(resume_ckpt.get("global_step", 0))
        best_success_rate = float(resume_ckpt.get("success_rate", -1.0))
        if "teacher_arm_std" in resume_ckpt:
            teacher_arm_std = resume_ckpt["teacher_arm_std"].to(device)
        if args_cli.beta_grasp_gated and not args_cli.beta_grasp_reset_on_resume:
            grasp_beta_state = GraspGatedBetaState.from_dict(resume_ckpt.get("grasp_beta_state"))
        elif args_cli.beta_grasp_gated and args_cli.beta_grasp_reset_on_resume:
            grasp_beta_state = GraspGatedBetaState()
        resume_ckpt.clear()
        del resume_ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[INFO] Resumed policy from {args_cli.resume_checkpoint}: "
            f"iteration={start_iteration}, global_step={global_step}"
        )
        if args_cli.beta_grasp_gated and args_cli.beta_grasp_reset_on_resume:
            print(
                f"[INFO] Grasp-gated β schedule resets at resume (β uses iter - {start_iteration})."
            )

    aggregate_storage_dir = args_cli.aggregate_storage_dir or os.path.join(args_cli.log_dir, "dagger_episodes")
    buffer = create_vision_dagger_buffer(
        backend=args_cli.aggregate_backend,
        max_episodes=args_cli.max_aggregate_episodes,
        device=device,
        success_reserve_ratio=args_cli.success_reserve_ratio,
        storage_dir=aggregate_storage_dir,
        cache_size=args_cli.aggregate_cache_size,
        visual_storage_dtype=args_cli.aggregate_visual_storage_dtype,
        max_store_episode_steps=args_cli.max_store_episode_steps,
    )
    buffer_loaded = False
    resume_buffer_path = ""
    if not args_cli.no_resume_buffer:
        resume_buffer_path = args_cli.resume_buffer or (
            buffer_path if args_cli.resume_checkpoint and os.path.isfile(buffer_path) else ""
        )
    elif args_cli.resume_buffer:
        resume_buffer_path = args_cli.resume_buffer
    if resume_buffer_path and os.path.isfile(resume_buffer_path):
        print(f"[INFO] Loading aggregate buffer ({os.path.getsize(resume_buffer_path) / 1e9:.1f} GB): {resume_buffer_path}")
        buffer = load_vision_dagger_buffer(
            resume_buffer_path,
            device=device,
            default_backend=args_cli.aggregate_backend,
            default_max_episodes=args_cli.max_aggregate_episodes,
            default_success_reserve_ratio=args_cli.success_reserve_ratio,
            storage_dir=aggregate_storage_dir,
            cache_size=args_cli.aggregate_cache_size,
            visual_storage_dtype=args_cli.aggregate_visual_storage_dtype,
            max_store_episode_steps=args_cli.max_store_episode_steps,
        )
        buffer_loaded = True
        print(f"[INFO] Loaded aggregate buffer: {resume_buffer_path} episodes={len(buffer)} steps={buffer.num_steps}")
    elif embedded_buffer_state is not None and not args_cli.no_resume_buffer:
        buffer.load_state_dict(embedded_buffer_state)
        buffer_loaded = True
        print(f"[INFO] Loaded aggregate buffer from checkpoint episodes={len(buffer)} steps={buffer.num_steps}")

    if args_cli.no_resume_buffer and args_cli.resume_checkpoint:
        print("[INFO] --no_resume_buffer: starting with empty aggregate buffer (policy-only resume).")

    if args_cli.oracle_demo_file:
        print("[WARN] oracle_demo_file is ignored for vision DAgger (use online aggregate only).")

    end_iteration = start_iteration + args_cli.max_iterations

    os.makedirs(args_cli.log_dir, exist_ok=True)
    tb_log_dir = os.path.join(args_cli.log_dir, "tb")
    writer = SummaryWriter(log_dir=tb_log_dir) if SummaryWriter is not None else None

    num_envs = env.num_envs
    num_eval_envs = min(max(args_cli.num_eval_envs, 1), num_envs)
    max_ep_len = env.max_episode_length
    episode_storage = torch.device("cpu")
    rnn_hidden: RNNState = policy.init_hidden(num_envs, device)
    ep_visual = torch.zeros(
        num_envs, max_ep_len, visual_channels, policy_visual_h, policy_visual_w, device=episode_storage
    )
    mosaic_history = None
    if visual_layout == VISUAL_LAYOUT_MOSAIC:
        mosaic_history = VisualMosaicHistory(
            num_envs,
            image_size,
            device,
            history_len=args_cli.mosaic_history_len,
        )
    ep_proprio = (
        torch.zeros(num_envs, max_ep_len, proprio_dim, device=episode_storage) if proprio_dim > 0 else None
    )
    ep_cube_ee_delta = (
        torch.zeros(num_envs, max_ep_len, 3, device=episode_storage)
        if args_cli.aux_cube_ee_weight > 0.0
        else None
    )
    ep_teacher = torch.zeros(num_envs, max_ep_len, 7, device=episode_storage)
    ep_teacher_phase = torch.zeros(num_envs, max_ep_len, dtype=torch.long, device=episode_storage)
    ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    sampler_stats = SamplerEpisodeStats()
    teacher_label_stats = SamplerTeacherLabelStats()
    phase_tracker = None
    if args_cli.early_done:
        phase_tracker = OraclePhaseStallTracker(
            num_envs,
            device,
            max_stall_steps=args_cli.phase_stall_steps,
            min_progress_m=args_cli.phase_min_progress_m,
            min_sm_state=args_cli.early_done_min_sm_state,
        )
    success_tracker = None
    if args_cli.success_early_done:
        success_tracker = EpisodeLiftSuccessTracker(num_envs, device)

    print(f"[INFO] Task: {args_cli.task}")
    print(
        f"[INFO] Vision DAgger: obs_mode={args_cli.obs_mode}, guidance_mode={args_cli.guidance_mode}, "
        f"visual_layout={visual_layout}, visual=({visual_channels},{policy_visual_h},{policy_visual_w}), "
        f"proprio_dim={proprio_dim}, num_eval_envs={num_eval_envs}/{num_envs}, episode_storage=cpu"
    )
    if visual_layout == VISUAL_LAYOUT_MOSAIC:
        print(
            f"[INFO] Mosaic input: (4, 2H, W) single frame — ee/top vertical, mask RGB + depth per row"
        )
    elif visual_layout == VISUAL_LAYOUT_EE_CHANNELS:
        print("[INFO] EE-only input: (4, H, W) — EE mask RGB + EE depth; third-person input excluded.")
    if args_cli.mask_palette_mode == "fixed":
        print(
            f"[INFO] Mask palette fixed: robot={tuple(args_cli.mask_robot_color)} "
            f"table={tuple(args_cli.mask_table_color)} ground={tuple(args_cli.mask_ground_color)} cube=(1,1,1)"
        )
    else:
        print("[INFO] Mask palette randomizes robot/table/ground colors on reset; cube=(1,1,1).")
    mosaic_debug: MosaicPolicyInputDumper | None = None
    if visual_layout == VISUAL_LAYOUT_MOSAIC and args_cli.debug_save_mosaic_dir:
        mosaic_debug_dir = os.path.abspath(args_cli.debug_save_mosaic_dir)
        mosaic_debug = MosaicPolicyInputDumper(
            mosaic_debug_dir,
            image_size,
            interval=args_cli.debug_save_mosaic_interval,
            max_saves=args_cli.debug_save_mosaic_max,
            env_ids=(args_cli.debug_save_mosaic_env_id,),
        )
        print(
            f"[INFO] Mosaic debug dump: dir={mosaic_debug_dir} "
            f"interval={args_cli.debug_save_mosaic_interval} max={args_cli.debug_save_mosaic_max} "
            f"env_id={args_cli.debug_save_mosaic_env_id}"
        )
    print(
        f"[INFO] Camera: native={cam_w}x{cam_h} topdown_close={args_cli.topdown_close} "
        f"(stored visual resize -> {image_size}x{image_size})"
    )
    if args_cli.beta_grasp_gated:
        print(
            f"[INFO] DAgger: grasp-gated β {args_cli.beta_start}->{args_cli.beta_grasp_floor} "
            f"(phase1, grasp-scaled) -> {args_cli.beta_end} @ iter {args_cli.beta_grasp_end_iter}, "
            f"warmup={args_cli.beta_warmup_iters}, grasp_target={args_cli.beta_grasp_target}, "
            f"stable_min={args_cli.beta_grasp_stable_min_rate}"
        )
    elif args_cli.beta_adaptive:
        print(
            f"[INFO] DAgger: adaptive β {args_cli.beta_start}->{args_cli.beta_end}, "
            f"warmup={args_cli.beta_warmup_iters}, success_target={args_cli.beta_success_target}, "
            f"success_ema={args_cli.beta_success_ema}"
        )
    else:
        print(
            f"[INFO] DAgger: linear β {args_cli.beta_start}->{args_cli.beta_end}, "
            f"warmup={args_cli.beta_warmup_iters}, decay={args_cli.beta_decay_iters}"
        )
    if args_cli.lift_goal_above_cube and not args_cli.randomize_lift_goal:
        pass  # vertical ACTUATE goal already logged above
    elif args_cli.fixed_lift_goal and not args_cli.randomize_lift_goal:
        print(
            f"[INFO] Fixed lift goal: "
            f"({args_cli.lift_goal_x}, {args_cli.lift_goal_y}, {args_cli.lift_goal_z})"
        )
    if args_cli.randomize_cube or args_cli.randomize_cube_z_only or args_cli.randomize_lift_goal:
        if args_cli.randomize_cube_z_only:
            cube_dr = f"z_only[z={args_cli.cube_z_min},{args_cli.cube_z_max}]"
        elif args_cli.randomize_cube:
            cube_dr = (
                f"xy[x={args_cli.cube_x_min}..{args_cli.cube_x_max},"
                f"y={args_cli.cube_y_min}..{args_cli.cube_y_max},z=0]"
            )
        else:
            cube_dr = "off"
        print(
            f"[INFO] Domain randomization: cube={cube_dr}, "
            f"lift_goal={args_cli.randomize_lift_goal}"
        )
        if args_cli.max_aggregate_episodes < 1000:
            print(
                "[WARN] Domain randomization usually needs a larger aggregate buffer "
                f"(current max_aggregate_episodes={args_cli.max_aggregate_episodes}; "
                "recommend >=1500, ideally 2000–3000)."
            )
    print(
        f"[INFO] Aggregate buffer max_episodes={args_cli.max_aggregate_episodes}, "
        f"success_reserve={args_cli.success_reserve_ratio:.0%}, "
        f"min_buffer_episodes={args_cli.min_buffer_episodes}, current={len(buffer)} "
        f"(success={buffer.num_success_episodes}, rolling={buffer.num_rolling_episodes}) "
        f"backend={args_cli.aggregate_backend} storage={buffer.storage_device} "
        f"max_store_episode_steps={args_cli.max_store_episode_steps} partial_cap={_partial_rollout_cap()}"
    )
    print(
        f"[INFO] Teacher: min_lin={args_cli.teacher_min_lin_speed} min_ang={args_cli.teacher_min_ang_speed} "
        f"max_lin={args_cli.teacher_max_lin_vel} max_ang={args_cli.teacher_max_ang_vel} "
        f"arm_scale={args_cli.arm_scale} grasp_hold={args_cli.teacher_grasp_hold_steps} "
        f"hover_z={args_cli.teacher_above_offset_z} goal_thr={args_cli.goal_threshold}m"
    )
    if args_cli.success_early_done:
        print("[INFO] Rollout ends on lift success (success_early_done).")
    if args_cli.buffer_success_only:
        print("[INFO] DAgger buffer stores successful episodes only (no timeout/failure segments).")
    if args_cli.aggregate_backend == "disk":
        print(
            f"[INFO] Aggregate disk store: dir={aggregate_storage_dir}, "
            f"visual_dtype={args_cli.aggregate_visual_storage_dtype}, cache={args_cli.aggregate_cache_size}"
        )
    print(
        f"[INFO] Collect: steps/iter={args_cli.collect_steps}, optim_steps={args_cli.optim_steps}, "
        f"batch={args_cli.batch_size}, seq_len={args_cli.seq_len}, grad_interval={args_cli.grad_interval}"
    )
    if args_cli.aux_cube_ee_weight > 0.0:
        print(
            f"[INFO] Aux loss: CNN -> normalized cube_pos_b - ee_pos_b Huber, "
            f"weight={args_cli.aux_cube_ee_weight}, scale={args_cli.aux_cube_ee_scale}m"
        )
    else:
        print("[INFO] Aux loss disabled (--aux_cube_ee_weight=0).")
    print(f"[INFO] TensorBoard: {tb_log_dir}")
    if _has_g_augmentation_enabled():
        print(
            "[INFO] B7 G augmentation enabled: "
            f"channels={guidance_channels}, noise_std={args_cli.g_aug_noise_std}, "
            f"dropout_prob={args_cli.g_aug_dropout_prob}, blend_gt_prob={args_cli.g_aug_blend_gt_prob}, "
            f"reach_only={args_cli.g_aug_reach_only}"
        )

    env.reset()
    oracle.reset()
    reset_teacher_phase_state(env)

    start_time = time.time()
    teacher_exec_steps = 0
    rollout_steps = 0
    beta_success_ema: float | None = None

    for iteration in range(start_iteration, end_iteration):
        beta_schedule_iter = iteration - start_iteration
        if args_cli.beta_grasp_gated:
            beta = compute_dagger_beta_grasp_gated(
                beta_schedule_iter,
                grasp_beta_state,
                beta_start=args_cli.beta_start,
                beta_end=args_cli.beta_end,
                beta_floor=args_cli.beta_grasp_floor,
                warmup_iters=args_cli.beta_warmup_iters,
                phase1_decay_iters=args_cli.beta_phase1_decay_iters,
                grasp_end_iter=args_cli.beta_grasp_end_iter,
                grasp_target=args_cli.beta_grasp_target,
            )
        elif args_cli.beta_adaptive:
            beta = compute_dagger_beta_adaptive(
                beta_schedule_iter,
                beta_success_ema,
                beta_start=args_cli.beta_start,
                beta_end=args_cli.beta_end,
                warmup_iters=args_cli.beta_warmup_iters,
                success_target=args_cli.beta_success_target,
            )
        else:
            beta = compute_dagger_beta(
                beta_schedule_iter,
                beta_start=args_cli.beta_start,
                beta_end=args_cli.beta_end,
                warmup_iters=args_cli.beta_warmup_iters,
                decay_iters=args_cli.beta_decay_iters,
            )
        sampler_stats.reset()
        teacher_label_stats.reset()
        g_align_sum = 0.0
        g_angle_sum = 0.0
        g_l2_sum = 0.0
        g_align_n = 0
        g_reach_sum = 0.0
        g_reach_n = 0
        g_actuate_sum = 0.0
        g_actuate_n = 0
        policy.eval()

        for _ in range(args_cli.collect_steps):
            student_obs = env.observation_manager.compute_group("policy")
            visual = stack_policy_visual_torch(
                student_obs,
                image_size=image_size,
                visual_layout=visual_layout,
                mosaic_history=mosaic_history,
            )
            if mosaic_debug is not None:
                mosaic_debug.maybe_save(visual, iteration=iteration, phase="collect")
            if guidance_field is not None:
                visual = guidance_field.augment_visual(
                    visual, student_obs, image_size=image_size, env=env
                )
            proprio = extract_proprio_from_policy_obs_torch(student_obs)
            proprio_step = None if proprio is None else proprio.unsqueeze(1)

            with torch.no_grad():
                teacher_twist = compute_teacher_twist_binary_gripper(
                    env,
                    oracle,
                    control_dt=control_dt,
                    min_lin_speed=args_cli.teacher_min_lin_speed,
                    min_ang_speed=args_cli.teacher_min_ang_speed,
                    max_lin_vel=args_cli.teacher_max_lin_vel,
                    max_ang_vel=args_cli.teacher_max_ang_vel,
                    teacher_phase_kwargs=_teacher_phase_kwargs(),
                )
                teacher_phase = infer_teacher_phase(env)
                pre_teacher_phase = teacher_phase
                cube_ee_delta = (
                    _cube_ee_delta_in_robot_root_frame(env) if ep_cube_ee_delta is not None else None
                )
                pre_cube_to_goal = compute_cube_to_goal_distance(env)
                pre_cube_z_w = env.scene["object"].data.root_pos_w[:, 2].clone()
                teacher_label_stats.observe(teacher_twist[:, 6], teacher_phase)
                if guidance_field is not None and hasattr(guidance_field, "alignment_metrics"):
                    g_metrics = guidance_field.alignment_metrics(
                        student_obs,
                        env=env,
                        image_size=image_size,
                        teacher_phase=teacher_phase,
                    )
                    n_all = num_envs
                    g_align_sum += g_metrics["cos"] * n_all
                    g_angle_sum += g_metrics["angle_deg"] * n_all
                    g_l2_sum += g_metrics["l2"] * n_all
                    g_align_n += n_all
                    if not math.isnan(g_metrics["reach_cos"]):
                        rn = int(g_metrics["reach_n"])
                        g_reach_sum += g_metrics["reach_cos"] * rn
                        g_reach_n += rn
                    if not math.isnan(g_metrics["actuate_cos"]):
                        an = int(g_metrics["actuate_n"])
                        g_actuate_sum += g_metrics["actuate_cos"] * an
                        g_actuate_n += an
                if phase_tracker is not None:
                    phase_tracker.on_pre_step(oracle, env)

                policy_visual = visual
                if _has_g_augmentation_enabled():
                    gt_guidance = None
                    if guidance_channels > 0 and args_cli.g_aug_blend_gt_prob > 0.0:
                        gt_guidance = OracleRuleGuidanceField.compute_subgoal_vectors_3d(
                            env,
                            teacher_phase,
                            grasp_phase_min=args_cli.grasp_sample_min_phase,
                        )
                    policy_visual = _augment_guidance_channels(
                        visual,
                        guidance_channels=guidance_channels,
                        teacher_phase=teacher_phase,
                        gt_guidance=gt_guidance,
                    )
                visual_step = policy_visual.unsqueeze(1)
                student_twist, rnn_hidden, _ = policy.act(
                    visual_step, rnn_hidden, proprio_step, deterministic=False
                )
                exec_twist, teacher_exec_mask = mix_rollout_twist(teacher_twist, student_twist, beta)
                teacher_exec_steps += int(teacher_exec_mask.sum().item())
                rollout_steps += num_envs

                current_opening = get_mean_finger_opening(env)
                env_actions = twist_to_env_action(
                    exec_twist,
                    control_dt=control_dt,
                    current_opening=current_opening,
                    arm_scale=args_cli.arm_scale,
                    max_lin_vel=args_cli.teacher_max_lin_vel,
                    max_ang_vel=args_cli.teacher_max_ang_vel,
                    num_gripper_actions=gripper_action_dim,
                    binary_gripper=True,
                )

            active = ep_step < max_ep_len
            if active.any():
                active_ids = active.nonzero(as_tuple=False).squeeze(-1)
                if active_ids.ndim == 0:
                    active_ids = active_ids.unsqueeze(0)
                step_ids = ep_step[active_ids]
                ep_visual[active_ids, step_ids] = visual[active_ids].to(episode_storage, non_blocking=True)
                if ep_cube_ee_delta is not None and cube_ee_delta is not None:
                    ep_cube_ee_delta[active_ids, step_ids] = cube_ee_delta[active_ids].to(
                        episode_storage, non_blocking=True
                    )
                if ep_proprio is not None and proprio is not None:
                    ep_proprio[active_ids, step_ids] = proprio[active_ids].to(episode_storage, non_blocking=True)
                ep_teacher[active_ids, step_ids] = teacher_twist[active_ids].to(episode_storage, non_blocking=True)
                ep_teacher_phase[active_ids, step_ids] = teacher_phase[active_ids].to(
                    episode_storage, non_blocking=True
                )
                ep_step[active_ids] += 1

            _, _, terminated, truncated, _ = env.step(env_actions)
            global_step += 1

            success_early = None
            if success_tracker is not None:
                success_early = success_tracker.on_step(
                    env,
                    terminated=terminated,
                    truncated=truncated,
                    pre_cube_to_goal=pre_cube_to_goal,
                    pre_cube_z_w=pre_cube_z_w,
                    lift_threshold=args_cli.goal_threshold,
                )

            early_done = None
            if phase_tracker is not None:
                early_done = phase_tracker.on_post_step(oracle, env)

            dones = terminated | truncated
            if early_done is not None:
                dones = dones | early_done
            if success_early is not None:
                dones = dones | success_early
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.ndim == 0:
                    done_ids = done_ids.unsqueeze(0)
                manual_reset_ids = early_reset_env_ids(early_done, success_early)
                _record_and_reset_episodes(
                    done_ids,
                    env=env,
                    oracle=oracle,
                    buffer=buffer,
                    ep_visual=ep_visual,
                    ep_proprio=ep_proprio,
                    ep_cube_ee_delta=ep_cube_ee_delta,
                    ep_teacher=ep_teacher,
                    ep_teacher_phase=ep_teacher_phase,
                    ep_step=ep_step,
                    sampler_stats=sampler_stats,
                    rnn_hidden=rnn_hidden,
                    phase_tracker=phase_tracker,
                    truncated=truncated,
                    terminated=terminated,
                    early_done=early_done,
                    goal_threshold=args_cli.goal_threshold,
                    pre_teacher_phase=pre_teacher_phase,
                    pre_cube_to_goal=pre_cube_to_goal,
                    pre_cube_z_w=pre_cube_z_w,
                    manual_reset_ids=manual_reset_ids,
                    mosaic_history=mosaic_history,
                    buffer_success_only=args_cli.buffer_success_only,
                    success_early=success_early,
                    ever_lift_success=success_tracker.ever_lift_success if success_tracker is not None else None,
                )
                if success_tracker is not None:
                    success_tracker.reset(done_ids)

        _flush_capped_partial_rollouts(
            env,
            oracle,
            buffer,
            ep_visual,
            ep_proprio,
            ep_cube_ee_delta,
            ep_teacher,
            ep_teacher_phase,
            ep_step,
            sampler_stats,
            rnn_hidden,
            phase_tracker,
            mosaic_history=mosaic_history,
        )

        policy.train()
        max_buf_ep_len = buffer.max_episode_length
        can_optimize = len(buffer) >= args_cli.min_buffer_episodes and max_buf_ep_len >= 2
        mean_total_loss = 0.0
        mean_arm_kl = 0.0
        mean_gripper_bce = 0.0
        mean_aux_delta = 0.0
        mean_aux_l2_m = 0.0
        grasp_sample_hit_rate = 0.0
        if can_optimize:
            visual_augment_fn = None
            if _has_g_augmentation_enabled():
                visual_augment_fn = lambda visual_batch, phase_batch: _augment_guidance_channels(
                    visual_batch,
                    guidance_channels=guidance_channels,
                    teacher_phase=phase_batch,
                )
            (
                mean_total_loss,
                mean_arm_kl,
                mean_gripper_bce,
                mean_aux_delta,
                mean_aux_l2_m,
                grasp_sample_hit_rate,
            ) = optimize_kl_bce_sequences_vision(
                policy,
                optimizer,
                buffer,
                teacher_arm_std,
                batch_size=args_cli.batch_size,
                seq_len=args_cli.seq_len,
                optim_steps=args_cli.optim_steps,
                gripper_bce_weight=args_cli.gripper_bce_weight,
                grad_interval=args_cli.grad_interval,
                grasp_sample_ratio=args_cli.grasp_sample_ratio,
                min_grasp_phase=args_cli.grasp_sample_min_phase,
                aux_cube_ee_weight=args_cli.aux_cube_ee_weight,
                aux_cube_ee_scale=args_cli.aux_cube_ee_scale,
                visual_augment_fn=visual_augment_fn,
            )
            optim_skip_reason = "on"
        elif len(buffer) < args_cli.min_buffer_episodes:
            optim_skip_reason = f"buffer {len(buffer)}/{args_cli.min_buffer_episodes}"
        else:
            optim_skip_reason = f"short_episodes max_len={max_buf_ep_len}"

        teacher_exec_rate = teacher_exec_steps / max(rollout_steps, 1)
        teacher_exec_steps = 0
        rollout_steps = 0

        if args_cli.beta_grasp_gated and sampler_stats.num_episodes > 0:
            collect_grasp = sampler_stats.grasp_success_count / sampler_stats.num_episodes
            grasp_beta_state.observe_collect_grasp(
                collect_grasp,
                ema=args_cli.beta_grasp_collect_ema,
            )
            latched = grasp_beta_state.maybe_latch_grasp_stable(
                beta_schedule_iter,
                warmup_iters=args_cli.beta_warmup_iters,
                stable_window=args_cli.beta_grasp_stable_window,
                stable_delta=args_cli.beta_grasp_stable_delta,
                min_rate=args_cli.beta_grasp_stable_min_rate,
                min_improvement=args_cli.beta_grasp_stable_min_improvement,
            )
            if latched:
                print(
                    f"[INFO] Grasp-gated β: collect grasp stable at iter {iteration + 1} "
                    f"(ema={grasp_beta_state.grasp_ema:.3f}) -> phase-2 decay to 0 @ iter {args_cli.beta_grasp_end_iter}"
                )

        elapsed = time.time() - start_time
        dagger_line = (
            f"[DAgger {iteration + 1}/{end_iteration}] β={beta:.3f} teacher_exec={teacher_exec_rate:.3f} "
            f"loss={mean_total_loss:.4f} arm_kl={mean_arm_kl:.4f} bce={mean_gripper_bce:.4f} "
            f"aux_delta={mean_aux_delta:.4f} aux_l2_m={mean_aux_l2_m:.4f} "
            f"buffer={len(buffer)} (ok={buffer.num_success_episodes}) steps={buffer.num_steps} time={elapsed:.1f}s"
        )
        optim_line = f"  optim={optim_skip_reason} | {sampler_stats.summary_line(max_ep_len)}"
        _log_progress_line(args_cli.log_dir, dagger_line)
        _log_progress_line(args_cli.log_dir, optim_line)
        _write_progress_json(
            args_cli.log_dir,
            {
                "iteration": iteration + 1,
                "max_iterations": end_iteration,
                "beta": beta,
                "buffer_episodes": len(buffer),
                "buffer_success_episodes": buffer.num_success_episodes,
                "buffer_steps": buffer.num_steps,
                "loss": mean_total_loss,
                "aux_cube_ee_delta_loss": mean_aux_delta,
                "aux_cube_ee_delta_l2_m": mean_aux_l2_m,
                "elapsed_sec": elapsed,
                "optim": optim_skip_reason,
                "guidance_mode": args_cli.guidance_mode,
                "visual_layout": visual_layout,
            },
        )

        _log_scalar(writer, "dagger/beta", beta, global_step)
        if args_cli.beta_grasp_gated:
            if grasp_beta_state.grasp_ema is not None:
                _log_scalar(writer, "dagger/collect_grasp_ema", grasp_beta_state.grasp_ema, global_step)
            _log_scalar(writer, "dagger/grasp_stable", float(grasp_beta_state.grasp_stable), global_step)
        if beta_success_ema is not None:
            _log_scalar(writer, "dagger/eval_success_ema", beta_success_ema, global_step)
        _log_scalar(writer, "dagger/teacher_exec_rate", teacher_exec_rate, global_step)
        _log_scalar(writer, "train/total_loss", mean_total_loss, global_step)
        _log_scalar(writer, "train/arm_kl_loss", mean_arm_kl, global_step)
        _log_scalar(writer, "train/gripper_bce_loss", mean_gripper_bce, global_step)
        _log_scalar(writer, "train/aux_cube_ee_delta_loss", mean_aux_delta, global_step)
        _log_scalar(writer, "train/aux_cube_ee_delta_l2_m", mean_aux_l2_m, global_step)
        _log_scalar(writer, "train/aux_cube_ee_weight", args_cli.aux_cube_ee_weight, global_step)
        _log_scalar(writer, "train/aux_cube_ee_scale", args_cli.aux_cube_ee_scale, global_step)
        _log_scalar(writer, "train/num_episodes", float(len(buffer)), global_step)
        _log_scalar(writer, "train/num_success_episodes", float(buffer.num_success_episodes), global_step)
        _log_scalar(writer, "train/num_rolling_episodes", float(buffer.num_rolling_episodes), global_step)
        _log_scalar(writer, "train/buffer_steps", float(buffer.num_steps), global_step)
        _log_scalar(writer, "train/iteration", float(iteration + 1), global_step)
        _log_scalar(writer, "g_aug/noise_std", args_cli.g_aug_noise_std, global_step)
        _log_scalar(writer, "g_aug/dropout_prob", args_cli.g_aug_dropout_prob, global_step)
        _log_scalar(writer, "g_aug/blend_gt_prob", args_cli.g_aug_blend_gt_prob, global_step)
        if g_align_n > 0:
            _log_scalar(writer, "train_g/cos_gt", g_align_sum / g_align_n, global_step)
            _log_scalar(writer, "train_g/angle_deg", g_angle_sum / g_align_n, global_step)
            _log_scalar(writer, "train_g/l2", g_l2_sum / g_align_n, global_step)
            if g_reach_n > 0:
                _log_scalar(writer, "train_g/reach_cos_gt", g_reach_sum / g_reach_n, global_step)
            if g_actuate_n > 0:
                _log_scalar(writer, "train_g/actuate_cos_gt", g_actuate_sum / g_actuate_n, global_step)
        _log_sampler_stats(writer, sampler_stats, global_step, max_ep_len)

        run_eval = (
            not args_cli.no_eval
            and (iteration + 1) % args_cli.eval_interval == 0
            and iteration > start_iteration
        )
        if run_eval:
            eval_env_ids = torch.arange(num_eval_envs, device=ep_step.device)
            _flush_capped_partial_rollouts(
                env,
                oracle,
                buffer,
                ep_visual,
                ep_proprio,
                ep_cube_ee_delta,
                ep_teacher,
                ep_teacher_phase,
                ep_step,
                sampler_stats,
                rnn_hidden,
                phase_tracker,
                env_ids=eval_env_ids,
                reset_after_flush=True,
                mosaic_history=mosaic_history,
            )
            # Clear lenient teacher phase latches before eval so GT G / B3 channels
            # are not stuck in LIFT from the preceding training rollout.
            reset_teacher_phase_state(env)
            eval_metrics = run_student_eval_binary_gripper_vision(
                policy,
                env,
                oracle,
                num_episodes=args_cli.eval_episodes,
                image_size=image_size,
                control_dt=control_dt,
                arm_scale=args_cli.arm_scale,
                teacher_arm_std=teacher_arm_std,
                gripper_bce_weight=args_cli.gripper_bce_weight,
                goal_threshold=args_cli.goal_threshold,
                device=device,
                max_envs=num_eval_envs,
                early_done=args_cli.early_done,
                phase_stall_steps=args_cli.eval_phase_stall_steps,
                phase_min_progress_m=args_cli.phase_min_progress_m,
                early_done_min_sm_state=args_cli.early_done_min_sm_state,
                num_gripper_actions=gripper_action_dim,
                guidance_field=guidance_field,
                collect_diagnostics=not args_cli.no_eval_diagnostics,
                visual_layout=visual_layout,
                mosaic_history_len=args_cli.mosaic_history_len,
            )
            rnn_hidden = policy.init_hidden(num_envs, device)
            if mosaic_history is not None:
                mosaic_history.reset(slice(None))
            ep_step.zero_()
            env.reset()
            oracle.reset()
            reset_teacher_phase_state(env)
            if phase_tracker is not None:
                phase_tracker.reset()
            eval_line = (
                f"[Eval {iteration + 1}] success={eval_metrics.success_rate:.3f} "
                f"success@5cm={eval_metrics.success_rate_5cm:.3f} "
                f"grasp={eval_metrics.grasp_rate:.3f} "
                f"pre_grasp_ee_cube={eval_metrics.mean_pre_grasp_min_ee_cube:.3f} "
                f"post_grasp_dg={eval_metrics.mean_post_grasp_dist_goal:.3f} "
                f"wander={eval_metrics.mean_post_grasp_wander_m:.3f} "
                f"len={eval_metrics.mean_episode_length:.1f} ({eval_metrics.num_episodes} eps)"
            )
            if eval_metrics.failure_mode_counts:
                modes = ",".join(f"{k}:{v}" for k, v in sorted(eval_metrics.failure_mode_counts.items()))
                eval_line += f" modes=[{modes}]"
            _log_progress_line(args_cli.log_dir, eval_line)
            _log_scalar(writer, "eval/success_rate", eval_metrics.success_rate, global_step)
            _log_scalar(writer, "eval/success_rate_5cm", eval_metrics.success_rate_5cm, global_step)
            _log_scalar(writer, "eval/grasp_rate", eval_metrics.grasp_rate, global_step)
            _log_scalar(writer, "eval/pre_grasp_min_ee_cube", eval_metrics.mean_pre_grasp_min_ee_cube, global_step)
            _log_scalar(writer, "eval/post_grasp_dist_goal", eval_metrics.mean_post_grasp_dist_goal, global_step)
            _log_scalar(writer, "eval/post_grasp_wander_m", eval_metrics.mean_post_grasp_wander_m, global_step)
            _log_scalar(writer, "eval/min_ee_cube_dist", eval_metrics.mean_min_ee_cube_dist, global_step)
            _log_scalar(writer, "eval_g/cos_gt", eval_metrics.mean_g_cos_gt, global_step)
            _log_scalar(writer, "eval_g/angle_deg", eval_metrics.mean_g_angle_deg, global_step)
            _log_scalar(writer, "eval_g/l2", eval_metrics.mean_g_l2, global_step)
            _log_scalar(writer, "eval_g/reach_cos_gt", eval_metrics.mean_g_reach_cos_gt, global_step)
            _log_scalar(writer, "eval_g/actuate_cos_gt", eval_metrics.mean_g_actuate_cos_gt, global_step)
            if eval_metrics.failure_mode_counts and not args_cli.no_eval_diagnostics:
                diag_path = os.path.join(args_cli.log_dir, f"eval_diagnostics_iter_{iteration + 1:04d}.json")
                with open(diag_path, "w") as f:
                    json.dump(
                        {
                            "iteration": iteration + 1,
                            "success_rate": eval_metrics.success_rate,
                            "grasp_rate": eval_metrics.grasp_rate,
                            "mean_pre_grasp_min_ee_cube": eval_metrics.mean_pre_grasp_min_ee_cube,
                            "mean_post_grasp_dist_goal": eval_metrics.mean_post_grasp_dist_goal,
                            "mean_post_grasp_wander_m": eval_metrics.mean_post_grasp_wander_m,
                            "mean_min_ee_cube_dist": eval_metrics.mean_min_ee_cube_dist,
                            "failure_mode_counts": eval_metrics.failure_mode_counts,
                            "success_rate_5cm": eval_metrics.success_rate_5cm,
                            "num_episodes": eval_metrics.num_episodes,
                        },
                        f,
                        indent=2,
                    )
            if args_cli.beta_adaptive:
                if args_cli.beta_eval_metric == "success_5cm":
                    beta_eval_value = eval_metrics.success_rate_5cm
                elif args_cli.beta_eval_metric == "grasp":
                    beta_eval_value = eval_metrics.grasp_rate
                else:
                    beta_eval_value = eval_metrics.success_rate
                if beta_success_ema is None:
                    beta_success_ema = beta_eval_value
                else:
                    ema = args_cli.beta_success_ema
                    beta_success_ema = ema * beta_success_ema + (1.0 - ema) * beta_eval_value
                _log_scalar(writer, f"dagger/beta_input_{args_cli.beta_eval_metric}", beta_eval_value, global_step)
                print(
                    f"[INFO] Adaptive β input ({args_cli.beta_eval_metric}): value={beta_eval_value:.3f} "
                    f"ema={beta_success_ema:.3f} -> next β≈"
                    f"{compute_dagger_beta_adaptive(iteration + 1, beta_success_ema, beta_start=args_cli.beta_start, beta_end=args_cli.beta_end, warmup_iters=args_cli.beta_warmup_iters, success_target=args_cli.beta_success_target):.3f}"
                )

            if eval_metrics.success_rate >= best_success_rate:
                best_success_rate = eval_metrics.success_rate
                best_path = os.path.join(args_cli.log_dir, "student_best.pt")
                _save_training_checkpoint(
                    best_path,
                    policy=policy,
                    optimizer=optimizer,
                    iteration=iteration + 1,
                    global_step=global_step,
                    teacher_arm_std=teacher_arm_std,
                    best_success_rate=best_success_rate,
                    proprio_dim=proprio_dim,
                    image_size=image_size,
                    visual_channels=visual_channels,
                    grasp_beta_state=grasp_beta_state if args_cli.beta_grasp_gated else None,
                )
                print(f"[INFO] Saved best: {best_path} (success={best_success_rate:.3f})")

        save_ckpt = (iteration + 1) % args_cli.save_interval == 0
        save_buffer = args_cli.buffer_save_interval > 0 and (iteration + 1) % args_cli.buffer_save_interval == 0
        if save_ckpt:
            ckpt_path = os.path.join(args_cli.log_dir, f"student_iter_{iteration + 1}.pt")
            _save_training_checkpoint(
                ckpt_path,
                policy=policy,
                optimizer=optimizer,
                iteration=iteration + 1,
                global_step=global_step,
                teacher_arm_std=teacher_arm_std,
                best_success_rate=best_success_rate,
                buffer=buffer,
                include_buffer=save_buffer,
                proprio_dim=proprio_dim,
                image_size=image_size,
                visual_channels=visual_channels,
                grasp_beta_state=grasp_beta_state if args_cli.beta_grasp_gated else None,
            )
            print(f"[INFO] Saved checkpoint: {ckpt_path}")
        if save_buffer or (save_ckpt and args_cli.buffer_save_interval == 0):
            buffer.save(buffer_path)
            print(f"[INFO] Saved aggregate buffer: {buffer_path} ({len(buffer)} episodes)")

        if env.sim.is_stopped():
            break

    buffer.add_in_progress_episodes(
        ep_visual,
        ep_teacher,
        ep_step,
        ep_teacher_phase=ep_teacher_phase,
        ep_proprio=ep_proprio,
        ep_cube_ee_delta=ep_cube_ee_delta,
    )
    buffer.save(buffer_path)
    final_path = os.path.join(args_cli.log_dir, "student_final.pt")
    _save_training_checkpoint(
        final_path,
        policy=policy,
        optimizer=optimizer,
        iteration=end_iteration,
        global_step=global_step,
        teacher_arm_std=teacher_arm_std,
        best_success_rate=best_success_rate,
        buffer=buffer,
        include_buffer=True,
        proprio_dim=proprio_dim,
        image_size=image_size,
        visual_channels=visual_channels,
        grasp_beta_state=grasp_beta_state if args_cli.beta_grasp_gated else None,
    )
    print(f"[INFO] Saved final policy: {final_path}")
    print(f"[INFO] Saved aggregate buffer: {buffer_path} ({len(buffer)} episodes, {buffer.num_steps} steps)")

    if writer is not None:
        writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
