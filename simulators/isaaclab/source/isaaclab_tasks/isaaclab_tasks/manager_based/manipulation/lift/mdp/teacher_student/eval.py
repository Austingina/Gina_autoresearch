# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation utilities for teacher-student lift-cube training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.envs import ManagerBasedRLEnv

from ..data_collection import (
    EpisodeLiftSuccessTracker,
    compute_cube_to_goal_distance,
    compute_grasp_success,
    get_ee_pose_in_robot_root_frame,
    get_lift_goal_position_in_robot_root_frame,
    get_mean_finger_opening,
)
from ..observations import object_position_in_robot_root_frame
from ..oracle.lift_cube_oracle import LiftCubeOraclePolicy
from .early_done import OraclePhaseStallTracker
from .kl_trainer import TwistEpisodeReplayBuffer, gaussian_kl_arm_gripper_unweighted, gaussian_kl_to_teacher, gripper_bce_to_teacher
from .lift_episode_diagnostics import EpisodeDiagnosticRecorder, aggregate_diagnostics
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.networks import reset_rnn_state
from .eval_utils import early_reset_env_ids as _early_reset_env_ids
from .teacher import compute_teacher_twist, compute_teacher_twist_binary_gripper, infer_teacher_phase, twist_to_env_action
from .teacher_phase import reset_teacher_phase_state
from .vision_policy_obs import (
    VISUAL_LAYOUT_CHANNELS,
    VISUAL_LAYOUT_MOSAIC,
    VisualMosaicHistory,
    extract_proprio_from_policy_obs_torch,
    stack_policy_visual_torch,
    stack_visual_from_policy_obs_torch,
)


@dataclass
class EvalMetrics:
    """Aggregated evaluation statistics."""

    success_rate: float
    mean_episode_length: float
    mean_kl: float
    mean_arm_kl: float
    mean_gripper_kl: float
    num_episodes: int


def run_student_eval(
    policy: StudentTwistPolicy,
    env: ManagerBasedRLEnv,
    oracle: LiftCubeOraclePolicy,
    *,
    num_episodes: int,
    obs_dim: int,
    control_dt: float,
    arm_scale: float,
    teacher_std: torch.Tensor,
    kl_dim_weights: torch.Tensor | None = None,
    goal_threshold: float = 0.02,
    device: str | torch.device = "cpu",
    max_envs: int | None = None,
    history_len: int = 1,
    early_done: bool = True,
    phase_stall_steps: int = 20,
    phase_min_progress_m: float = 0.002,
    early_done_min_sm_state: int = 2,
    num_gripper_actions: int = 2,
) -> EvalMetrics:
    """Roll out the student deterministically and measure task success."""
    del history_len, obs_dim  # rollout uses single-step obs + carried RNN hidden state.

    policy.eval()
    num_envs = env.num_envs
    eval_envs = num_envs if max_envs is None else min(max(max_envs, 1), num_envs)
    rnn_hidden = policy.init_hidden(num_envs, device)
    ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    success_tracker = EpisodeLiftSuccessTracker(num_envs, device)
    phase_tracker = None
    if early_done:
        phase_tracker = OraclePhaseStallTracker(
            num_envs,
            device,
            max_stall_steps=phase_stall_steps,
            min_progress_m=phase_min_progress_m,
            min_sm_state=early_done_min_sm_state,
        )

    env.reset()
    oracle.reset()

    episodes_done = 0
    success_total = 0
    length_sum = 0.0
    kl_sum = 0.0
    arm_kl_sum = 0.0
    gripper_kl_sum = 0.0
    kl_count = 0

    max_total_steps = max(num_episodes * env.max_episode_length, env.max_episode_length)
    global_steps = 0

    with torch.no_grad():
        while episodes_done < num_episodes and global_steps < max_total_steps:
            student_obs = env.observation_manager.compute_group("policy")
            obs_step = student_obs.unsqueeze(1)

            dist, rnn_hidden = policy.policy_dist(obs_step, rnn_hidden)
            teacher_twist = compute_teacher_twist(env, oracle, control_dt=control_dt)
            if phase_tracker is not None:
                phase_tracker.on_pre_step(oracle, env)
            kl_sum += gaussian_kl_to_teacher(
                dist, teacher_twist, teacher_std, dim_weights=kl_dim_weights
            ).mean().item()
            arm_kl, gripper_kl = gaussian_kl_arm_gripper_unweighted(dist, teacher_twist, teacher_std)
            arm_kl_sum += arm_kl.mean().item()
            gripper_kl_sum += gripper_kl.mean().item()
            kl_count += 1

            actions = dist.mean
            current_opening = get_mean_finger_opening(env)
            pre_cube_to_goal = compute_cube_to_goal_distance(env)
            pre_cube_z_w = env.scene["object"].data.root_pos_w[:, 2].clone()
            env_actions = twist_to_env_action(
                actions,
                control_dt=control_dt,
                current_opening=current_opening,
                arm_scale=arm_scale,
                num_gripper_actions=num_gripper_actions,
            )
            _, _, terminated, truncated, _ = env.step(env_actions)
            ep_step += 1
            global_steps += 1

            success_early = success_tracker.on_step(
                env,
                terminated=terminated,
                truncated=truncated,
                pre_cube_to_goal=pre_cube_to_goal,
                pre_cube_z_w=pre_cube_z_w,
                lift_threshold=goal_threshold,
            )
            early_mask = None
            if phase_tracker is not None:
                early_mask = phase_tracker.on_post_step(oracle, env)
            dones = terminated | truncated | success_early
            if early_mask is not None:
                dones = dones | early_mask
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.ndim == 0:
                    done_ids = done_ids.unsqueeze(0)
                manual_reset_ids = _early_reset_env_ids(early_mask, success_early, max_env_id=eval_envs)
                for env_id in done_ids.tolist():
                    if env_id >= eval_envs:
                        continue
                    if episodes_done >= num_episodes:
                        break
                    episodes_done += 1
                    length_sum += float(ep_step[env_id].item())
                    if success_tracker.ever_lift_success[env_id].item():
                        success_total += 1

                oracle.reset(done_ids.tolist())
                reset_teacher_phase_state(env, done_ids.tolist())
                reset_rnn_state(rnn_hidden, done_ids)
                ep_step[done_ids] = 0
                success_tracker.reset(done_ids)
                if phase_tracker is not None:
                    phase_tracker.reset(done_ids)
                if manual_reset_ids.numel() > 0:
                    env.reset(env_ids=manual_reset_ids)

                if episodes_done >= num_episodes:
                    break

    if episodes_done == 0:
        return EvalMetrics(
            success_rate=0.0,
            mean_episode_length=0.0,
            mean_kl=0.0,
            mean_arm_kl=0.0,
            mean_gripper_kl=0.0,
            num_episodes=0,
        )

    denom = max(kl_count, 1)
    return EvalMetrics(
        success_rate=success_total / episodes_done,
        mean_episode_length=length_sum / episodes_done,
        mean_kl=kl_sum / denom,
        mean_arm_kl=arm_kl_sum / denom,
        mean_gripper_kl=gripper_kl_sum / denom,
        num_episodes=episodes_done,
    )


@dataclass
class BinaryGripperEvalMetrics:
    """Aggregated evaluation statistics for binary-gripper student policies."""

    success_rate: float
    mean_episode_length: float
    mean_total_loss: float
    mean_arm_kl: float
    mean_gripper_bce: float
    num_episodes: int
    grasp_rate: float = 0.0
    mean_post_grasp_wander_m: float = 0.0
    mean_pre_grasp_min_ee_cube: float = 0.0
    mean_post_grasp_dist_goal: float = 0.0
    mean_min_ee_cube_dist: float = 0.0
    failure_mode_counts: dict[str, int] | None = None
    success_rate_5cm: float = 0.0
    mean_g_cos_gt: float = 0.0
    mean_g_angle_deg: float = 0.0
    mean_g_l2: float = 0.0
    mean_g_reach_cos_gt: float = 0.0
    mean_g_actuate_cos_gt: float = 0.0


def _ee_cube_distance(env: ManagerBasedRLEnv, env_id: int) -> float:
    obj = env.scene["object"].data.root_pos_w[env_id, :3]
    ee = env.scene["ee_frame"].data.target_pos_w[env_id, 0, :3]
    return float(torch.norm(obj - ee).item())


def run_student_eval_binary_gripper(
    policy: StudentTwistPolicyBinaryGripper,
    env: ManagerBasedRLEnv,
    oracle: LiftCubeOraclePolicy,
    *,
    num_episodes: int,
    obs_dim: int,
    control_dt: float,
    arm_scale: float,
    teacher_arm_std: torch.Tensor,
    gripper_bce_weight: float = 1.0,
    goal_threshold: float = 0.02,
    device: str | torch.device = "cpu",
    max_envs: int | None = None,
    history_len: int = 1,
    early_done: bool = True,
    phase_stall_steps: int = 20,
    phase_min_progress_m: float = 0.002,
    early_done_min_sm_state: int = 2,
    num_gripper_actions: int = 2,
) -> BinaryGripperEvalMetrics:
    """Roll out the binary-gripper student deterministically and measure task success."""
    del history_len, obs_dim

    policy.eval()
    num_envs = env.num_envs
    eval_envs = num_envs if max_envs is None else min(max(max_envs, 1), num_envs)
    rnn_hidden = policy.init_hidden(num_envs, device)
    ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    success_tracker = EpisodeLiftSuccessTracker(num_envs, device)
    phase_tracker = None
    if early_done:
        phase_tracker = OraclePhaseStallTracker(
            num_envs,
            device,
            max_stall_steps=phase_stall_steps,
            min_progress_m=phase_min_progress_m,
            min_sm_state=early_done_min_sm_state,
        )

    env.reset()
    oracle.reset()

    episodes_done = 0
    success_total = 0
    length_sum = 0.0
    total_loss_sum = 0.0
    arm_kl_sum = 0.0
    gripper_bce_sum = 0.0
    loss_count = 0
    g_align_sum = 0.0
    g_angle_sum = 0.0
    g_l2_sum = 0.0
    g_align_n = 0
    g_reach_sum = 0.0
    g_reach_n = 0
    g_actuate_sum = 0.0
    g_actuate_n = 0

    max_total_steps = max(num_episodes * env.max_episode_length, env.max_episode_length)
    global_steps = 0

    with torch.no_grad():
        while episodes_done < num_episodes and global_steps < max_total_steps:
            student_obs = env.observation_manager.compute_group("policy")
            obs_step = student_obs.unsqueeze(1)

            arm_dist, gripper_logit, rnn_hidden = policy.policy_dist(obs_step, rnn_hidden)
            teacher_twist = compute_teacher_twist_binary_gripper(env, oracle, control_dt=control_dt)
            if phase_tracker is not None:
                phase_tracker.on_pre_step(oracle, env)

            arm_kl = gaussian_kl_to_teacher(arm_dist, teacher_twist[:, :6], teacher_arm_std).mean().item()
            gripper_bce = gripper_bce_to_teacher(gripper_logit, teacher_twist[:, 6]).item()
            total_loss_sum += arm_kl + gripper_bce_weight * gripper_bce
            arm_kl_sum += arm_kl
            gripper_bce_sum += gripper_bce
            loss_count += 1

            actions, rnn_hidden, _ = policy.act(obs_step, rnn_hidden, deterministic=True)
            current_opening = get_mean_finger_opening(env)
            pre_cube_to_goal = compute_cube_to_goal_distance(env)
            pre_cube_z_w = env.scene["object"].data.root_pos_w[:, 2].clone()
            env_actions = twist_to_env_action(
                actions,
                control_dt=control_dt,
                current_opening=current_opening,
                arm_scale=arm_scale,
                num_gripper_actions=num_gripper_actions,
                binary_gripper=True,
            )
            _, _, terminated, truncated, _ = env.step(env_actions)
            ep_step += 1
            global_steps += 1

            success_early = success_tracker.on_step(
                env,
                terminated=terminated,
                truncated=truncated,
                pre_cube_to_goal=pre_cube_to_goal,
                pre_cube_z_w=pre_cube_z_w,
                lift_threshold=goal_threshold,
            )
            early_mask = None
            if phase_tracker is not None:
                early_mask = phase_tracker.on_post_step(oracle, env)
            dones = terminated | truncated | success_early
            if early_mask is not None:
                dones = dones | early_mask
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.ndim == 0:
                    done_ids = done_ids.unsqueeze(0)
                manual_reset_ids = _early_reset_env_ids(early_mask, success_early, max_env_id=eval_envs)
                for env_id in done_ids.tolist():
                    if env_id >= eval_envs:
                        continue
                    if episodes_done >= num_episodes:
                        break
                    episodes_done += 1
                    length_sum += float(ep_step[env_id].item())
                    if success_tracker.ever_lift_success[env_id].item():
                        success_total += 1

                oracle.reset(done_ids.tolist())
                reset_teacher_phase_state(env, done_ids.tolist())
                reset_rnn_state(rnn_hidden, done_ids)
                ep_step[done_ids] = 0
                success_tracker.reset(done_ids)
                if phase_tracker is not None:
                    phase_tracker.reset(done_ids)
                if manual_reset_ids.numel() > 0:
                    env.reset(env_ids=manual_reset_ids)

                if episodes_done >= num_episodes:
                    break

    if episodes_done == 0:
        return BinaryGripperEvalMetrics(
            success_rate=0.0,
            mean_episode_length=0.0,
            mean_total_loss=0.0,
            mean_arm_kl=0.0,
            mean_gripper_bce=0.0,
            num_episodes=0,
        )

    denom = max(loss_count, 1)
    return BinaryGripperEvalMetrics(
        success_rate=success_total / episodes_done,
        mean_episode_length=length_sum / episodes_done,
        mean_total_loss=total_loss_sum / denom,
        mean_arm_kl=arm_kl_sum / denom,
        mean_gripper_bce=gripper_bce_sum / denom,
        num_episodes=episodes_done,
    )


def run_student_eval_binary_gripper_vision(
    policy,
    env: ManagerBasedRLEnv,
    oracle: LiftCubeOraclePolicy,
    *,
    num_episodes: int,
    image_size: int,
    control_dt: float,
    arm_scale: float,
    teacher_arm_std: torch.Tensor,
    gripper_bce_weight: float = 1.0,
    goal_threshold: float = 0.02,
    device: str | torch.device = "cpu",
    max_envs: int | None = None,
    early_done: bool = True,
    phase_stall_steps: int = 20,
    phase_min_progress_m: float = 0.002,
    early_done_min_sm_state: int = 2,
    num_gripper_actions: int = 2,
    guidance_field=None,
    collect_diagnostics: bool = False,
    eval_max_episode_steps: int | None = None,
    visual_layout: str = VISUAL_LAYOUT_CHANNELS,
    mosaic_history_len: int = 1,
) -> BinaryGripperEvalMetrics:
    """Roll out vision binary-gripper student deterministically."""
    policy.eval()
    num_envs = env.num_envs
    eval_envs = num_envs if max_envs is None else min(max(max_envs, 1), num_envs)
    rnn_hidden = policy.init_hidden(num_envs, device)
    ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    success_tracker = EpisodeLiftSuccessTracker(num_envs, device)
    recorders = [EpisodeDiagnosticRecorder() for _ in range(num_envs)] if collect_diagnostics else None
    episode_diagnostics = []
    phase_tracker = None
    if early_done:
        phase_tracker = OraclePhaseStallTracker(
            num_envs,
            device,
            max_stall_steps=phase_stall_steps,
            min_progress_m=phase_min_progress_m,
            min_sm_state=early_done_min_sm_state,
        )

    env.reset()
    oracle.reset()
    reset_teacher_phase_state(env)

    mosaic_history = None
    if visual_layout == VISUAL_LAYOUT_MOSAIC:
        mosaic_history = VisualMosaicHistory(
            num_envs,
            image_size,
            device,
            history_len=mosaic_history_len,
        )

    episodes_done = 0
    success_total = 0
    length_sum = 0.0
    total_loss_sum = 0.0
    arm_kl_sum = 0.0
    gripper_bce_sum = 0.0
    loss_count = 0
    g_align_sum = 0.0
    g_angle_sum = 0.0
    g_l2_sum = 0.0
    g_align_n = 0
    g_reach_sum = 0.0
    g_reach_n = 0
    g_actuate_sum = 0.0
    g_actuate_n = 0

    max_total_steps = max(num_episodes * env.max_episode_length, env.max_episode_length)
    global_steps = 0

    with torch.no_grad():
        while episodes_done < num_episodes and global_steps < max_total_steps:
            student_obs = env.observation_manager.compute_group("policy")
            visual = stack_policy_visual_torch(
                student_obs,
                image_size=image_size,
                visual_layout=visual_layout,
                mosaic_history=mosaic_history,
            )
            if guidance_field is not None:
                visual = guidance_field.augment_visual(
                    visual, student_obs, image_size=image_size, env=env
                )
            visual_step = visual.unsqueeze(1)
            proprio = extract_proprio_from_policy_obs_torch(student_obs)
            proprio_step = None if proprio is None else proprio.unsqueeze(1)

            arm_dist, gripper_logit, rnn_hidden = policy.policy_dist(visual_step, rnn_hidden, proprio_step)
            teacher_twist = compute_teacher_twist_binary_gripper(env, oracle, control_dt=control_dt)
            teacher_phase = infer_teacher_phase(env)
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
                if g_metrics["reach_cos"] == g_metrics["reach_cos"]:
                    rn = int(g_metrics["reach_n"])
                    g_reach_sum += g_metrics["reach_cos"] * rn
                    g_reach_n += rn
                if g_metrics["actuate_cos"] == g_metrics["actuate_cos"]:
                    an = int(g_metrics["actuate_n"])
                    g_actuate_sum += g_metrics["actuate_cos"] * an
                    g_actuate_n += an
            if phase_tracker is not None:
                phase_tracker.on_pre_step(oracle, env)

            arm_kl = gaussian_kl_to_teacher(arm_dist, teacher_twist[:, :6], teacher_arm_std).mean().item()
            gripper_bce = gripper_bce_to_teacher(gripper_logit, teacher_twist[:, 6]).item()
            total_loss_sum += arm_kl + gripper_bce_weight * gripper_bce
            arm_kl_sum += arm_kl
            gripper_bce_sum += gripper_bce
            loss_count += 1

            actions, rnn_hidden, _ = policy.act(visual_step, rnn_hidden, proprio_step, deterministic=True)
            current_opening = get_mean_finger_opening(env)
            pre_cube_to_goal = compute_cube_to_goal_distance(env)
            pre_cube_z_w = env.scene["object"].data.root_pos_w[:, 2].clone()
            env_actions = twist_to_env_action(
                actions,
                control_dt=control_dt,
                current_opening=current_opening,
                arm_scale=arm_scale,
                num_gripper_actions=num_gripper_actions,
                binary_gripper=True,
            )
            _, _, terminated, truncated, _ = env.step(env_actions)
            ep_step += 1
            global_steps += 1

            success_early = success_tracker.on_step(
                env,
                terminated=terminated,
                truncated=truncated,
                pre_cube_to_goal=pre_cube_to_goal,
                pre_cube_z_w=pre_cube_z_w,
                lift_threshold=goal_threshold,
            )
            if collect_diagnostics and recorders is not None:
                ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
                cube_pos_b = object_position_in_robot_root_frame(env)
                goal_pos_b = get_lift_goal_position_in_robot_root_frame(env)
                grasp_now = compute_grasp_success(env)
                lift_now = success_tracker.ever_lift_success.clone()
                for env_id in range(eval_envs):
                    recorders[env_id].record_step(
                        step=int(ep_step[env_id].item()),
                        ee_pos_b=ee_pos_b[env_id].detach().cpu().numpy(),
                        cube_pos_b=cube_pos_b[env_id].detach().cpu().numpy(),
                        goal_pos_b=goal_pos_b[env_id].detach().cpu().numpy(),
                        dist_to_goal=float(pre_cube_to_goal[env_id].item()),
                        ee_cube_dist=_ee_cube_distance(env, env_id),
                        gripper_open=float(current_opening[env_id].item()),
                        gripper_cmd=float(actions[env_id, 6].item()),
                        teacher_phase=int(teacher_phase[env_id].item()),
                        grasp_now=bool(grasp_now[env_id].item()),
                        lift_now=bool(lift_now[env_id].item()),
                    )
            early_mask = None
            if phase_tracker is not None:
                early_mask = phase_tracker.on_post_step(oracle, env)
            dones = terminated | truncated | success_early
            if early_mask is not None:
                dones = dones | early_mask
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.ndim == 0:
                    done_ids = done_ids.unsqueeze(0)
                manual_reset_ids = _early_reset_env_ids(early_mask, success_early, max_env_id=eval_envs)
                for env_id in done_ids.tolist():
                    if env_id >= eval_envs:
                        continue
                    if episodes_done >= num_episodes:
                        break
                    if collect_diagnostics and recorders is not None:
                        if success_tracker.ever_lift_success[env_id].item():
                            end_reason = "success"
                        elif truncated[env_id].item():
                            end_reason = "timeout"
                        elif terminated[env_id].item():
                            end_reason = "terminated"
                        else:
                            end_reason = "done"
                        episode_diagnostics.append(
                            recorders[env_id].finalize(
                                episode_id=episodes_done,
                                end_reason=end_reason,
                                ever_grasp_flag=bool(success_tracker.ever_grasp_success[env_id].item()),
                                ever_lift_flag=bool(success_tracker.ever_lift_success[env_id].item()),
                                save_trajectory=False,
                            )
                        )
                        recorders[env_id].reset()
                    episodes_done += 1
                    length_sum += float(ep_step[env_id].item())
                    if success_tracker.ever_lift_success[env_id].item():
                        success_total += 1

                done_list = done_ids.tolist()
                oracle.reset(done_list)
                reset_teacher_phase_state(env, done_list)
                reset_rnn_state(rnn_hidden, done_ids)
                if mosaic_history is not None:
                    mosaic_history.reset(done_ids)
                ep_step[done_ids] = 0
                success_tracker.reset(done_ids)
                if phase_tracker is not None:
                    phase_tracker.reset(done_ids)
                if manual_reset_ids.numel() > 0:
                    env.reset(env_ids=manual_reset_ids)

                if episodes_done >= num_episodes:
                    break

    if episodes_done == 0:
        return BinaryGripperEvalMetrics(
            success_rate=0.0,
            mean_episode_length=0.0,
            mean_total_loss=0.0,
            mean_arm_kl=0.0,
            mean_gripper_bce=0.0,
            num_episodes=0,
        )

    denom = max(loss_count, 1)
    diag_summary = aggregate_diagnostics(episode_diagnostics) if episode_diagnostics else None
    return BinaryGripperEvalMetrics(
        success_rate=success_total / episodes_done,
        mean_episode_length=length_sum / episodes_done,
        mean_total_loss=total_loss_sum / denom,
        mean_arm_kl=arm_kl_sum / denom,
        mean_gripper_bce=gripper_bce_sum / denom,
        num_episodes=episodes_done,
        grasp_rate=float(diag_summary["grasp_rate"]) if diag_summary else 0.0,
        mean_post_grasp_wander_m=float(diag_summary["mean_post_grasp_wander_m"]) if diag_summary else 0.0,
        mean_pre_grasp_min_ee_cube=float(diag_summary["mean_pre_grasp_min_ee_cube"]) if diag_summary else 0.0,
        mean_post_grasp_dist_goal=float(diag_summary["mean_post_grasp_dist_goal"]) if diag_summary else 0.0,
        mean_min_ee_cube_dist=float(diag_summary["mean_min_ee_cube_dist"]) if diag_summary else 0.0,
        failure_mode_counts=diag_summary.get("failure_mode_counts") if diag_summary else None,
        success_rate_5cm=float(diag_summary["success_rate_5cm"]) if diag_summary else 0.0,
        mean_g_cos_gt=g_align_sum / max(g_align_n, 1),
        mean_g_angle_deg=g_angle_sum / max(g_align_n, 1),
        mean_g_l2=g_l2_sum / max(g_align_n, 1),
        mean_g_reach_cos_gt=g_reach_sum / max(g_reach_n, 1),
        mean_g_actuate_cos_gt=g_actuate_sum / max(g_actuate_n, 1),
    )
