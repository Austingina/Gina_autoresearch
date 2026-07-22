# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Train a proprio student policy with oracle KL supervision (active_tracking_rl style).

Teacher: oracle pick-and-lift SM mapped to 7D EE command (6D velocity + gripper opening rate).
Student: LSTM policy on 10D state, outputs 7D EE command.

Training loop (aligned with active_tracking train_mode=6):
1. Sampler: student rollout, store completed episodes (obs sequence + teacher twist).
2. Optimizer: sample trajectory segments, BPTT KL(student || teacher) with carried RNN state.
3. Eval: deterministic rollout, log success metrics to TensorBoard.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train lift-cube student policy with oracle KL supervision.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Teacher-Student-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--collect_steps", type=int, default=32, help="Environment steps collected per iteration.")
parser.add_argument("--buffer_size", type=int, default=200, help="Maximum number of episodes kept in replay buffer.")
parser.add_argument("--init_steps", type=int, default=2000, help="Sampler env-steps before KL optimization starts.")
parser.add_argument("--batch_size", type=int, default=8, help="Number of trajectory segments per KL update.")
parser.add_argument("--seq_len", type=int, default=64, help="Length of each sampled trajectory segment for BPTT.")
parser.add_argument("--optim_steps", type=int, default=4, help="KL optimization passes per iteration.")
parser.add_argument("--grad_interval", type=int, default=20, help="BPTT steps between gradient updates.")
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--teacher_lin_std", type=float, default=0.05)
parser.add_argument("--teacher_ang_std", type=float, default=0.10)
parser.add_argument(
    "--teacher_gripper_std",
    type=float,
    default=0.02,
    help="Teacher std for gripper opening rate (m/s).",
)
parser.add_argument(
    "--teacher_min_lin_speed",
    type=float,
    default=0.10,
    help="Minimum teacher linear speed norm (m/s) for the 6D arm twist.",
)
parser.add_argument(
    "--teacher_min_ang_speed",
    type=float,
    default=0.20,
    help="Minimum teacher angular speed norm (rad/s) for the 6D arm twist.",
)
parser.add_argument("--kl_arm_weight", type=float, default=1.0, help="KL loss weight for the 6D arm twist dims.")
parser.add_argument(
    "--kl_gripper_weight",
    type=float,
    default=1.0,
    help="KL loss weight for the gripper opening-rate dim.",
)
parser.add_argument("--arm_scale", type=float, default=0.5, help="IK-Rel arm action scale in the env cfg.")
parser.add_argument(
    "--log_dir",
    type=str,
    default="./runs/lift_cube_teacher_student_kl",
    help="Directory for checkpoints and tensorboard logs.",
)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--position_threshold", type=float, default=0.01)
parser.add_argument("--eval_interval", type=int, default=50, help="Run eval every N training iterations.")
parser.add_argument("--eval_episodes", type=int, default=50, help="Number of eval episodes per eval run.")
parser.add_argument(
    "--num_eval_envs",
    type=int,
    default=16,
    help="Only the first N parallel env slots contribute to eval episode statistics.",
)
parser.add_argument("--no_eval", action="store_true", help="Disable periodic evaluation.")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Override env episode timeout (seconds). Default uses task cfg (15s for teacher-student).",
)
parser.add_argument(
    "--early_done",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Terminate episodes early when oracle phase progress stalls.",
)
parser.add_argument(
    "--phase_stall_steps",
    type=int,
    default=20,
    help="Consecutive steps without phase progress before early termination.",
)
parser.add_argument(
    "--phase_min_progress_m",
    type=float,
    default=0.002,
    help="Minimum EE distance improvement (m) per step to count as phase progress.",
)
parser.add_argument(
    "--early_done_min_sm_state",
    type=int,
    default=2,
    help="Disable early done once oracle SM reaches this state (2=APPROACH).",
)
parser.add_argument(
    "--oracle_demo_file",
    type=str,
    default="",
    help="Optional HDF5 oracle demos to preload into the replay buffer.",
)
parser.add_argument(
    "--oracle_demo_max_episodes",
    type=int,
    default=100,
    help="Maximum oracle demo episodes to preload when --oracle_demo_file is set.",
)
parser.add_argument(
    "--resume_checkpoint",
    type=str,
    default="",
    help="Optional checkpoint (.pt) to resume policy weights and training counters from.",
)
parser.add_argument(
    "--grasp_sample_ratio",
    type=float,
    default=0.7,
    help="Probability of sampling BPTT segments that contain teacher_phase >= min_grasp_phase.",
)
parser.add_argument(
    "--grasp_sample_min_phase",
    type=int,
    default=3,
    help="Minimum teacher phase (SM_GRASP=3) required in a sampled segment for grasp-priority pool.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift.mdp.data_collection import (
    compute_cube_to_goal_distance,
    compute_grasp_success_from_snapshot,
    compute_lift_success_from_snapshot,
    get_mean_finger_opening,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.oracle import LiftCubeOraclePolicy
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student import (
    EvalMetrics,
    OraclePhaseStallTracker,
    SamplerEpisodeStats,
    SamplerTeacherLabelStats,
    StudentTwistPolicy,
    TwistEpisodeReplayBuffer,
    compute_teacher_twist,
    get_gripper_action_dim,
    infer_teacher_phase,
    load_oracle_demos_into_buffer,
    make_kl_dim_weights,
    make_teacher_std,
    optimize_kl_sequences,
    reset_teacher_phase_state,
    reset_rnn_state,
    run_student_eval,
    twist_to_env_action,
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


def _log_training_metrics(
    writer,
    *,
    global_step: int,
    arm_kl: float,
    gripper_kl: float,
    grasp_sample_hit_rate: float,
    teacher_stats: SamplerTeacherLabelStats,
    sampler_stats: SamplerEpisodeStats,
) -> None:
    if writer is None:
        return
    writer.add_scalar("train/arm_kl", arm_kl, global_step)
    writer.add_scalar("train/gripper_kl", gripper_kl, global_step)
    writer.add_scalar("train/grasp_sample_hit_rate", grasp_sample_hit_rate, global_step)
    writer.add_scalar("teacher/close_rate", teacher_stats.close_rate(), global_step)
    writer.add_scalar("teacher/grasp_phase_rate", teacher_stats.grasp_phase_rate(), global_step)
    if sampler_stats.num_episodes > 0:
        n = float(sampler_stats.num_episodes)
        writer.add_scalar("rollout/grasp_success_rate", sampler_stats.grasp_success_count / n, global_step)
        writer.add_scalar("rollout/sm_reached_grasp_rate", sampler_stats.sm_rate_at_least(3), global_step)
        writer.add_scalar("rollout/sm_reached_approach_rate", sampler_stats.sm_rate_at_least(2), global_step)
    writer.flush()


def _log_eval_metrics(writer, metrics: EvalMetrics, global_step: int) -> None:
    if writer is None:
        return
    writer.add_scalar("eval/success_rate", metrics.success_rate, global_step)
    writer.add_scalar("eval/arm_kl", metrics.mean_arm_kl, global_step)
    writer.add_scalar("eval/gripper_kl", metrics.mean_gripper_kl, global_step)
    writer.flush()


def _record_and_reset_episodes(
    done_ids: torch.Tensor,
    *,
    env,
    oracle,
    buffer: TwistEpisodeReplayBuffer,
    ep_obs: torch.Tensor,
    ep_teacher: torch.Tensor,
    ep_teacher_phase: torch.Tensor,
    ep_step: torch.Tensor,
    sampler_stats: SamplerEpisodeStats,
    rnn_hidden,
    phase_tracker: OraclePhaseStallTracker | None,
    truncated: torch.Tensor,
    terminated: torch.Tensor,
    early_done: torch.Tensor | None,
    goal_threshold: float,
    pre_teacher_phase: torch.Tensor,
    pre_cube_to_goal: torch.Tensor,
    pre_cube_z_w: torch.Tensor,
    manual_reset_ids: torch.Tensor | None = None,
) -> None:
    """Flush finished episodes to the replay buffer and reset rollout state."""
    if done_ids.numel() == 0:
        return

    grasp_ok = compute_grasp_success_from_snapshot(pre_cube_z_w)
    lift_ok = compute_lift_success_from_snapshot(
        pre_cube_to_goal, pre_cube_z_w, threshold=goal_threshold
    )
    early_mask = early_done[done_ids] if early_done is not None else None

    for idx, env_id in enumerate(done_ids.tolist()):
        length = int(ep_step[env_id].item())
        if length > 0:
            buffer.add_episode(
                ep_obs[env_id, :length],
                ep_teacher[env_id, :length],
                teacher_phase=ep_teacher_phase[env_id, :length],
            )
        is_early = bool(early_mask[idx].item()) if early_mask is not None else False
        sampler_stats.record(
            length=length,
            timed_out=bool(truncated[env_id].item()),
            terminated=bool(terminated[env_id].item()),
            early_done=is_early,
            sm_state=int(pre_teacher_phase[env_id].item()),
            grasp_success=bool(grasp_ok[env_id].item()),
            lift_success=bool(lift_ok[env_id].item()),
        )
        ep_step[env_id] = 0

    done_list = done_ids.tolist()
    oracle.reset(done_list)
    reset_teacher_phase_state(env, done_list)
    reset_rnn_state(rnn_hidden, done_ids)
    if phase_tracker is not None:
        phase_tracker.reset(done_ids)

    if manual_reset_ids is not None and manual_reset_ids.numel() > 0:
        env.reset(env_ids=manual_reset_ids)


def _load_resume_checkpoint(
    checkpoint_path: str,
    policy: StudentTwistPolicy,
    *,
    obs_dim: int,
    device: str | torch.device,
) -> dict:
    """Load policy weights and resume metadata from a training checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' state dict: {checkpoint_path}")

    ckpt_obs_dim = int(ckpt.get("obs_dim", obs_dim))
    if ckpt_obs_dim != obs_dim:
        raise ValueError(f"Checkpoint obs_dim={ckpt_obs_dim} does not match env obs_dim={obs_dim}")

    policy_type = ckpt.get("policy_type", "continuous")
    if policy_type != "continuous":
        raise ValueError(
            f"Checkpoint policy_type={policy_type!r} is not compatible with continuous KL training "
            f"(StudentTwistPolicy). Use the matching training script."
        )

    policy.load_state_dict(ckpt["model"])
    return ckpt


def main():
    torch.manual_seed(args_cli.seed)
    device = args_cli.device

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    num_envs = env.num_envs

    control_dt = env_cfg.sim.dt * env_cfg.decimation
    gripper_action_dim = get_gripper_action_dim(env)
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    if obs_dim != 10:
        print(f"[WARN] Expected student obs dim 10, got {obs_dim}.")

    policy = StudentTwistPolicy(obs_dim=obs_dim, action_dim=7).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.learning_rate, amsgrad=True)
    oracle = LiftCubeOraclePolicy(env, position_threshold=args_cli.position_threshold)
    teacher_std = make_teacher_std(
        7,
        lin_std=args_cli.teacher_lin_std,
        ang_std=args_cli.teacher_ang_std,
        gripper_std=args_cli.teacher_gripper_std,
        device=device,
    )
    kl_dim_weights = make_kl_dim_weights(
        7,
        arm_weight=args_cli.kl_arm_weight,
        gripper_weight=args_cli.kl_gripper_weight,
        device=device,
    )

    start_iteration = 0
    global_step = 0
    total_sampler_steps = 0
    best_success_rate = -1.0
    resume_ckpt: dict | None = None
    if args_cli.resume_checkpoint:
        resume_ckpt = _load_resume_checkpoint(
            args_cli.resume_checkpoint,
            policy,
            obs_dim=obs_dim,
            device=device,
        )
        start_iteration = int(resume_ckpt.get("iteration", 0))
        global_step = int(resume_ckpt.get("global_step", 0))
        total_sampler_steps = max(global_step * num_envs, args_cli.init_steps)
        best_success_rate = float(resume_ckpt.get("success_rate", -1.0))
        if "teacher_std" in resume_ckpt:
            teacher_std = resume_ckpt["teacher_std"].to(device)
        if "kl_dim_weights" in resume_ckpt:
            kl_dim_weights = resume_ckpt["kl_dim_weights"].to(device)
        print(
            f"[INFO] Resumed from {args_cli.resume_checkpoint}: "
            f"iteration={start_iteration}, global_step={global_step}, "
            f"best_success={best_success_rate:.3f}"
        )

    end_iteration = start_iteration + args_cli.max_iterations

    buffer = TwistEpisodeReplayBuffer(max_episodes=args_cli.buffer_size, device=device)
    if args_cli.oracle_demo_file:
        demo_count = load_oracle_demos_into_buffer(
            args_cli.oracle_demo_file,
            buffer,
            device=device,
            max_demos=args_cli.oracle_demo_max_episodes,
            control_dt=control_dt,
        )
        print(f"[INFO] Preloaded {demo_count} oracle demo episodes from {args_cli.oracle_demo_file}")

    os.makedirs(args_cli.log_dir, exist_ok=True)
    tb_log_dir = os.path.join(args_cli.log_dir, "tb")
    writer = SummaryWriter(log_dir=tb_log_dir) if SummaryWriter is not None else None
    if writer is None:
        print("[WARN] TensorBoard SummaryWriter unavailable; metrics will only print to stdout.")

    num_envs = env.num_envs
    max_ep_len = env.max_episode_length
    rnn_hidden = policy.init_hidden(num_envs, device)
    ep_obs = torch.zeros(num_envs, max_ep_len, obs_dim, device=device)
    ep_teacher = torch.zeros(num_envs, max_ep_len, 7, device=device)
    ep_teacher_phase = torch.zeros(num_envs, max_ep_len, dtype=torch.long, device=device)
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

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] episode_length_s={env_cfg.episode_length_s} max_episode_steps={max_ep_len}")
    if phase_tracker is not None:
        print(
            "[INFO] Early done:",
            f"stall_steps={args_cli.phase_stall_steps},",
            f"min_progress_m={args_cli.phase_min_progress_m},",
            f"disabled_from_sm>={args_cli.early_done_min_sm_state}",
        )
    print(
        "[INFO] KL weights:",
        f"arm={args_cli.kl_arm_weight}, gripper={args_cli.kl_gripper_weight},",
        f"teacher_gripper_std={args_cli.teacher_gripper_std}",
    )
    print(f"[INFO] Student obs dim: {obs_dim}, action dim: 7")
    print("[INFO] Training mode: episode buffer + BPTT KL(student || teacher) + lenient teacher phase")
    if resume_ckpt is not None:
        end_iteration = start_iteration + args_cli.max_iterations
        print(
            "[INFO] Resume:",
            f"start_iter={start_iteration}, run_iters={args_cli.max_iterations}, end_iter={end_iteration}",
        )
    print(
        "[INFO] Buffer:",
        f"max_episodes={args_cli.buffer_size}, init_steps={args_cli.init_steps},",
        f"seq_len={args_cli.seq_len}, grad_interval={args_cli.grad_interval},",
        f"grasp_sample_ratio={args_cli.grasp_sample_ratio},",
        f"grasp_sample_min_phase={args_cli.grasp_sample_min_phase}",
    )
    print(f"[INFO] TensorBoard log dir: {tb_log_dir}")
    if not args_cli.no_eval:
        print(
            "[INFO] Eval:",
            f"every {args_cli.eval_interval} iters, {args_cli.eval_episodes} episodes,",
            f"first {min(args_cli.num_eval_envs, num_envs)} env slots",
        )

    env.reset()
    oracle.reset()
    reset_teacher_phase_state(env)

    for iteration in range(start_iteration, end_iteration):
        sampler_stats.reset()
        teacher_label_stats.reset()
        policy.eval()
        for _ in range(args_cli.collect_steps):
            student_obs = env.observation_manager.compute_group("policy")
            obs_step = student_obs.unsqueeze(1)

            with torch.no_grad():
                teacher_twist = compute_teacher_twist(
                    env,
                    oracle,
                    control_dt=control_dt,
                    min_lin_speed=args_cli.teacher_min_lin_speed,
                    min_ang_speed=args_cli.teacher_min_ang_speed,
                )
                teacher_phase = infer_teacher_phase(env)
                pre_teacher_phase = teacher_phase
                pre_cube_to_goal = compute_cube_to_goal_distance(env)
                pre_cube_z_w = env.scene["object"].data.root_pos_w[:, 2].clone()
                teacher_label_stats.observe(teacher_twist[:, 6], teacher_phase)
                if phase_tracker is not None:
                    phase_tracker.on_pre_step(oracle, env)
                actions, rnn_hidden, _ = policy.act(obs_step, rnn_hidden, deterministic=False)
                current_opening = get_mean_finger_opening(env)
                env_actions = twist_to_env_action(
                    actions,
                    control_dt=control_dt,
                    current_opening=current_opening,
                    arm_scale=args_cli.arm_scale,
                    num_gripper_actions=gripper_action_dim,
                )

            active = ep_step < max_ep_len
            if active.any():
                active_ids = active.nonzero(as_tuple=False).squeeze(-1)
                if active_ids.ndim == 0:
                    active_ids = active_ids.unsqueeze(0)
                step_ids = ep_step[active_ids]
                ep_obs[active_ids, step_ids] = student_obs[active_ids]
                ep_teacher[active_ids, step_ids] = teacher_twist[active_ids]
                ep_teacher_phase[active_ids, step_ids] = teacher_phase[active_ids]
                ep_step[active_ids] += 1

            _, _, terminated, truncated, _ = env.step(env_actions)
            total_sampler_steps += num_envs
            global_step += 1

            early_done = None
            if phase_tracker is not None:
                early_done = phase_tracker.on_post_step(oracle, env)

            dones = terminated | truncated
            if early_done is not None:
                dones = dones | early_done
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.ndim == 0:
                    done_ids = done_ids.unsqueeze(0)
                manual_reset_ids = None
                if early_done is not None and early_done.any():
                    manual_reset_ids = early_done.nonzero(as_tuple=False).squeeze(-1)
                    if manual_reset_ids.ndim == 0:
                        manual_reset_ids = manual_reset_ids.unsqueeze(0)
                _record_and_reset_episodes(
                    done_ids,
                    env=env,
                    oracle=oracle,
                    buffer=buffer,
                    ep_obs=ep_obs,
                    ep_teacher=ep_teacher,
                    ep_teacher_phase=ep_teacher_phase,
                    ep_step=ep_step,
                    sampler_stats=sampler_stats,
                    rnn_hidden=rnn_hidden,
                    phase_tracker=phase_tracker,
                    truncated=truncated,
                    terminated=terminated,
                    early_done=early_done,
                    goal_threshold=args_cli.position_threshold,
                    pre_teacher_phase=pre_teacher_phase,
                    pre_cube_to_goal=pre_cube_to_goal,
                    pre_cube_z_w=pre_cube_z_w,
                    manual_reset_ids=manual_reset_ids,
                )

        policy.train()
        mean_arm_kl = 0.0
        mean_gripper_kl = 0.0
        grasp_sample_hit_rate = 0.0
        max_buf_ep_len = buffer.max_episode_length
        can_optimize = (
            total_sampler_steps >= args_cli.init_steps
            and len(buffer) > 0
            and max_buf_ep_len >= 2
        )
        if can_optimize:
            _, mean_arm_kl, mean_gripper_kl, grasp_sample_hit_rate = optimize_kl_sequences(
                policy,
                optimizer,
                buffer,
                teacher_std,
                batch_size=args_cli.batch_size,
                seq_len=args_cli.seq_len,
                optim_steps=args_cli.optim_steps,
                dim_weights=kl_dim_weights,
                grad_interval=args_cli.grad_interval,
                grasp_sample_ratio=args_cli.grasp_sample_ratio,
                min_grasp_phase=args_cli.grasp_sample_min_phase,
            )
        elif len(buffer) == 0:
            optim_skip_reason = "empty_buffer"
        elif total_sampler_steps < args_cli.init_steps:
            optim_skip_reason = f"warmup {total_sampler_steps}/{args_cli.init_steps}"
        else:
            optim_skip_reason = f"short_episodes max_len={max_buf_ep_len}"

        print(
            f"[Iter {iteration + 1}/{end_iteration}] "
            f"arm_kl={mean_arm_kl:.2f} gripper_kl={mean_gripper_kl:.2f} "
            f"grasp_sample={grasp_sample_hit_rate:.2f} "
            f"| teacher_close={teacher_label_stats.close_rate():.3f} "
            f"teacher_grasp={teacher_label_stats.grasp_phase_rate():.3f} "
            f"| {sampler_stats.summary_line(max_ep_len)}"
        )
        if not can_optimize:
            print(f"  optim=off ({optim_skip_reason})")
        _log_training_metrics(
            writer,
            global_step=global_step,
            arm_kl=mean_arm_kl,
            gripper_kl=mean_gripper_kl,
            grasp_sample_hit_rate=grasp_sample_hit_rate,
            teacher_stats=teacher_label_stats,
            sampler_stats=sampler_stats,
        )

        run_eval = (
            not args_cli.no_eval
            and (iteration + 1) % args_cli.eval_interval == 0
            and iteration > 0
        )
        if run_eval:
            flushed = buffer.add_in_progress_episodes(
                ep_obs, ep_teacher, ep_step, ep_teacher_phase=ep_teacher_phase
            )
            if flushed > 0:
                print(f"  flushed {flushed} in-progress episodes to buffer before eval")
            eval_metrics = run_student_eval(
                policy,
                env,
                oracle,
                num_episodes=args_cli.eval_episodes,
                obs_dim=obs_dim,
                control_dt=control_dt,
                arm_scale=args_cli.arm_scale,
                teacher_std=teacher_std,
                kl_dim_weights=kl_dim_weights,
                goal_threshold=args_cli.position_threshold,
                device=device,
                max_envs=args_cli.num_eval_envs,
                early_done=args_cli.early_done,
                phase_stall_steps=args_cli.phase_stall_steps,
                phase_min_progress_m=args_cli.phase_min_progress_m,
                early_done_min_sm_state=args_cli.early_done_min_sm_state,
                num_gripper_actions=gripper_action_dim,
            )
            rnn_hidden = policy.init_hidden(num_envs, device)
            ep_step.zero_()
            env.reset()
            oracle.reset()
            if phase_tracker is not None:
                phase_tracker.reset()
            print(
                f"[Eval {iteration + 1}] "
                f"success={eval_metrics.success_rate:.3f} "
                f"arm_kl={eval_metrics.mean_arm_kl:.2f} "
                f"gripper_kl={eval_metrics.mean_gripper_kl:.2f}"
            )
            _log_eval_metrics(writer, eval_metrics, global_step)

            if eval_metrics.success_rate >= best_success_rate:
                best_success_rate = eval_metrics.success_rate
                best_path = os.path.join(args_cli.log_dir, "student_best.pt")
                torch.save(
                    {
                        "model": policy.state_dict(),
                        "iteration": iteration + 1,
                        "global_step": global_step,
                        "teacher_std": teacher_std,
                        "kl_dim_weights": kl_dim_weights,
                        "obs_dim": obs_dim,
                        "policy_type": "continuous",
                        "success_rate": eval_metrics.success_rate,
                    },
                    best_path,
                )
                print(f"[INFO] Saved best checkpoint: {best_path} (success={best_success_rate:.3f})")

        if (iteration + 1) % args_cli.save_interval == 0:
            ckpt_path = os.path.join(args_cli.log_dir, f"student_iter_{iteration + 1}.pt")
            torch.save(
                {
                    "model": policy.state_dict(),
                    "iteration": iteration + 1,
                    "global_step": global_step,
                    "teacher_std": teacher_std,
                    "kl_dim_weights": kl_dim_weights,
                    "obs_dim": obs_dim,
                    "policy_type": "continuous",
                },
                ckpt_path,
            )
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

        if env.sim.is_stopped():
            break

    flushed = buffer.add_in_progress_episodes(
        ep_obs, ep_teacher, ep_step, ep_teacher_phase=ep_teacher_phase
    )
    if flushed > 0:
        print(f"[INFO] Flushed {flushed} in-progress episodes at end of training")

    final_path = os.path.join(args_cli.log_dir, "student_final.pt")
    torch.save(
        {
            "model": policy.state_dict(),
            "iteration": end_iteration,
            "global_step": global_step,
            "teacher_std": teacher_std,
            "kl_dim_weights": kl_dim_weights,
            "obs_dim": obs_dim,
            "policy_type": "continuous",
        },
        final_path,
    )
    print(f"[INFO] Saved final checkpoint: {final_path}")

    if writer is not None:
        writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
