# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
DAgger training for lift-cube dual-head binary-gripper student.

Algorithm (Ross et al., 2011) adapted to sim parallel rollouts:

1. **Initial aggregate** (optional oracle HDF5 preload + BC warmup with β=1).
2. Each iteration:
   - Roll out with mixed policy: execute teacher twist w.p. β, else student twist.
   - **Always** label visited states with teacher twist (lenient analytic expert).
   - Append completed episodes to a **growing aggregate buffer** (bounded FIFO).
   - Retrain student on samples drawn uniformly from the full aggregate (BPTT KL+BCE).

Key differences vs online ring-buffer imitation:
- β schedule stabilizes early training (teacher executes near-contact phases).
- Aggregate retains student-visited states (covariate shift correction).
- Buffer checkpointed alongside policy for true resume.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAgger training for lift-cube binary-gripper student.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Teacher-Student-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--collect_steps", type=int, default=64, help="Env steps collected per DAgger iteration.")
parser.add_argument(
    "--max_aggregate_episodes",
    type=int,
    default=5000,
    help="Maximum aggregated episodes kept (FIFO). Classic DAgger keeps all; this bounds GPU RAM.",
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
parser.add_argument("--teacher_min_lin_speed", type=float, default=0.10)
parser.add_argument("--teacher_min_ang_speed", type=float, default=0.20)
parser.add_argument("--arm_scale", type=float, default=0.5)
parser.add_argument("--log_dir", type=str, default="./runs/lift_cube_dagger_binary")
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument(
    "--buffer_save_interval",
    type=int,
    default=200,
    help="Save dagger_aggregate.pt every N iterations (0 = only with checkpoints).",
)
parser.add_argument("--position_threshold", type=float, default=0.01)
parser.add_argument("--eval_interval", type=int, default=50)
parser.add_argument("--eval_episodes", type=int, default=50)
parser.add_argument("--num_eval_envs", type=int, default=16)
parser.add_argument("--no_eval", action="store_true")
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--early_done", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--phase_stall_steps", type=int, default=180)
parser.add_argument("--phase_min_progress_m", type=float, default=0.0005)
parser.add_argument("--early_done_min_sm_state", type=int, default=2)
parser.add_argument("--oracle_demo_file", type=str, default="")
parser.add_argument("--oracle_demo_max_episodes", type=int, default=200)
parser.add_argument("--resume_checkpoint", type=str, default="")
parser.add_argument(
    "--resume_buffer",
    type=str,
    default="",
    help="Optional dagger_aggregate.pt. Defaults to <log_dir>/dagger_aggregate.pt when resuming.",
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
    "--randomize_cube",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Randomize cube spawn offset on reset (Isaac default x/y ranges).",
)
parser.add_argument(
    "--randomize_lift_goal",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Sample lift goal from default Isaac-Lift-Cube command ranges each episode.",
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
    OraclePhaseStallTracker,
    SamplerEpisodeStats,
    SamplerTeacherLabelStats,
    StudentTwistPolicyBinaryGripper,
    RNNState,
    apply_lift_teacher_student_env_randomization,
    reset_rnn_state,
    compute_teacher_twist_binary_gripper,
    get_gripper_action_dim,
    infer_teacher_phase,
    load_oracle_demos_into_buffer,
    make_teacher_arm_std,
    optimize_kl_bce_sequences,
    reset_teacher_phase_state,
    run_student_eval_binary_gripper,
    twist_to_env_action,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.dagger_trainer import (
    DaggerAggregateBuffer,
    compute_dagger_beta,
    compute_dagger_beta_adaptive,
    mix_rollout_twist,
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


def _record_and_reset_episodes(
    done_ids: torch.Tensor,
    *,
    env,
    oracle,
    buffer: DaggerAggregateBuffer,
    ep_obs: torch.Tensor,
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
) -> None:
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
                teacher_success=bool(lift_ok[env_id].item()),
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


def _save_training_checkpoint(
    path: str,
    *,
    policy,
    optimizer,
    iteration: int,
    global_step: int,
    teacher_arm_std: torch.Tensor,
    best_success_rate: float,
    buffer: DaggerAggregateBuffer | None = None,
    include_buffer: bool = False,
) -> None:
    payload = {
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "global_step": global_step,
        "teacher_arm_std": teacher_arm_std,
        "gripper_bce_weight": args_cli.gripper_bce_weight,
        "obs_dim": policy.obs_dim,
        "policy_type": "binary_gripper_dual_head_dagger",
        "success_rate": best_success_rate,
        "beta_start": args_cli.beta_start,
        "beta_end": args_cli.beta_end,
        "beta_warmup_iters": args_cli.beta_warmup_iters,
        "beta_decay_iters": args_cli.beta_decay_iters,
        "beta_adaptive": args_cli.beta_adaptive,
        "beta_success_target": args_cli.beta_success_target,
        "success_reserve_ratio": args_cli.success_reserve_ratio,
    }
    if include_buffer and buffer is not None:
        payload["aggregate_buffer"] = buffer.state_dict()
    torch.save(payload, path)


def main():
    torch.manual_seed(args_cli.seed)
    device = args_cli.device

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.randomize_lift_goal and args_cli.fixed_lift_goal:
        print("[WARN] --randomize_lift_goal overrides --fixed_lift_goal.")
    apply_lift_teacher_student_env_randomization(
        env_cfg,
        randomize_cube=args_cli.randomize_cube,
        randomize_lift_goal=args_cli.randomize_lift_goal,
        fixed_lift_goal=args_cli.fixed_lift_goal and not args_cli.randomize_lift_goal,
        lift_goal_x=args_cli.lift_goal_x,
        lift_goal_y=args_cli.lift_goal_y,
        lift_goal_z=args_cli.lift_goal_z,
        debug_vis_goal=True if (args_cli.randomize_cube or args_cli.randomize_lift_goal) else None,
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    control_dt = env_cfg.sim.dt * env_cfg.decimation
    gripper_action_dim = get_gripper_action_dim(env)
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    if obs_dim != 10:
        print(f"[WARN] Expected student obs dim 10, got {obs_dim}.")

    policy = StudentTwistPolicyBinaryGripper(obs_dim=obs_dim).to(device)
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
    buffer_path = os.path.join(args_cli.log_dir, "dagger_aggregate.pt")
    resume_ckpt: dict | None = None

    if args_cli.resume_checkpoint:
        if not os.path.isfile(args_cli.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {args_cli.resume_checkpoint}")
        resume_ckpt = torch.load(args_cli.resume_checkpoint, map_location="cpu", weights_only=False)
        StudentTwistPolicyBinaryGripper.load_from_checkpoint(policy, resume_ckpt, obs_dim=obs_dim)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_iteration = int(resume_ckpt.get("iteration", 0))
        global_step = int(resume_ckpt.get("global_step", 0))
        best_success_rate = float(resume_ckpt.get("success_rate", -1.0))
        if "teacher_arm_std" in resume_ckpt:
            teacher_arm_std = resume_ckpt["teacher_arm_std"].to(device)
        print(
            f"[INFO] Resumed policy from {args_cli.resume_checkpoint}: "
            f"iteration={start_iteration}, global_step={global_step}"
        )

    buffer = DaggerAggregateBuffer(
        max_episodes=args_cli.max_aggregate_episodes,
        device=device,
        success_reserve_ratio=args_cli.success_reserve_ratio,
    )
    buffer_loaded = False
    resume_buffer_path = args_cli.resume_buffer or (
        buffer_path if args_cli.resume_checkpoint and os.path.isfile(buffer_path) else ""
    )
    if resume_buffer_path and os.path.isfile(resume_buffer_path):
        buffer = DaggerAggregateBuffer.load(resume_buffer_path, device=device)
        buffer_loaded = True
        print(f"[INFO] Loaded aggregate buffer: {resume_buffer_path} episodes={len(buffer)} steps={buffer.num_steps}")
    elif args_cli.resume_checkpoint and "aggregate_buffer" in resume_ckpt:
        buffer.load_state_dict(resume_ckpt["aggregate_buffer"])
        buffer_loaded = True
        print(f"[INFO] Loaded aggregate buffer from checkpoint episodes={len(buffer)} steps={buffer.num_steps}")

    if args_cli.oracle_demo_file and not buffer_loaded:
        demo_count = load_oracle_demos_into_buffer(
            args_cli.oracle_demo_file,
            buffer,
            device=device,
            max_demos=args_cli.oracle_demo_max_episodes,
            control_dt=control_dt,
            binary_gripper=True,
        )
        print(f"[INFO] Preloaded {demo_count} oracle demo episodes from {args_cli.oracle_demo_file}")

    end_iteration = start_iteration + args_cli.max_iterations

    os.makedirs(args_cli.log_dir, exist_ok=True)
    tb_log_dir = os.path.join(args_cli.log_dir, "tb")
    writer = SummaryWriter(log_dir=tb_log_dir) if SummaryWriter is not None else None

    num_envs = env.num_envs
    max_ep_len = env.max_episode_length
    rnn_hidden: RNNState = policy.init_hidden(num_envs, device)
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
    if args_cli.beta_adaptive:
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
    if args_cli.fixed_lift_goal and not args_cli.randomize_lift_goal:
        print(
            f"[INFO] Fixed lift goal: "
            f"({args_cli.lift_goal_x}, {args_cli.lift_goal_y}, {args_cli.lift_goal_z})"
        )
    if args_cli.randomize_cube or args_cli.randomize_lift_goal:
        print(
            f"[INFO] Domain randomization: cube={args_cli.randomize_cube}, "
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
        f"(success={buffer.num_success_episodes}, rolling={buffer.num_rolling_episodes})"
    )
    print(
        f"[INFO] Collect: steps/iter={args_cli.collect_steps}, optim_steps={args_cli.optim_steps}, "
        f"batch={args_cli.batch_size}, seq_len={args_cli.seq_len}, grad_interval={args_cli.grad_interval}"
    )
    print(f"[INFO] TensorBoard: {tb_log_dir}")

    env.reset()
    oracle.reset()
    reset_teacher_phase_state(env)

    start_time = time.time()
    teacher_exec_steps = 0
    rollout_steps = 0
    beta_success_ema: float | None = None

    for iteration in range(start_iteration, end_iteration):
        if args_cli.beta_adaptive:
            beta = compute_dagger_beta_adaptive(
                iteration,
                beta_success_ema,
                beta_start=args_cli.beta_start,
                beta_end=args_cli.beta_end,
                warmup_iters=args_cli.beta_warmup_iters,
                success_target=args_cli.beta_success_target,
            )
        else:
            beta = compute_dagger_beta(
                iteration,
                beta_start=args_cli.beta_start,
                beta_end=args_cli.beta_end,
                warmup_iters=args_cli.beta_warmup_iters,
                decay_iters=args_cli.beta_decay_iters,
            )
        sampler_stats.reset()
        teacher_label_stats.reset()
        policy.eval()

        for _ in range(args_cli.collect_steps):
            student_obs = env.observation_manager.compute_group("policy")
            obs_step = student_obs.unsqueeze(1)

            with torch.no_grad():
                teacher_twist = compute_teacher_twist_binary_gripper(
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

                student_twist, rnn_hidden, _ = policy.act(obs_step, rnn_hidden, deterministic=False)
                exec_twist, teacher_exec_mask = mix_rollout_twist(teacher_twist, student_twist, beta)
                teacher_exec_steps += int(teacher_exec_mask.sum().item())
                rollout_steps += num_envs

                current_opening = get_mean_finger_opening(env)
                env_actions = twist_to_env_action(
                    exec_twist,
                    control_dt=control_dt,
                    current_opening=current_opening,
                    arm_scale=args_cli.arm_scale,
                    num_gripper_actions=gripper_action_dim,
                    binary_gripper=True,
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
        max_buf_ep_len = buffer.max_episode_length
        can_optimize = len(buffer) >= args_cli.min_buffer_episodes and max_buf_ep_len >= 2
        mean_total_loss = 0.0
        mean_arm_kl = 0.0
        mean_gripper_bce = 0.0
        grasp_sample_hit_rate = 0.0
        if can_optimize:
            mean_total_loss, mean_arm_kl, mean_gripper_bce, grasp_sample_hit_rate = optimize_kl_bce_sequences(
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
            )
            optim_skip_reason = "on"
        elif len(buffer) < args_cli.min_buffer_episodes:
            optim_skip_reason = f"buffer {len(buffer)}/{args_cli.min_buffer_episodes}"
        else:
            optim_skip_reason = f"short_episodes max_len={max_buf_ep_len}"

        teacher_exec_rate = teacher_exec_steps / max(rollout_steps, 1)
        teacher_exec_steps = 0
        rollout_steps = 0

        elapsed = time.time() - start_time
        print(
            f"[DAgger {iteration + 1}/{end_iteration}] β={beta:.3f} teacher_exec={teacher_exec_rate:.3f} "
            f"loss={mean_total_loss:.4f} arm_kl={mean_arm_kl:.4f} bce={mean_gripper_bce:.4f} "
            f"buffer={len(buffer)} (ok={buffer.num_success_episodes}) steps={buffer.num_steps} time={elapsed:.1f}s"
        )
        print(f"  optim={optim_skip_reason} | {sampler_stats.summary_line(max_ep_len)}")

        _log_scalar(writer, "dagger/beta", beta, global_step)
        if beta_success_ema is not None:
            _log_scalar(writer, "dagger/eval_success_ema", beta_success_ema, global_step)
        _log_scalar(writer, "dagger/teacher_exec_rate", teacher_exec_rate, global_step)
        _log_scalar(writer, "train/total_loss", mean_total_loss, global_step)
        _log_scalar(writer, "train/arm_kl_loss", mean_arm_kl, global_step)
        _log_scalar(writer, "train/gripper_bce_loss", mean_gripper_bce, global_step)
        _log_scalar(writer, "train/num_episodes", float(len(buffer)), global_step)
        _log_scalar(writer, "train/num_success_episodes", float(buffer.num_success_episodes), global_step)
        _log_scalar(writer, "train/num_rolling_episodes", float(buffer.num_rolling_episodes), global_step)
        _log_scalar(writer, "train/buffer_steps", float(buffer.num_steps), global_step)
        _log_scalar(writer, "train/iteration", float(iteration + 1), global_step)
        _log_sampler_stats(writer, sampler_stats, global_step, max_ep_len)

        run_eval = (
            not args_cli.no_eval
            and (iteration + 1) % args_cli.eval_interval == 0
            and iteration > start_iteration
        )
        if run_eval:
            buffer.add_in_progress_episodes(
                ep_obs, ep_teacher, ep_step, ep_teacher_phase=ep_teacher_phase
            )
            eval_metrics = run_student_eval_binary_gripper(
                policy,
                env,
                oracle,
                num_episodes=args_cli.eval_episodes,
                obs_dim=obs_dim,
                control_dt=control_dt,
                arm_scale=args_cli.arm_scale,
                teacher_arm_std=teacher_arm_std,
                gripper_bce_weight=args_cli.gripper_bce_weight,
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
            reset_teacher_phase_state(env)
            if phase_tracker is not None:
                phase_tracker.reset()
            print(
                f"[Eval {iteration + 1}] success={eval_metrics.success_rate:.3f} "
                f"len={eval_metrics.mean_episode_length:.1f} ({eval_metrics.num_episodes} eps)"
            )
            _log_scalar(writer, "eval/success_rate", eval_metrics.success_rate, global_step)
            if args_cli.beta_adaptive:
                if beta_success_ema is None:
                    beta_success_ema = eval_metrics.success_rate
                else:
                    ema = args_cli.beta_success_ema
                    beta_success_ema = ema * beta_success_ema + (1.0 - ema) * eval_metrics.success_rate
                print(
                    f"[INFO] Adaptive β input: eval_success={eval_metrics.success_rate:.3f} "
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
            )
            print(f"[INFO] Saved checkpoint: {ckpt_path}")
        if save_buffer or (save_ckpt and args_cli.buffer_save_interval == 0):
            buffer.save(buffer_path)
            print(f"[INFO] Saved aggregate buffer: {buffer_path} ({len(buffer)} episodes)")

        if env.sim.is_stopped():
            break

    buffer.add_in_progress_episodes(ep_obs, ep_teacher, ep_step, ep_teacher_phase=ep_teacher_phase)
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
    )
    print(f"[INFO] Saved final policy: {final_path}")
    print(f"[INFO] Saved aggregate buffer: {buffer_path} ({len(buffer)} episodes, {buffer.num_steps} steps)")

    if writer is not None:
        writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
