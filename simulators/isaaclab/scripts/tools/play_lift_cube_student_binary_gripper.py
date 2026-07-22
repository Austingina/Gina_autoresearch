# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visualize a trained dual-head binary-gripper student checkpoint.

Supports ``policy_type``:
- ``binary_gripper_dual_head`` (default for new training)
- ``binary_gripper`` (legacy single-actor)
- ``continuous`` (loads encoder + arm only; gripper head stays random init)

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/play_lift_cube_student_binary_gripper.py \\
        --checkpoint ./runs/lift_cube_ts_binary_gripper_bce/student_iter_500.pt \\
        --num_envs 1

    # Headless + save preview mp4:
    HEADLESS=1 ./scripts/play_lift_cube_student_binary_gripper.sh \\
        CHECKPOINT=./runs/lift_cube_ts_binary_gripper_bce/student_best.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play/visualize dual-head binary-gripper lift-cube student policy."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-Oracle-Twist-Play-v0",
    help="IK-Rel env with cameras (same action interface as training).",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./runs/lift_cube_ts_binary_gripper_bce/student_best.pt",
    help="Path to student checkpoint (.pt).",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--position_threshold", type=float, default=0.01)
parser.add_argument("--arm_scale", type=float, default=0.5)
parser.add_argument("--max_episodes", type=int, default=20)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=30.0,
    help="Episode timeout; should match training (default 30s).",
)
parser.add_argument(
    "--max_preview_steps",
    type=int,
    default=0,
    help="Stop after N sim steps. 0 = until max_episodes or window closed.",
)
parser.add_argument(
    "--loop",
    action="store_true",
    default=False,
    help="After max_episodes, keep running new episodes until quit.",
)
parser.add_argument("--display_scale", type=int, default=4)
parser.add_argument("--velocity_scale", type=float, default=0.35)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./runs/student_binary_gripper_preview",
    help="Directory for preview_latest.png and preview.mp4.",
)
parser.add_argument("--video_fps", type=int, default=20)
parser.add_argument("--show_opencv", action="store_true", default=False)
parser.add_argument(
    "--fixed_lift_goal",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use a fixed object_pose lift target (default matches successful DAgger run).",
)
parser.add_argument("--lift_goal_x", type=float, default=0.5)
parser.add_argument("--lift_goal_y", type=float, default=0.0)
parser.add_argument("--lift_goal_z", type=float, default=0.375)
parser.add_argument(
    "--randomize_cube",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Randomize cube spawn offset on reset.",
)
parser.add_argument(
    "--randomize_lift_goal",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Sample lift goal from default Isaac-Lift-Cube command ranges each episode.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift.mdp.data_collection import (
    EpisodeLiftSuccessTracker,
    compute_cube_to_goal_distance,
    get_ee_pose_in_robot_root_frame,
    get_lift_goal_position_in_robot_root_frame,
    get_mean_finger_opening,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student import (
    StudentTwistPolicyBinaryGripper,
    apply_lift_teacher_student_env_randomization,
    get_gripper_action_dim,
    infer_teacher_phase,
    twist_to_env_action,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.sampler_logging import (
    ORACLE_SM_STATE_NAMES,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from oracle_demo_preview import CollectionPreviewWriter, read_camera_bgr, render_topdown_bgr

PHASE_NAMES = ORACLE_SM_STATE_NAMES

VISION_STUDENT_POLICY_TYPES = frozenset({"binary_gripper_dual_head_vision_dagger"})
PROPRIO_STUDENT_POLICY_TYPES = frozenset(
    {
        "binary_gripper_dual_head",
        "binary_gripper_dual_head_dagger",
        "binary_gripper",
        "continuous",
        "StudentTwistPolicy",
    }
)


def _validate_proprio_student_checkpoint(checkpoint_path: str, ckpt: dict) -> None:
    if "model" not in ckpt and "agent" in ckpt:
        raise ValueError(
            f"Checkpoint looks like an offline CQL-SAC LSTM checkpoint, not a DAgger student checkpoint "
            f"(missing key 'model', found 'agent'). Use:\n"
            f"  CHECKPOINT={checkpoint_path} ./scripts/play_lift_cube_offline_cql_lstm.sh"
        )
    if "model" not in ckpt:
        raise KeyError(
            f"Unsupported checkpoint format in {checkpoint_path}. "
            f"Expected DAgger student keys like 'model'; found: {sorted(ckpt.keys())}"
        )
    policy_type = str(ckpt.get("policy_type", "binary_gripper_dual_head"))
    if policy_type in VISION_STUDENT_POLICY_TYPES:
        raise ValueError(
            f"Checkpoint policy_type={policy_type!r} is a vision DAgger student (CNN+LSTM on mask+depth), "
            f"not proprio dual-head. Use:\n"
            f"  CHECKPOINT={checkpoint_path} ./scripts/play_lift_cube_student_vision_binary_gripper.sh"
        )


def load_student_policy(checkpoint_path: str, device: str) -> tuple[StudentTwistPolicyBinaryGripper, dict]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _validate_proprio_student_checkpoint(checkpoint_path, ckpt)
    obs_dim = int(ckpt.get("obs_dim", 10))
    policy = StudentTwistPolicyBinaryGripper(obs_dim=obs_dim).to(device)
    StudentTwistPolicyBinaryGripper.load_from_checkpoint(policy, ckpt, obs_dim=obs_dim)
    policy.eval()
    return policy, ckpt


def build_student_preview_panel(
    env,
    env_index: int,
    *,
    ckpt_label: str,
    policy_type: str,
    ee_pos_b: np.ndarray,
    goal_pos_b: np.ndarray,
    lin_vel: np.ndarray,
    gripper_cmd: float,
    gripper_prob: float,
    teacher_phase: int,
    episode_step: int,
    episode_id: int,
    success_lift: bool,
    success_grasp: bool,
    display_scale: int,
    velocity_scale: float,
) -> np.ndarray:
    camera_bgr = read_camera_bgr(env, "third_person_camera", env_index, display_scale)
    plot_size = camera_bgr.shape[0]
    topdown_bgr = render_topdown_bgr(
        ee_pos_b,
        goal_pos_b,
        lin_vel,
        goal_phase=1.0 if success_grasp else 0.0,
        gripper_rate=gripper_cmd,
        plot_size=plot_size,
        velocity_scale=velocity_scale,
    )
    panel = np.concatenate([camera_bgr, topdown_bgr], axis=1)

    status = "LIFT OK" if success_lift else ("GRASP OK" if success_grasp else "running")
    color = (0, 220, 0) if success_lift else (0, 180, 255) if success_grasp else (255, 255, 255)
    grip_label = "CLOSE" if gripper_cmd < 0.0 else "OPEN"
    phase_label = PHASE_NAMES[min(max(teacher_phase, 0), len(PHASE_NAMES) - 1)]
    cv2.putText(
        panel,
        f"dual-head {ckpt_label} ep={episode_id} step={episode_step} {status}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"grip={grip_label} p={gripper_prob:.2f} phase={phase_label} |v|={np.linalg.norm(lin_vel):.3f}",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"LSTM -> arm KL head + gripper BCE head ({policy_type})",
        (8, panel.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return panel


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
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
    policy, ckpt = load_student_policy(args_cli.checkpoint, args_cli.device)
    preview = CollectionPreviewWriter(
        output_dir=args_cli.output_dir,
        video_fps=args_cli.video_fps,
        show_opencv=args_cli.show_opencv,
        window_name="student_binary_gripper_preview",
    )

    ckpt_label = os.path.basename(args_cli.checkpoint)
    policy_type = str(ckpt.get("policy_type", "binary_gripper_dual_head"))
    train_success = ckpt.get("success_rate")
    train_iter = ckpt.get("iteration")

    env.reset()

    num_envs = env.num_envs
    preview_env = 0
    rnn_hidden = policy.init_hidden(num_envs, env.device)
    episode_step = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    success_tracker = EpisodeLiftSuccessTracker(num_envs, env.device)
    episodes_done = 0
    lift_success_count = 0
    grasp_success_count = 0
    global_step = 0

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Checkpoint: {os.path.abspath(args_cli.checkpoint)}")
    print(f"[INFO] policy_type: {policy_type}")
    if args_cli.randomize_cube or args_cli.randomize_lift_goal:
        print(
            f"[INFO] Domain randomization: cube={args_cli.randomize_cube}, "
            f"lift_goal={args_cli.randomize_lift_goal}"
        )
        print("[WARN] Checkpoint may have been trained on fixed cube/goal; expect lower success under DR.")
    elif args_cli.fixed_lift_goal:
        print(
            f"[INFO] Fixed lift goal: "
            f"({args_cli.lift_goal_x}, {args_cli.lift_goal_y}, {args_cli.lift_goal_z})"
        )
    if train_iter is not None:
        print(f"[INFO] Trained iteration: {train_iter}")
    if train_success is not None:
        print(f"[INFO] Checkpoint eval success_rate (from training): {train_success:.3f}")
    print(f"[INFO] episode_length_s={args_cli.episode_length_s} max_steps={env.max_episode_length}")
    print(f"[INFO] Preview: {preview.latest_path}")
    if policy_type == "continuous":
        print("[WARN] Loaded continuous checkpoint: encoder+arm only; gripper head was not trained.")

    while simulation_app.is_running():
        with torch.no_grad():
            student_obs = env.observation_manager.compute_group("policy")
            obs_step = student_obs.unsqueeze(1)
            arm_dist, gripper_logit, rnn_hidden = policy.policy_dist(obs_step, rnn_hidden)
            arm_action = arm_dist.mean
            gripper_binary = policy._logit_to_binary(gripper_logit, deterministic=True)
            twist = torch.cat([arm_action, gripper_binary], dim=-1)
            gripper_prob = torch.sigmoid(gripper_logit).squeeze(-1)
            teacher_phase = infer_teacher_phase(env)

        current_opening = get_mean_finger_opening(env)
        pre_cube_to_goal = compute_cube_to_goal_distance(env)
        pre_cube_z_w = env.scene["object"].data.root_pos_w[:, 2].clone()
        env_actions = twist_to_env_action(
            twist,
            control_dt=control_dt,
            current_opening=current_opening,
            arm_scale=args_cli.arm_scale,
            num_gripper_actions=gripper_action_dim,
            binary_gripper=True,
        )
        _, _, terminated, truncated, _ = env.step(env_actions)
        episode_step += 1
        global_step += 1

        success_early = success_tracker.on_step(
            env,
            terminated=terminated,
            truncated=truncated,
            pre_cube_to_goal=pre_cube_to_goal,
            pre_cube_z_w=pre_cube_z_w,
            lift_threshold=args_cli.position_threshold,
        )

        ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
        grasp_ok = success_tracker.ever_grasp_success
        lift_ok = success_tracker.ever_lift_success
        goal_pos_b = get_lift_goal_position_in_robot_root_frame(env)

        panel = build_student_preview_panel(
            env,
            preview_env,
            ckpt_label=ckpt_label,
            policy_type=policy_type,
            ee_pos_b=ee_pos_b[preview_env].detach().cpu().numpy(),
            goal_pos_b=goal_pos_b[preview_env].detach().cpu().numpy(),
            lin_vel=twist[preview_env, :3].detach().cpu().numpy(),
            gripper_cmd=float(twist[preview_env, 6].item()),
            gripper_prob=float(gripper_prob[preview_env].item()),
            teacher_phase=int(teacher_phase[preview_env].item()),
            episode_step=int(episode_step[preview_env].item()),
            episode_id=episodes_done,
            success_lift=bool(lift_ok[preview_env].item()),
            success_grasp=bool(grasp_ok[preview_env].item()),
            display_scale=args_cli.display_scale,
            velocity_scale=args_cli.velocity_scale,
        )
        key = preview.write(panel)
        if key == ord("q"):
            break

        dones = terminated | truncated | success_early
        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.ndim == 0:
                done_ids = done_ids.unsqueeze(0)
            manual_reset_ids = None
            if success_early.any():
                manual_reset_ids = success_early.nonzero(as_tuple=False).squeeze(-1)
                if manual_reset_ids.ndim == 0:
                    manual_reset_ids = manual_reset_ids.unsqueeze(0)
            for env_id in done_ids.tolist():
                episodes_done += 1
                if success_tracker.ever_lift_success[env_id].item():
                    end_reason = "success"
                elif truncated[env_id].item():
                    end_reason = "timeout"
                elif terminated[env_id].item():
                    end_reason = "terminated"
                else:
                    end_reason = "done"
                if lift_ok[env_id].item():
                    lift_success_count += 1
                if grasp_ok[env_id].item():
                    grasp_success_count += 1
                print(
                    f"[Episode {episodes_done}] steps={episode_step[env_id].item()} end={end_reason} "
                    f"lift={'OK' if lift_ok[env_id].item() else 'FAIL'} "
                    f"grasp={'OK' if grasp_ok[env_id].item() else 'FAIL'} "
                    f"phase={PHASE_NAMES[int(teacher_phase[env_id].item())]}"
                )
            policy.reset_hidden(rnn_hidden, done_ids)
            success_tracker.reset(done_ids)
            episode_step[done_ids] = 0
            if manual_reset_ids is not None and manual_reset_ids.numel() > 0:
                env.reset(env_ids=manual_reset_ids)

        if not args_cli.loop and episodes_done >= args_cli.max_episodes:
            break
        if args_cli.max_preview_steps > 0 and global_step >= args_cli.max_preview_steps:
            break

    success_rate = lift_success_count / max(episodes_done, 1)
    grasp_rate = grasp_success_count / max(episodes_done, 1)
    print("[INFO] Summary")
    print(f"  episodes={episodes_done}")
    print(f"  lift_success_rate={success_rate:.3f}")
    print(f"  grasp_success_rate={grasp_rate:.3f}")
    print(f"  preview_video={preview.video_path}")

    preview.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
