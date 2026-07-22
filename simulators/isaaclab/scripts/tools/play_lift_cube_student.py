# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visualize a trained student twist policy checkpoint (e.g. student_best.pt).

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/play_lift_cube_student.py \\
        --checkpoint ./runs/lift_cube_teacher_student_kl/student_best.pt \\
        --num_envs 1

    # Headless + save preview mp4:
    ./isaaclab.sh -p scripts/tools/play_lift_cube_student.py \\
        --checkpoint ./runs/lift_cube_teacher_student_kl/student_best.pt \\
        --headless --output_dir ./runs/student_best_preview
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play/visualize trained lift-cube student policy.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-Oracle-Twist-Play-v0",
    help="IK-Rel env with cameras (same action interface as training).",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./runs/lift_cube_teacher_student_kl/student_best.pt",
    help="Path to student checkpoint (student_best.pt or student_final.pt).",
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
    default="./runs/student_policy_preview",
    help="Directory for preview_latest.png and preview.mp4.",
)
parser.add_argument("--video_fps", type=int, default=20)
parser.add_argument("--show_opencv", action="store_true", default=False)
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
    StudentTwistPolicy,
    get_gripper_action_dim,
    twist_to_env_action,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from oracle_demo_preview import CollectionPreviewWriter, read_camera_bgr, render_topdown_bgr


def load_student_policy(checkpoint_path: str, device: str) -> tuple[StudentTwistPolicy, dict]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    obs_dim = int(ckpt.get("obs_dim", 10))
    policy = StudentTwistPolicy(obs_dim=obs_dim, action_dim=7).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    return policy, ckpt


def build_student_preview_panel(
    env,
    env_index: int,
    *,
    ckpt_label: str,
    ee_pos_b: np.ndarray,
    goal_pos_b: np.ndarray,
    lin_vel: np.ndarray,
    gripper_rate: float,
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
        gripper_rate=gripper_rate,
        plot_size=plot_size,
        velocity_scale=velocity_scale,
    )
    panel = np.concatenate([camera_bgr, topdown_bgr], axis=1)

    status = "LIFT OK" if success_lift else ("GRASP OK" if success_grasp else "running")
    color = (0, 220, 0) if success_lift else (0, 180, 255) if success_grasp else (255, 255, 255)
    cv2.putText(
        panel,
        f"student {ckpt_label} ep={episode_id} step={episode_step} {status}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"|v|={np.linalg.norm(lin_vel):.3f} grip={gripper_rate:+.0f}",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "student LSTM -> 7D twist -> IK-Rel",
        (8, panel.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return panel


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.episode_length_s = args_cli.episode_length_s
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    control_dt = env_cfg.sim.dt * env_cfg.decimation
    gripper_action_dim = get_gripper_action_dim(env)
    policy, ckpt = load_student_policy(args_cli.checkpoint, args_cli.device)
    preview = CollectionPreviewWriter(
        output_dir=args_cli.output_dir,
        video_fps=args_cli.video_fps,
        show_opencv=args_cli.show_opencv,
        window_name="student_policy_preview",
    )

    ckpt_label = os.path.basename(args_cli.checkpoint)
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
    global_step = 0

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Checkpoint: {os.path.abspath(args_cli.checkpoint)}")
    if train_iter is not None:
        print(f"[INFO] Trained iteration: {train_iter}")
    if train_success is not None:
        print(f"[INFO] Checkpoint eval success_rate (from training): {train_success:.3f}")
    print(f"[INFO] episode_length_s={args_cli.episode_length_s} max_steps={env.max_episode_length}")
    print(f"[INFO] Preview: {preview.latest_path}")

    while simulation_app.is_running():
        with torch.no_grad():
            student_obs = env.observation_manager.compute_group("policy")
            obs_step = student_obs.unsqueeze(1)
            actions, rnn_hidden, _ = policy.act(obs_step, rnn_hidden, deterministic=True)
            twist = actions

        current_opening = get_mean_finger_opening(env)
        pre_cube_to_goal = compute_cube_to_goal_distance(env)
        pre_cube_z_w = env.scene["object"].data.root_pos_w[:, 2].clone()
        env_actions = twist_to_env_action(
            twist,
            control_dt=control_dt,
            current_opening=current_opening,
            arm_scale=args_cli.arm_scale,
            num_gripper_actions=gripper_action_dim,
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
            ee_pos_b=ee_pos_b[preview_env].detach().cpu().numpy(),
            goal_pos_b=goal_pos_b[preview_env].detach().cpu().numpy(),
            lin_vel=twist[preview_env, :3].detach().cpu().numpy(),
            gripper_rate=float(twist[preview_env, 6].item()),
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
                print(
                    f"[Episode {episodes_done}] steps={episode_step[env_id].item()} end={end_reason} "
                    f"lift={'OK' if lift_ok[env_id].item() else 'FAIL'} "
                    f"grasp={'OK' if grasp_ok[env_id].item() else 'FAIL'}"
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
    print("[INFO] Summary")
    print(f"  episodes={episodes_done}")
    print(f"  lift_success_rate={success_rate:.3f}")
    print(f"  preview_video={preview.video_path}")

    preview.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
