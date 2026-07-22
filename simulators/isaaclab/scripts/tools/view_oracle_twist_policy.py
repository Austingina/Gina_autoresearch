# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visualize oracle pick-and-lift mapped to 7D EE twist and executed on the arm.

This uses the same pipeline as teacher-student training:
1. Read env state.
2. Oracle SM -> desired EE pose / gripper.
3. Map to 7D twist via ``compute_teacher_twist``.
4. Execute via IK-Rel + continuous finger position using ``twist_to_env_action``.

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/view_oracle_twist_policy.py --num_envs 1

    # Headless with saved preview mp4:
    ./isaaclab.sh -p scripts/tools/view_oracle_twist_policy.py \\
        --num_envs 1 --headless --output_dir ./runs/oracle_twist_preview
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize oracle twist execution on Franka lift-cube.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-Oracle-Twist-Play-v0",
    help="IK-Rel env with cameras for oracle twist visualization.",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--position_threshold", type=float, default=0.01)
parser.add_argument("--arm_scale", type=float, default=0.5)
parser.add_argument("--max_episodes", type=int, default=20, help="Number of episodes to evaluate.")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=15.0,
    help="Episode timeout in seconds. Default 15s (750 steps) for slow twist->IK-Rel execution.",
)
parser.add_argument(
    "--max_preview_steps",
    type=int,
    default=0,
    help="Stop after this many simulation steps. 0 means run until max_episodes or Isaac Sim closes.",
)
parser.add_argument("--display_scale", type=int, default=4)
parser.add_argument("--velocity_scale", type=float, default=0.35)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./runs/oracle_twist_preview",
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
    compute_grasp_success,
    compute_lift_success,
    get_ee_pose_in_robot_root_frame,
    get_lift_goal_position_in_robot_root_frame,
    get_mean_finger_opening,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.oracle import LiftCubeOraclePolicy
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student import (
    compute_teacher_twist,
    compute_desired_gripper_opening,
    get_gripper_action_dim,
    get_oracle_desired_ee_in_base_frame,
    twist_to_env_action,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from oracle_demo_preview import CollectionPreviewWriter, read_camera_bgr, render_topdown_bgr


def build_twist_preview_panel(
    env,
    env_index: int,
    *,
    ee_pos_b: np.ndarray,
    goal_pos_b: np.ndarray,
    lin_vel: np.ndarray,
    gripper_rate: float,
    pos_error: float,
    episode_step: int,
    episode_id: int,
    success_lift: bool,
    success_grasp: bool,
    display_scale: int,
    velocity_scale: float,
) -> np.ndarray:
    """Build camera + top-down panel for oracle twist execution."""
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
        f"oracle_twist ep={episode_id} step={episode_step} {status}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"pos_err={pos_error:.4f}m |v|={np.linalg.norm(lin_vel):.3f} grip={gripper_rate:+.0f}",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "pipeline: oracle SM -> 7D twist -> IK-Rel",
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
    oracle = LiftCubeOraclePolicy(env, position_threshold=args_cli.position_threshold)
    preview = CollectionPreviewWriter(
        output_dir=args_cli.output_dir,
        video_fps=args_cli.video_fps,
        show_opencv=args_cli.show_opencv,
        window_name="oracle_twist_preview",
    )

    env.reset()
    oracle.reset()

    num_envs = env.num_envs
    preview_env = 0
    episode_step = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    episodes_done = 0
    lift_success_count = 0
    pos_error_sum = 0.0
    pos_error_count = 0
    global_step = 0

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] control_dt={control_dt:.4f}s arm_scale={args_cli.arm_scale}")
    print(
        f"[INFO] episode_length_s={args_cli.episode_length_s} "
        f"(max_steps={env.max_episode_length}, was 5s/250 steps in base lift env)"
    )
    print("[INFO] Oracle -> compute_teacher_twist -> twist_to_env_action -> IK-Rel env")
    print(f"[INFO] Preview: {preview.latest_path}")

    while simulation_app.is_running():
        teacher_twist = compute_teacher_twist(env, oracle, control_dt=control_dt)
        desired_opening = compute_desired_gripper_opening(env, oracle)
        current_opening = get_mean_finger_opening(env)
        env_actions = twist_to_env_action(
            teacher_twist,
            control_dt=control_dt,
            current_opening=current_opening,
            arm_scale=args_cli.arm_scale,
            num_gripper_actions=gripper_action_dim,
            desired_opening=desired_opening,
        )
        _, _, terminated, truncated, _ = env.step(env_actions)
        episode_step += 1
        global_step += 1

        ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
        des_pos_b, _, _ = get_oracle_desired_ee_in_base_frame(env, oracle)
        pos_error = torch.norm(des_pos_b - ee_pos_b, dim=1)

        pos_error_sum += float(pos_error.sum().item())
        pos_error_count += num_envs

        grasp_ok = compute_grasp_success(env)
        lift_ok = compute_lift_success(env, threshold=args_cli.position_threshold)
        goal_pos_b = get_lift_goal_position_in_robot_root_frame(env)

        panel = build_twist_preview_panel(
            env,
            preview_env,
            ee_pos_b=ee_pos_b[preview_env].detach().cpu().numpy(),
            goal_pos_b=goal_pos_b[preview_env].detach().cpu().numpy(),
            lin_vel=teacher_twist[preview_env, :3].detach().cpu().numpy(),
            gripper_rate=float(teacher_twist[preview_env, 6].item()),
            pos_error=float(pos_error[preview_env].item()),
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

        dones = terminated | truncated
        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.ndim == 0:
                done_ids = done_ids.unsqueeze(0)
            for env_id in done_ids.tolist():
                episodes_done += 1
                if truncated[env_id].item():
                    end_reason = "timeout"
                elif terminated[env_id].item():
                    end_reason = "terminated"
                else:
                    end_reason = "done"
                if lift_ok[env_id].item():
                    lift_success_count += 1
                print(
                    f"[Episode {episodes_done}] env={env_id} steps={episode_step[env_id].item()} "
                    f"end={end_reason} "
                    f"lift={'OK' if lift_ok[env_id].item() else 'FAIL'} "
                    f"grasp={'OK' if grasp_ok[env_id].item() else 'FAIL'} "
                    f"final_pos_err={pos_error[env_id].item():.4f}"
                )
            oracle.reset(done_ids.tolist())
            episode_step[done_ids] = 0

        if episodes_done >= args_cli.max_episodes:
            break
        if args_cli.max_preview_steps > 0 and global_step >= args_cli.max_preview_steps:
            break

    mean_pos_error = pos_error_sum / max(pos_error_count, 1)
    success_rate = lift_success_count / max(episodes_done, 1)
    print("[INFO] Summary")
    print(f"  episodes={episodes_done}")
    print(f"  lift_success_rate={success_rate:.3f}")
    print(f"  mean_pos_error={mean_pos_error:.4f} m")
    print(f"  preview_video={preview.video_path}")

    preview.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
