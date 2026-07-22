# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visualize lenient teacher labels (7D EE twist) executed on the arm.

Uses the same pipeline as teacher-student training:
1. Env state -> lenient teacher phase + desired EE / gripper schedule.
2. ``compute_teacher_twist`` or ``compute_teacher_twist_binary_gripper`` (min-speed arm twist).
3. ``twist_to_env_action`` -> IK-Rel env.

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/view_lenient_teacher.py --num_envs 1

    # Binary gripper teacher + headless mp4:
    HEADLESS=1 ./scripts/view_lenient_teacher.sh --binary_gripper

    # Random cube spawn + random lift goal (new env):
    ./scripts/view_lenient_teacher_random.sh
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import random
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize lenient teacher twist on Franka lift-cube.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-Oracle-Twist-Play-v0",
    help="IK-Rel env with cameras.",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--position_threshold", type=float, default=0.01)
parser.add_argument("--arm_scale", type=float, default=0.5)
parser.add_argument("--max_episodes", type=int, default=20)
parser.add_argument("--episode_length_s", type=float, default=30.0)
parser.add_argument(
    "--max_preview_steps",
    type=int,
    default=0,
    help="Stop after N sim steps. 0 = until max_episodes or window closed.",
)
parser.add_argument(
    "--binary_gripper",
    action="store_true",
    default=False,
    help="Use binary gripper teacher labels (+1 open / -1 close).",
)
parser.add_argument(
    "--teacher_min_lin_speed",
    type=float,
    default=0.10,
    help="Minimum teacher linear speed norm (m/s).",
)
parser.add_argument(
    "--teacher_min_ang_speed",
    type=float,
    default=0.20,
    help="Minimum teacher angular speed norm (rad/s).",
)
parser.add_argument("--display_scale", type=int, default=4)
parser.add_argument("--velocity_scale", type=float, default=0.35)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./runs/lenient_teacher_preview",
    help="Directory for preview_latest.png and preview.mp4.",
)
parser.add_argument(
    "--randomize_cube",
    action="store_true",
    default=False,
    help="Randomize cube spawn on reset (same xy range as default lift task). "
    "Alternatively use --task Isaac-Lift-Cube-Franka-Oracle-Twist-Play-Random-v0.",
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
    compute_cube_to_goal_distance,
    get_ee_pose_in_robot_root_frame,
    get_lift_goal_position_in_robot_root_frame,
    get_mean_finger_opening,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.oracle import LiftCubeOraclePolicy
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student import (
    compute_teacher_twist,
    compute_teacher_twist_binary_gripper,
    compute_desired_gripper_opening,
    infer_teacher_phase,
    is_teacher_gripper_close_latched,
    reset_teacher_phase_state,
    teacher_desired_ee_pos_in_base_frame,
    get_gripper_action_dim,
    twist_to_env_action,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.observations import object_position_in_robot_root_frame
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.sampler_logging import (
    ORACLE_SM_STATE_NAMES,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from oracle_demo_preview import CollectionPreviewWriter, read_camera_bgr, render_topdown_bgr

PHASE_NAMES = ORACLE_SM_STATE_NAMES


def build_teacher_preview_panel(
    env,
    env_index: int,
    *,
    ee_pos_b: np.ndarray,
    cube_pos_b: np.ndarray,
    goal_pos_b: np.ndarray,
    lin_vel: np.ndarray,
    ang_vel: np.ndarray,
    gripper_cmd: float,
    teacher_phase: int,
    pos_error: float,
    episode_step: int,
    episode_id: int,
    success_lift: bool,
    success_grasp: bool,
    cube_to_goal: float,
    binary_gripper: bool,
    display_scale: int,
    velocity_scale: float,
    randomize_cube: bool,
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
        cube_pos=cube_pos_b if randomize_cube else None,
    )
    panel = np.concatenate([camera_bgr, topdown_bgr], axis=1)

    status = "LIFT OK" if success_lift else ("GRASP OK" if success_grasp else "running")
    color = (0, 220, 0) if success_lift else (0, 180, 255) if success_grasp else (255, 255, 255)
    phase_label = PHASE_NAMES[min(max(teacher_phase, 0), len(PHASE_NAMES) - 1)]
    lin_norm = float(np.linalg.norm(lin_vel))
    ang_norm = float(np.linalg.norm(ang_vel))
    grip_text = f"{'CLOSE' if gripper_cmd < 0 else 'OPEN'}" if binary_gripper else f"rate={gripper_cmd:+.3f}"

    cv2.putText(
        panel,
        f"lenient_teacher ep={episode_id} step={episode_step} phase={phase_label} {status}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"pos_err={pos_error:.4f}m cube_to_goal={cube_to_goal:.3f}m "
        f"|v|={lin_norm:.3f} |w|={ang_norm:.3f} grip={grip_text}",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    if randomize_cube:
        cv2.putText(
            panel,
            f"cube=({cube_pos_b[0]:.2f},{cube_pos_b[1]:.2f},{cube_pos_b[2]:.2f}) "
            f"goal=({goal_pos_b[0]:.2f},{goal_pos_b[1]:.2f},{goal_pos_b[2]:.2f})",
            (8, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        panel,
        "lenient teacher -> 7D twist -> IK-Rel (min-speed arm)",
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
    if args_cli.randomize_cube:
        env_cfg.events.reset_object_position.params["pose_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.25, 0.25),
            "z": (0.0, 0.0),
        }
        env_cfg.commands.object_pose.debug_vis = True
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    control_dt = env_cfg.sim.dt * env_cfg.decimation
    gripper_action_dim = get_gripper_action_dim(env)
    oracle = LiftCubeOraclePolicy(env, position_threshold=args_cli.position_threshold)
    preview = CollectionPreviewWriter(
        output_dir=args_cli.output_dir,
        show_opencv=args_cli.show_opencv,
        video_fps=args_cli.video_fps,
        window_name="lenient_teacher_preview",
    )

    teacher_kwargs = {
        "control_dt": control_dt,
        "min_lin_speed": args_cli.teacher_min_lin_speed,
        "min_ang_speed": args_cli.teacher_min_ang_speed,
    }

    env.reset()
    oracle.reset()
    reset_teacher_phase_state(env)

    num_envs = env.num_envs
    preview_env = 0
    episode_step = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    episodes_done = 0
    lift_success_count = 0
    lin_norm_sum = 0.0
    ang_norm_sum = 0.0
    norm_count = 0
    global_step = 0

    mode = "binary_gripper" if args_cli.binary_gripper else "continuous_gripper_rate"
    randomize_cube = args_cli.randomize_cube or args_cli.task.endswith("-Random-v0")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Teacher mode: {mode}")
    print(f"[INFO] randomize_cube={randomize_cube}")
    print(
        f"[INFO] min_lin_speed={args_cli.teacher_min_lin_speed} m/s, "
        f"min_ang_speed={args_cli.teacher_min_ang_speed} rad/s"
    )
    print(f"[INFO] control_dt={control_dt:.4f}s arm_scale={args_cli.arm_scale}")
    print(f"[INFO] Preview: {preview.latest_path}")
    flag=0
    while simulation_app.is_running():
        # Snapshot before step(): ManagerBasedRLEnv resets inside step(), so on timeout/done the
        # post-step object pose is the reset spawn (~table height), not the pose you just saw.
        pre_ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
        pre_cube_pos_b = object_position_in_robot_root_frame(env)
        pre_goal_pos_b = get_lift_goal_position_in_robot_root_frame(env)
        pre_cube_to_goal = compute_cube_to_goal_distance(env)
        pre_teacher_phase = infer_teacher_phase(env)
        pre_close_latched = is_teacher_gripper_close_latched(env)
        pre_opening = get_mean_finger_opening(env)
        pre_cube_z_w = env.scene["object"].data.root_pos_w[:, 2].clone()
        pre_des_pos_b = teacher_desired_ee_pos_in_base_frame(env, oracle, pre_teacher_phase)
        pre_pos_error = torch.norm(pre_des_pos_b - pre_ee_pos_b, dim=1)

        teacher_phase = pre_teacher_phase
        if args_cli.binary_gripper:
            teacher_twist = compute_teacher_twist_binary_gripper(env, oracle, **teacher_kwargs)
        else:
            teacher_twist = compute_teacher_twist(env, oracle, **teacher_kwargs)

        desired_opening = compute_desired_gripper_opening(env, oracle, teacher_phase=teacher_phase)
        current_opening = get_mean_finger_opening(env)
        env_actions = twist_to_env_action(
            teacher_twist,
            control_dt=control_dt,
            current_opening=current_opening,
            arm_scale=args_cli.arm_scale,
            num_gripper_actions=gripper_action_dim,
            desired_opening=desired_opening,
            binary_gripper=args_cli.binary_gripper,
        )
        print(f"env_actions: {env_actions}")
        flag -= 1
        if random.random() < 0.3 or flag > 0:
            env_actions = env_actions + torch.randn_like(env_actions) * 0.05 #对action加噪声
            if flag <= 0:
                action_tmp = env_actions
                flag = random.randint(3, 5)
            else:
                env_actions= action_tmp
        


        _, _, terminated, truncated, _ = env.step(env_actions)
        episode_step += 1
        global_step += 1

        dones = terminated | truncated
        if dones.any():
            ee_pos_b = pre_ee_pos_b
            cube_pos_b = pre_cube_pos_b
            goal_pos_b = pre_goal_pos_b
            cube_to_goal = pre_cube_to_goal
            teacher_phase = pre_teacher_phase
            grasp_ok = pre_close_latched
            opening = pre_opening
            cube_z_w = pre_cube_z_w
            pos_error = pre_pos_error
        else:
            ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
            cube_pos_b = object_position_in_robot_root_frame(env)
            goal_pos_b = get_lift_goal_position_in_robot_root_frame(env)
            cube_to_goal = compute_cube_to_goal_distance(env)
            teacher_phase = infer_teacher_phase(env)
            grasp_ok = is_teacher_gripper_close_latched(env)
            opening = get_mean_finger_opening(env)
            cube_z_w = env.scene["object"].data.root_pos_w[:, 2]
            des_pos_b = teacher_desired_ee_pos_in_base_frame(env, oracle, teacher_phase)
            pos_error = torch.norm(des_pos_b - ee_pos_b, dim=1)

        lift_ok = cube_to_goal <= args_cli.position_threshold

        lin_vel = teacher_twist[:, :3]
        ang_vel = teacher_twist[:, 3:6]
        lin_norm_sum += float(torch.linalg.norm(lin_vel, dim=-1).sum().item())
        ang_norm_sum += float(torch.linalg.norm(ang_vel, dim=-1).sum().item())
        norm_count += num_envs

        panel = build_teacher_preview_panel(
            env,
            preview_env,
            ee_pos_b=ee_pos_b[preview_env].detach().cpu().numpy(),
            cube_pos_b=cube_pos_b[preview_env].detach().cpu().numpy(),
            goal_pos_b=goal_pos_b[preview_env].detach().cpu().numpy(),
            lin_vel=lin_vel[preview_env].detach().cpu().numpy(),
            ang_vel=ang_vel[preview_env].detach().cpu().numpy(),
            gripper_cmd=float(teacher_twist[preview_env, 6].item()),
            teacher_phase=int(teacher_phase[preview_env].item()),
            pos_error=float(pos_error[preview_env].item()),
            episode_step=int(episode_step[preview_env].item()),
            episode_id=episodes_done,
            success_lift=bool(lift_ok[preview_env].item()),
            success_grasp=bool(grasp_ok[preview_env].item()),
            cube_to_goal=float(cube_to_goal[preview_env].item()),
            binary_gripper=args_cli.binary_gripper,
            display_scale=args_cli.display_scale,
            velocity_scale=args_cli.velocity_scale,
            randomize_cube=randomize_cube,
        )
        key = preview.write(panel)
        if key == ord("q"):
            break

        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.ndim == 0:
                done_ids = done_ids.unsqueeze(0)
            for env_id in done_ids.tolist():
                episodes_done += 1
                end_reason = "timeout" if truncated[env_id].item() else "terminated"
                if lift_ok[env_id].item():
                    lift_success_count += 1
                lv = teacher_twist[env_id, :3]
                av = teacher_twist[env_id, 3:6]
                close_latched = int(grasp_ok[env_id].item())
                cube_pos_ep = cube_pos_b[env_id]
                goal_pos_ep = goal_pos_b[env_id]
                cube_to_goal_ep = float(cube_to_goal[env_id].item())
                cube_z = float(cube_z_w[env_id].item())
                ee_to_ee_goal = float(torch.norm(ee_pos_b[env_id] - goal_pos_ep).item())
                spawn_cube = (
                    f" cube_spawn=({cube_pos_ep[0]:.3f},{cube_pos_ep[1]:.3f},{cube_pos_ep[2]:.3f})"
                    if randomize_cube
                    else ""
                )
                spawn_goal = (
                    f" goal=({goal_pos_ep[0]:.3f},{goal_pos_ep[1]:.3f},{goal_pos_ep[2]:.3f})"
                    if randomize_cube
                    else ""
                )
                print(
                    f"[Episode {episodes_done}] steps={episode_step[env_id].item()} end={end_reason} "
                    f"phase={PHASE_NAMES[int(teacher_phase[env_id].item())]} "
                    f"close_latched={close_latched} opening={float(opening[env_id].item()):.4f} "
                    f"cube_z={cube_z:.3f} cube_to_goal={cube_to_goal_ep:.3f}m "
                    f"ee_to_goal={ee_to_ee_goal:.3f}m{spawn_cube}{spawn_goal} "
                    f"|v|={torch.linalg.norm(lv).item():.3f} |w|={torch.linalg.norm(av).item():.3f} "
                    f"lift={'OK' if lift_ok[env_id].item() else 'FAIL'} "
                    f"grasp={'OK' if grasp_ok[env_id].item() else 'FAIL'}"
                )
            oracle.reset(done_ids.tolist())
            reset_teacher_phase_state(env, done_ids.tolist())
            episode_step[done_ids] = 0

        if episodes_done >= args_cli.max_episodes:
            break
        if args_cli.max_preview_steps > 0 and global_step >= args_cli.max_preview_steps:
            break

    print("[INFO] Summary")
    print(f"  episodes={episodes_done}")
    print(f"  lift_success_rate={lift_success_count / max(episodes_done, 1):.3f}")
    print(f"  mean_|v|={lin_norm_sum / max(norm_count, 1):.4f} m/s")
    print(f"  mean_|w|={ang_norm_sum / max(norm_count, 1):.4f} rad/s")
    print(f"  preview_video={preview.video_path}")

    preview.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
