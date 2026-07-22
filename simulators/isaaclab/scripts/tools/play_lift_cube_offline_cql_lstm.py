# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Roll out an offline CQL-SAC LSTM checkpoint in simulation.

The policy outputs normalized IK-Rel ``env_actions`` (8D) directly from mask+depth
observations with an LSTM history buffer.

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/play_lift_cube_offline_cql_lstm.py \\
        --checkpoint ./runs/lift_cube_offline_cql_lstm/cql_sac_lstm/checkpoint_final.pt \\
        --num_envs 1 --max_episodes 20
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sim evaluation / preview for offline CQL-SAC LSTM policy.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-Vision-MaskDepth-Play-v0",
)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_episodes", type=int, default=20)
parser.add_argument("--position_threshold", type=float, default=0.02)
parser.add_argument("--lstm_seq_len", type=int, default=0, help="0 = read from checkpoint.")
parser.add_argument("--image_size", type=int, default=0, help="0 = read from checkpoint; -1 = native resolution.")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=30.0,
    help="Episode timeout; should match data collection (default 30s).",
)
parser.add_argument(
    "--max_preview_steps",
    type=int,
    default=0,
    help="Stop after N sim steps. 0 = until max_episodes or window closed.",
)
parser.add_argument("--loop", action="store_true", default=False)
parser.add_argument("--display_scale", type=int, default=4)
parser.add_argument("--velocity_scale", type=float, default=0.35)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./runs/offline_cql_lstm_preview",
)
parser.add_argument("--video_fps", type=int, default=20)
parser.add_argument("--show_opencv", action="store_true", default=False)
parser.add_argument(
    "--fixed_lift_goal",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--lift_goal_x", type=float, default=0.5)
parser.add_argument("--lift_goal_y", type=float, default=0.0)
parser.add_argument("--lift_goal_z", type=float, default=0.375)
parser.add_argument("--randomize_cube", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--randomize_lift_goal", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    "--metrics_only",
    action="store_true",
    default=False,
    help="Run sim eval without preview rendering and exit.",
)
parser.add_argument(
    "--output_json",
    type=str,
    default="",
    help="Write sim eval metrics to this JSON file (used by training subprocess eval).",
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
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student import apply_lift_teacher_student_env_randomization
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.offline_cql.obs_utils import (
    stack_visual_from_policy_obs,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.offline_cql.checkpoint import load_checkpoint
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.offline_cql.sim_eval import (
    load_policy_from_checkpoint,
    run_sim_eval,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from oracle_demo_preview import CollectionPreviewWriter, read_camera_bgr, render_topdown_bgr


def _resolve_image_size(checkpoint_args: dict) -> int | None:
    if args_cli.image_size == -1:
        return None
    if args_cli.image_size > 0:
        return args_cli.image_size
    ckpt_size = int(checkpoint_args.get("image_size", 64))
    return None if ckpt_size <= 0 else ckpt_size


def _resolve_seq_len(checkpoint_args: dict) -> int:
    if args_cli.lstm_seq_len > 0:
        return args_cli.lstm_seq_len
    return int(checkpoint_args.get("lstm_seq_len", 20))


def build_preview_panel(
    env,
    env_index: int,
    *,
    ckpt_label: str,
    ee_pos_b: np.ndarray,
    goal_pos_b: np.ndarray,
    env_action: np.ndarray,
    episode_step: int,
    episode_id: int,
    success_lift: bool,
    success_grasp: bool,
    cube_to_goal: float,
    display_scale: int,
    velocity_scale: float,
) -> np.ndarray:
    camera_bgr = read_camera_bgr(env, "third_person_camera", env_index, display_scale)
    plot_size = camera_bgr.shape[0]
    topdown_bgr = render_topdown_bgr(
        ee_pos_b,
        goal_pos_b,
        env_action[:3],
        goal_phase=1.0 if success_grasp else 0.0,
        gripper_rate=float(env_action[-1]),
        plot_size=plot_size,
        velocity_scale=velocity_scale,
    )
    panel = np.concatenate([camera_bgr, topdown_bgr], axis=1)
    status = "LIFT OK" if success_lift else ("GRASP OK" if success_grasp else "running")
    color = (0, 220, 0) if success_lift else (0, 180, 255) if success_grasp else (255, 255, 255)
    cv2.putText(
        panel,
        f"CQL-LSTM {ckpt_label} ep={episode_id} step={episode_step} {status}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"cube_to_goal={cube_to_goal:.3f} |a|={np.linalg.norm(env_action):.3f}",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")

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

    device = torch.device(args_cli.device)
    ckpt_args = load_checkpoint(args_cli.checkpoint, map_location=device).get("args", {})
    seq_len = _resolve_seq_len(ckpt_args)
    image_size = _resolve_image_size(ckpt_args)
    policy, _ = load_policy_from_checkpoint(
        args_cli.checkpoint,
        seq_len=seq_len,
        device=device,
    )

    ckpt_label = os.path.basename(args_cli.checkpoint)
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Checkpoint: {os.path.abspath(args_cli.checkpoint)}")
    print(f"[INFO] lstm_seq_len={seq_len} image_size={image_size}")
    print(f"[INFO] episode_length_s={args_cli.episode_length_s} max_steps={env.max_episode_length}")

    if args_cli.metrics_only:
        metrics = run_sim_eval(
            policy,
            env,
            num_episodes=args_cli.max_episodes,
            goal_threshold=args_cli.position_threshold,
            image_size=image_size,
        )
        result = {
            "checkpoint": os.path.abspath(args_cli.checkpoint),
            "task": args_cli.task,
            "num_episodes": metrics.num_episodes,
            "lift_success_rate": metrics.success_rate,
            "grasp_success_rate": metrics.grasp_rate,
            "mean_episode_length": metrics.mean_episode_length,
            "mean_final_cube_to_goal": metrics.mean_final_cube_to_goal,
        }
        print("[INFO] Sim evaluation summary")
        print(f"  episodes={metrics.num_episodes}")
        print(f"  lift_success_rate={metrics.success_rate:.3f}")
        print(f"  grasp_success_rate={metrics.grasp_rate:.3f}")
        print(f"  mean_episode_length={metrics.mean_episode_length:.1f}")
        print(f"  mean_final_cube_to_goal={metrics.mean_final_cube_to_goal:.4f}")
        if args_cli.output_json:
            with open(args_cli.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"[INFO] Wrote metrics to {args_cli.output_json}")
        env.close()
        return

    preview = CollectionPreviewWriter(
        output_dir=args_cli.output_dir,
        video_fps=args_cli.video_fps,
        show_opencv=args_cli.show_opencv,
        window_name="offline_cql_lstm_preview",
    )
    print(f"[INFO] Preview: {preview.latest_path}")

    env.reset()
    num_envs = env.num_envs
    preview_env = 0
    policy.reset(num_envs)
    episode_step = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    success_tracker = EpisodeLiftSuccessTracker(num_envs, env.device)
    episodes_done = 0
    lift_success_count = 0
    grasp_success_count = 0
    global_step = 0
    obs_mode = policy.obs_mode

    while simulation_app.is_running():
        obs = env.obs_buf["policy"]
        visual = proprio = None
        if obs_mode in ("vision", "vision_proprio"):
            visual = stack_visual_from_policy_obs(obs, image_size=image_size)
        if obs_mode in ("proprio", "vision_proprio"):
            proprio = obs["student_state"].detach().cpu().numpy()

        pre_cube_to_goal = compute_cube_to_goal_distance(env)
        pre_cube_z_w = env.scene["object"].data.root_pos_w[:, 2].clone()
        env_actions_np = policy.act(visual=visual, proprio=proprio)
        env_actions = torch.as_tensor(env_actions_np, dtype=torch.float32, device=env.device)
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
        goal_pos_b = get_lift_goal_position_in_robot_root_frame(env)
        lift_ok = success_tracker.ever_lift_success
        grasp_ok = success_tracker.ever_grasp_success
        panel = build_preview_panel(
            env,
            preview_env,
            ckpt_label=ckpt_label,
            ee_pos_b=ee_pos_b[preview_env].detach().cpu().numpy(),
            goal_pos_b=goal_pos_b[preview_env].detach().cpu().numpy(),
            env_action=env_actions_np[preview_env],
            episode_step=int(episode_step[preview_env].item()),
            episode_id=episodes_done,
            success_lift=bool(lift_ok[preview_env].item()),
            success_grasp=bool(grasp_ok[preview_env].item()),
            cube_to_goal=float(pre_cube_to_goal[preview_env].item()),
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
            for env_id in done_ids.tolist():
                episodes_done += 1
                if lift_ok[env_id].item():
                    lift_success_count += 1
                if grasp_ok[env_id].item():
                    grasp_success_count += 1
                end_reason = "success" if lift_ok[env_id].item() else ("timeout" if truncated[env_id].item() else "done")
                print(
                    f"[Episode {episodes_done}] steps={episode_step[env_id].item()} end={end_reason} "
                    f"lift={'OK' if lift_ok[env_id].item() else 'FAIL'} "
                    f"grasp={'OK' if grasp_ok[env_id].item() else 'FAIL'}"
                )
            policy.reset_envs(done_ids)
            success_tracker.reset(done_ids)
            episode_step[done_ids] = 0

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
