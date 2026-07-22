# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preview lenient teacher on lift-cube with GT mask + depth (OpenCV)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview lift-cube teacher + GT mask/depth cameras.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-Vision-MaskDepth-Play-v0",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="./runs/lift_cube_mask_depth_preview")
parser.add_argument("--display_scale", type=int, default=2)
parser.add_argument("--camera_height", type=int, default=240)
parser.add_argument("--camera_width", type=int, default=240)
parser.add_argument("--topdown_close", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--max_preview_steps", type=int, default=0, help="0 = until max_episodes")
parser.add_argument("--max_episodes", type=int, default=50)
parser.add_argument("--episode_length_s", type=float, default=25.0)
parser.add_argument("--video_fps", type=int, default=20)
parser.add_argument("--show_opencv", action="store_true", default=False)
parser.add_argument("--depth_colormap", action="store_true", default=True)
parser.add_argument("--arm_scale", type=float, default=0.5)
parser.add_argument("--position_threshold", type=float, default=0.01)
parser.add_argument(
    "--goal_threshold",
    type=float,
    default=0.02,
    help="Cube-to-goal distance for lift success early-done (matches datacollection export).",
)
parser.add_argument("--teacher_min_lin_speed", type=float, default=0.10)
parser.add_argument("--teacher_min_ang_speed", type=float, default=0.20)
parser.add_argument("--teacher_max_lin_vel", type=float, default=0.5)
parser.add_argument("--teacher_max_ang_vel", type=float, default=1.0)
parser.add_argument(
    "--teacher_grasp_hold_steps",
    type=int,
    default=3,
    help="Closed-gripper steps before LIFT phase (default oracle is 15).",
)
parser.add_argument(
    "--teacher_above_offset_z",
    type=float,
    default=0.05,
    help="Hover height above cube during ABOVE phase (m). Lower = shorter reach.",
)
parser.add_argument("--robot_x_offset", type=float, default=0.0)
parser.add_argument("--panda_joint6", type=float, default=None)
parser.add_argument("--randomize_cube", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--fixed_lift_goal", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--lift_goal_above_cube", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--lift_height_above_cube", type=float, default=0.20)
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
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.vision_teacher_student_env_cfg import (
    apply_vision_camera_settings,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.data_collection import (
    EpisodeLiftSuccessTracker,
    compute_cube_to_goal_distance,
    get_mean_finger_opening,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.oracle import LiftCubeOraclePolicy
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student import (
    apply_lift_teacher_student_env_randomization,
    compute_teacher_twist_binary_gripper,
    get_gripper_action_dim,
    infer_teacher_phase,
    reset_teacher_phase_state,
    twist_to_env_action,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.eval_utils import early_reset_env_ids
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _float01_to_bgr(tensor: torch.Tensor, scale: int) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if arr.shape[-1] >= 3:
        rgb = (np.clip(arr[..., :3].astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        gray = (np.clip(arr[..., 0], 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = np.repeat(gray[..., np.newaxis], 3, axis=-1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if scale > 1:
        bgr = cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    return bgr


def _depth_to_bgr(tensor: torch.Tensor, scale: int) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    depth = np.clip(arr.astype(np.float32), 0.0, None)
    dmax = float(depth.max())
    if dmax < 1.0e-6:
        dmax = 1.0
    gray = (np.clip(depth / dmax, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    if scale > 1:
        bgr = cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    return bgr


def _label_panel(bgr: np.ndarray, text: str) -> np.ndarray:
    out = bgr.copy()
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _phase_name(phase: int) -> str:
    names = {0: "rest", 1: "reach_hover", 2: "reach_contact", 3: "grasp", 4: "lift"}
    return names.get(phase, f"phase_{phase}")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.episode_length_s = args_cli.episode_length_s

    if abs(args_cli.robot_x_offset) > 1.0e-9:
        env_cfg.scene.robot.init_state.pos = (args_cli.robot_x_offset, 0.0, 0.0)
    if args_cli.panda_joint6 is not None:
        joint_pos = dict(env_cfg.scene.robot.init_state.joint_pos)
        joint_pos["panda_joint6"] = args_cli.panda_joint6
        env_cfg.scene.robot.init_state.joint_pos = joint_pos

    apply_lift_teacher_student_env_randomization(
        env_cfg,
        randomize_cube=args_cli.randomize_cube,
        randomize_lift_goal=False,
        fixed_lift_goal=args_cli.fixed_lift_goal,
        lift_goal_above_cube=args_cli.lift_goal_above_cube,
        lift_height_above_cube=args_cli.lift_height_above_cube,
        debug_vis_goal=True,
    )
    apply_vision_camera_settings(
        env_cfg,
        camera_height=args_cli.camera_height,
        camera_width=args_cli.camera_width,
        topdown_close=args_cli.topdown_close,
    )
    # Match IK-Rel execution scale to twist_to_env_action arm_scale for faster play.
    env_cfg.actions.arm_action.scale = args_cli.arm_scale

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    oracle = LiftCubeOraclePolicy(env, position_threshold=args_cli.position_threshold)
    gripper_action_dim = get_gripper_action_dim(env)
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    teacher_kwargs = {
        "control_dt": control_dt,
        "min_lin_speed": args_cli.teacher_min_lin_speed,
        "min_ang_speed": args_cli.teacher_min_ang_speed,
        "max_lin_vel": args_cli.teacher_max_lin_vel,
        "max_ang_vel": args_cli.teacher_max_ang_vel,
        "teacher_phase_kwargs": {
            "grasp_hold_steps": args_cli.teacher_grasp_hold_steps,
            "above_offset_z_m": args_cli.teacher_above_offset_z,
        },
    }

    os.makedirs(args_cli.output_dir, exist_ok=True)
    latest_path = os.path.join(args_cli.output_dir, "preview_latest.png")
    video_path = os.path.join(args_cli.output_dir, "preview.mp4")
    writer = None

    teacher_phase_kwargs = {
        "grasp_hold_steps": args_cli.teacher_grasp_hold_steps,
        "above_offset_z_m": args_cli.teacher_above_offset_z,
    }

    env.reset()
    oracle.reset()
    reset_teacher_phase_state(env)
    success_tracker = EpisodeLiftSuccessTracker(env.num_envs, env.device)

    episodes_done = 0
    success_lengths: list[int] = []
    timeout_lengths: list[int] = []
    episode_step = 0
    global_step = 0
    depth_fn = _depth_to_bgr if args_cli.depth_colormap else _float01_to_bgr

    print(f"[INFO] Task: {args_cli.task}")
    print(
        f"[INFO] teacher min_lin={args_cli.teacher_min_lin_speed} min_ang={args_cli.teacher_min_ang_speed} "
        f"max_lin={args_cli.teacher_max_lin_vel} control_dt={control_dt:.4f}s goal_thr={args_cli.goal_threshold}m"
    )
    print(f"[INFO] camera={args_cli.camera_height}px topdown_close={args_cli.topdown_close}")
    print(f"[INFO] Preview: {latest_path}")

    while simulation_app.is_running() and episodes_done < args_cli.max_episodes:
        if args_cli.max_preview_steps > 0 and global_step >= args_cli.max_preview_steps:
            break

        with torch.inference_mode():
            teacher_twist = compute_teacher_twist_binary_gripper(env, oracle, **teacher_kwargs)
            teacher_phase = infer_teacher_phase(env, **teacher_phase_kwargs)
            current_opening = get_mean_finger_opening(env)
            pre_cube_to_goal = compute_cube_to_goal_distance(env)
            pre_cube_z_w = env.scene["object"].data.root_pos_w[:, 2].clone()
            actions = twist_to_env_action(
                teacher_twist,
                control_dt=control_dt,
                current_opening=current_opening,
                arm_scale=args_cli.arm_scale,
                max_lin_vel=args_cli.teacher_max_lin_vel,
                max_ang_vel=args_cli.teacher_max_ang_vel,
                num_gripper_actions=gripper_action_dim,
                binary_gripper=True,
            )
            _, _, terminated, truncated, _ = env.step(actions)
            success_early = success_tracker.on_step(
                env,
                terminated=terminated,
                truncated=truncated,
                pre_cube_to_goal=pre_cube_to_goal,
                pre_cube_z_w=pre_cube_z_w,
                lift_threshold=args_cli.goal_threshold,
            )
            dist_goal = float(pre_cube_to_goal[0].item())

            episode_step += 1
            global_step += 1

            dones = terminated | truncated | success_early
            if dones.any():
                success = bool(success_tracker.ever_lift_success[0].item())
                if success_early.any():
                    reason = "success"
                elif truncated.any():
                    reason = "timeout"
                elif terminated.any():
                    reason = "terminated"
                else:
                    reason = "done"
                print(
                    f"[Episode {episodes_done}] steps={episode_step} success={success} "
                    f"dist={dist_goal:.3f}m reason={reason}"
                )
                if success:
                    success_lengths.append(episode_step)
                elif reason == "timeout":
                    timeout_lengths.append(episode_step)
                episodes_done += 1
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.ndim == 0:
                    done_ids = done_ids.unsqueeze(0)
                reset_teacher_phase_state(env, done_ids.tolist())
                success_tracker.reset(done_ids)
                oracle.reset()
                manual_reset_ids = early_reset_env_ids(success_early)
                if manual_reset_ids.numel() > 0:
                    env.reset(env_ids=manual_reset_ids)
                episode_step = 0

            ee_mask = mdp.camera_colored_semantic_mask(env, sensor_cfg=SceneEntityCfg("ee_camera"))[0]
            tp_mask = mdp.camera_colored_semantic_mask(env, sensor_cfg=SceneEntityCfg("third_person_camera"))[0]
            ee_depth = mdp.camera_depth(env, sensor_cfg=SceneEntityCfg("ee_camera"), max_depth=2.0)[0]
            tp_depth = mdp.camera_depth(env, sensor_cfg=SceneEntityCfg("third_person_camera"), max_depth=5.0)[0]

            lin_speed = float(torch.linalg.norm(teacher_twist[0, :3]).item())
            phase_name = _phase_name(int(teacher_phase[0].item()))

            row_ee = np.concatenate(
                [
                    _label_panel(_float01_to_bgr(ee_mask, args_cli.display_scale), "ee mask"),
                    _label_panel(depth_fn(ee_depth, args_cli.display_scale), "ee depth"),
                ],
                axis=1,
            )
            row_tp = np.concatenate(
                [
                    _label_panel(_float01_to_bgr(tp_mask, args_cli.display_scale), "top mask"),
                    _label_panel(depth_fn(tp_depth, args_cli.display_scale), "top depth"),
                ],
                axis=1,
            )
            panel = np.concatenate([row_ee, row_tp], axis=0)
            status = (
                f"ep={episodes_done} step={episode_step} |v|={lin_speed:.3f} m/s phase={phase_name} "
                f"dist_goal={dist_goal * 100:.1f}cm"
            )
            cv2.putText(panel, status, (8, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 255, 220), 1, cv2.LINE_AA)

            cv2.imwrite(latest_path, panel)
            if writer is None:
                h, w = panel.shape[:2]
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), args_cli.video_fps, (w, h))
            if writer is not None:
                writer.write(panel)
            if args_cli.show_opencv:
                cv2.imshow("teacher_mask_depth_preview", panel)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    if writer is not None:
        writer.release()
    if args_cli.show_opencv:
        cv2.destroyAllWindows()
    env.close()
    if success_lengths:
        mean_success = sum(success_lengths) / len(success_lengths)
        max_success = max(success_lengths)
        print(
            f"[INFO] Success episodes: n={len(success_lengths)} "
            f"mean_steps={mean_success:.1f} max_steps={max_success}"
        )
    else:
        mean_success = 0.0
        max_success = 0
        print("[INFO] Success episodes: n=0")
    if timeout_lengths:
        print(f"[INFO] Timeout episodes: n={len(timeout_lengths)} mean_steps={sum(timeout_lengths) / len(timeout_lengths):.1f}")
    summary_path = os.path.join(args_cli.output_dir, "episodes_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"episodes_done={episodes_done}\n")
        f.write(f"success_count={len(success_lengths)}\n")
        f.write(f"success_lengths={success_lengths}\n")
        f.write(f"timeout_lengths={timeout_lengths}\n")
        if success_lengths:
            f.write(f"success_mean_steps={mean_success:.2f}\n")
            f.write(f"success_max_steps={max_success}\n")
    print(f"[INFO] Episode summary: {summary_path}")
    print(f"[INFO] Saved preview: {latest_path}")
    print(f"[INFO] Saved video: {video_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
