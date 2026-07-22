# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize clean oracle teacher with G_θ overlay (mask-only field check)."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Oracle teacher play with G_θ vs GT subgoal overlay.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Lenient-Teacher-MaskDepth-Data-v0")
parser.add_argument("--field_checkpoint", type=str, required=True)
parser.add_argument("--autoresearch_root", type=str, default=os.environ.get("AUTORESEARCH_ROOT", ""))
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--image_size", type=int, default=240)
parser.add_argument("--camera_height", type=int, default=None)
parser.add_argument("--camera_width", type=int, default=None)
parser.add_argument("--topdown_close", action="store_true", default=False)
parser.add_argument("--position_threshold", type=float, default=0.01)
parser.add_argument("--arm_scale", type=float, default=0.5)
parser.add_argument("--max_episodes", type=int, default=20)
parser.add_argument("--episode_length_s", type=float, default=30.0)
parser.add_argument("--display_scale", type=int, default=2)
parser.add_argument("--velocity_scale", type=float, default=0.35)
parser.add_argument("--output_dir", type=str, default="./runs/teacher_guidance_preview")
parser.add_argument("--video_fps", type=int, default=20)
parser.add_argument("--show_opencv", action="store_true", default=False)
parser.add_argument("--fixed_lift_goal", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--randomize_cube", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--randomize_lift_goal", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--teacher_min_lin_speed", type=float, default=0.10)
parser.add_argument("--teacher_min_ang_speed", type=float, default=0.20)
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
from isaaclab_tasks.manager_based.manipulation.lift.mdp.observations import object_position_in_robot_root_frame
from isaaclab_tasks.manager_based.manipulation.lift.mdp.oracle import LiftCubeOraclePolicy
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student import (
    apply_lift_teacher_student_env_randomization,
    compute_desired_gripper_opening,
    compute_teacher_twist_binary_gripper,
    get_gripper_action_dim,
    infer_teacher_phase,
    reset_teacher_phase_state,
    twist_to_env_action,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.sampler_logging import ORACLE_SM_STATE_NAMES
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from oracle_demo_preview import CollectionPreviewWriter, read_camera_bgr, render_topdown_bgr, render_xz_direction_bgr

PHASE_NAMES = ORACLE_SM_STATE_NAMES


def _unit2(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float64)
    return (vec / norm).astype(np.float64)


def compute_subgoal_gt_xy(ee, cube, goal, *, grasped: bool) -> np.ndarray:
    if not grasped:
        return _unit2([cube[0] - ee[0], cube[1] - ee[1]])
    return _unit2([goal[0] - cube[0], goal[1] - cube[1]])


def compute_subgoal_gt_xz(ee, cube, goal, *, grasped: bool) -> np.ndarray:
    if not grasped:
        return _unit2([cube[0] - ee[0], cube[2] - ee[2]])
    return _unit2([goal[0] - cube[0], goal[2] - cube[2]])


def build_teacher_g_panel(
    env,
    env_index: int,
    *,
    ee_pos_b: np.ndarray,
    cube_pos_b: np.ndarray,
    goal_pos_b: np.ndarray,
    dist_to_goal: float,
    lin_vel: np.ndarray,
    teacher_phase: int,
    episode_step: int,
    episode_id: int,
    guidance_xz: np.ndarray | None,
    gt_subgoal_xy: np.ndarray,
    gt_subgoal_xz: np.ndarray,
    display_scale: int,
    velocity_scale: float,
) -> np.ndarray:
    camera_bgr = read_camera_bgr(env, "third_person_camera", env_index, display_scale)
    plot_size = camera_bgr.shape[0]
    grasped = teacher_phase >= 3
    topdown_bgr = render_topdown_bgr(
        ee_pos_b,
        goal_pos_b,
        lin_vel,
        goal_phase=1.0 if grasped else 0.0,
        gripper_rate=None,
        plot_size=plot_size,
        velocity_scale=velocity_scale,
        cube_pos=cube_pos_b,
        dist_to_goal=dist_to_goal,
        goal_success_radius_m=args_cli.position_threshold,
        gt_subgoal_xy=gt_subgoal_xy,
    )
    xz_bgr = render_xz_direction_bgr(
        ee_pos_b,
        lin_vel,
        guidance_xz=guidance_xz,
        gt_subgoal_xz=gt_subgoal_xz,
        plot_size=plot_size,
    )
    panel = np.concatenate([camera_bgr, topdown_bgr, xz_bgr], axis=1)
    phase_label = PHASE_NAMES[min(max(teacher_phase, 0), len(PHASE_NAMES) - 1)]
    cv2.putText(
        panel,
        f"oracle teacher + G_theta ep={episode_id} step={episode_step} phase={phase_label}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if guidance_xz is not None:
        g = guidance_xz
        t = gt_subgoal_xz
        cos = float(np.dot(g, t) / (np.linalg.norm(g) * np.linalg.norm(t) + 1e-8))
        cv2.putText(
            panel,
            f"Gθ=({g[0]:+.2f},{g[1]:+.2f}) cos(Gθ,G*)={cos:.2f} dist_goal={dist_to_goal*100:.1f}cm",
            (8, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 0, 200),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        panel,
        "oracle teacher (clean) | orange=teacher vel green=GT purple=Gθ",
        (8, panel.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return panel


def main():
    if args_cli.autoresearch_root:
        import sys

        sys.path.insert(0, args_cli.autoresearch_root)
    from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.guidance_field_bridge import (
        LearnedGuidanceField,
    )

    cam_h = args_cli.camera_height or args_cli.image_size
    cam_w = args_cli.camera_width or args_cli.image_size
    guidance_field = LearnedGuidanceField(args_cli.field_checkpoint, device=args_cli.device)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.episode_length_s = args_cli.episode_length_s
    apply_lift_teacher_student_env_randomization(
        env_cfg,
        randomize_cube=args_cli.randomize_cube,
        randomize_lift_goal=args_cli.randomize_lift_goal,
        fixed_lift_goal=args_cli.fixed_lift_goal and not args_cli.randomize_lift_goal,
        debug_vis_goal=True,
    )
    env_cfg.scene.ee_camera.height = cam_h
    env_cfg.scene.ee_camera.width = cam_w
    env_cfg.scene.third_person_camera.height = cam_h
    env_cfg.scene.third_person_camera.width = cam_w
    if args_cli.topdown_close:
        env_cfg.scene.third_person_camera.offset.pos = (1.15, 0.0, 0.72)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    gripper_action_dim = get_gripper_action_dim(env)
    oracle = LiftCubeOraclePolicy(env, position_threshold=args_cli.position_threshold)
    preview = CollectionPreviewWriter(
        output_dir=args_cli.output_dir,
        video_fps=args_cli.video_fps,
        show_opencv=args_cli.show_opencv,
        window_name="teacher_guidance_preview",
    )

    teacher_kwargs = {
        "control_dt": control_dt,
        "min_lin_speed": args_cli.teacher_min_lin_speed,
        "min_ang_speed": args_cli.teacher_min_ang_speed,
    }

    env.reset()
    oracle.reset()
    reset_teacher_phase_state(env)

    preview_env = 0
    episode_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    episodes_done = 0

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] field: {args_cli.field_checkpoint}")
    print(f"[INFO] camera={cam_h}x{cam_w} topdown_close={args_cli.topdown_close}")
    print(f"[INFO] G_θ input: mask+depth 8ch (ee_mask+ee_depth+tp_mask+tp_depth)")

    while simulation_app.is_running():
        student_obs = env.observation_manager.compute_group("policy")
        with torch.no_grad():
            guidance_xz_t = guidance_field.predict_vectors(student_obs, image_size=args_cli.image_size)
        teacher_phase = infer_teacher_phase(env)
        teacher_twist = compute_teacher_twist_binary_gripper(env, oracle, **teacher_kwargs)
        desired_opening = compute_desired_gripper_opening(env, oracle, teacher_phase=teacher_phase)
        current_opening = get_mean_finger_opening(env)
        env_actions = twist_to_env_action(
            teacher_twist,
            control_dt=control_dt,
            current_opening=current_opening,
            arm_scale=args_cli.arm_scale,
            num_gripper_actions=gripper_action_dim,
            desired_opening=desired_opening,
            binary_gripper=True,
        )
        _, _, terminated, truncated, _ = env.step(env_actions)
        episode_step += 1

        ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
        cube_pos_b = object_position_in_robot_root_frame(env)
        goal_pos_b = get_lift_goal_position_in_robot_root_frame(env)
        dist_goal = float(compute_cube_to_goal_distance(env)[preview_env].item())

        ee_np = ee_pos_b[preview_env].detach().cpu().numpy()
        cube_np = cube_pos_b[preview_env].detach().cpu().numpy()
        goal_np = goal_pos_b[preview_env].detach().cpu().numpy()
        ph = int(teacher_phase[preview_env].item())
        grasped = ph >= 3
        gxz = guidance_xz_t[preview_env].detach().cpu().numpy()

        panel = build_teacher_g_panel(
            env,
            preview_env,
            ee_pos_b=ee_np,
            cube_pos_b=cube_np,
            goal_pos_b=goal_np,
            dist_to_goal=dist_goal,
            lin_vel=teacher_twist[preview_env, :3].detach().cpu().numpy(),
            teacher_phase=ph,
            episode_step=int(episode_step[preview_env].item()),
            episode_id=episodes_done,
            guidance_xz=gxz,
            gt_subgoal_xy=compute_subgoal_gt_xy(ee_np, cube_np, goal_np, grasped=grasped),
            gt_subgoal_xz=compute_subgoal_gt_xz(ee_np, cube_np, goal_np, grasped=grasped),
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
                print(f"[Episode {episodes_done}] steps={episode_step[env_id].item()} phase={PHASE_NAMES[ph]}")
            oracle.reset(done_ids.tolist())
            reset_teacher_phase_state(env, done_ids)
            episode_step[done_ids] = 0

        if episodes_done >= args_cli.max_episodes:
            break

    preview.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
