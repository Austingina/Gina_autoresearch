#!/usr/bin/env python3
"""Check whether the initial wrist/EE camera sees the randomized lift cube.

The cube mask is encoded as white in the colored semantic mask.  This script
resets the IsaacLab lift vision env multiple times with uniform cube
randomization and reports the fraction of initial EE-camera frames containing
enough white pixels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check initial EE-mask cube visibility for lift random cube.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_resets", type=int, default=8)
parser.add_argument("--camera_size", type=int, default=240)
parser.add_argument("--min_white_pixels", type=int, default=25)
parser.add_argument("--cube_x_min", type=float, default=-0.10)
parser.add_argument("--cube_x_max", type=float, default=0.10)
parser.add_argument("--cube_y_min", type=float, default=-0.25)
parser.add_argument("--cube_y_max", type=float, default=0.25)
parser.add_argument("--grid_nx", type=int, default=0, help="If >0, validate fixed cube x grid points instead of random resets.")
parser.add_argument("--grid_ny", type=int, default=0, help="If >0, validate fixed cube y grid points instead of random resets.")
parser.add_argument("--robot_x_offset", type=float, default=0.0)
parser.add_argument("--ee_camera_focal_length", type=float, default=None)
parser.add_argument("--rollout_teacher", action="store_true", default=False)
parser.add_argument("--max_rollout_steps", type=int, default=900)
parser.add_argument("--success_lift_height", type=float, default=0.12)
parser.add_argument("--arm_scale", type=float, default=0.5)
parser.add_argument("--show_opencv", action="store_true", default=False)
parser.add_argument("--show_every_steps", type=int, default=10)
for _joint_id in range(1, 8):
    parser.add_argument(f"--panda_joint{_joint_id}", type=float, default=None)
parser.add_argument(
    "--start_pose",
    choices=["default", "back_high"],
    default="default",
    help="Optional robot initial joint pose candidate for better wrist-camera coverage.",
)
parser.add_argument("--save_panel", type=Path, default=None)
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
from isaaclab_tasks.manager_based.manipulation.lift.mdp.data_collection import get_mean_finger_opening
from isaaclab_tasks.manager_based.manipulation.lift.mdp.oracle import LiftCubeOraclePolicy
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student import (
    compute_teacher_twist_binary_gripper,
    get_gripper_action_dim,
    reset_teacher_phase_state,
    twist_to_env_action,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.env_randomization import (
    apply_lift_teacher_student_env_randomization,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


BACK_HIGH_START_JOINT_POS = {
    # Candidate reset posture: wrist camera starts farther back and high while
    # keeping a downward-looking camera orientation.
    "panda_joint1": 0.0,
    "panda_joint2": -1.10,
    "panda_joint3": 0.0,
    "panda_joint4": -2.45,
    "panda_joint5": 0.0,
    "panda_joint6": 2.45,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.04,
}


def _white_pixel_counts(mask: torch.Tensor) -> torch.Tensor:
    # camera_colored_semantic_mask returns float RGB in [0, 1].  The cube is white.
    rgb = mask[..., :3]
    white = (rgb[..., 0] > 0.9) & (rgb[..., 1] > 0.9) & (rgb[..., 2] > 0.9)
    return white.flatten(start_dim=1).sum(dim=1)


def _mask_panel(mask: torch.Tensor, counts: torch.Tensor, cols: int = 4) -> np.ndarray:
    imgs = []
    for i in range(mask.shape[0]):
        rgb = (mask[i, ..., :3].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"white={int(counts[i].item())}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        imgs.append(bgr)
    rows = []
    for start in range(0, len(imgs), cols):
        chunk = imgs[start : start + cols]
        if len(chunk) < cols:
            chunk += [np.zeros_like(imgs[0]) for _ in range(cols - len(chunk))]
        rows.append(np.concatenate(chunk, axis=1))
    return np.concatenate(rows, axis=0)


def _show_ee_mask(mask: torch.Tensor, title: str, wait_ms: int = 1) -> bool:
    counts = _white_pixel_counts(mask)
    panel = _mask_panel(mask, counts, cols=max(1, min(4, mask.shape[0])))
    cv2.putText(panel, title, (8, panel.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imshow("lift ee_mask", panel)
    key = cv2.waitKey(wait_ms) & 0xFF
    return key in (ord("q"), 27)


def _rollout_teacher(env, oracle: LiftCubeOraclePolicy, control_dt: float, gripper_action_dim: int) -> dict[str, torch.Tensor]:
    init_cube_pos = env.scene["object"].data.root_pos_w.clone()
    max_height_gain = torch.zeros(env.num_envs, device=init_cube_pos.device)
    max_xy_drift = torch.zeros(env.num_envs, device=init_cube_pos.device)
    oracle.reset()
    reset_teacher_phase_state(env)
    should_stop = False
    for step in range(args_cli.max_rollout_steps):
        with torch.no_grad():
            teacher_twist = compute_teacher_twist_binary_gripper(env, oracle, control_dt=control_dt)
            current_opening = get_mean_finger_opening(env)
            actions = twist_to_env_action(
                teacher_twist,
                control_dt=control_dt,
                current_opening=current_opening,
                arm_scale=args_cli.arm_scale,
                num_gripper_actions=gripper_action_dim,
                binary_gripper=True,
            )
            env.step(actions)
            cube_pos = env.scene["object"].data.root_pos_w
            height_gain = cube_pos[:, 2] - init_cube_pos[:, 2]
            xy_drift = torch.linalg.norm(cube_pos[:, :2] - init_cube_pos[:, :2], dim=-1)
            max_height_gain = torch.maximum(max_height_gain, height_gain)
            max_xy_drift = torch.maximum(max_xy_drift, xy_drift)
            if args_cli.show_opencv and step % max(1, args_cli.show_every_steps) == 0:
                env.sim.render()
                ee_mask = mdp.camera_colored_semantic_mask(env, sensor_cfg=SceneEntityCfg("ee_camera"))
                should_stop = _show_ee_mask(ee_mask, f"teacher rollout step={step}", wait_ms=1)
                if should_stop:
                    break
            if torch.all(max_height_gain >= args_cli.success_lift_height):
                break
    final_cube_pos = env.scene["object"].data.root_pos_w.clone()
    final_height_gain = final_cube_pos[:, 2] - init_cube_pos[:, 2]
    success = max_height_gain >= args_cli.success_lift_height
    return {
        "success": success.detach().cpu(),
        "max_height_gain": max_height_gain.detach().cpu(),
        "final_height_gain": final_height_gain.detach().cpu(),
        "max_xy_drift": max_xy_drift.detach().cpu(),
    }


def main() -> int:
    env_cfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-Vision-Teacher-Student-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = 11
    env_cfg.scene.robot.init_state.pos = (args_cli.robot_x_offset, 0.0, 0.0)
    apply_vision_camera_settings(
        env_cfg,
        camera_height=args_cli.camera_size,
        camera_width=args_cli.camera_size,
        topdown_close=True,
    )
    if args_cli.ee_camera_focal_length is not None:
        env_cfg.scene.ee_camera.spawn.focal_length = args_cli.ee_camera_focal_length
    apply_lift_teacher_student_env_randomization(
        env_cfg,
        randomize_cube=True,
        lift_goal_above_cube=True,
        cube_pose_range={
            "x": (args_cli.cube_x_min, args_cli.cube_x_max),
            "y": (args_cli.cube_y_min, args_cli.cube_y_max),
            "z": (0.0, 0.0),
        },
        debug_vis_goal=False,
    )
    if args_cli.start_pose == "back_high":
        env_cfg.scene.robot.init_state.joint_pos = BACK_HIGH_START_JOINT_POS.copy()
    joint_overrides = {
        f"panda_joint{joint_id}": getattr(args_cli, f"panda_joint{joint_id}")
        for joint_id in range(1, 8)
        if getattr(args_cli, f"panda_joint{joint_id}") is not None
    }
    if joint_overrides:
        joint_pos = dict(env_cfg.scene.robot.init_state.joint_pos)
        joint_pos.update(joint_overrides)
        env_cfg.scene.robot.init_state.joint_pos = joint_pos
        print(f"[INFO] robot joint overrides: {joint_overrides}")

    env = gym.make("Isaac-Lift-Cube-Franka-Vision-Teacher-Student-v0", cfg=env_cfg).unwrapped
    oracle = LiftCubeOraclePolicy(env) if args_cli.rollout_teacher else None
    gripper_action_dim = get_gripper_action_dim(env) if args_cli.rollout_teacher else 0
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    all_counts = []
    all_visible = []
    all_teacher_success = []
    all_teacher_max_height = []
    all_teacher_final_height = []
    all_teacher_xy_drift = []
    first_panel = None
    grid_points = None
    if args_cli.grid_nx > 0 or args_cli.grid_ny > 0:
        grid_nx = max(1, args_cli.grid_nx)
        grid_ny = max(1, args_cli.grid_ny)
        xs = np.linspace(args_cli.cube_x_min, args_cli.cube_x_max, grid_nx)
        ys = np.linspace(args_cli.cube_y_min, args_cli.cube_y_max, grid_ny)
        grid_points = [(float(x), float(y)) for x in xs for y in ys]
        args_cli.num_resets = len(grid_points)
        print(f"[INFO] Grid validation: {grid_nx}x{grid_ny} fixed cube positions")

    reset_object_cfg = env.event_manager.get_term_cfg("reset_object_position")
    for reset_id in range(args_cli.num_resets):
        if grid_points is not None:
            cube_x, cube_y = grid_points[reset_id]
            reset_object_cfg.params["pose_range"] = {
                "x": (cube_x, cube_x),
                "y": (cube_y, cube_y),
                "z": (0.0, 0.0),
            }
            env.event_manager.set_term_cfg("reset_object_position", reset_object_cfg)
        env.reset()
        for _ in range(max(1, getattr(env_cfg, "num_rerenders_on_reset", 1))):
            env.sim.render()
        ee_mask = mdp.camera_colored_semantic_mask(env, sensor_cfg=SceneEntityCfg("ee_camera"))
        ee_pos = env.scene["robot"].data.body_pos_w[:, env.scene["robot"].data.body_names.index("panda_hand")]
        cube_pos = env.scene["object"].data.root_pos_w
        cube_pos_local = cube_pos - env.scene.env_origins
        counts = _white_pixel_counts(ee_mask)
        visible = counts >= args_cli.min_white_pixels
        all_counts.append(counts.detach().cpu())
        all_visible.append(visible.detach().cpu())
        if reset_id == 0:
            first_panel = _mask_panel(ee_mask, counts)
        if args_cli.show_opencv:
            if _show_ee_mask(ee_mask, f"initial reset={reset_id}", wait_ms=300):
                break
        print(
            f"[RESET {reset_id:03d}] visible={int(visible.sum().item())}/{args_cli.num_envs} "
            f"min={int(counts.min().item())} median={float(counts.float().median().item()):.1f} "
            f"max={int(counts.max().item())} "
            f"mean_ee=({ee_pos[:,0].mean().item():.3f},{ee_pos[:,1].mean().item():.3f},{ee_pos[:,2].mean().item():.3f}) "
            f"cube_xy_local=({cube_pos_local[:,0].mean().item():+.3f},{cube_pos_local[:,1].mean().item():+.3f}) "
            f"cube_y_local=({cube_pos_local[:,1].min().item():+.3f},{cube_pos_local[:,1].max().item():+.3f})"
        )
        if args_cli.rollout_teacher:
            rollout = _rollout_teacher(env, oracle, control_dt, gripper_action_dim)
            all_teacher_success.append(rollout["success"])
            all_teacher_max_height.append(rollout["max_height_gain"])
            all_teacher_final_height.append(rollout["final_height_gain"])
            all_teacher_xy_drift.append(rollout["max_xy_drift"])
            success = rollout["success"]
            max_height = rollout["max_height_gain"]
            final_height = rollout["final_height_gain"]
            xy_drift = rollout["max_xy_drift"]
            print(
                f"[TEACHER {reset_id:03d}] success={int(success.sum().item())}/{args_cli.num_envs} "
                f"max_h_median={float(max_height.float().median().item()):.3f} "
                f"max_h_min={float(max_height.min().item()):.3f} "
                f"final_h_median={float(final_height.float().median().item()):.3f} "
                f"xy_drift_median={float(xy_drift.float().median().item()):.3f}"
            )

    counts_cat = torch.cat(all_counts)
    visible_cat = torch.cat(all_visible)
    rate = float(visible_cat.float().mean().item())
    print(
        f"[RESULT] start_pose={args_cli.start_pose} cube_x=({args_cli.cube_x_min},{args_cli.cube_x_max}) "
        f"cube_y=({args_cli.cube_y_min},{args_cli.cube_y_max}) robot_x_offset={args_cli.robot_x_offset} "
        f"grid={args_cli.grid_nx}x{args_cli.grid_ny} "
        f"ee_camera_focal_length={args_cli.ee_camera_focal_length} visibility_rate={rate:.3f} "
        f"min_white={int(counts_cat.min().item())} median_white={float(counts_cat.float().median().item()):.1f} "
        f"max_white={int(counts_cat.max().item())}"
    )
    teacher_rate = None
    if args_cli.rollout_teacher:
        teacher_success_cat = torch.cat(all_teacher_success)
        teacher_max_height_cat = torch.cat(all_teacher_max_height)
        teacher_final_height_cat = torch.cat(all_teacher_final_height)
        teacher_xy_drift_cat = torch.cat(all_teacher_xy_drift)
        teacher_rate = float(teacher_success_cat.float().mean().item())
        print(
            f"[TEACHER_RESULT] success_rate={teacher_rate:.3f} "
            f"success_lift_height={args_cli.success_lift_height:.3f} "
            f"max_height_min={float(teacher_max_height_cat.min().item()):.3f} "
            f"max_height_median={float(teacher_max_height_cat.float().median().item()):.3f} "
            f"final_height_median={float(teacher_final_height_cat.float().median().item()):.3f} "
            f"xy_drift_median={float(teacher_xy_drift_cat.float().median().item()):.3f}"
        )
    if args_cli.save_panel is not None and first_panel is not None:
        args_cli.save_panel.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args_cli.save_panel), first_panel)
        print(f"[INFO] Saved panel: {args_cli.save_panel}")
    if args_cli.show_opencv:
        cv2.destroyAllWindows()
    env.close()
    if rate < 0.99:
        return 2
    if teacher_rate is not None and teacher_rate < 0.99:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
