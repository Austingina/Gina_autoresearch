# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visualize a trained vision DAgger dual-head binary-gripper student checkpoint.

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/play_lift_cube_student_vision_binary_gripper.py \\
        --checkpoint ./runs/lift_cube_dagger_vision/student_best.pt \\
        --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play/visualize vision DAgger dual-head binary-gripper lift-cube student."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-Vision-Teacher-Student-v0",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./runs/lift_cube_dagger_vision/student_best.pt",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--image_size", type=int, default=64)
parser.add_argument("--camera_height", type=int, default=0)
parser.add_argument("--camera_width", type=int, default=0)
parser.add_argument("--topdown_close", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--randomize_cube_z_only", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--cube_z_min", type=float, default=0.0)
parser.add_argument("--cube_z_max", type=float, default=0.0)
parser.add_argument("--cube_x_min", type=float, default=-0.1)
parser.add_argument("--cube_x_max", type=float, default=0.1)
parser.add_argument("--cube_y_min", type=float, default=-0.25)
parser.add_argument("--cube_y_max", type=float, default=0.25)
parser.add_argument("--position_threshold", type=float, default=0.025)
parser.add_argument("--arm_scale", type=float, default=1.0)
parser.add_argument("--max_episodes", type=int, default=20)
parser.add_argument("--episode_length_s", type=float, default=25.0)
parser.add_argument("--max_preview_steps", type=int, default=0)
parser.add_argument("--loop", action="store_true", default=False)
parser.add_argument("--display_scale", type=int, default=4)
parser.add_argument("--velocity_scale", type=float, default=0.35)
parser.add_argument("--output_dir", type=str, default="./runs/student_vision_binary_gripper_preview")
parser.add_argument("--video_fps", type=int, default=20)
parser.add_argument("--show_opencv", action="store_true", default=False)
parser.add_argument(
    "--show_mask_depth",
    action="store_true",
    default=False,
    help="Show ee/top mask+depth in a second OpenCV window (requires --show_opencv).",
)
parser.add_argument("--fixed_lift_goal", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--lift_goal_x", type=float, default=0.5)
parser.add_argument("--lift_goal_y", type=float, default=0.0)
parser.add_argument("--lift_goal_z", type=float, default=0.375)
parser.add_argument("--lift_goal_above_cube", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--lift_height_above_cube", type=float, default=0.20)
parser.add_argument("--randomize_cube", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--randomize_lift_goal", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    "--robot_x_offset",
    type=float,
    default=-0.10,
    help="Initial robot base x offset. Negative values move the EE camera farther back for cube visibility.",
)
parser.add_argument(
    "--panda_joint6",
    type=float,
    default=2.7,
    help="Initial panda_joint6 override for wrist-camera visibility. Pass nothing via play_student ROBOT defaults.",
)
parser.add_argument(
    "--guidance_mode",
    type=str,
    default="anchor_only",
    choices=("anchor_only", "anchor_field_demo", "anchor_field_rule"),
)
parser.add_argument("--field_checkpoint", type=str, default="")
parser.add_argument("--autoresearch_root", type=str, default=os.environ.get("AUTORESEARCH_ROOT", ""))
parser.add_argument(
    "--mask_palette_mode",
    type=str,
    default="checkpoint",
    choices=("checkpoint", "random", "fixed"),
    help="Semantic mask palette for non-cube classes; checkpoint uses saved training metadata when present.",
)
parser.add_argument("--mask_robot_color", type=float, nargs=3, default=(0.10, 0.35, 0.95))
parser.add_argument("--mask_table_color", type=float, nargs=3, default=(0.55, 0.35, 0.18))
parser.add_argument("--mask_ground_color", type=float, nargs=3, default=(0.10, 0.10, 0.10))
parser.add_argument(
    "--debug_save_mosaic_dir",
    type=str,
    default="",
    help="If set, save (4,2H,W) mosaic panels + .pt before policy inference (mosaic layout only).",
)
parser.add_argument("--debug_save_mosaic_interval", type=int, default=1)
parser.add_argument("--debug_save_mosaic_max", type=int, default=500)
parser.add_argument("--debug_save_mosaic_env_id", type=int, default=0)
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
from isaaclab_tasks.manager_based.manipulation.lift.mdp.observations import object_position_in_robot_root_frame
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student import (
    StudentTwistPolicyBinaryGripperVision,
    apply_lift_teacher_student_env_randomization,
    extract_proprio_from_policy_obs_torch,
    get_gripper_action_dim,
    infer_teacher_phase,
    reset_teacher_phase_state,
    twist_to_env_action,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.vision_mosaic_debug import (
    MosaicPolicyInputDumper,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.vision_policy_obs import (
    VISUAL_LAYOUT_CHANNELS,
    VISUAL_LAYOUT_MOSAIC,
    VisualMosaicHistory,
    stack_policy_visual_torch,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.vision_observations import (
    set_fixed_colored_mask_palette,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.sampler_logging import (
    ORACLE_SM_STATE_NAMES,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.vision_teacher_student_env_cfg import (
    apply_vision_camera_settings,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.env_randomization import (
    cube_pose_range_z_only,
)

from oracle_demo_preview import CollectionPreviewWriter, read_camera_bgr, render_topdown_bgr, render_xz_direction_bgr

PHASE_NAMES = ORACLE_SM_STATE_NAMES


def _unit2(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float64)
    return (vec / norm).astype(np.float64)


def compute_subgoal_gt_xy(ee_pos_b: np.ndarray, cube_pos_b: np.ndarray, goal_pos_b: np.ndarray, *, grasped: bool) -> np.ndarray:
    if not grasped:
        return _unit2(np.array([cube_pos_b[0] - ee_pos_b[0], cube_pos_b[1] - ee_pos_b[1]], dtype=np.float64))
    return _unit2(np.array([goal_pos_b[0] - cube_pos_b[0], goal_pos_b[1] - cube_pos_b[1]], dtype=np.float64))


def compute_subgoal_gt_xz(ee_pos_b: np.ndarray, cube_pos_b: np.ndarray, goal_pos_b: np.ndarray, *, grasped: bool) -> np.ndarray:
    if not grasped:
        return _unit2(np.array([cube_pos_b[0] - ee_pos_b[0], cube_pos_b[2] - ee_pos_b[2]], dtype=np.float64))
    return _unit2(np.array([goal_pos_b[0] - cube_pos_b[0], goal_pos_b[2] - cube_pos_b[2]], dtype=np.float64))

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


def _validate_vision_student_checkpoint(checkpoint_path: str, ckpt: dict) -> None:
    if "model" not in ckpt and "agent" in ckpt:
        raise ValueError(
            f"Checkpoint looks like an offline CQL-SAC LSTM checkpoint, not a vision DAgger student "
            f"(missing key 'model', found 'agent'). Use:\n"
            f"  CHECKPOINT={checkpoint_path} ./scripts/play_lift_cube_offline_cql_lstm.sh"
        )
    if "model" not in ckpt:
        raise KeyError(
            f"Unsupported checkpoint format in {checkpoint_path}. "
            f"Expected vision DAgger keys like 'model'; found: {sorted(ckpt.keys())}"
        )
    policy_type = str(ckpt.get("policy_type", "binary_gripper_dual_head_vision_dagger"))
    if policy_type in PROPRIO_STUDENT_POLICY_TYPES:
        raise ValueError(
            f"Checkpoint policy_type={policy_type!r} is a proprio student (LSTM on student_state), "
            f"not vision DAgger. Use:\n"
            f"  CHECKPOINT={checkpoint_path} ./scripts/play_lift_cube_student_binary_gripper.sh"
        )
    if policy_type not in VISION_STUDENT_POLICY_TYPES:
        raise ValueError(
            f"Unsupported vision checkpoint policy_type={policy_type!r} in {checkpoint_path}. "
            f"Expected one of {sorted(VISION_STUDENT_POLICY_TYPES)}."
        )


def load_vision_student_policy(checkpoint_path: str, device: str) -> tuple[StudentTwistPolicyBinaryGripperVision, dict]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _validate_vision_student_checkpoint(checkpoint_path, ckpt)
    visual_channels = StudentTwistPolicyBinaryGripperVision.resolve_visual_channels_from_checkpoint(ckpt)
    enable_aux_delta_head = StudentTwistPolicyBinaryGripperVision.resolve_aux_delta_head_from_checkpoint(ckpt)
    proprio_dim = int(ckpt.get("proprio_dim", 0))
    image_size = int(ckpt.get("image_size", args_cli.image_size))
    policy = StudentTwistPolicyBinaryGripperVision(
        visual_channels=visual_channels,
        proprio_dim=proprio_dim,
        enable_aux_delta_head=enable_aux_delta_head,
    ).to(device)
    StudentTwistPolicyBinaryGripperVision.load_from_checkpoint(
        policy, ckpt, visual_channels=visual_channels, proprio_dim=proprio_dim
    )
    policy.eval()
    ckpt["_resolved_image_size"] = image_size
    return policy, ckpt


def _apply_mask_palette_settings(env_cfg, checkpoint: dict) -> None:
    mode = args_cli.mask_palette_mode
    if mode == "checkpoint":
        mode = str(checkpoint.get("mask_palette_mode", "random"))
    if mode != "fixed":
        print("[INFO] Mask palette: random robot/table/ground colors.")
        return
    robot_color = tuple(checkpoint.get("mask_robot_color", args_cli.mask_robot_color))
    table_color = tuple(checkpoint.get("mask_table_color", args_cli.mask_table_color))
    ground_color = tuple(checkpoint.get("mask_ground_color", args_cli.mask_ground_color))
    env_cfg.events.randomize_mask_palette.func = set_fixed_colored_mask_palette
    env_cfg.events.randomize_mask_palette.params = {
        "robot_color": tuple(float(x) for x in robot_color),
        "table_color": tuple(float(x) for x in table_color),
        "ground_color": tuple(float(x) for x in ground_color),
    }
    print(
        f"[INFO] Mask palette fixed: robot={robot_color} table={table_color} "
        f"ground={ground_color} cube=(1, 1, 1)"
    )


def _obs_term_to_bgr(
    obs_tensor: torch.Tensor,
    env_index: int,
    *,
    is_depth: bool,
    display_scale: int,
) -> np.ndarray:
    arr = obs_tensor[env_index].detach().cpu().numpy()
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]

    if is_depth:
        depth = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
        depth = np.clip(depth.astype(np.float32), 0.0, None)
        depth_max = float(depth.max())
        if depth_max < 1.0e-6:
            depth_max = 1.0
        gray = (np.clip(depth / depth_max, 0.0, 1.0) * 255.0).astype(np.uint8)
        bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    elif arr.shape[-1] >= 3:
        rgb_f = np.clip(arr[..., :3].astype(np.float32), 0.0, None)
        if float(rgb_f.max()) <= 1.0 + 1.0e-3:
            rgb = (rgb_f * 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb_f, 0.0, 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        gray = (np.clip(arr[..., 0], 0.0, 1.0) * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if display_scale > 1:
        bgr = cv2.resize(
            bgr,
            (bgr.shape[1] * display_scale, bgr.shape[0] * display_scale),
            interpolation=cv2.INTER_NEAREST,
        )
    return bgr


def build_mask_depth_preview_panel(
    student_obs: dict[str, torch.Tensor],
    env_index: int,
    *,
    display_scale: int,
    visual_layout: str = VISUAL_LAYOUT_CHANNELS,
) -> np.ndarray:
    tiles_spec = [
        ("ee_mask", "ee mask", False),
        ("ee_depth", "ee depth", True),
        ("third_person_mask", "top mask", False),
        ("third_person_depth", "top depth", True),
    ]
    labeled: list[np.ndarray] = []
    for obs_key, label, is_depth in tiles_spec:
        tile = _obs_term_to_bgr(student_obs[obs_key], env_index, is_depth=is_depth, display_scale=display_scale)
        arr = student_obs[obs_key][env_index].detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        stat_txt = f"min={float(arr.min()):.3f} max={float(arr.max()):.3f} mean={float(arr.mean()):.3f}"
        cv2.putText(
            tile,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            stat_txt,
            (8, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 255, 200),
            1,
            cv2.LINE_AA,
        )
        labeled.append(tile)
    top = np.concatenate(labeled[:2], axis=1)
    bottom = np.concatenate(labeled[2:], axis=1)
    panel = np.concatenate([top, bottom], axis=0)
    if visual_layout == VISUAL_LAYOUT_MOSAIC:
        cv2.putText(
            panel,
            "policy input: mosaic (4,2H,W) ee/top mask+depth, single frame",
            (8, panel.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return panel


def build_student_preview_panel(
    env,
    env_index: int,
    *,
    ckpt_label: str,
    ee_pos_b: np.ndarray,
    cube_pos_b: np.ndarray,
    goal_pos_b: np.ndarray,
    dist_to_goal: float,
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
    guidance_xz: np.ndarray | None = None,
    gt_subgoal_xy: np.ndarray | None = None,
    gt_subgoal_xz: np.ndarray | None = None,
) -> np.ndarray:
    if guidance_xz is not None:
        guidance_xz = np.asarray(guidance_xz, dtype=np.float64)
        if guidance_xz.shape[0] >= 3:
            guidance_xz = guidance_xz[[0, 2]]
    if gt_subgoal_xz is not None:
        gt_subgoal_xz = np.asarray(gt_subgoal_xz, dtype=np.float64)
        if gt_subgoal_xz.shape[0] >= 3:
            gt_subgoal_xz = gt_subgoal_xz[[0, 2]]
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

    status = "LIFT OK" if success_lift else ("GRASP OK" if success_grasp else "running")
    color = (0, 220, 0) if success_lift else (0, 180, 255) if success_grasp else (255, 255, 255)
    grip_label = "CLOSE" if gripper_cmd < 0.0 else "OPEN"
    phase_label = PHASE_NAMES[min(max(teacher_phase, 0), len(PHASE_NAMES) - 1)]
    cv2.putText(
        panel,
        f"vision DAgger {ckpt_label} ep={episode_id} step={episode_step} {status}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"grip={grip_label} p={gripper_prob:.2f} dist_goal={dist_to_goal * 100:.1f}cm "
        f"phase={phase_label} |v|={np.linalg.norm(lin_vel):.3f}",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "top: orange=policy green=GT(xy) | right: orange=policy purple=Gθ green=G*(xz)",
        (8, panel.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    if guidance_xz is not None:
        cv2.putText(
            panel,
            f"Gθ=({guidance_xz[0]:+.2f},{guidance_xz[1]:+.2f})",
            (8, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 0, 200),
            1,
            cv2.LINE_AA,
        )
    return panel


def main():
    guidance_field = None
    guidance_mode = args_cli.guidance_mode
    ckpt_preview = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    inferred_visual = StudentTwistPolicyBinaryGripperVision.resolve_visual_channels_from_checkpoint(ckpt_preview)
    if inferred_visual > 8 and guidance_mode == "anchor_only":
        if inferred_visual == 11:
            guidance_mode = "anchor_field_rule"
            print(
                f"[INFO] Checkpoint has {inferred_visual} visual channels; "
                "auto-enabling guidance_mode=anchor_field_rule"
            )
        else:
            guidance_mode = "anchor_field_demo"
            print(
                f"[INFO] Checkpoint has {inferred_visual} visual channels; "
                "auto-enabling guidance_mode=anchor_field_demo"
            )
    if guidance_mode == "anchor_field_demo":
        field_ckpt = args_cli.field_checkpoint
        if not field_ckpt:
            root = args_cli.autoresearch_root or "/home/l/code/AutoResearch"
            field_ckpt = os.path.join(root, "checkpoints/lift/guidance_field_s0_fixed.pt")
        if not os.path.isfile(field_ckpt):
            raise ValueError(f"--field_checkpoint required for B7 play; missing {field_ckpt}")
        if args_cli.autoresearch_root:
            import sys

            sys.path.insert(0, args_cli.autoresearch_root)
        from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.guidance_field_bridge import (
            LearnedGuidanceField,
        )

        guidance_field = LearnedGuidanceField(field_ckpt, device=args_cli.device)
    elif guidance_mode == "anchor_field_rule":
        from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.guidance_field_bridge import (
            OracleRuleGuidanceField,
        )

        guidance_field = OracleRuleGuidanceField(device=args_cli.device)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.actions.arm_action.scale = args_cli.arm_scale
    _apply_mask_palette_settings(env_cfg, ckpt_preview)
    if abs(args_cli.robot_x_offset) > 1.0e-9:
        env_cfg.scene.robot.init_state.pos = (args_cli.robot_x_offset, 0.0, 0.0)
        print(f"[INFO] robot init_state.pos=({args_cli.robot_x_offset}, 0.0, 0.0)")
    if args_cli.panda_joint6 is not None:
        joint_pos = dict(env_cfg.scene.robot.init_state.joint_pos)
        joint_pos["panda_joint6"] = args_cli.panda_joint6
        env_cfg.scene.robot.init_state.joint_pos = joint_pos
        print(f"[INFO] robot init panda_joint6={args_cli.panda_joint6}")
    if args_cli.randomize_lift_goal and args_cli.fixed_lift_goal:
        print("[WARN] --randomize_lift_goal overrides --fixed_lift_goal.")
    for key in ("cube_x_min", "cube_x_max", "cube_y_min", "cube_y_max"):
        if key in ckpt_preview:
            setattr(args_cli, key, float(ckpt_preview[key]))
            print(f"[INFO] cube pose range from checkpoint: {key}={getattr(args_cli, key)}")
    cube_pose_range = None
    if args_cli.randomize_cube_z_only:
        cube_pose_range = cube_pose_range_z_only(z_min=args_cli.cube_z_min, z_max=args_cli.cube_z_max)
    elif args_cli.randomize_cube:
        from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.env_randomization import (
            cube_pose_range_xy,
        )

        cube_pose_range = cube_pose_range_xy(
            x_min=args_cli.cube_x_min,
            x_max=args_cli.cube_x_max,
            y_min=args_cli.cube_y_min,
            y_max=args_cli.cube_y_max,
        )
    apply_lift_teacher_student_env_randomization(
        env_cfg,
        randomize_cube=args_cli.randomize_cube or args_cli.randomize_cube_z_only,
        randomize_lift_goal=args_cli.randomize_lift_goal,
        fixed_lift_goal=args_cli.fixed_lift_goal and not args_cli.randomize_lift_goal,
        lift_goal_x=args_cli.lift_goal_x,
        lift_goal_y=args_cli.lift_goal_y,
        lift_goal_z=args_cli.lift_goal_z,
        lift_goal_above_cube=args_cli.lift_goal_above_cube,
        lift_height_above_cube=args_cli.lift_height_above_cube,
        cube_pose_range=cube_pose_range,
        debug_vis_goal=True,
    )
    cam_h = args_cli.camera_height or args_cli.image_size
    cam_w = args_cli.camera_width or args_cli.image_size
    apply_vision_camera_settings(
        env_cfg,
        camera_height=cam_h,
        camera_width=cam_w,
        topdown_close=args_cli.topdown_close,
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    control_dt = env_cfg.sim.dt * env_cfg.decimation
    gripper_action_dim = get_gripper_action_dim(env)
    policy, ckpt = load_vision_student_policy(args_cli.checkpoint, args_cli.device)
    image_size = int(ckpt.get("_resolved_image_size", args_cli.image_size))
    visual_layout = str(ckpt.get("visual_layout", "channels"))
    mosaic_history_len = int(ckpt.get("mosaic_history_len", 1))
    preview = CollectionPreviewWriter(
        output_dir=args_cli.output_dir,
        video_fps=args_cli.video_fps,
        show_opencv=args_cli.show_opencv,
        window_name="student_vision_binary_gripper_preview",
    )
    mask_depth_preview = None
    if args_cli.show_mask_depth:
        if not args_cli.show_opencv:
            print("[WARN] --show_mask_depth requires --show_opencv; enabling OpenCV preview.")
        mask_depth_dir = os.path.join(args_cli.output_dir, "mask_depth")
        mask_depth_preview = CollectionPreviewWriter(
            output_dir=mask_depth_dir,
            video_fps=args_cli.video_fps,
            show_opencv=True,
            window_name="student_mask_depth_preview",
        )

    ckpt_label = os.path.basename(args_cli.checkpoint)
    train_success = ckpt.get("success_rate")
    train_iter = ckpt.get("iteration")
    obs_mode = ckpt.get("obs_mode", "vision")

    env.reset()

    num_envs = env.num_envs
    mosaic_history = None
    if visual_layout == VISUAL_LAYOUT_MOSAIC:
        mosaic_history = VisualMosaicHistory(
            num_envs,
            image_size,
            env.device,
            history_len=mosaic_history_len,
        )
    mosaic_debug: MosaicPolicyInputDumper | None = None
    if visual_layout == VISUAL_LAYOUT_MOSAIC and args_cli.debug_save_mosaic_dir:
        mosaic_debug_dir = os.path.abspath(args_cli.debug_save_mosaic_dir)
        mosaic_debug = MosaicPolicyInputDumper(
            mosaic_debug_dir,
            image_size,
            interval=args_cli.debug_save_mosaic_interval,
            max_saves=args_cli.debug_save_mosaic_max,
            env_ids=(args_cli.debug_save_mosaic_env_id,),
        )
        print(
            f"[INFO] Mosaic debug dump: dir={mosaic_debug_dir} "
            f"interval={args_cli.debug_save_mosaic_interval} max={args_cli.debug_save_mosaic_max}"
        )
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
    print(f"[INFO] obs_mode={obs_mode} image_size={image_size} proprio_dim={ckpt.get('proprio_dim', 0)}")
    print(f"[INFO] guidance_mode={guidance_mode} visual_channels={policy.visual_channels}")
    print(f"[INFO] visual_layout={visual_layout} image_size={image_size}")
    if train_iter is not None:
        print(f"[INFO] Trained iteration: {train_iter}")
    if train_success is not None:
        print(f"[INFO] Checkpoint eval success_rate: {train_success:.3f}")
    print(f"[INFO] Preview: {preview.latest_path}")

    while simulation_app.is_running():
        with torch.no_grad():
            student_obs = env.observation_manager.compute_group("policy")
            visual = stack_policy_visual_torch(
                student_obs,
                image_size=image_size,
                visual_layout=visual_layout,
                mosaic_history=mosaic_history,
            )
            if mosaic_debug is not None:
                mosaic_debug.maybe_save(visual, phase="play")
            guidance_xz = None
            if guidance_field is not None:
                teacher_phase = infer_teacher_phase(env)
                if guidance_mode == "anchor_field_rule":
                    guidance_vec = guidance_field.predict_vectors(env).detach().cpu().numpy()
                    guidance_xz = guidance_vec[preview_env, [0, 2]]
                else:
                    guidance_xz_t = guidance_field.predict_vectors(
                        student_obs,
                        image_size=image_size,
                        env=env,
                        teacher_phase=teacher_phase,
                    )
                    guidance_xz = guidance_xz_t[preview_env].detach().cpu().numpy()
                visual = guidance_field.augment_visual(
                    visual,
                    student_obs,
                    image_size=image_size,
                    env=env,
                    teacher_phase=teacher_phase,
                )
            visual_step = visual.unsqueeze(1)
            proprio = extract_proprio_from_policy_obs_torch(student_obs)
            proprio_step = None if proprio is None else proprio.unsqueeze(1)
            arm_dist, gripper_logit, rnn_hidden = policy.policy_dist(visual_step, rnn_hidden, proprio_step)
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
        cube_pos_b = object_position_in_robot_root_frame(env)
        grasp_ok = success_tracker.ever_grasp_success
        lift_ok = success_tracker.ever_lift_success
        goal_pos_b = get_lift_goal_position_in_robot_root_frame(env)
        dist_goal = float(pre_cube_to_goal[preview_env].item())
        ee_np = ee_pos_b[preview_env].detach().cpu().numpy()
        cube_np = cube_pos_b[preview_env].detach().cpu().numpy()
        goal_np = goal_pos_b[preview_env].detach().cpu().numpy()
        grasped = bool(grasp_ok[preview_env].item())
        gt_xy = compute_subgoal_gt_xy(ee_np, cube_np, goal_np, grasped=grasped)
        gt_xz = compute_subgoal_gt_xz(ee_np, cube_np, goal_np, grasped=grasped)

        panel = build_student_preview_panel(
            env,
            preview_env,
            ckpt_label=ckpt_label,
            ee_pos_b=ee_np,
            cube_pos_b=cube_np,
            goal_pos_b=goal_np,
            dist_to_goal=dist_goal,
            lin_vel=twist[preview_env, :3].detach().cpu().numpy(),
            gripper_cmd=float(twist[preview_env, 6].item()),
            gripper_prob=float(gripper_prob[preview_env].item()),
            teacher_phase=int(teacher_phase[preview_env].item()),
            episode_step=int(episode_step[preview_env].item()),
            episode_id=episodes_done,
            success_lift=bool(lift_ok[preview_env].item()),
            success_grasp=grasped,
            display_scale=args_cli.display_scale,
            velocity_scale=args_cli.velocity_scale,
            guidance_xz=guidance_xz,
            gt_subgoal_xy=gt_xy,
            gt_subgoal_xz=gt_xz,
        )
        key = preview.write(panel)
        if mask_depth_preview is not None:
            mask_depth_panel = build_mask_depth_preview_panel(
                student_obs,
                preview_env,
                display_scale=args_cli.display_scale,
                visual_layout=visual_layout,
            )
            md_key = mask_depth_preview.write(mask_depth_panel)
            if md_key == ord("q"):
                key = md_key
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
            if mosaic_history is not None:
                mosaic_history.reset(done_ids)
            reset_teacher_phase_state(env, done_ids.tolist())
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
    if mask_depth_preview is not None:
        mask_depth_preview.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
