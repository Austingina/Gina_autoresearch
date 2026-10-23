# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Record lift-cube demonstrations with lenient teacher labels and dual-camera RGB+mask+depth observations.

Each exported episode contains:
- obs: ee_rgb, ee_mask, ee_depth, third_person_rgb, third_person_mask, third_person_depth, student_state  (s_t)
- env_actions: noisy IK-Rel actions actually executed  (a_t, for CQL)
- actions: lenient teacher 7D twist (expert label, pre-noise)
- rewards/progress_reward: PROGRESS_REWARD_SCALE * Δprogress  (r_t for CQL; terminal bonus logged only)
- rewards/progress, reach_score, grasp_score, lift_score, goal_score, success_score
- ee_pose, goals (position, phase), rewards (grasp, lift), states, initial_state

CQL transition tuple per step t:
    (s_t, a_t, r_t, s_{t+1}) = (obs[t], env_actions[t], rewards/progress_reward[t], obs[t+1])

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/lenient_teacher_datacollection_maskdepth.py \\
        --num_envs 32 \\
        --num_demos 500 \\
        --dataset_file ./datasets/lift_cube_lenient_teacher_maskdepth.hdf5
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Record lenient-teacher lift-cube demos with GT mask+depth cameras."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-Lenient-Teacher-MaskDepth-Data-v0",
    help="Vision mask+depth data-collection task.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/lift_cube_lenient_teacher_maskdepth.hdf5",
    help="Output HDF5 dataset path.",
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel collecting environments.")
parser.add_argument("--num_demos", type=int, default=100, help="Total episodes to export (success + failure).")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Consecutive successful steps required before exporting an episode.",
)
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=0,
    help="Max env steps before reset on failure. 0 = auto: episode_length_s / control_dt × 1.5 (noise buffer).",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=0.0,
    help="Override env episode_length_s (0 = use task default, typically 30s for lenient teacher).",
)
parser.add_argument(
    "--position_threshold",
    type=float,
    default=0.01,
    help="Position threshold for oracle / lift success checks.",
)
parser.add_argument(
    "--goal_threshold",
    type=float,
    default=0.02,
    help="Distance threshold for task success (success_score and export latch).",
)
parser.add_argument(
    "--progress_terminal_bonus",
    type=float,
    default=0.0,
    help="Terminal bonus coefficient logged in HDF5 only (not added to progress_reward).",
)
parser.add_argument(
    "--table_z",
    type=float,
    default=0.0,
    help="Table surface height in world frame for lift_score.",
)
parser.add_argument("--camera_height", type=int, default=84, help="Camera height in pixels.")
parser.add_argument("--camera_width", type=int, default=84, help="Camera width in pixels.")
parser.add_argument(
    "--topdown_close",
    action="store_true",
    default=False,
    help="Move third_person_camera closer to the table (offset 1.15, 0, 0.72 vs default 2.2, 0, 1.05).",
)
parser.add_argument("--third_person_offset_x", type=float, default=None)
parser.add_argument("--third_person_offset_y", type=float, default=None)
parser.add_argument("--third_person_offset_z", type=float, default=None)
parser.add_argument(
    "--lift_goal_above_cube",
    action="store_true",
    default=False,
    help="Interaction-geometry vertical lift: sync goal to cube spawn xy + height (teacher lifts straight up).",
)
parser.add_argument(
    "--lift_height_above_cube",
    type=float,
    default=0.20,
    help="Vertical ACTUATE goal height above cube spawn (with --lift_goal_above_cube).",
)
parser.add_argument("--arm_scale", type=float, default=0.5, help="IK-Rel arm action scale.")
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
parser.add_argument(
    "--noise_trigger_prob",
    type=float,
    default=0.2,
    help="Per-step probability of starting an action-noise burst.",
)
parser.add_argument(
    "--noise_std",
    type=float,
    default=0.03,
    help="Gaussian noise std added to env actions during a burst.",
)
parser.add_argument(
    "--noise_hold_min",
    type=int,
    default=3,
    help="Minimum hold steps for a noisy action burst.",
)
parser.add_argument(
    "--noise_hold_max",
    type=int,
    default=5,
    help="Maximum hold steps for a noisy action burst.",
)
parser.add_argument(
    "--randomize_cube",
    action="store_true",
    default=False,
    help="Randomize cube spawn on reset.",
)
parser.add_argument(
    "--robot_x_offset",
    type=float,
    default=0.0,
    help="Initial robot base x offset. Use negative values to start the EE camera farther back.",
)
parser.add_argument(
    "--panda_joint6",
    type=float,
    default=None,
    help="Optional initial panda_joint6 override for wrist-camera visibility.",
)
parser.add_argument(
    "--enable_preview",
    action="store_true",
    default=False,
    help="Write a live mask+depth preview image/video while collecting.",
)
parser.add_argument(
    "--preview_dir",
    type=str,
    default="./runs/lift_cube_lenient_teacher_maskdepth_preview",
    help="Directory for preview_latest.png and preview.mp4.",
)
parser.add_argument(
    "--preview_env_idx",
    type=int,
    default=0,
    help="Environment index shown in the live preview.",
)
parser.add_argument("--display_scale", type=int, default=4, help="Upscale factor for preview panels.")
parser.add_argument("--preview_fps", type=int, default=20, help="FPS for preview.mp4.")
parser.add_argument(
    "--show_opencv",
    action="store_true",
    default=False,
    help="Open an OpenCV window for live preview.",
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=50,
    help="Print collection progress every N global steps (0 = disable).",
)
parser.add_argument(
    "--max_total_steps",
    type=int,
    default=0,
    help="Stop after N global steps regardless of demos collected (0 = no limit).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import numpy as np
import random
import sys
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.mdp.data_collection import (
    compute_cube_to_goal_distance,
    compute_lift_progress,
    get_mean_finger_opening,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.oracle import LiftCubeOraclePolicy
from isaaclab_tasks.manager_based.manipulation.lift.mdp.recorders_cfg import (
    LiftLenientTeacherMaskDepthRecorderManagerCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student import (
    compute_desired_gripper_opening,
    compute_teacher_twist_binary_gripper,
    get_gripper_action_dim,
    infer_teacher_phase,
    reset_teacher_phase_state,
    twist_to_env_action,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _float01_to_bgr(tensor: torch.Tensor, scale: int) -> np.ndarray:
    rgb = (tensor.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if scale > 1:
        bgr = cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    return bgr


def build_maskdepth_preview_panel(
    env,
    env_index: int,
    *,
    display_scale: int,
    step: int,
    exported: int,
    success_latched: bool,
    success_steps: int,
    num_success_steps: int,
    cube_to_goal: float,
) -> np.ndarray:
    obs = env.obs_buf["policy"]
    ee_mask = obs["ee_mask"][env_index]
    tp_mask = obs["third_person_mask"][env_index]
    ee_depth = obs["ee_depth"][env_index]
    tp_depth = obs["third_person_depth"][env_index]

    row_ee = np.concatenate(
        [
            _float01_to_bgr(ee_mask, display_scale),
            _float01_to_bgr(ee_depth, display_scale),
        ],
        axis=1,
    )
    row_tp = np.concatenate(
        [
            _float01_to_bgr(tp_mask, display_scale),
            _float01_to_bgr(tp_depth, display_scale),
        ],
        axis=1,
    )
    panel = np.concatenate([row_ee, row_tp], axis=0)
    cv2.putText(
        panel,
        f"step={step} exported={exported} success={success_steps}/{num_success_steps} "
        f"latched={int(success_latched)} cube_to_goal={cube_to_goal:.3f}m",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


class MaskDepthPreviewWriter:
    def __init__(self, output_dir: str, *, show_opencv: bool, video_fps: int):
        os.makedirs(output_dir, exist_ok=True)
        self.latest_path = os.path.join(output_dir, "preview_latest.png")
        self.video_path = os.path.join(output_dir, "preview.mp4")
        self.show_opencv = show_opencv
        self.video_fps = video_fps
        self._writer = None

    def write(self, panel: np.ndarray) -> None:
        cv2.imwrite(self.latest_path, panel)
        if self._writer is None:
            h, w = panel.shape[:2]
            self._writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.video_fps, (w, h))
        if self._writer is not None:
            self._writer.write(panel)
        if self.show_opencv:
            cv2.imshow("lenient_teacher_maskdepth", panel)
            cv2.waitKey(1)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
        if self.show_opencv:
            cv2.destroyAllWindows()


def create_environment_config(output_dir: str, output_file_name: str):
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.terminations.object_dropping = None

    if args_cli.episode_length_s > 0.0:
        env_cfg.episode_length_s = args_cli.episode_length_s

    if abs(args_cli.robot_x_offset) > 1.0e-9:
        env_cfg.scene.robot.init_state.pos = (args_cli.robot_x_offset, 0.0, 0.0)
        print(f"[INFO] robot init_state.pos=({args_cli.robot_x_offset}, 0.0, 0.0)")
    if args_cli.panda_joint6 is not None:
        joint_pos = dict(env_cfg.scene.robot.init_state.joint_pos)
        joint_pos["panda_joint6"] = args_cli.panda_joint6
        env_cfg.scene.robot.init_state.joint_pos = joint_pos
        print(f"[INFO] robot init panda_joint6={args_cli.panda_joint6}")

    if args_cli.randomize_cube:
        env_cfg.events.reset_object_position.params["pose_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.25, 0.25),
            "z": (0.0, 0.0),
        }
        env_cfg.commands.object_pose.debug_vis = True

    if args_cli.lift_goal_above_cube:
        from isaaclab_tasks.manager_based.manipulation.lift.mdp.lift_goal_command import (
            LiftGoalAboveCubeCommand,
        )

        env_cfg.commands.object_pose.class_type = LiftGoalAboveCubeCommand
        env_cfg.commands.object_pose.object_name = "object"
        env_cfg.commands.object_pose.height_above = float(args_cli.lift_height_above_cube)
        env_cfg.commands.object_pose.debug_vis = True
        print(
            f"[INFO] Interaction-geometry lift goal: vertical, cube xy + {args_cli.lift_height_above_cube} m above spawn"
        )

    env_cfg.scene.ee_camera.height = args_cli.camera_height
    env_cfg.scene.ee_camera.width = args_cli.camera_width
    env_cfg.scene.third_person_camera.height = args_cli.camera_height
    env_cfg.scene.third_person_camera.width = args_cli.camera_width
    if args_cli.topdown_close:
        env_cfg.scene.third_person_camera.offset.pos = (1.15, 0.0, 0.72)
        print("[INFO] third_person_camera close top-down offset=(1.15, 0.0, 0.72)")
    elif args_cli.third_person_offset_x is not None:
        ox = args_cli.third_person_offset_x
        oy = args_cli.third_person_offset_y
        oz = args_cli.third_person_offset_z
        env_cfg.scene.third_person_camera.offset.pos = (ox, oy, oz)
        print(f"[INFO] third_person_camera offset=({ox}, {oy}, {oz})")

    env_cfg.recorders = LiftLenientTeacherMaskDepthRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

    return env_cfg


def export_recorded_episodes(env, env_ids: list[int], *, success: bool) -> None:
    """Finalize recorder buffers and export episodes with an explicit success flag."""
    if not env_ids:
        return
    env.recorder_manager.record_pre_reset(env_ids, force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes(
        env_ids,
        torch.full((len(env_ids), 1), success, dtype=torch.bool, device=env.device),
    )
    env.recorder_manager.export_episodes(env_ids)


def reset_envs_after_export(
    env,
    oracle,
    env_ids: list[int],
    *,
    success_latched: torch.Tensor,
    success_step_count: torch.Tensor,
    episode_step_count: torch.Tensor,
) -> None:
    env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    success_latched[env_ids_tensor] = False
    success_step_count[env_ids_tensor] = 0
    episode_step_count[env_ids_tensor] = 0
    env.reset(env_ids=env_ids_tensor)
    env.recorder_manager.reset(env_ids=env_ids_tensor)
    oracle.reset(env_ids)
    reset_teacher_phase_state(env, env_ids)


def print_collection_balance(env) -> None:
    success_count = env.recorder_manager.exported_successful_episode_count
    failure_count = env.recorder_manager.exported_failed_episode_count
    total = success_count + failure_count
    success_rate = success_count / max(total, 1)
    failure_rate = failure_count / max(total, 1)
    print(
        f"[INFO] Exported episodes: total={total} success={success_count} failure={failure_count} "
        f"success_rate={success_rate:.3f} failure_rate={failure_rate:.3f}"
    )


def resolve_max_episode_steps(env_cfg, cli_value: int, *, noise_buffer_factor: float = 1.5) -> int:
    """Derive step budget from env episode length; lenient teacher + noise needs more than oracle 400."""
    if cli_value > 0:
        return cli_value
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    return max(1, int(env_cfg.episode_length_s / control_dt * noise_buffer_factor))


def validate_nonzero_camera_outputs(env, *, attempts: int = 5) -> dict[str, dict[str, float | str]]:
    """Render and fail closed unless every camera produces non-zero raw annotator data."""
    sensor_names = ("ee_camera", "third_person_camera")
    last_stats: dict[str, dict[str, float | str]] = {}
    for _ in range(attempts):
        env.sim.render()
        last_stats = {}
        ready = True
        for sensor_name in sensor_names:
            sensor = env.scene.sensors[sensor_name]
            try:
                sensor.update(env.physics_dt, force_recompute=True)
                output = sensor.data.output
            except RuntimeError as exc:
                last_stats[sensor_name] = {"error": str(exc)}
                ready = False
                continue
            stats = {}
            for key in ("rgb", "distance_to_image_plane", "instance_id_segmentation_fast"):
                value = output.get(key)
                stats[key] = float(torch.count_nonzero(value).item()) if value is not None else -1.0
            last_stats[sensor_name] = stats
            ready &= all(count > 0 for count in stats.values())
        if ready:
            print(
                f"[INFO] Vision preflight PASS has_rtx_sensors={env.sim.has_rtx_sensors()} "
                f"render_mode={env.sim.render_mode} stats={last_stats}"
            )
            return last_stats
    raise RuntimeError(
        "Vision preflight failed: camera annotators remained missing/all-zero after "
        f"{attempts} explicit renders; has_rtx_sensors={env.sim.has_rtx_sensors()} "
        f"render_mode={env.sim.render_mode} stats={last_stats}"
    )


def main():
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    env_cfg = create_environment_config(output_dir, output_file_name)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env._progress_reward_cfg = {
        "goal_threshold": args_cli.goal_threshold,
        "terminal_bonus": args_cli.progress_terminal_bonus,
        "table_z": args_cli.table_z,
    }

    oracle = LiftCubeOraclePolicy(env, position_threshold=args_cli.position_threshold)
    gripper_action_dim = get_gripper_action_dim(env)
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    max_episode_steps = resolve_max_episode_steps(env_cfg, args_cli.max_episode_steps)
    teacher_kwargs = {
        "control_dt": control_dt,
        "min_lin_speed": args_cli.teacher_min_lin_speed,
        "min_ang_speed": args_cli.teacher_min_ang_speed,
    }

    preview = None
    if args_cli.enable_preview:
        preview = MaskDepthPreviewWriter(
            args_cli.preview_dir,
            show_opencv=args_cli.show_opencv,
            video_fps=args_cli.preview_fps,
        )
    preview_env_idx = min(max(args_cli.preview_env_idx, 0), env.num_envs - 1)

    flag = 0
    action_tmp = None

    success_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    success_step_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    episode_step_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    global_step = 0
    preview_scale = 1 if max(args_cli.camera_width, args_cli.camera_height) >= 128 else args_cli.display_scale

    print(f"[INFO] Collecting with task: {args_cli.task}")
    print(f"[INFO] Camera resolution: {args_cli.camera_width}x{args_cli.camera_height}")
    print(
        f"[INFO] episode_length_s={env_cfg.episode_length_s:.1f}, control_dt={control_dt:.4f}s, "
        f"max_episode_steps={max_episode_steps} (~{max_episode_steps * control_dt:.1f}s sim time)"
    )
    print(f"[INFO] Export after {args_cli.num_success_steps} steps once lift goal is latched (success episodes)")
    print("[INFO] Export failure episodes on manual timeout (max_episode_steps)")
    print(
        f"[INFO] Teacher noise: trigger_prob={args_cli.noise_trigger_prob}, std={args_cli.noise_std}, "
        f"hold=[{args_cli.noise_hold_min}, {args_cli.noise_hold_max}]"
    )
    print("[INFO] Recorded keys: obs (RGB+mask+depth+student_state), env_actions (CQL action),")
    print("       rewards/progress_reward (scaled Δprogress), actions (teacher twist),")
    print("       ee_pose, goals, rewards/grasp|lift|progress*, states, demo attr success")
    print(
        f"[INFO] Progress reward: scale=50*Δprogress, terminal_bonus(log-only)={args_cli.progress_terminal_bonus}, "
        f"goal_threshold={args_cli.goal_threshold}, table_z={args_cli.table_z}"
    )

    env.reset()
    validate_nonzero_camera_outputs(env)
    oracle.reset()
    reset_teacher_phase_state(env)

    while simulation_app.is_running():
        with torch.inference_mode():
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
            flag -= 1
            if random.random() < args_cli.noise_trigger_prob or flag > 0:
                env_actions = env_actions + torch.randn_like(env_actions) * args_cli.noise_std
                if flag <= 0:
                    action_tmp = env_actions
                    flag = random.randint(args_cli.noise_hold_min, args_cli.noise_hold_max)
                else:
                    env_actions = action_tmp

            env._lenient_teacher_record = {
                "teacher_twist": teacher_twist.clone(),
                "env_actions": env_actions.clone(),
            }

            _, _, terminated, truncated, _ = env.step(env_actions)
            episode_step_count += 1
            global_step += 1

            done_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if done_env_ids.numel() > 0:
                if done_env_ids.ndim == 0:
                    done_env_ids = done_env_ids.unsqueeze(0)
                done_list = done_env_ids.tolist()
                oracle.reset(done_list)
                reset_teacher_phase_state(env, done_list)
                success_latched[done_env_ids] = False
                success_step_count[done_env_ids] = 0
                episode_step_count[done_env_ids] = 0

            timeout_env_ids = (episode_step_count >= max_episode_steps).nonzero(as_tuple=False).squeeze(-1)
            if timeout_env_ids.numel() > 0:
                if timeout_env_ids.ndim == 0:
                    timeout_env_ids = timeout_env_ids.unsqueeze(0)
                timeout_list = timeout_env_ids.tolist()
                if args_cli.log_interval > 0:
                    for env_id in timeout_list:
                        dist = float(compute_cube_to_goal_distance(env)[env_id].item())
                        print(
                            f"[INFO] env={env_id} episode timeout at {max_episode_steps} steps "
                            f"(cube_to_goal={dist:.3f}m, latched={bool(success_latched[env_id].item())}), exporting failure"
                        )
                export_recorded_episodes(env, timeout_list, success=False)
                print_collection_balance(env)
                reset_envs_after_export(
                    env,
                    oracle,
                    timeout_list,
                    success_latched=success_latched,
                    success_step_count=success_step_count,
                    episode_step_count=episode_step_count,
                )

            success_now = mdp.object_reached_goal(env, threshold=args_cli.goal_threshold)
            success_latched |= success_now
            # Once the cube reaches the goal, keep counting even if it briefly drifts (lenient teacher
            # keeps min-speed motion near the goal, so strict consecutive checks rarely reach 10/10).
            success_step_count = torch.where(
                success_latched,
                success_step_count + 1,
                torch.zeros_like(success_step_count),
            )
            cube_to_goal = compute_cube_to_goal_distance(env)

            if args_cli.log_interval > 0 and global_step % args_cli.log_interval == 0:
                env_id = preview_env_idx
                _, progress_now = compute_lift_progress(
                    env,
                    goal_threshold=args_cli.goal_threshold,
                    table_z=args_cli.table_z,
                )
                print(
                    f"[INFO] step={global_step} ep_step={episode_step_count[env_id].item()} "
                    f"cube_to_goal={cube_to_goal[env_id].item():.3f}m progress={progress_now[env_id].item():.3f} "
                    f"latched={bool(success_latched[env_id].item())} "
                    f"success_steps={success_step_count[env_id].item()}/{args_cli.num_success_steps} "
                    f"exported={env.recorder_manager.exported_successful_episode_count + env.recorder_manager.exported_failed_episode_count}/{args_cli.num_demos}"
                )

            if preview is not None:
                env_id = preview_env_idx
                panel = build_maskdepth_preview_panel(
                    env,
                    env_id,
                    display_scale=preview_scale,
                    step=global_step,
                    exported=env.recorder_manager.exported_successful_episode_count
                    + env.recorder_manager.exported_failed_episode_count,
                    success_latched=bool(success_latched[env_id].item()),
                    success_steps=int(success_step_count[env_id].item()),
                    num_success_steps=args_cli.num_success_steps,
                    cube_to_goal=float(cube_to_goal[env_id].item()),
                )
                preview.write(panel)

            ready_env_ids = (success_step_count >= args_cli.num_success_steps).nonzero(as_tuple=False).squeeze(-1)
            if ready_env_ids.numel() > 0:
                if ready_env_ids.ndim == 0:
                    ready_env_ids = ready_env_ids.unsqueeze(0)
                ready_list = ready_env_ids.tolist()

                export_recorded_episodes(env, ready_list, success=True)
                print_collection_balance(env)
                reset_envs_after_export(
                    env,
                    oracle,
                    ready_list,
                    success_latched=success_latched,
                    success_step_count=success_step_count,
                    episode_step_count=episode_step_count,
                )

            exported_total = (
                env.recorder_manager.exported_successful_episode_count
                + env.recorder_manager.exported_failed_episode_count
            )
            if args_cli.num_demos > 0 and exported_total >= args_cli.num_demos:
                print(f"[INFO] Collected {exported_total} episodes (success + failure). Done.")
                break

            if args_cli.max_total_steps > 0 and global_step >= args_cli.max_total_steps:
                print(
                    f"[WARN] Reached --max_total_steps={args_cli.max_total_steps} "
                    f"with exported={exported_total}/{args_cli.num_demos}. Stopping."
                )
                break

            if env.sim.is_stopped():
                break

    if preview is not None:
        preview.close()
    env.close()

    from analyze_offline_cql_dataset import analyze_dataset, print_export_report

    stats_path = os.path.splitext(args_cli.dataset_file)[0] + "_export_stats.json"
    if os.path.isfile(args_cli.dataset_file):
        report = analyze_dataset(args_cli.dataset_file)
        with open(stats_path, "w", encoding="utf-8") as f:
            import json

            json.dump(report["export_stats"], f, indent=2)
        print_export_report(report)
        print(f"[INFO] Export statistics saved to: {stats_path}")
    print(f"[INFO] Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
