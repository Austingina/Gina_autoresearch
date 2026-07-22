# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Record noisy oracle demonstrations for offline RL on a fixed cube pose.

The oracle policy is mixed with held random sub-optimal actions to create
sub-optimal -> optimal transitions. Each sampled noise action is executed for
a random number of consecutive steps in ``[noise_hold_min, noise_hold_max]``.

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/record_oracle_noisy_offline_demos.py \\
        --num_envs 8 \\
        --num_demos 100 \\
        --enable_preview \\
        --preview_dir ./runs/lift_cube_offline_preview \\
        --dataset_file ./datasets/lift_cube_offline_noisy.hdf5
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Record noisy oracle lift-cube demonstrations for offline RL on a fixed cube pose."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-Oracle-Offline-v0",
    help="Task with fixed cube pose and oracle data-collection sensors.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/lift_cube_offline_noisy.hdf5",
    help="Output HDF5 dataset path.",
)
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel collecting environments.")
parser.add_argument("--num_demos", type=int, default=100, help="Number of successful episodes to export.")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Consecutive successful steps required before exporting an episode.",
)
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=400,
    help="Maximum steps per episode before forcing a reset on failure.",
)
parser.add_argument(
    "--position_threshold",
    type=float,
    default=0.01,
    help="Position threshold for the pick-and-lift state machine.",
)
parser.add_argument(
    "--goal_threshold",
    type=float,
    default=0.02,
    help="Distance threshold for considering the object to have reached the goal.",
)
parser.add_argument(
    "--noise_prob",
    type=float,
    default=0.2,
    help="Probability of starting a new held noise action segment when not already noisy.",
)
parser.add_argument(
    "--noise_hold_min",
    type=int,
    default=5,
    help="Minimum number of consecutive steps to hold a sampled noise action.",
)
parser.add_argument(
    "--noise_hold_max",
    type=int,
    default=10,
    help="Maximum number of consecutive steps to hold a sampled noise action.",
)
parser.add_argument(
    "--pos_noise_std",
    type=float,
    default=0.03,
    help="Gaussian position perturbation std for noisy IK targets (meters).",
)
parser.add_argument(
    "--rot_noise_std",
    type=float,
    default=0.15,
    help="Gaussian axis-angle perturbation std for noisy IK targets (radians).",
)
parser.add_argument(
    "--gripper_flip_prob",
    type=float,
    default=0.3,
    help="Probability of flipping the gripper command sign inside a noisy action.",
)
parser.add_argument("--camera_height", type=int, default=84, help="RGB camera height in pixels.")
parser.add_argument("--camera_width", type=int, default=84, help="RGB camera width in pixels.")
parser.add_argument(
    "--enable_preview",
    action="store_true",
    default=False,
    help="Write a live preview image/video while collecting (third-person camera + top-down goal/velocity).",
)
parser.add_argument(
    "--preview_dir",
    type=str,
    default="./runs/lift_cube_offline_preview",
    help="Directory for preview_latest.png and preview.mp4.",
)
parser.add_argument(
    "--preview_env_idx",
    type=int,
    default=0,
    help="Environment index shown in the live preview.",
)
parser.add_argument(
    "--display_scale",
    type=int,
    default=4,
    help="Upscale factor for preview camera and top-down panel.",
)
parser.add_argument(
    "--velocity_scale",
    type=float,
    default=0.25,
    help="Arrow length scale in the top-down preview panel.",
)
parser.add_argument(
    "--preview_fps",
    type=int,
    default=20,
    help="FPS for preview.mp4.",
)
parser.add_argument(
    "--show_opencv",
    action="store_true",
    default=False,
    help="Open an OpenCV window for live preview (requires GUI-enabled OpenCV).",
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

import gymnasium as gym
import sys
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.mdp.oracle import LiftCubeOraclePolicy, NoisyOracleActionController
from isaaclab_tasks.manager_based.manipulation.lift.mdp.recorders_cfg import LiftOracleRecorderManagerCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from oracle_demo_preview import build_collection_preview_panel, get_step_data_for_preview, maybe_create_preview_writer


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

    if hasattr(env_cfg.scene, "ee_camera"):
        env_cfg.scene.ee_camera.height = args_cli.camera_height
        env_cfg.scene.ee_camera.width = args_cli.camera_width
    if hasattr(env_cfg.scene, "third_person_camera"):
        env_cfg.scene.third_person_camera.height = args_cli.camera_height
        env_cfg.scene.third_person_camera.width = args_cli.camera_width

    env_cfg.recorders = LiftOracleRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg


def main():
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    env_cfg = create_environment_config(output_dir, output_file_name)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    oracle = LiftCubeOraclePolicy(env, position_threshold=args_cli.position_threshold)
    controller = NoisyOracleActionController(
        oracle,
        noise_prob=args_cli.noise_prob,
        hold_min=args_cli.noise_hold_min,
        hold_max=args_cli.noise_hold_max,
        pos_noise_std=args_cli.pos_noise_std,
        rot_noise_std=args_cli.rot_noise_std,
        gripper_flip_prob=args_cli.gripper_flip_prob,
    )

    print(f"[INFO] Collecting with task: {args_cli.task}")
    print("[INFO] Cube pose: fixed at default centered spawn position")
    print(
        "[INFO] Noise: "
        f"p={args_cli.noise_prob}, hold=[{args_cli.noise_hold_min}, {args_cli.noise_hold_max}], "
        f"pos_std={args_cli.pos_noise_std}, rot_std={args_cli.rot_noise_std}"
    )
    print(f"[INFO] Target successful episodes: {args_cli.num_demos}")

    preview = maybe_create_preview_writer(args_cli)
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    preview_env_idx = min(max(args_cli.preview_env_idx, 0), env.num_envs - 1)

    success_step_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    episode_step_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    exported_demo_count = 0

    env.reset()
    controller.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = controller.compute()
            _, _, terminated, truncated, _ = env.step(actions)
            episode_step_count += 1

            done_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if done_env_ids.numel() > 0:
                if done_env_ids.ndim == 0:
                    done_env_ids = done_env_ids.unsqueeze(0)
                done_ids_list = done_env_ids.tolist()
                controller.reset(done_ids_list)
                success_step_count[done_env_ids] = 0
                episode_step_count[done_env_ids] = 0

            timeout_env_ids = (episode_step_count >= args_cli.max_episode_steps).nonzero(as_tuple=False).squeeze(-1)
            if timeout_env_ids.numel() > 0:
                if timeout_env_ids.ndim == 0:
                    timeout_env_ids = timeout_env_ids.unsqueeze(0)
                timeout_env_ids_list = timeout_env_ids.tolist()
                controller.reset(timeout_env_ids_list)
                success_step_count[timeout_env_ids] = 0
                episode_step_count[timeout_env_ids] = 0
                env_ids_tensor = torch.tensor(timeout_env_ids_list, device=env.device, dtype=torch.long)
                env.reset(env_ids=env_ids_tensor)
                env.recorder_manager.reset(env_ids=env_ids_tensor)

            success_mask = mdp.object_reached_goal(env, threshold=args_cli.goal_threshold)
            success_step_count = torch.where(
                success_mask,
                success_step_count + 1,
                torch.zeros_like(success_step_count),
            )

            if preview is not None:
                step_data = get_step_data_for_preview(env, control_dt)
                panel = build_collection_preview_panel(
                    env,
                    preview_env_idx,
                    step_data,
                    episode_step=int(episode_step_count[preview_env_idx].item()),
                    exported_demo_count=exported_demo_count,
                    display_scale=args_cli.display_scale,
                    velocity_scale=args_cli.velocity_scale,
                    using_noise=True,
                    noise_flag=int(controller.noise_flag[preview_env_idx].item()),
                    reached_goal=bool(success_mask[preview_env_idx].item()),
                )
                preview.write(panel)

            ready_env_ids = (success_step_count >= args_cli.num_success_steps).nonzero(as_tuple=False).squeeze(-1)
            if ready_env_ids.numel() > 0:
                if ready_env_ids.ndim == 0:
                    ready_env_ids = ready_env_ids.unsqueeze(0)
                ready_env_ids_list = ready_env_ids.tolist()

                env.recorder_manager.record_pre_reset(ready_env_ids_list, force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    ready_env_ids_list,
                    torch.ones((len(ready_env_ids_list), 1), dtype=torch.bool, device=env.device),
                )
                env.recorder_manager.export_episodes(ready_env_ids_list)

                exported_demo_count = env.recorder_manager.exported_successful_episode_count
                print(f"[INFO] Exported successful demonstrations: {exported_demo_count}")
                success_step_count[ready_env_ids] = 0
                episode_step_count[ready_env_ids] = 0
                controller.reset(ready_env_ids_list)

                env_ids_tensor = torch.tensor(ready_env_ids_list, device=env.device, dtype=torch.long)
                env.reset(env_ids=env_ids_tensor)
                env.recorder_manager.reset(env_ids=env_ids_tensor)

            if args_cli.num_demos > 0 and exported_demo_count >= args_cli.num_demos:
                print(f"[INFO] Collected {exported_demo_count} demonstrations. Done.")
                break

            if env.sim.is_stopped():
                break

    if preview is not None:
        preview.close()
    env.close()
    print(f"[INFO] Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
