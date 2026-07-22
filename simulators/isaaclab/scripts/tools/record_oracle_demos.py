# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Record lift-cube demonstrations with the built-in pick-and-lift oracle policy.

The oracle uses a GPU state machine to reach, grasp, and lift the cube for
randomized cube positions and target commands. Demonstrations are exported to
HDF5 using Isaac Lab's recorder manager.

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/record_oracle_demos.py \\
        --task Isaac-Lift-Cube-Franka-v0 \\
        --num_envs 64 \\
        --num_demos 1000 \\
        --headless \\
        --dataset_file ./datasets/lift_cube_oracle.hdf5
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record lift-cube demonstrations with the oracle policy.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-v0",
    help="Name of the task. Joint-position and IK-Abs lift tasks are supported.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/lift_cube_oracle.hdf5",
    help="File path to export recorded demonstrations.",
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--num_demos", type=int, default=100, help="Number of successful demonstrations to record.")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of consecutive successful steps before exporting an episode.",
)
parser.add_argument(
    "--position_threshold",
    type=float,
    default=0.01,
    help="Position threshold used by the pick-and-lift state machine.",
)
parser.add_argument(
    "--goal_threshold",
    type=float,
    default=0.02,
    help="Distance threshold for considering the object to have reached the goal.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.mdp.oracle import LiftCubeOraclePolicy
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


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

    env_cfg.recorders = ActionStateRecorderManagerCfg()
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
    print(f"[INFO] Oracle action space: {'IK absolute pose' if oracle.uses_ik_abs else 'joint position'}")

    success_step_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    exported_demo_count = 0

    env.reset()
    oracle.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = oracle.compute()
            _, _, terminated, truncated, _ = env.step(actions)

            done_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if done_env_ids.numel() > 0:
                if done_env_ids.ndim == 0:
                    done_env_ids = done_env_ids.unsqueeze(0)
                oracle.reset(done_env_ids.tolist())
                success_step_count[done_env_ids] = 0

            success_mask = mdp.object_reached_goal(env, threshold=args_cli.goal_threshold)
            success_step_count = torch.where(
                success_mask,
                success_step_count + 1,
                torch.zeros_like(success_step_count),
            )

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
                oracle.reset(ready_env_ids_list)

                env_ids_tensor = torch.tensor(ready_env_ids_list, device=env.device, dtype=torch.long)
                env.reset(env_ids=env_ids_tensor)
                env.recorder_manager.reset(env_ids=env_ids_tensor)

            if args_cli.num_demos > 0 and exported_demo_count >= args_cli.num_demos:
                print(f"[INFO] Collected {exported_demo_count} demonstrations. Done.")
                break

            if env.sim.is_stopped():
                break

    env.close()
    print(f"[INFO] Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
