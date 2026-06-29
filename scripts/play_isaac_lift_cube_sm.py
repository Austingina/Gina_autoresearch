#!/usr/bin/env python3
"""Clean Isaac Lab lift-cube state-machine teacher.

This script defines lift as Interaction Geometry ``κ=vertical_free``:
after grasping, the cube is lifted straight up above its contact/spawn xy.
There is no hidden xy relocation goal.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Lift cube with a clean vertical-free Franka state machine.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_seed", type=int, default=7)
parser.add_argument("--max_steps", type=int, default=650)
parser.add_argument("--position_threshold", type=float, default=0.018)
parser.add_argument("--approach_height", type=float, default=0.13)
parser.add_argument("--grasp_height_offset", type=float, default=0.005)
parser.add_argument("--lift_height", type=float, default=0.22)
parser.add_argument("--min_success_height", type=float, default=0.16)
parser.add_argument("--max_success_xy_drift", type=float, default=0.08)
parser.add_argument("--fixed_quat", type=float, nargs=4, default=[0.0, 1.0, 0.0, 0.0])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.sensors import FrameTransformer
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class GripperState:
    OPEN = 1.0
    CLOSE = -1.0


class LiftCubeState:
    APPROACH_ABOVE = 0
    APPROACH_CUBE = 1
    GRASP = 2
    LIFT = 3
    HOLD = 4


class VerticalLiftSm:
    """Small task-space state machine for vertical-free cube lifting."""

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str,
        *,
        approach_height: float,
        grasp_height_offset: float,
        lift_height: float,
        position_threshold: float,
        fixed_quat: Sequence[float],
    ):
        self.dt = float(dt)
        self.num_envs = int(num_envs)
        self.device = device
        self.approach_height = float(approach_height)
        self.grasp_height_offset = float(grasp_height_offset)
        self.lift_height = float(lift_height)
        self.position_threshold = float(position_threshold)
        self.state = torch.full((num_envs,), LiftCubeState.APPROACH_ABOVE, dtype=torch.long, device=device)
        self.wait = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        self.fixed_quat = torch.tensor(fixed_quat, dtype=torch.float32, device=device).repeat(num_envs, 1)
        self.lift_anchor = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self.hold_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self.hold_quat = self.fixed_quat.clone()

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = LiftCubeState.APPROACH_ABOVE
        self.wait[env_ids] = 0.0
        self.lift_anchor[env_ids] = 0.0
        self.hold_pos[env_ids] = 0.0
        self.hold_quat[env_ids] = self.fixed_quat[env_ids]

    def compute(self, ee_pose: torch.Tensor, cube_pos: torch.Tensor) -> torch.Tensor:
        ee_pos = ee_pose[:, :3]
        desired_pos = ee_pos.clone()
        desired_quat = self.fixed_quat.clone()
        gripper = torch.full((self.num_envs,), GripperState.OPEN, dtype=torch.float32, device=self.device)

        above_pos = cube_pos.clone()
        above_pos[:, 2] += self.approach_height
        grasp_pos = cube_pos.clone()
        grasp_pos[:, 2] += self.grasp_height_offset

        for state_id in range(5):
            mask = self.state == state_id
            if not mask.any():
                continue
            if state_id == LiftCubeState.APPROACH_ABOVE:
                desired_pos[mask] = above_pos[mask]
                gripper[mask] = GripperState.OPEN
                reached = torch.linalg.norm(ee_pos[mask] - above_pos[mask], dim=-1) < self.position_threshold
                if (reached & (self.wait[mask] >= 0.2)).any():
                    transition_mask = mask.clone()
                    transition_mask[mask] = reached
                    self.state[transition_mask] = LiftCubeState.APPROACH_CUBE
                    self.wait[transition_mask] = 0.0
            elif state_id == LiftCubeState.APPROACH_CUBE:
                desired_pos[mask] = grasp_pos[mask]
                gripper[mask] = GripperState.OPEN
                reached = torch.linalg.norm(ee_pos[mask] - grasp_pos[mask], dim=-1) < self.position_threshold
                if (reached | (self.wait[mask] >= 0.9)).any():
                    transition = reached | (self.wait[mask] >= 0.9)
                    transition_mask = mask.clone()
                    transition_mask[mask] = transition
                    gripper[transition_mask] = GripperState.CLOSE
                    self.state[transition_mask] = LiftCubeState.GRASP
                    self.wait[transition_mask] = 0.0
            elif state_id == LiftCubeState.GRASP:
                desired_pos[mask] = grasp_pos[mask]
                gripper[mask] = GripperState.CLOSE
                if (self.wait[mask] >= 0.45).any():
                    self.lift_anchor[mask] = cube_pos[mask]
                    self.state[mask] = LiftCubeState.LIFT
                    self.wait[mask] = 0.0
            elif state_id == LiftCubeState.LIFT:
                lift_pos = self.lift_anchor[mask].clone()
                lift_pos[:, 2] += self.lift_height
                desired_pos[mask] = lift_pos
                gripper[mask] = GripperState.CLOSE
                reached = torch.linalg.norm(ee_pos[mask] - lift_pos, dim=-1) < self.position_threshold
                if (reached | (self.wait[mask] >= 1.5)).any():
                    transition = reached | (self.wait[mask] >= 1.5)
                    transition_mask = mask.clone()
                    transition_mask[mask] = transition
                    self.hold_pos[transition_mask] = desired_pos[transition_mask]
                    self.hold_quat[transition_mask] = desired_quat[transition_mask]
                    self.state[transition_mask] = LiftCubeState.HOLD
                    self.wait[transition_mask] = 0.0
            else:
                desired_pos[mask] = self.hold_pos[mask]
                desired_quat[mask] = self.hold_quat[mask]
                gripper[mask] = GripperState.CLOSE

        self.wait += self.dt
        return torch.cat([desired_pos, desired_quat, gripper.unsqueeze(-1)], dim=-1)


def main() -> None:
    env_cfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.env_seed
    env_cfg.episode_length_s = 8.0
    env_cfg.commands.object_pose.debug_vis = False
    # Deterministic smoke task: vertical-free lift at the cube's reset xy.
    env_cfg.events.reset_object_position.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}

    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    sm = VerticalLiftSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        approach_height=args_cli.approach_height,
        grasp_height_offset=args_cli.grasp_height_offset,
        lift_height=args_cli.lift_height,
        position_threshold=args_cli.position_threshold,
        fixed_quat=args_cli.fixed_quat,
    )

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    actions[:, 7] = GripperState.OPEN
    initial_cube_pos = None
    max_height_gain = torch.zeros((env.unwrapped.num_envs,), device=env.unwrapped.device)
    max_xy_drift = torch.zeros((env.unwrapped.num_envs,), device=env.unwrapped.device)

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]
            ee_frame_tf: FrameTransformer = env.unwrapped.scene["ee_frame"]
            ee_pos = ee_frame_tf.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            ee_quat = ee_frame_tf.data.target_quat_w[..., 0, :].clone()
            cube_pos = env.unwrapped.scene["object"].data.root_pos_w.clone() - env.unwrapped.scene.env_origins
            if initial_cube_pos is None:
                initial_cube_pos = cube_pos.clone()

            actions = sm.compute(torch.cat([ee_pos, ee_quat], dim=-1), cube_pos)

            height_gain = cube_pos[:, 2] - initial_cube_pos[:, 2]
            xy_drift = torch.linalg.norm(cube_pos[:, :2] - initial_cube_pos[:, :2], dim=-1)
            max_height_gain = torch.maximum(max_height_gain, height_gain)
            max_xy_drift = torch.maximum(max_xy_drift, xy_drift)

            if step % 50 == 0:
                ee_cube = torch.linalg.norm(ee_pos - cube_pos, dim=-1)
                print(
                    f"[STEP {step:04d}] state={int(sm.state[0].item())} "
                    f"height_gain={height_gain[0].item():.3f} max_height={max_height_gain[0].item():.3f} "
                    f"xy_drift={xy_drift[0].item():.3f} ee_cube={ee_cube[0].item():.3f}"
                )
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                sm.reset_idx(done_ids)
                actions[done_ids] = 0.0
                actions[done_ids, 3] = 1.0
                actions[done_ids, 7] = GripperState.OPEN
            step += 1
            if args_cli.headless and step >= args_cli.max_steps:
                break

    success = (max_height_gain >= args_cli.min_success_height) & (max_xy_drift <= args_cli.max_success_xy_drift)
    print(
        f"[RESULT] success={bool(success[0].item())} "
        f"max_height_gain={max_height_gain[0].item():.4f} max_xy_drift={max_xy_drift[0].item():.4f}"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
