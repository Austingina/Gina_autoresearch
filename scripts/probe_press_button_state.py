#!/usr/bin/env python3
"""Probe coffee-machine button state and EE/target geometry (headless)."""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_steps", type=int, default=3)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena_environments.press_button_environment import PressButtonEnvironment  # noqa: E402


def main() -> int:
    env_args = argparse.Namespace(
        enable_cameras=False,
        embodiment="franka",
        teleop_device=None,
        object=None,
        kitchen_layout=1,
        kitchen_style=2,
        coffee_button_joint="Button002_joint",
        pressedness_threshold=0.5,
        reset_pressedness=0.0,
        episode_length_s=20.0,
        press_target_x=0.455,
        press_target_y=-0.331,
        press_target_z=1.117,
        franka_x=0.60,
        franka_y=-1.05,
        franka_z=None,
        franka_yaw_deg=90.0,
        platform_height=0.45,
        platform_size_x=0.80,
        platform_size_y=0.80,
        num_envs=1,
        env_spacing=30.0,
        mimic=False,
        solve_relations=True,
        placement_seed=None,
        disable_fabric=getattr(args_cli, "disable_fabric", False),
        device=getattr(args_cli, "device", "cuda:0"),
        seed=getattr(args_cli, "seed", None),
    )

    arena_env = PressButtonEnvironment().get_env(env_args)
    name, cfg = ArenaEnvBuilder(arena_env, env_args).build_registered()
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    task = arena_env.task
    pressable = task.pressable_object
    pressed = pressable.is_pressed(env)
    joint_pos = env.scene.articulations["coffee_machine"].data.joint_pos[0]
    joint_names = env.scene.articulations["coffee_machine"].data.joint_names
    btn_idx = joint_names.index("Button002_joint")
    btn_raw = float(joint_pos[btn_idx].item())

    ee = env.scene["ee_frame"].data.target_pos_w[0, 0] - env.scene.env_origins[0]
    target = torch.tensor(task.press_target_position, device=env.device)
    dist = torch.linalg.norm(ee - target).item()

    body_names = env.scene.articulations["coffee_machine"].data.body_names
    body_pos = env.scene.articulations["coffee_machine"].data.body_pos_w[0] - env.scene.env_origins[0]

    out_path = "/home/l/code/AutoResearch/runs/probe_button_result.txt"
    lines = []
    lines.append("=== press_button probe ===")
    lines.append(f"is_pressed={bool(pressed.item())} joint_raw={btn_raw:.6f}")
    lines.append(f"eef_pos=({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f})")
    lines.append(f"target =({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}) dist={dist:.4f}")
    lines.append(f"joint_names containing Button: {[n for n in joint_names if 'Button' in n]}")
    for i, name in enumerate(body_names):
        if "Button" in name or "button" in name.lower():
            p = body_pos[i]
            lines.append(f"body {name}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
