#!/usr/bin/env python3
"""Verify oracle button press sets is_pressed and triggers termination."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena.policy.interaction_geometry_teacher_policy import (  # noqa: E402
    InteractionGeometryTeacherArgs,
    InteractionGeometryTeacherPolicy,
)
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
        franka_y=-0.85,
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
    env = gym.make(name, cfg=cfg)
    pressable = arena_env.task.pressable_object

    obs, _ = env.reset()
    policy = InteractionGeometryTeacherPolicy(
        InteractionGeometryTeacherArgs(
            mode="press_button",
            press_axis="y",
            reach_gain=6.0,
            actuate_gain=4.0,
            max_delta_pos=0.12,
            position_tolerance=0.12,
            oracle_press_distance=0.50,
            oracle_press_after_actuate_steps=0,
            press_surface_offset=(0.0, 0.0, 0.0),
        )
    )
    policy.reset()
    policy.set_task_description(arena_env.task.get_task_description())

    lines = []
    for step in range(80):
        with torch.inference_mode():
            action = policy.get_action(env, obs)
            obs, _, terminated, truncated, _ = env.step(action)
        pressed = bool(pressable.is_pressed(env.unwrapped).item())
        dbg = policy.last_debug
        lines.append(
            f"step={step:03d} phase={dbg.get('phase')} dist={float(dbg['distance'][0]):.3f} "
            f"pressed={pressed} term={bool(terminated.any())}"
        )
        if terminated.any() or truncated.any() or pressed:
            break

    out = "\n".join(lines)
    print(out, flush=True)
    with open("/home/l/code/AutoResearch/runs/probe_oracle_press.txt", "w", encoding="utf-8") as f:
        f.write(out + "\n")
    env.close()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
