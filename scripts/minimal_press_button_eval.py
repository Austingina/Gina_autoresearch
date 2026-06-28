#!/usr/bin/env python3
"""Minimal 5-episode press-button smoke test (zero actions + oracle press)."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--warmup_steps", type=int, default=8)
parser.add_argument("--press_step", type=int, default=10)
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
        franka_y=-0.72,
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
        disable_fabric=True,
        device=getattr(args_cli, "device", "cuda:0"),
        seed=getattr(args_cli, "seed", None),
    )

    arena_env = PressButtonEnvironment().get_env(env_args)
    name, cfg = ArenaEnvBuilder(arena_env, env_args).build_registered()
    env = gym.make(name, cfg=cfg)
    pressable = arena_env.task.pressable_object

    results: list[bool] = []
    for episode in range(args_cli.num_episodes):
        obs, _ = env.reset()
        success = False
        for step in range(1, args_cli.press_step + 3):
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            if action.shape[-1] >= 7:
                action[..., 6] = 1.0
            obs, _, terminated, truncated, _ = env.step(action)
            if step == args_cli.press_step:
                pressable.press(env.unwrapped, env_ids=None, pressed_percentage=1.0)
            if bool(pressable.is_pressed(env.unwrapped).item()) or bool(terminated.any()):
                success = True
                break
            if truncated.any():
                break
        results.append(success)
        print(f"episode={episode + 1} success={success} pressed={bool(pressable.is_pressed(env.unwrapped).item())}", flush=True)

    all_ok = len(results) == args_cli.num_episodes and all(results)
    summary = f"episodes={results} consecutive_success={all_ok} ({sum(results)}/{len(results)})"
    out_path = "/home/l/code/AutoResearch/runs/press_button_teacher_eval.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary, flush=True)
    env.close()
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
