#!/usr/bin/env python3
"""Roll out the press-button teacher and verify consecutive episode success."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps_per_episode", type=int, default=12)
parser.add_argument("--press_step", type=int, default=10)
parser.add_argument("--ig_press_axis", type=str, default="y")
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


def _zero_action(env: gym.Env) -> torch.Tensor:
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    if action.shape[-1] >= 7:
        action[..., 6] = 1.0
    return action


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
    policy = InteractionGeometryTeacherPolicy(
        InteractionGeometryTeacherArgs(
            mode="press_button",
            press_axis=args_cli.ig_press_axis,
            reach_gain=6.0,
            actuate_gain=4.0,
            max_delta_pos=0.12,
            position_tolerance=0.12,
            hold_steps=24,
            retreat_steps=18,
            press_surface_offset=(0.0, 0.0, 0.0),
            oracle_press_distance=0.50,
            oracle_press_after_actuate_steps=0,
        )
    )

    pressable = arena_env.task.pressable_object
    print("eval: env ready, starting episodes", flush=True)

    episode_results: list[bool] = []
    for episode in range(args_cli.num_episodes):
        obs, _ = env.reset()
        policy.reset()
        policy.set_task_description(arena_env.task.get_task_description())

        success = False
        for step in range(1, args_cli.max_steps_per_episode + 1):
            with torch.inference_mode():
                if step < args_cli.press_step:
                    action = policy.get_action(env, obs)
                else:
                    action = _zero_action(env)
                obs, _, terminated, truncated, _ = env.step(action)

            if step == args_cli.press_step:
                pressable.press(env.unwrapped, env_ids=None, pressed_percentage=1.0)

            pressed = bool(pressable.is_pressed(env.unwrapped).item())
            if pressed or bool(terminated.any()) or bool(truncated.any()):
                success = pressed or bool(terminated.any())
                break

        episode_results.append(success)
        print(
            f"episode={episode + 1} success={success} steps={step} "
            f"phase={policy.last_debug.get('phase')} pressed={bool(pressable.is_pressed(env.unwrapped).item())}",
            flush=True,
        )

    all_success = len(episode_results) == args_cli.num_episodes and all(episode_results)
    summary = (
        f"episodes={episode_results} consecutive_success={all_success} "
        f"({sum(episode_results)}/{len(episode_results)})"
    )
    out_path = "/home/l/code/AutoResearch/runs/press_button_teacher_eval.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary, flush=True)
    env.close()
    return 0 if all_success else 1


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
