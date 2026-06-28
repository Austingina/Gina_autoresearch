#!/usr/bin/env python3
"""Verify geometry teacher reaches the coffee button, not the top lid."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_steps", type=int, default=35)
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

    obs, _ = env.reset()
    policy.reset()
    policy.set_task_description(arena_env.task.get_task_description())
    unwrapped = env.unwrapped

    body_names = unwrapped.scene.articulations["coffee_machine"].data.body_names
    btn_body_idx = next(i for i, n in enumerate(body_names) if "Button002" in n)
    btn_body = (
        unwrapped.scene.articulations["coffee_machine"].data.body_pos_w[0, btn_body_idx]
        - unwrapped.scene.env_origins[0]
    )

    lines = [f"button_body=({btn_body[0]:.4f}, {btn_body[1]:.4f}, {btn_body[2]:.4f})"]
    for step in range(1, args_cli.num_steps + 1):
        with torch.inference_mode():
            action = policy.get_action(env, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            policy.apply_pending_oracle_press(env)
        ee = unwrapped.scene["ee_frame"].data.target_pos_w[0, 0] - unwrapped.scene.env_origins[0]
        dist_btn = torch.linalg.norm(ee - btn_body).item()
        dbg = policy.last_debug
        if step in (1, 8, 16, 24, args_cli.num_steps) or step % 10 == 0:
            lines.append(
                f"step={step:02d} phase={dbg.get('phase')} ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                f"dist_btn={dist_btn:.4f} dz_btn={float(ee[2]-btn_body[2]):+.4f}"
            )
        if terminated.any() or truncated.any():
            lines.append(f"terminated at step={step}")
            break

    pressed = bool(arena_env.task.pressable_object.is_pressed(unwrapped).item())
    lines.append(f"pressed={pressed}")
    text = "\n".join(lines)
    out_path = "/home/l/code/AutoResearch/runs/probe_press_button_reach.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text, flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
