"""Inspect IsaacLab Sektion cabinet prims/joints for drawer/door extension."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Open-Drawer-Franka-IK-Abs-v0")
parser.add_argument("--contains", default="door,handle,drawer")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.sim.utils.stage import get_current_stage

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()
    stage = get_current_stage()
    needles = tuple(x.strip().lower() for x in args.contains.split(",") if x.strip())
    print(f"task={args.task}")
    print("=== prims containing", needles, "===")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        low = path.lower()
        if "/world/envs/env_0/cabinet" in low and any(n in low for n in needles):
            print(path)
    print("=== cabinet joints ===")
    cabinet = env.unwrapped.scene["cabinet"]
    print("joint_names:", cabinet.data.joint_names)
    print("body_names:", cabinet.data.body_names)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
