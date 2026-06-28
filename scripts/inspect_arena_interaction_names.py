#!/usr/bin/env python3
"""Visualize Arena articulation body/joint names with numbered 3D markers."""

from __future__ import annotations

import argparse
import os
import re
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect Arena interaction body/joint names in the 3D scene.")
parser.add_argument("--arena_task", choices=["button", "door", "microwave"], default="button")
parser.add_argument("--output_dir", type=str, default="/home/l/code/AutoResearch/runs/arena_name_inspect")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=30.0)
parser.add_argument("--marker_radius", type=float, default=0.018)
parser.add_argument("--max_steps", type=int, default=100000)
parser.add_argument("--all_bodies", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena_environments.franka_open_fridge_environment import FrankaOpenFridgeEnvironment  # noqa: E402
from isaaclab_arena_environments.franka_open_microwave_environment import FrankaOpenMicrowaveEnvironment  # noqa: E402
from isaaclab_arena_environments.press_button_environment import PressButtonEnvironment  # noqa: E402

_UI_WINDOW = None


def _env_args() -> SimpleNamespace:
    common = dict(
        enable_cameras=True,
        embodiment="franka",
        teleop_device=None,
        object=None,
        num_envs=args_cli.num_envs,
        env_spacing=args_cli.env_spacing,
        mimic=False,
        solve_relations=True,
        placement_seed=None,
        disable_fabric=getattr(args_cli, "disable_fabric", False),
        device=getattr(args_cli, "device", "cuda:0"),
        seed=getattr(args_cli, "seed", None),
    )
    if args_cli.arena_task == "button":
        return SimpleNamespace(
            **common,
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
        )
    if args_cli.arena_task == "door":
        return SimpleNamespace(
            **common,
            kitchen_layout=1,
            kitchen_style=2,
            franka_x=4.45,
            franka_y=-1.20,
            franka_z=None,
            franka_yaw_deg=90.0,
            reset_openness=0.0,
            episode_length_s=20.0,
            open_target_x=4.25,
            open_target_y=-0.62,
            open_target_z=1.05,
            platform_height=0.45,
            platform_size_x=0.80,
            platform_size_y=0.80,
        )
    return SimpleNamespace(**common)


def _build_env(env_args: SimpleNamespace):
    env_cls = {
        "button": PressButtonEnvironment,
        "door": FrankaOpenFridgeEnvironment,
        "microwave": FrankaOpenMicrowaveEnvironment,
    }[args_cli.arena_task]
    arena_env = env_cls().get_env(env_args)
    name, cfg = ArenaEnvBuilder(arena_env, env_args).build_registered()
    env = gym.make(name, cfg=cfg).unwrapped
    return env, arena_env


def _interesting(name: str) -> bool:
    lowered = name.lower()
    tokens = ("button", "btn", "door", "handle", "knob", "fridge", "microwave")
    return args_cli.all_bodies or any(token in lowered for token in tokens)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)[:96]


def _make_marker(stage, idx: int, name: str, pos: torch.Tensor, color: tuple[float, float, float]) -> None:
    path = f"/World/NameInspect/marker_{idx:03d}_{_safe_name(name)}"
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(args_cli.marker_radius)
    sphere.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _draw_points(records: list[dict]) -> None:
    try:
        from isaacsim.util.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
        points = [tuple(record["world_pos"]) for record in records]
        colors = [record["color"] + (1.0,) for record in records]
        sizes = [14.0 for _ in records]
        draw.draw_points(points, colors, sizes)
    except Exception as exc:
        print(f"[WARN] debug_draw points unavailable: {exc}", flush=True)


def _show_ui(records: list[dict]) -> None:
    global _UI_WINDOW
    try:
        import omni.ui as ui

        _UI_WINDOW = ui.Window("Arena Interaction Name Map", width=760, height=520)
        with _UI_WINDOW.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=2):
                    ui.Label("3D markers: index | articulation | body | joint_hint | world_xyz")
                    for record in records:
                        xyz = record["world_pos"]
                        ui.Label(
                            f"{record['idx']:03d} | {record['articulation']} | {record['body']} | "
                            f"{record['joint_hint']} | ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})"
                        )
    except Exception as exc:
        print(f"[WARN] UI label window unavailable: {exc}", flush=True)


def _collect_records(env) -> list[dict]:
    records: list[dict] = []
    scene = env.scene
    for articulation_name, articulation in scene.articulations.items():
        body_names = list(articulation.data.body_names)
        joint_names = list(articulation.data.joint_names)
        body_pos_w = articulation.data.body_pos_w[0].detach().cpu()
        for body_idx, body_name in enumerate(body_names):
            if not _interesting(body_name) and not _interesting(articulation_name):
                continue
            joint_hint = next((j for j in joint_names if j.replace("_joint", "") in body_name), "")
            idx = len(records) + 1
            color = (1.0, 0.15, 0.15) if "button" in body_name.lower() else (0.15, 0.85, 1.0)
            records.append(
                {
                    "idx": idx,
                    "articulation": articulation_name,
                    "body": body_name,
                    "joint_hint": joint_hint,
                    "world_pos": tuple(float(v) for v in body_pos_w[body_idx, :3]),
                    "color": color,
                }
            )
    return records


def _write_records(records: list[dict]) -> str:
    os.makedirs(args_cli.output_dir, exist_ok=True)
    path = os.path.join(args_cli.output_dir, f"{args_cli.arena_task}_name_markers.tsv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("idx\tarticulation\tbody\tjoint_hint\tworld_x\tworld_y\tworld_z\n")
        for record in records:
            x, y, z = record["world_pos"]
            f.write(
                f"{record['idx']:03d}\t{record['articulation']}\t{record['body']}\t{record['joint_hint']}\t"
                f"{x:.6f}\t{y:.6f}\t{z:.6f}\n"
            )
    return path


def main() -> None:
    env_args = _env_args()
    env, _ = _build_env(env_args)
    env.reset()

    stage = env.sim.stage
    records = _collect_records(env)
    for record in records:
        _make_marker(stage, record["idx"], record["body"], torch.tensor(record["world_pos"]), record["color"])
    _draw_points(records)
    _show_ui(records)
    path = _write_records(records)

    print(f"[INFO] Created {len(records)} 3D name markers.", flush=True)
    print(f"[INFO] Marker table: {path}", flush=True)
    for record in records:
        x, y, z = record["world_pos"]
        print(
            f"{record['idx']:03d} | {record['articulation']} | {record['body']} | "
            f"{record['joint_hint']} | ({x:.3f}, {y:.3f}, {z:.3f})",
            flush=True,
        )

    zero_action = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break
        env.step(zero_action)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
