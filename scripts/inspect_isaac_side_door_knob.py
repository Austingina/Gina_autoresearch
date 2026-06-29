"""Inspect Sektion side-door knob geometry and link/frame offsets."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Open-Drawer-Franka-IK-Abs-v0")
parser.add_argument("--door", choices=["left", "right"], default="right")
parser.add_argument("--hold", action="store_true", default=False, help="Keep the Isaac Sim window open after printing.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.sim.utils.stage import get_current_stage
from pxr import Gf, Usd, UsdGeom

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _fmt_vec(v) -> str:
    return f"({float(v[0]): .5f}, {float(v[1]): .5f}, {float(v[2]): .5f})"


def _world_translation(prim: Usd.Prim) -> Gf.Vec3d:
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return mat.ExtractTranslation()


def _print_prim(
    stage: Usd.Stage,
    bbox_cache: UsdGeom.BBoxCache,
    path: str,
    ref: Gf.Vec3d | None = None,
    ref_inv: Gf.Matrix4d | None = None,
) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        print(f"missing: {path}")
        return
    trans = _world_translation(prim)
    print(f"\n{path}")
    print(f"  type={prim.GetTypeName()} world_translation={_fmt_vec(trans)}")
    if ref is not None:
        print(f"  delta_from_link_origin={_fmt_vec(trans - ref)}")
    try:
        bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = bound.GetMin()
        mx = bound.GetMax()
        center = (mn + mx) * 0.5
        size = mx - mn
        print(f"  bbox_min={_fmt_vec(mn)}")
        print(f"  bbox_max={_fmt_vec(mx)}")
        print(f"  bbox_center={_fmt_vec(center)}")
        print(f"  bbox_size={_fmt_vec(size)}")
        if ref is not None:
            print(f"  bbox_center_delta_from_link_origin={_fmt_vec(center - ref)}")
        if ref_inv is not None:
            print(f"  bbox_center_in_link_frame={_fmt_vec(ref_inv.Transform(center))}")
    except Exception as exc:
        print(f"  bbox_error={exc}")


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()

    stage = get_current_stage()
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    base = f"/World/envs/env_0/Cabinet"
    door = args.door
    link_path = f"{base}/door_{door}_nob_link"
    link_prim = stage.GetPrimAtPath(link_path)
    link_origin = _world_translation(link_prim)
    link_world = UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    link_inv = link_world.GetInverse()

    cabinet = env.unwrapped.scene["cabinet"]
    print("joint_names:", cabinet.data.joint_names)
    print("body_names:", cabinet.data.body_names)
    print(f"door={door}")
    print(f"nob_link_origin={_fmt_vec(link_origin)}")

    paths = [
        f"{base}/door_{door}_link",
        f"{base}/door_{door}_link/door_{door}_nob_transform",
        link_path,
        f"{link_path}/visuals",
        f"{link_path}/collisions",
    ]
    for path in paths:
        _print_prim(stage, bbox_cache, path, ref=link_origin, ref_inv=link_inv)

    print("\nchildren under knob link:")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(link_path + "/"):
            _print_prim(stage, bbox_cache, path, ref=link_origin, ref_inv=link_inv)

    if args.hold:
        print("\n[INFO] Holding Isaac Sim window open. Close the window or press Ctrl+C to exit.")
        while simulation_app.is_running():
            env.unwrapped.sim.step(render=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
