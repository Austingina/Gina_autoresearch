#!/usr/bin/env python3
"""Offline visualization of collected lift teacher demos (maskdepth HDF5)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]


def _demo_sort_key(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def mask_to_rgb(mask_hwc: np.ndarray) -> np.ndarray:
    x = np.clip(mask_hwc, 0.0, 1.0)
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        return x if x.shape[-1] == 3 else np.repeat(x, 3, axis=-1)
    return x


def load_episode_summary(hdf5_path: Path) -> list[dict]:
    rows: list[dict] = []
    with h5py.File(hdf5_path, "r") as f:
        for name in sorted(f["data"].keys(), key=_demo_sort_key):
            g = f["data"][name]
            st = np.asarray(g["obs/student_state"], dtype=np.float32)
            phase = np.asarray(g["goals/phase"], dtype=np.float32).reshape(-1)
            tp_mask = np.asarray(g["obs/third_person_mask"])
            ee_mask = np.asarray(g["obs/ee_mask"])
            success = bool(g.attrs.get("success", False))
            cube0 = st[0, 4:7]
            goal0 = st[0, 7:10]
            ee0 = st[0, 0:3]
            rows.append(
                {
                    "name": name,
                    "success": success,
                    "length": len(st),
                    "cube0": cube0,
                    "goal0": goal0,
                    "ee0": ee0,
                    "student_state": st,
                    "phase": phase,
                    "tp_mask0": tp_mask[0],
                    "ee_mask0": ee_mask[0],
                }
            )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Visualize collected teacher demo HDF5")
    p.add_argument(
        "--hdf5",
        type=Path,
        default=ROOT / "data/lift/s1_random_cube/demos/s1_random_cube_20260622_170659.hdf5",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--num-mask-examples", type=int, default=8)
    p.add_argument("--num-trajectories", type=int, default=4)
    args = p.parse_args()

    if not args.hdf5.is_file():
        print(f"[ERROR] not found: {args.hdf5}", file=sys.stderr)
        return 1

    episodes = load_episode_summary(args.hdf5)
    if not episodes:
        print("[ERROR] no episodes", file=sys.stderr)
        return 1

    out = args.output or (
        ROOT / "outputs/figures" / f"demo_collection_{args.hdf5.stem}.png"
    )

    cube0 = np.stack([e["cube0"] for e in episodes])
    goal0 = episodes[0]["goal0"]
    n_succ = sum(e["success"] for e in episodes)

    # Pick diverse cube spawns for mask panel (spread in y)
    order = np.argsort(cube0[:, 1])
    pick_idx = np.linspace(0, len(order) - 1, min(args.num_mask_examples, len(order)), dtype=int)
    mask_picks = [episodes[order[i]] for i in pick_idx]

    succ_eps = [e for e in episodes if e["success"]]
    traj_picks = succ_eps[: args.num_trajectories] if succ_eps else episodes[: args.num_trajectories]

    n_mask = len(mask_picks)
    n_traj = len(traj_picks)
    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs = GridSpec(3, max(n_mask, 4), figure=fig, height_ratios=[1.1, 1.0, 1.2], hspace=0.45, wspace=0.35)

    fig.suptitle(
        f"Teacher demos: {args.hdf5.name}\n"
        f"n={len(episodes)} success={n_succ} ({100*n_succ/len(episodes):.0f}%)  "
        f"goal=({goal0[0]:.3f},{goal0[1]:.3f},{goal0[2]:.3f})",
        fontsize=13,
        fontweight="bold",
    )

    # Row 0: cube spawn scatter
    ax_spawn = fig.add_subplot(gs[0, :2])
    ok = cube0[[i for i, e in enumerate(episodes) if e["success"]]]
    bad = cube0[[i for i, e in enumerate(episodes) if not e["success"]]]
    if len(ok):
        ax_spawn.scatter(ok[:, 0], ok[:, 1], c="#2ca02c", s=36, alpha=0.8, label=f"success ({len(ok)})")
    if len(bad):
        ax_spawn.scatter(bad[:, 0], bad[:, 1], c="#d62728", s=36, alpha=0.7, label=f"fail ({len(bad)})")
    ax_spawn.scatter([goal0[0]], [goal0[1]], c="gold", s=220, marker="*", edgecolors="red", linewidths=1.2, label="lift goal", zorder=6)
    ax_spawn.axvline(0.5, color="gray", ls=":", alpha=0.5, label="default cube x=0.5")
    ax_spawn.axhline(0.0, color="gray", ls=":", alpha=0.5)
    ax_spawn.set_xlabel("cube start x (m)")
    ax_spawn.set_ylabel("cube start y (m)")
    ax_spawn.set_title("Cube spawn at episode start (teacher collection)")
    ax_spawn.set_aspect("equal", adjustable="box")
    ax_spawn.grid(True, alpha=0.3)
    ax_spawn.legend(fontsize=8, loc="upper right")

    ax_hist = fig.add_subplot(gs[0, 2:])
    ax_hist.hist(cube0[:, 0], bins=20, alpha=0.7, color="#1f77b4", label="x")
    ax_hist.hist(cube0[:, 1], bins=20, alpha=0.7, color="#ff7f0e", label="y")
    ax_hist.axvline(0.5, color="gray", ls="--", label="default x")
    ax_hist.set_xlabel("position (m)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Spawn distribution")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    # Row 1: third-person masks at t=0
    for col, ep in enumerate(mask_picks):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(mask_to_rgb(ep["tp_mask0"]))
        c = ep["cube0"]
        ax.set_title(
            f"{ep['name']}\n{'OK' if ep['success'] else 'FAIL'}  "
            f"cube=({c[0]:.2f},{c[1]:.2f})",
            fontsize=7,
        )
        ax.axis("off")
    if n_mask < gs.ncols:
        for col in range(n_mask, gs.ncols):
            fig.add_subplot(gs[1, col]).axis("off")

    # Row 2: trajectories (top-down)
    for col, ep in enumerate(traj_picks):
        ax = fig.add_subplot(gs[2, col])
        st = ep["student_state"]
        ee_x, ee_y = st[:, 0], st[:, 1]
        cube_x, cube_y = st[:, 4], st[:, 5]
        goal = ep["goal0"]
        ax.plot(ee_x, ee_y, "b-", lw=1.2, alpha=0.8, label="EE")
        ax.plot(cube_x, cube_y, "g-", lw=1.5, label="cube")
        ax.scatter([cube_x[0]], [cube_y[0]], c="orange", s=40, zorder=5)
        ax.scatter([goal[0]], [goal[1]], c="gold", s=120, marker="*", edgecolors="red", zorder=6)
        ax.set_title(f"{ep['name']} ({'OK' if ep['success'] else 'FAIL'})", fontsize=8)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[OK] {args.hdf5}")
    print(f"     episodes={len(episodes)} success={n_succ}")
    print(f"     cube x [{cube0[:,0].min():.3f}, {cube0[:,0].max():.3f}]  "
          f"y [{cube0[:,1].min():.3f}, {cube0[:,1].max():.3f}]")
    print(f"[OK] Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
