#!/usr/bin/env python3
"""Plot EE/cube/goal trajectories from eval_diagnostics.json (offline, no Isaac)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


SUCCESS_RADIUS_M = 0.02


def plot_episode(
    ax_xy,
    ax_z,
    traj: list[dict],
    *,
    title: str,
    final_dist: float | None = None,
    success_radius_m: float = SUCCESS_RADIUS_M,
) -> None:
    ee_x = [s["ee_x"] for s in traj]
    ee_y = [s["ee_y"] for s in traj]
    cube_x = [s["cube_x"] for s in traj]
    cube_y = [s["cube_y"] for s in traj]
    cube_z = [s["cube_z"] for s in traj]
    goal_x = traj[0]["goal_x"]
    goal_y = traj[0]["goal_y"]
    goal_z = traj[0]["goal_z"]
    steps = [s["step"] for s in traj]
    dist_series = [s["dist_to_goal"] for s in traj]

    ax_xy.plot(ee_x, ee_y, "b-", label="EE", linewidth=1.2, alpha=0.8)
    ax_xy.plot(cube_x, cube_y, "r-", label="cube path", linewidth=1.8)
    ax_xy.scatter([cube_x[-1]], [cube_y[-1]], c="darkred", s=60, zorder=6, label="cube end")
    ax_xy.scatter([goal_x], [goal_y], c="gold", s=200, marker="*", edgecolors="red", linewidths=1.2, label="GOAL", zorder=7)
    ax_xy.add_patch(
        Circle(
            (goal_x, goal_y),
            success_radius_m,
            fill=False,
            edgecolor="limegreen",
            linewidth=2,
            linestyle="--",
            label=f"success ({success_radius_m * 100:.0f}cm)",
        )
    )
    ax_xy.plot([cube_x[-1], goal_x], [cube_y[-1], goal_y], "m--", linewidth=1.5, alpha=0.9, label="final gap")
    final_d = final_dist if final_dist is not None else dist_series[-1]
    mid_x = 0.5 * (cube_x[-1] + goal_x)
    mid_y = 0.5 * (cube_y[-1] + goal_y)
    ax_xy.annotate(
        f"{final_d * 100:.1f} cm",
        (mid_x, mid_y),
        fontsize=9,
        color="purple",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
    )
    ax_xy.scatter([cube_x[0]], [cube_y[0]], c="orange", s=40, label="cube start", zorder=4)
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title(title)
    ax_xy.legend(loc="best", fontsize=7)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.3)

    ax_z.plot(steps, cube_z, "r-", label="cube z")
    ax_z.axhline(goal_z, color="green", linestyle="--", linewidth=1.5, label="goal z")
    ax_z.axhline(0.04, color="gray", linestyle=":", label="table")
    ax_z2 = ax_z.twinx()
    ax_z2.plot(steps, [d * 100 for d in dist_series], "m-", alpha=0.7, label="dist goal (cm)")
    ax_z2.set_ylabel("dist to goal (cm)", color="purple")
    ax_z2.tick_params(axis="y", labelcolor="purple")
    ax_z.set_xlabel("step")
    ax_z.set_ylabel("z (m)")
    lines1, labels1 = ax_z.get_legend_handles_labels()
    lines2, labels2 = ax_z2.get_legend_handles_labels()
    ax_z.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=7)
    ax_z.grid(True, alpha=0.3)


def _load_episodes(json_path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(json_path.read_text())
    return data, data.get("episodes", [])


def main() -> int:
    p = argparse.ArgumentParser(description="Plot diagnostic eval trajectories with goal distance")
    p.add_argument("json_path", type=Path)
    p.add_argument("--compare", type=Path, default=None, help="Second eval JSON (e.g. B7 vs B1)")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="Episode ids (default: first 4)")
    p.add_argument("--success-radius", type=float, default=SUCCESS_RADIUS_M)
    args = p.parse_args()

    data, episodes = _load_episodes(args.json_path)
    if not episodes:
        raise SystemExit("No episodes in JSON")

    compare_data = None
    compare_eps: dict[int, dict] = {}
    if args.compare:
        compare_data, compare_list = _load_episodes(args.compare)
        compare_eps = {e["episode_id"]: e for e in compare_list}

    ep_ids = args.episodes if args.episodes else [e["episode_id"] for e in episodes[:4]]
    selected = [e for e in episodes if e["episode_id"] in ep_ids]
    n = len(selected)
    cols = 4 if args.compare else 2
    fig, axes = plt.subplots(n, cols, figsize=(5 * cols, 4 * n), squeeze=False)
    tag = Path(data.get("checkpoint", "ckpt")).parent.name
    fig.suptitle(
        f"{tag} | success@2cm={data['aggregate'].get('success_rate', 0):.0%} "
        f"success@5cm={data['aggregate'].get('success_rate_5cm', 0):.0%} "
        f"grasp={data['aggregate'].get('grasp_rate', 0):.0%} "
        f"wander={data['aggregate'].get('mean_post_grasp_wander_m', 0):.3f}m",
        fontsize=12,
    )

    for row, ep in enumerate(selected):
        traj = ep.get("trajectory", [])
        if not traj:
            continue
        title = (
            f"{tag} ep{ep['episode_id']} | final={ep['final_dist_to_goal'] * 100:.1f}cm "
            f"max_z={ep['max_cube_z']:.2f}m"
        )
        plot_episode(
            axes[row, 0],
            axes[row, 1],
            traj,
            title=title,
            final_dist=ep["final_dist_to_goal"],
            success_radius_m=args.success_radius,
        )
        if args.compare:
            tag2 = Path(compare_data.get("checkpoint", "cmp")).parent.name  # type: ignore[union-attr]
            cep = compare_eps.get(ep["episode_id"])
            if cep and cep.get("trajectory"):
                title2 = (
                    f"{tag2} ep{cep['episode_id']} | final={cep['final_dist_to_goal'] * 100:.1f}cm "
                    f"max_z={cep['max_cube_z']:.2f}m"
                )
                plot_episode(
                    axes[row, 2],
                    axes[row, 3],
                    cep["trajectory"],
                    title=title2,
                    final_dist=cep["final_dist_to_goal"],
                    success_radius_m=args.success_radius,
                )
            else:
                axes[row, 2].text(0.5, 0.5, "no matching ep", ha="center", va="center")
                axes[row, 3].axis("off")

    fig.tight_layout()
    if args.output:
        out = args.output
    elif args.compare:
        out = args.json_path.parent / "trajectory_compare_b1_b7.png"
    else:
        out = args.json_path.parent / "trajectory_plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"[OK] Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
