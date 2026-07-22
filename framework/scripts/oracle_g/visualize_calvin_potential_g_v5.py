#!/usr/bin/env python3
"""Visualize CALVIN articulation potential-field G-v5 tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

POLICY_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "policy_state_bc"
sys.path.insert(0, POLICY_SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as pol  # noqa: E402


TASKS = ("close_drawer", "move_slider_left", "move_slider_right", "open_drawer")
TOKEN_DIM = pol.ARTICULATION_V5_TOKEN_DIM
TOKEN_COUNT = pol.ARTICULATION_V5_TOKEN_COUNT


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def episode_file(split_dir: Path, idx: int) -> Path:
    return split_dir / f"episode_{idx:07d}.npz"


def load_task_states(data_root: Path, split: str, tasks: tuple[str, ...], samples_per_task: int) -> dict[str, list[dict[str, Any]]]:
    split_dir = data_root / split
    segments = pol.load_lang_segments(split_dir)
    by_task = {task: [] for task in tasks}
    for seg in segments:
        task = seg["task"]
        if task not in by_task or len(by_task[task]) >= samples_per_task:
            continue
        idx = int(seg["start"])
        f = episode_file(split_dir, idx)
        if not f.exists():
            continue
        try:
            d = np.load(f, allow_pickle=True)
            robot = np.asarray(d["robot_obs"], dtype=np.float32)
            scene = np.asarray(d["scene_obs"], dtype=np.float32)
        except Exception:
            continue
        by_task[task].append({"task": task, "episode": idx, "robot_obs": robot, "scene_obs": scene})
        if all(len(items) >= samples_per_task for items in by_task.values()):
            break
    return by_task


def joint_points(joint_model: dict[str, Any], q: float) -> dict[str, np.ndarray]:
    intercept = np.asarray(joint_model["intercept"], dtype=np.float32)
    slope = np.asarray(joint_model["slope"], dtype=np.float32)
    q_min = float(joint_model.get("q_min_sample", 0.0))
    q_max = float(joint_model.get("q_max_sample", q_min + float(joint_model["joint_range"])))
    return {
        "current": intercept + slope * np.float32(q),
        "min": intercept + slope * np.float32(q_min),
        "max": intercept + slope * np.float32(q_max),
    }


def decode_tokens(robot: np.ndarray, tokens_flat: np.ndarray) -> list[dict[str, Any]]:
    tokens = tokens_flat.reshape(TOKEN_COUNT, TOKEN_DIM)
    decoded: list[dict[str, Any]] = []
    for i, token in enumerate(tokens):
        anchor_delta = token[:3]
        attractive_flow = token[3:6]
        repulsive_flow = token[6:9]
        attract_weight = float(token[9])
        repel_weight = float(token[10])
        decoded.append(
            {
                "index": i,
                "anchor": robot[:3] + anchor_delta,
                "anchor_delta": anchor_delta,
                "attractive_flow": attractive_flow,
                "repulsive_flow": repulsive_flow,
                "attract_weight": attract_weight,
                "repel_weight": repel_weight,
            }
        )
    return decoded


def axis_limits(points: list[np.ndarray], dims: tuple[int, int], pad: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    arr = np.stack(points)
    lo = np.min(arr[:, list(dims)], axis=0) - pad
    hi = np.max(arr[:, list(dims)], axis=0) + pad
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def draw_view(
    ax: plt.Axes,
    sample: dict[str, Any],
    geometry: dict[str, Any],
    decoded: list[dict[str, Any]],
    dims: tuple[int, int],
    labels: tuple[str, str],
    arrow_scale: float,
) -> None:
    task = sample["task"]
    robot = sample["robot_obs"]
    scene = sample["scene_obs"]
    joint_name = pol.ARTICULATION_TASK_TO_JOINT[task]
    joint_model = geometry["joints"][joint_name]
    q = float(scene[int(joint_model["joint_index"])])
    target = float(pol.articulation_target_controls(task, scene)[int(joint_model["joint_index"])])
    residual_norm = (target - q) / max(float(joint_model["joint_range"]), 1e-6)
    pts = joint_points(joint_model, q)

    all_points = [robot[:3], pts["current"], pts["min"], pts["max"]]
    for token in decoded:
        all_points.append(token["anchor"])
    xlim, ylim = axis_limits(all_points, dims)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    d0, d1 = dims
    ax.plot([pts["min"][d0], pts["max"][d0]], [pts["min"][d1], pts["max"][d1]], color="#777777", lw=2.0, alpha=0.55)
    ax.scatter([pts["current"][d0]], [pts["current"][d1]], s=80, marker="*", color="#1f77b4", label="part(q)", zorder=4)
    ax.scatter([robot[d0]], [robot[d1]], s=65, marker="x", color="#111111", label="eef", zorder=5)

    for token in decoded:
        anchor = token["anchor"]
        aw = token["attract_weight"]
        rw = token["repel_weight"]
        if aw == 0.0 and rw == 0.0:
            continue
        ax.scatter([anchor[d0]], [anchor[d1]], s=34, color="#444444", zorder=3)
        ax.text(anchor[d0], anchor[d1], str(token["index"]), fontsize=7, color="#222222")
        att = token["attractive_flow"] * arrow_scale
        rep = token["repulsive_flow"] * arrow_scale
        if float(np.linalg.norm(att)) > 1e-5:
            ax.arrow(
                anchor[d0],
                anchor[d1],
                att[d0],
                att[d1],
                color="#2ca02c",
                width=0.0015,
                head_width=0.012,
                length_includes_head=True,
                alpha=0.9,
            )
        if float(np.linalg.norm(rep)) > 1e-5:
            ax.arrow(
                anchor[d0],
                anchor[d1],
                rep[d0],
                rep[d1],
                color="#d62728",
                width=0.0015,
                head_width=0.012,
                length_includes_head=True,
                alpha=0.9,
            )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(f"{task} ep={sample['episode']} q={q:.3f} target={target:.3f} res={residual_norm:+.2f}", fontsize=9)


def render_task_figure(
    task: str,
    samples: list[dict[str, Any]],
    geometry: dict[str, Any],
    out_dir: Path,
    arrow_scale: float,
) -> list[dict[str, Any]]:
    rows = 2
    cols = max(1, len(samples))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 8.0), squeeze=False, constrained_layout=True)
    records: list[dict[str, Any]] = []

    for col, sample in enumerate(samples):
        robot = sample["robot_obs"]
        scene = sample["scene_obs"]
        tokens_flat = pol.make_articulation_potential_g_v5(
            robot[None, :],
            scene[None, :],
            np.asarray([task], dtype=object),
            geometry,
        )[0]
        decoded = decode_tokens(robot, tokens_flat)
        draw_view(axes[0][col], sample, geometry, decoded, dims=(0, 1), labels=("x", "y"), arrow_scale=arrow_scale)
        draw_view(axes[1][col], sample, geometry, decoded, dims=(0, 2), labels=("x", "z"), arrow_scale=arrow_scale)
        records.append(
            {
                "task": task,
                "episode": sample["episode"],
                "robot_eef": robot[:3],
                "scene_controls": scene[:2],
                "tokens": decoded,
            }
        )

    fig.suptitle(f"CALVIN G-v5 potential field: {task} (green=attraction, red=repulsion)", fontsize=14)
    path = out_dir / f"g_v5_potential_{task}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--geometry-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--arrow-scale", type=float, default=0.055)
    args = parser.parse_args()

    geometry = pol.load_articulation_geometry(args.geometry_json)
    if geometry is None:
        raise RuntimeError("Missing geometry")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    states_by_task = load_task_states(args.data_root, args.split, TASKS, args.samples_per_task)
    summary: dict[str, Any] = {
        "data_root": args.data_root,
        "geometry_json": args.geometry_json,
        "split": args.split,
        "samples_per_task": args.samples_per_task,
        "arrow_scale": args.arrow_scale,
        "tasks": {},
    }
    all_records: list[dict[str, Any]] = []
    for task in TASKS:
        records = render_task_figure(task, states_by_task[task], geometry, args.out_dir, args.arrow_scale)
        summary["tasks"][task] = {
            "num_samples": len(records),
            "figure": args.out_dir / f"g_v5_potential_{task}.png",
            "episodes": [r["episode"] for r in records],
        }
        all_records.extend(records)

    with open(args.out_dir / "g_v5_potential_tokens.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, default=json_default)
    with open(args.out_dir / "g_v5_potential_visualization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
