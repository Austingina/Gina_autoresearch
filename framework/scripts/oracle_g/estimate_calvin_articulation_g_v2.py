#!/usr/bin/env python3
"""Estimate compact oracle articulation geometry for CALVIN demos.

The estimator fits a simple per-joint contact trajectory model:

    tcp_contact_point(q) ~= intercept + slope * q

using frames where the articulation joint is moving. This gives a compact
oracle source for:

- target_affordance: fitted contact point at current q minus eef position
- residual_motion: normalized remaining joint motion projected onto slope axis
- interaction_stage: approach / actuate / done
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
from typing import Any

import numpy as np


ARTICULATION_TASKS = ("open_drawer", "close_drawer", "move_slider_left", "move_slider_right")

TASK_TO_JOINT = {
    "move_slider_left": "slider",
    "move_slider_right": "slider",
    "open_drawer": "drawer",
    "close_drawer": "drawer",
}

JOINT_META = {
    "slider": {"index": 0, "target_open": 0.28, "range": 0.28},
    "drawer": {"index": 1, "target_open": 0.22, "range": 0.22},
}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def episode_file(split_dir: Path, index: int) -> Path:
    return split_dir / f"episode_{index:07d}.npz"


def load_lang_segments(split_dir: Path) -> list[dict[str, Any]]:
    lang_file = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    lang = np.load(lang_file, allow_pickle=True).item()
    tasks = lang["language"]["task"]
    anns = lang["language"]["ann"]
    indices = lang["info"]["indx"]
    return [
        {"task": tasks[i], "ann": anns[i], "start": int(start), "end": int(end)}
        for i, (start, end) in enumerate(indices)
    ]


def collect_actuation_points(
    data_root: Path,
    max_segments_per_task: int,
    seed: int,
    stride: int,
    motion_eps: float,
) -> dict[str, dict[str, np.ndarray]]:
    split_dir = data_root / "training"
    segments = [seg for seg in load_lang_segments(split_dir) if seg["task"] in ARTICULATION_TASKS]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seg in segments:
        by_task[seg["task"]].append(seg)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for task in ARTICULATION_TASKS:
        task_segments = list(by_task[task])
        rng.shuffle(task_segments)
        if max_segments_per_task > 0:
            task_segments = task_segments[:max_segments_per_task]
        selected.extend(task_segments)
        selected_counts[task] = len(task_segments)

    qs: dict[str, list[float]] = defaultdict(list)
    points: dict[str, list[np.ndarray]] = defaultdict(list)
    velocities: dict[str, list[np.ndarray]] = defaultdict(list)
    joint_deltas: dict[str, list[float]] = defaultdict(list)

    for seg in selected:
        task = seg["task"]
        joint = TASK_TO_JOINT[task]
        joint_idx = JOINT_META[joint]["index"]
        frames = list(range(seg["start"], seg["end"] + 1, max(1, stride)))
        if len(frames) < 2:
            continue
        obs: list[tuple[np.ndarray, np.ndarray]] = []
        for idx in frames:
            f = episode_file(split_dir, idx)
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
                robot = np.asarray(d["robot_obs"], dtype=np.float32)
                scene = np.asarray(d["scene_obs"], dtype=np.float32)
            except Exception:
                continue
            obs.append((robot, scene))
        for i in range(len(obs) - 1):
            robot, scene = obs[i]
            next_robot, next_scene = obs[i + 1]
            dq = float(next_scene[joint_idx] - scene[joint_idx])
            if abs(dq) < motion_eps:
                continue
            tcp = robot[:3].astype(np.float64)
            next_tcp = next_robot[:3].astype(np.float64)
            qs[joint].append(float(scene[joint_idx]))
            points[joint].append(tcp)
            velocities[joint].append(next_tcp - tcp)
            joint_deltas[joint].append(dq)

    out: dict[str, dict[str, np.ndarray]] = {}
    for joint in JOINT_META:
        out[joint] = {
            "q": np.asarray(qs[joint], dtype=np.float64),
            "points": np.asarray(points[joint], dtype=np.float64),
            "velocities": np.asarray(velocities[joint], dtype=np.float64),
            "joint_deltas": np.asarray(joint_deltas[joint], dtype=np.float64),
            "selected_counts": selected_counts,
        }
    return out


def fit_joint_model(q: np.ndarray, points: np.ndarray, joint_range: float) -> dict[str, Any]:
    if len(q) < 4:
        raise RuntimeError(f"Need at least 4 actuation samples, got {len(q)}")

    x = np.stack([np.ones_like(q), q], axis=1)
    coef, *_ = np.linalg.lstsq(x, points, rcond=None)
    intercept = coef[0]
    slope = coef[1]
    slope_norm = float(np.linalg.norm(slope))
    if slope_norm < 1e-8:
        centered = points - np.mean(points, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    else:
        axis = slope / slope_norm

    pred = x @ coef
    residual = np.linalg.norm(points - pred, axis=1)
    contact_threshold = max(0.04, float(np.percentile(residual, 90) + 0.02))
    q_mid = joint_range * 0.5

    return {
        "num_actuation_samples": int(len(q)),
        "intercept": intercept,
        "slope": slope,
        "axis_unit": axis,
        "slope_norm": slope_norm,
        "q_min_sample": float(np.min(q)),
        "q_max_sample": float(np.max(q)),
        "q_mid_point": intercept + slope * q_mid,
        "fit_residual_mean": float(np.mean(residual)),
        "fit_residual_p50": float(np.percentile(residual, 50)),
        "fit_residual_p90": float(np.percentile(residual, 90)),
        "contact_dist_threshold": contact_threshold,
    }


def estimate_geometry_model(
    data_root: Path,
    max_segments_per_task: int,
    seed: int,
    stride: int,
    motion_eps: float,
) -> dict[str, Any]:
    samples = collect_actuation_points(
        data_root=data_root,
        max_segments_per_task=max_segments_per_task,
        seed=seed,
        stride=stride,
        motion_eps=motion_eps,
    )
    joints = {}
    for joint, meta in JOINT_META.items():
        joint_samples = samples[joint]
        fit = fit_joint_model(joint_samples["q"], joint_samples["points"], joint_range=meta["range"])
        joints[joint] = {
            "joint_index": meta["index"],
            "joint_range": meta["range"],
            **fit,
        }
    return {
        "max_segments_per_task": max_segments_per_task,
        "seed": seed,
        "stride": stride,
        "motion_eps": motion_eps,
        "selected_counts": samples["slider"]["selected_counts"],
        "joints": joints,
    }


def compare_model(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for joint in JOINT_META:
        cand = candidate["joints"][joint]
        ref = reference["joints"][joint]
        cand_axis = np.asarray(cand["axis_unit"], dtype=np.float64)
        ref_axis = np.asarray(ref["axis_unit"], dtype=np.float64)
        axis_cos = float(np.dot(cand_axis, ref_axis) / (np.linalg.norm(cand_axis) * np.linalg.norm(ref_axis) + 1e-12))
        q_mid_delta = float(np.linalg.norm(np.asarray(cand["q_mid_point"]) - np.asarray(ref["q_mid_point"])))
        slope_delta = float(np.linalg.norm(np.asarray(cand["slope"]) - np.asarray(ref["slope"])))
        rows[joint] = {
            "axis_cosine_to_full": axis_cos,
            "q_mid_point_delta_m": q_mid_delta,
            "slope_delta": slope_delta,
            "threshold_delta_m": float(abs(cand["contact_dist_threshold"] - ref["contact_dist_threshold"])),
            "num_actuation_samples": cand["num_actuation_samples"],
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[20, 50, 100, 150])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--motion-eps", type=float, default=1e-4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_model = estimate_geometry_model(
        data_root=args.data_root,
        max_segments_per_task=0,
        seed=0,
        stride=args.stride,
        motion_eps=args.motion_eps,
    )
    with open(args.out_dir / "oracle_g_v2_articulation_geometry_full.json", "w", encoding="utf-8") as f:
        json.dump(full_model, f, indent=2, default=json_default)

    rows = []
    models = {}
    for size in args.sample_sizes:
        for seed in args.seeds:
            model = estimate_geometry_model(
                data_root=args.data_root,
                max_segments_per_task=size,
                seed=seed,
                stride=args.stride,
                motion_eps=args.motion_eps,
            )
            models[f"size_{size}_seed_{seed}"] = model
            comparison = compare_model(model, full_model)
            for joint, metrics in comparison.items():
                rows.append(
                    {
                        "max_segments_per_task": size,
                        "seed": seed,
                        "joint": joint,
                        **metrics,
                    }
                )

    with open(args.out_dir / "oracle_g_v2_articulation_geometry_models.json", "w", encoding="utf-8") as f:
        json.dump(models, f, indent=2, default=json_default)
    with open(args.out_dir / "oracle_g_v2_articulation_geometry_stability.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out_dir / "oracle_g_v2_articulation_geometry_stability.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=json_default)

    print(json.dumps({"full_model": full_model, "stability_rows": rows}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
