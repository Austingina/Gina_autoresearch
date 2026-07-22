#!/usr/bin/env python3
"""Audit whether CALVIN demonstrations collapse to task-specific EEF templates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402


DEFAULT_TASKS = [
    "open_drawer",
    "move_slider_left",
    "move_slider_right",
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
]


def scalar_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def vector_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "std_norm": float(np.linalg.norm(np.std(values, axis=0))),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "span": np.ptp(values, axis=0).tolist(),
    }


def coefficient_of_variation(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values)) if len(values) else 0.0
    if abs(mean) < 1e-9:
        return None
    return float(np.std(values) / abs(mean))


def resample_path(path: np.ndarray, steps: int) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if len(path) == 1:
        return np.repeat(path, steps, axis=0)
    source = np.linspace(0.0, 1.0, len(path))
    target = np.linspace(0.0, 1.0, steps)
    return np.stack([np.interp(target, source, path[:, axis]) for axis in range(3)], axis=1)


def pca_summary(features: np.ndarray) -> dict[str, Any]:
    features = np.asarray(features, dtype=np.float64)
    if len(features) < 2:
        return {"samples": int(len(features)), "effective_rank": 0.0, "explained": []}
    centered = features - np.mean(features, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = singular**2
    total = float(np.sum(variance))
    if total <= 1e-12:
        return {"samples": int(len(features)), "effective_rank": 0.0, "explained": []}
    explained = variance / total
    nonzero = explained[explained > 1e-12]
    effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    return {
        "samples": int(len(features)),
        "effective_rank": effective_rank,
        "top1_explained": float(np.sum(explained[:1])),
        "top3_explained": float(np.sum(explained[:3])),
        "top5_explained": float(np.sum(explained[:5])),
        "explained": explained[:20].tolist(),
    }


def trajectory_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    trajectories = np.stack([record["trajectory"] for record in records])
    aligned = trajectories - trajectories[:, :1]
    displacement = trajectories[:, -1] - trajectories[:, 0]
    net = np.linalg.norm(displacement, axis=1)
    path_length = np.asarray([record["path_length"] for record in records], dtype=np.float64)
    frames = np.asarray([record["frames"] for record in records], dtype=np.float64)
    direction = np.zeros_like(displacement)
    valid = net > 1e-6
    direction[valid] = displacement[valid] / net[valid, None]
    mean_direction_raw = np.mean(direction[valid], axis=0) if np.any(valid) else np.zeros(3)
    resultant = float(np.linalg.norm(mean_direction_raw))
    mean_direction = mean_direction_raw / max(resultant, 1e-9)
    cosine = np.clip(direction[valid] @ mean_direction, -1.0, 1.0) if np.any(valid) else np.asarray([])
    angular_deg = np.degrees(np.arccos(cosine))

    absolute_template = np.mean(trajectories, axis=0, keepdims=True)
    aligned_template = np.mean(aligned, axis=0, keepdims=True)
    absolute_rmse = np.sqrt(np.mean((trajectories - absolute_template) ** 2, axis=(1, 2)))
    aligned_rmse = np.sqrt(np.mean((aligned - aligned_template) ** 2, axis=(1, 2)))
    phase_dispersion = np.sqrt(np.sum(np.var(aligned, axis=0), axis=1))
    median_path = float(np.median(path_length))

    normalized = aligned / np.maximum(net[:, None, None], 1e-6)
    normalized_template = np.mean(normalized, axis=0, keepdims=True)
    normalized_rmse = np.sqrt(np.mean((normalized - normalized_template) ** 2, axis=(1, 2)))

    return {
        "segments": int(len(records)),
        "frames": scalar_stats(frames),
        "start_xyz": vector_stats(trajectories[:, 0]),
        "end_xyz": vector_stats(trajectories[:, -1]),
        "displacement_xyz": vector_stats(displacement),
        "net_displacement_m": scalar_stats(net),
        "path_length_m": scalar_stats(path_length),
        "path_length_cv": coefficient_of_variation(path_length),
        "straightness": scalar_stats(net / np.maximum(path_length, 1e-9)),
        "direction": {
            "valid_segments": int(np.sum(valid)),
            "mean_unit_vector": mean_direction.tolist(),
            "mean_resultant_length": resultant,
            "angular_deviation_deg": scalar_stats(angular_deg),
        },
        "absolute_template_rmse_m": scalar_stats(absolute_rmse),
        "start_aligned_template_rmse_m": scalar_stats(aligned_rmse),
        "start_aligned_rmse_over_median_path": (
            None if median_path <= 1e-9 else float(np.median(aligned_rmse) / median_path)
        ),
        "displacement_normalized_template_rmse": scalar_stats(normalized_rmse),
        "phase_dispersion_m": scalar_stats(phase_dispersion),
        "start_aligned_trajectory_pca": pca_summary(aligned.reshape(len(aligned), -1)),
        "absolute_trajectory_pca": pca_summary(trajectories.reshape(len(trajectories), -1)),
        "mean_absolute_trajectory": absolute_template[0].tolist(),
        "mean_start_aligned_trajectory": aligned_template[0].tolist(),
    }


def stratified_nearest_centroid(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    class_folds: dict[str, list[np.ndarray]] = {}
    for label in classes:
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        class_folds[label] = list(np.array_split(indices, folds))

    fold_accuracy: list[float] = []
    fold_balanced_accuracy: list[float] = []
    all_true: list[str] = []
    all_pred: list[str] = []
    for fold in range(folds):
        test_indices = np.concatenate([class_folds[label][fold] for label in classes])
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[test_indices] = False
        train_x = features[train_mask]
        mean = np.mean(train_x, axis=0)
        std = np.std(train_x, axis=0)
        std[std < 1e-6] = 1.0
        train_z = (train_x - mean) / std
        test_z = (features[test_indices] - mean) / std
        train_y = labels[train_mask]
        centroids = np.stack([np.mean(train_z[train_y == label], axis=0) for label in classes])
        distance = np.sum((test_z[:, None] - centroids[None]) ** 2, axis=2)
        pred = classes[np.argmin(distance, axis=1)]
        true = labels[test_indices]
        recalls = [float(np.mean(pred[true == label] == label)) for label in classes]
        fold_accuracy.append(float(np.mean(pred == true)))
        fold_balanced_accuracy.append(float(np.mean(recalls)))
        all_true.extend(true.tolist())
        all_pred.extend(pred.tolist())

    all_true_array = np.asarray(all_true)
    all_pred_array = np.asarray(all_pred)
    return {
        "samples": int(len(labels)),
        "classes": classes.tolist(),
        "chance_balanced_accuracy": float(1.0 / len(classes)),
        "folds": int(folds),
        "accuracy": scalar_stats(np.asarray(fold_accuracy)),
        "balanced_accuracy": scalar_stats(np.asarray(fold_balanced_accuracy)),
        "per_task_recall": {
            label: float(np.mean(all_pred_array[all_true_array == label] == label)) for label in classes
        },
    }


def make_plots(records_by_task: dict[str, list[dict[str, Any]]], out_path: Path, seed: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib unavailable; skipping trajectory plot", flush=True)
        return

    tasks = sorted(records_by_task)
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(len(tasks), 3, figsize=(14, 3.2 * len(tasks)), squeeze=False)
    for row, task in enumerate(tasks):
        trajectories = np.stack([record["trajectory"] for record in records_by_task[task]])
        aligned = trajectories - trajectories[:, :1]
        chosen = rng.choice(len(trajectories), size=min(40, len(trajectories)), replace=False)
        panels = (
            (trajectories, (0, 1), "absolute XY"),
            (aligned, (0, 1), "start-aligned XY"),
            (aligned, (0, 2), "start-aligned XZ"),
        )
        for column, (values, dimensions, title) in enumerate(panels):
            axis = axes[row, column]
            for index in chosen:
                axis.plot(
                    values[index, :, dimensions[0]],
                    values[index, :, dimensions[1]],
                    color="#4c78a8",
                    alpha=0.18,
                    linewidth=0.8,
                )
            mean = np.mean(values, axis=0)
            axis.plot(mean[:, dimensions[0]], mean[:, dimensions[1]], color="#d62728", linewidth=2.2)
            axis.scatter(mean[0, dimensions[0]], mean[0, dimensions[1]], color="#2ca02c", s=24, zorder=3)
            axis.scatter(mean[-1, dimensions[0]], mean[-1, dimensions[1]], color="#d62728", s=24, zorder=3)
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=0.2)
            axis.set_title(f"{task}: {title}")
    figure.tight_layout()
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# CALVIN EEF Trajectory Distribution Audit",
        "",
        "This audit tests whether diverse segment starts still collapse to narrow, task-specific EEF motion templates.",
        "",
        "## Per-task trajectory concentration",
        "",
        "| Task | Segments | Frames p50 | Direction R | Angle p90 | Path CV | Aligned RMSE/path | End std norm | PCA effective rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, record in sorted(report["tasks"].items()):
        lines.append(
            f"| `{task}` | {record['segments']} | {record['frames']['p50']:.0f} | "
            f"{record['direction']['mean_resultant_length']:.3f} | "
            f"{record['direction']['angular_deviation_deg']['p90']:.1f} | "
            f"{record['path_length_cv']:.3f} | "
            f"{record['start_aligned_rmse_over_median_path']:.3f} | "
            f"{record['end_xyz']['std_norm']:.3f} | "
            f"{record['start_aligned_trajectory_pca']['effective_rank']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Task predictability without language or G",
            "",
            "| Feature | Balanced accuracy mean | Fold std | Chance |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, record in report["task_classification"].items():
        lines.append(
            f"| `{name}` | {record['balanced_accuracy']['mean']:.3f} | "
            f"{record['balanced_accuracy']['std']:.3f} | {record['chance_balanced_accuracy']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Reading the diagnostics",
            "",
            "- High initial-state accuracy means the physical state itself leaks task identity, allowing a policy to ignore G.",
            "- High start-aligned trajectory accuracy means each task has a recognizable motor template even after removing absolute EEF position.",
            "- Direction R near 1, low aligned RMSE/path, and low path-length CV indicate a narrow canonical trajectory family.",
            "- These are shortcut-risk diagnostics, not proof that a trained policy uses the shortcut; causal G corruption remains necessary.",
            "",
            "The trajectory overlay is saved as `eef_trajectory_templates.png`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--resample-steps", type=int, default=32)
    parser.add_argument("--classification-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split_dir = args.data_root.resolve() / "training"
    tasks = set(args.tasks)
    segments = [
        segment for segment in base.load_lang_segments(split_dir) if str(segment["task"]) in tasks
    ]
    records_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for segment_index, segment in enumerate(segments, start=1):
        task = str(segment["task"])
        start = int(segment["start"])
        end = int(segment["end"])
        eef: list[np.ndarray] = []
        initial_robot: np.ndarray | None = None
        initial_scene: np.ndarray | None = None
        for frame in range(start, end + 1):
            with np.load(base.episode_file(split_dir, frame), allow_pickle=True) as data:
                robot = np.asarray(data["robot_obs"], dtype=np.float64)
                scene = np.asarray(data["scene_obs"], dtype=np.float64)
            if initial_robot is None:
                initial_robot = robot
                initial_scene = scene
            eef.append(robot[:3])
        path = np.stack(eef)
        path_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        records_by_task[task].append(
            {
                "start": start,
                "end": end,
                "frames": int(len(path)),
                "trajectory": resample_path(path, args.resample_steps),
                "path_length": path_length,
                "initial_robot": initial_robot,
                "initial_scene": initial_scene,
            }
        )
        if segment_index == 1 or segment_index == len(segments) or segment_index % 100 == 0:
            print(f"[scan] {segment_index}/{len(segments)} task={task}", flush=True)

    task_summaries = {
        task: trajectory_summary(records) for task, records in sorted(records_by_task.items())
    }
    all_records = [record for task in sorted(records_by_task) for record in records_by_task[task]]
    labels = np.asarray(
        [task for task in sorted(records_by_task) for _ in records_by_task[task]], dtype=str
    )
    trajectories = np.stack([record["trajectory"] for record in all_records])
    aligned = trajectories - trajectories[:, :1]
    displacement = trajectories[:, -1] - trajectories[:, 0]
    path_features = np.stack(
        [
            np.concatenate(
                [
                    displacement[index],
                    [record["path_length"], float(record["frames"])],
                ]
            )
            for index, record in enumerate(all_records)
        ]
    )
    initial_eef = trajectories[:, 0]
    initial_state = np.stack(
        [np.concatenate([record["initial_robot"], record["initial_scene"]]) for record in all_records]
    )
    net = np.linalg.norm(displacement, axis=1)
    normalized_shape = aligned / np.maximum(net[:, None, None], 1e-6)
    classification_features = {
        "initial_eef_xyz": initial_eef,
        "initial_robot_plus_scene_state": initial_state,
        "eef_displacement_pathlength_duration": path_features,
        "start_aligned_eef_trajectory": aligned.reshape(len(aligned), -1),
        "displacement_normalized_eef_shape": normalized_shape.reshape(len(normalized_shape), -1),
    }
    classification = {
        name: stratified_nearest_centroid(
            features,
            labels,
            folds=args.classification_folds,
            seed=args.seed,
        )
        for name, features in classification_features.items()
    }

    report = {
        "data_root": args.data_root.resolve().as_posix(),
        "split": "training",
        "tasks_requested": args.tasks,
        "resample_steps": int(args.resample_steps),
        "segments": int(len(all_records)),
        "tasks": task_summaries,
        "task_classification": classification,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "eef_trajectory_distribution.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    rows: list[dict[str, Any]] = []
    for task, record in task_summaries.items():
        rows.append(
            {
                "task": task,
                "segments": record["segments"],
                "frames_p50": record["frames"]["p50"],
                "direction_resultant": record["direction"]["mean_resultant_length"],
                "direction_angle_p90_deg": record["direction"]["angular_deviation_deg"]["p90"],
                "path_length_p50_m": record["path_length_m"]["p50"],
                "path_length_cv": record["path_length_cv"],
                "aligned_template_rmse_p50_m": record["start_aligned_template_rmse_m"]["p50"],
                "aligned_rmse_over_median_path": record["start_aligned_rmse_over_median_path"],
                "end_xyz_std_norm_m": record["end_xyz"]["std_norm"],
                "pca_effective_rank": record["start_aligned_trajectory_pca"]["effective_rank"],
                "pca_top3_explained": record["start_aligned_trajectory_pca"]["top3_explained"],
            }
        )
    with open(args.out_dir / "eef_trajectory_distribution.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(report, args.out_dir / "summary.md")
    make_plots(records_by_task, args.out_dir / "eef_trajectory_templates.png", args.seed)
    print(json.dumps({"segments": len(all_records), "classification": classification}, indent=2))


if __name__ == "__main__":
    main()
