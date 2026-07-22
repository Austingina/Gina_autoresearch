#!/usr/bin/env python3
"""Measure state overlap and task leakage for inverse CALVIN goal pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402
import train_eval_state_chunk_flow_policy as chunk  # noqa: E402


PAIR_SPECS = {
    "slider": ("move_slider_left", "move_slider_right", 0),
    "lightbulb": ("turn_off_lightbulb", "turn_on_lightbulb", 4),
    "led": ("turn_off_led", "turn_on_led", 5),
    "red_block": ("push_red_block_left", "push_red_block_right", slice(6, 9)),
    "blue_block": ("push_blue_block_left", "push_blue_block_right", slice(12, 15)),
    "pink_block": ("push_pink_block_left", "push_pink_block_right", slice(18, 21)),
}


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def segment_folds(segment_ids: np.ndarray, labels: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    output: list[list[int]] = [[] for _ in range(folds)]
    for label in np.unique(labels):
        label_segments = np.unique(segment_ids[labels == label])
        rng.shuffle(label_segments)
        for fold, values in enumerate(np.array_split(label_segments, folds)):
            output[fold].extend(int(value) for value in values)
    return [np.asarray(values, dtype=np.int64) for values in output]


def ridge_binary_cv(
    features: np.ndarray,
    labels: np.ndarray,
    segment_ids: np.ndarray,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    results: list[float] = []
    recalls: list[list[float]] = []
    for test_segments in segment_folds(segment_ids, labels, folds, seed):
        test = np.isin(segment_ids, test_segments)
        train = ~test
        mean = np.mean(features[train], axis=0)
        std = np.std(features[train], axis=0)
        std[std < 1e-6] = 1.0
        train_x = (features[train] - mean) / std
        test_x = (features[test] - mean) / std
        train_x = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
        test_x = np.concatenate([test_x, np.ones((len(test_x), 1))], axis=1)
        regularizer = np.eye(train_x.shape[1]) * 1e-3
        regularizer[-1, -1] = 0.0
        weights = np.linalg.solve(train_x.T @ train_x + regularizer, train_x.T @ labels[train])
        pred = (test_x @ weights >= 0.5).astype(np.int64)
        fold_recalls = [float(np.mean(pred[labels[test] == value] == value)) for value in (0, 1)]
        recalls.append(fold_recalls)
        results.append(float(np.mean(fold_recalls)))
    recalls_array = np.asarray(recalls)
    return {
        "folds": int(folds),
        "balanced_accuracy": stats(np.asarray(results)),
        "class0_recall": stats(recalls_array[:, 0]),
        "class1_recall": stats(recalls_array[:, 1]),
    }


def nearest_indices(query: np.ndarray, reference: np.ndarray, batch_size: int = 512) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(query), batch_size):
        batch = query[start : start + batch_size]
        distance_sq = np.sum((batch[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        output.append(np.argmin(distance_sq, axis=1))
    return np.concatenate(output)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    result = np.zeros(len(a), dtype=np.float64)
    valid = denom > 1e-8
    result[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return result


def pair_report(
    arrays: dict[str, np.ndarray],
    task0: str,
    task1: str,
    mechanism_index: int | slice,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    names = arrays["task_name"].astype(str)
    selected = np.isin(names, [task0, task1])
    robot = arrays["robot_obs"][selected].astype(np.float64)
    scene = arrays["scene_obs"][selected].astype(np.float64)
    labels = (names[selected] == task1).astype(np.int64)
    segment_ids = arrays["segment_id"][selected]
    action = arrays["action_chunks"][selected, 0, :3].astype(np.float64)

    mechanism_state = np.asarray(scene[:, mechanism_index], dtype=np.float64)
    if mechanism_state.ndim == 1:
        mechanism_state = mechanism_state[:, None]
    compact = np.concatenate([robot[:, :3], mechanism_state], axis=1)
    full = np.concatenate([robot, scene], axis=1)
    state_classification = {
        "eef_xyz_plus_target_mechanism": ridge_binary_cv(compact, labels, segment_ids, folds, seed),
        "full_robot_plus_scene": ridge_binary_cv(full, labels, segment_ids, folds, seed),
    }

    compact_std = np.std(compact, axis=0)
    compact_std[compact_std < 1e-6] = 1.0
    standardized = (compact - np.mean(compact, axis=0)) / compact_std
    mask0 = labels == 0
    mask1 = labels == 1
    index0_to_1 = nearest_indices(standardized[mask0], standardized[mask1])
    index1_to_0 = nearest_indices(standardized[mask1], standardized[mask0])

    def direction(source_mask: np.ndarray, target_mask: np.ndarray, nearest: np.ndarray) -> dict[str, Any]:
        source_robot = robot[source_mask]
        target_robot = robot[target_mask][nearest]
        source_scene = scene[source_mask]
        target_scene = scene[target_mask][nearest]
        source_action = action[source_mask]
        target_action = action[target_mask][nearest]
        eef_distance = np.linalg.norm(source_robot[:, :3] - target_robot[:, :3], axis=1)
        source_mechanism = np.asarray(source_scene[:, mechanism_index], dtype=np.float64)
        target_mechanism = np.asarray(target_scene[:, mechanism_index], dtype=np.float64)
        if source_mechanism.ndim == 1:
            mechanism_difference = np.abs(source_mechanism - target_mechanism)
            mechanism_span = max(float(np.ptp(mechanism_state[:, 0])), 1e-6)
        else:
            mechanism_difference = np.linalg.norm(source_mechanism - target_mechanism, axis=1)
            mechanism_span = max(float(np.linalg.norm(np.ptp(mechanism_state, axis=0))), 1e-6)
        matched_2cm = (eef_distance < 0.02) & (mechanism_difference < 0.10 * mechanism_span)
        matched_5cm = (eef_distance < 0.05) & (mechanism_difference < 0.10 * mechanism_span)
        return {
            "samples": int(np.sum(source_mask)),
            "nearest_eef_distance_m": stats(eef_distance),
            "nearest_mechanism_abs_difference": stats(mechanism_difference),
            "mechanism_span": mechanism_span,
            "matched_2cm_10pct_mechanism_fraction": float(np.mean(matched_2cm)),
            "matched_5cm_10pct_mechanism_fraction": float(np.mean(matched_5cm)),
            "first_action_cosine_at_5cm_matches": stats(cosine(source_action[matched_5cm], target_action[matched_5cm])),
        }

    return {
        "task0": task0,
        "task1": task1,
        "windows": int(np.sum(selected)),
        "segments": int(len(np.unique(segment_ids))),
        "state_only_goal_classification": state_classification,
        "task0_to_task1_overlap": direction(mask0, mask1, index0_to_1),
        "task1_to_task0_overlap": direction(mask1, mask0, index1_to_0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--articulation-geometry-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    text_to_task, _ = base.load_eval_annotation_maps(args.calvin_code.resolve())
    segments = base.load_lang_segments(args.data_root.resolve() / "training")
    task_to_id = base.build_task_index(segments, text_to_task)
    geometry = base.load_articulation_geometry(args.articulation_geometry_json.resolve())
    arrays, used_segments = chunk.collect_chunk_records(
        args.data_root.resolve(),
        task_to_id,
        chunk_size=args.chunk_size,
        max_windows=0,
        max_segments=0,
        stride=args.stride,
        seed=0,
        include_tasks={task for pair in PAIR_SPECS.values() for task in pair[:2]},
        articulation_geometry=geometry,
        minimal_g_scope="articulation",
    )
    report = {
        "windows": int(len(arrays["task_name"])),
        "segments": int(len(used_segments)),
        "pairs": {
            name: pair_report(arrays, task0, task1, mechanism_index, args.folds, args.seed)
            for name, (task0, task1, mechanism_index) in PAIR_SPECS.items()
        },
        "missing_inverse_pair": {
            "mechanism": "drawer",
            "present": "open_drawer",
            "held_out": "close_drawer",
            "interpretation": "No state-matched opposite drawer goal is supervised during policy training.",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "inverse_goal_state_overlap.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
