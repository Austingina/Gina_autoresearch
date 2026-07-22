#!/usr/bin/env python3
"""Measure whether held-out minimal-G variants lie near seen-task G support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402
import train_eval_state_chunk_flow_policy as chunk  # noqa: E402


SEEN_TASKS = ("open_drawer", "move_slider_left", "move_slider_right")
HELD_OUT_TASK = "close_drawer"


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def sample_indices(task_names: np.ndarray, task: str, limit: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.flatnonzero(task_names.astype(str) == task)
    if limit > 0 and len(indices) > limit:
        indices = np.sort(rng.choice(indices, size=limit, replace=False))
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--articulation-geometry-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-samples-per-task", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    text_to_task, _ = base.load_eval_annotation_maps(args.calvin_code.resolve())
    segments = base.load_lang_segments(args.data_root.resolve() / "training")
    task_to_id = base.build_task_index(segments, text_to_task)
    geometry = base.load_articulation_geometry(args.articulation_geometry_json.resolve())
    if geometry is None:
        raise RuntimeError("Missing articulation geometry")

    arrays, used_segments = chunk.collect_chunk_records(
        args.data_root.resolve(),
        task_to_id,
        chunk_size=args.chunk_size,
        max_windows=0,
        max_segments=0,
        stride=args.stride,
        seed=args.seed,
        include_tasks=set(SEEN_TASKS + (HELD_OUT_TASK,)),
        articulation_geometry=geometry,
    )
    task_names = arrays["task_name"].astype(str)
    rng = np.random.default_rng(args.seed)
    selected = {
        task: sample_indices(task_names, task, args.max_samples_per_task, rng)
        for task in SEEN_TASKS + (HELD_OUT_TASK,)
    }

    seen_indices = np.concatenate([selected[task] for task in SEEN_TASKS])
    seen_g = arrays["articulation_minimal_g"][seen_indices].astype(np.float32)
    mean = seen_g.mean(axis=0)
    std = np.maximum(seen_g.std(axis=0), 1e-4)
    seen_g_n = (seen_g - mean) / std

    trees: dict[str, cKDTree] = {}
    offset = 0
    for task in SEEN_TASKS:
        count = len(selected[task])
        trees[task] = cKDTree(seen_g_n[offset : offset + count])
        offset += count
    trees["seen_union"] = cKDTree(seen_g_n)

    held_idx = selected[HELD_OUT_TASK]
    held_query = arrays["articulation_query_g"][held_idx].astype(np.float32)
    variants: dict[str, np.ndarray] = {
        "normal": chunk.make_minimal_articulation_g(held_query),
        "flow_flip": chunk.make_minimal_articulation_g(
            chunk.apply_query_g_control(held_query, "sign_flip")
        ),
        "anchor_flip": chunk.make_minimal_articulation_g(
            chunk.apply_query_g_control(held_query, "sign_flip_attraction")
        ),
        "flow_only": chunk.make_minimal_articulation_g(
            chunk.apply_query_g_control(held_query, "zero_attraction")
        ),
        "anchor_only": chunk.make_minimal_articulation_g(
            chunk.apply_query_g_control(held_query, "zero_flow")
        ),
    }
    wrong_query = chunk.make_articulation_query_g(
        arrays["robot_obs"][held_idx],
        arrays["scene_obs"][held_idx],
        np.full(len(held_idx), "open_drawer", dtype=object),
        geometry,
    )
    variants["wrong_task_open"] = chunk.make_minimal_articulation_g(wrong_query)

    report: dict[str, Any] = {
        "seen_tasks": list(SEEN_TASKS),
        "held_out_task": HELD_OUT_TASK,
        "used_segments": int(len(used_segments)),
        "sample_counts": {task: int(len(indices)) for task, indices in selected.items()},
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "variants": {},
    }
    markdown = [
        "# Minimal-G Seen-Support Diagnostic",
        "",
        "Distances are Euclidean in six-dimensional G standardized only by seen-task samples.",
        "",
        "| Held-out G variant | Nearest seen union p50 | p90 | Nearest open p50 | Nearest left p50 | Nearest right p50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in variants.items():
        values_n = (np.asarray(values, dtype=np.float32) - mean) / std
        distances = {
            task: stats(tree.query(values_n, k=1, workers=-1)[0])
            for task, tree in trees.items()
        }
        report["variants"][name] = {
            "nearest_distance": distances,
            "g_mean": np.mean(values, axis=0).tolist(),
            "g_std": np.std(values, axis=0).tolist(),
        }
        markdown.append(
            f"| {name} | {distances['seen_union']['p50']:.3f} | "
            f"{distances['seen_union']['p90']:.3f} | {distances['open_drawer']['p50']:.3f} | "
            f"{distances['move_slider_left']['p50']:.3f} | "
            f"{distances['move_slider_right']['p50']:.3f} |"
        )

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    (args.out_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
