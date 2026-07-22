#!/usr/bin/env python3
"""Check whether minimal attraction/object-flow G agrees with demo action chunks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402
import train_eval_state_chunk_flow_policy as chunk  # noqa: E402


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros(len(a), dtype=np.float32)
    valid = denom > 1e-8
    out[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return out


def field_decomposition(
    action: np.ndarray,
    direction: np.ndarray,
    *,
    min_positive_parallel: float,
) -> dict[str, np.ndarray]:
    parallel = np.sum(action * direction, axis=1)
    orthogonal = action - parallel[:, None] * direction
    orthogonal_norm = np.linalg.norm(orthogonal, axis=1)
    ratio = np.full(len(action), np.nan, dtype=np.float32)
    usable = parallel >= min_positive_parallel
    ratio[usable] = orthogonal_norm[usable] / parallel[usable]
    return {
        "cosine": cosine(action, direction),
        "parallel": parallel,
        "orthogonal_norm": orthogonal_norm,
        "orthogonal_to_positive_parallel_ratio": ratio,
        "positive_parallel": parallel > 0.0,
        "usable_parallel": usable,
    }


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"count": 0, "mean": None, "p50": None, "p10": None, "p90": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--articulation-geometry-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--include-tasks",
        nargs="+",
        default=["open_drawer", "move_slider_left", "move_slider_right"],
    )
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--near-anchor-threshold", type=float, default=0.10)
    parser.add_argument("--readiness-temperature", type=float, default=0.025)
    parser.add_argument("--min-positive-parallel", type=float, default=0.02)
    args = parser.parse_args()

    args.calvin_code = args.calvin_code.resolve()
    args.data_root = args.data_root.resolve()
    args.articulation_geometry_json = args.articulation_geometry_json.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    text_to_task, _ = base.load_eval_annotation_maps(args.calvin_code)
    segments = base.load_lang_segments(args.data_root / "training")
    task_to_id = base.build_task_index(segments, text_to_task)
    geometry = base.load_articulation_geometry(args.articulation_geometry_json)
    if geometry is None:
        raise RuntimeError("Missing articulation geometry")
    arrays, used_segments = chunk.collect_chunk_records(
        args.data_root,
        task_to_id,
        chunk_size=args.chunk_size,
        max_windows=0,
        max_segments=0,
        stride=args.stride,
        seed=args.data_seed,
        include_tasks=set(args.include_tasks),
        articulation_geometry=geometry,
    )

    query = arrays["articulation_query_g"].reshape(-1, chunk.QUERY_G_TOKEN_COUNT, chunk.QUERY_G_TOKEN_DIM)
    minimal = chunk.make_minimal_articulation_g(arrays["articulation_query_g"])
    attraction = minimal[:, :3]
    flow = minimal[:, 3:]
    anchor_delta = query[:, 2, 4:7]
    anchor_dist = np.linalg.norm(anchor_delta, axis=1)
    action_xyz = arrays["action_chunks"][:, :, :3]
    low_action = np.mean(action_xyz, axis=1)
    first_action = action_xyz[:, 0]
    attraction_norm = np.linalg.norm(attraction, axis=1)
    flow_norm = np.linalg.norm(flow, axis=1)
    attraction_dir = attraction / np.maximum(attraction_norm[:, None], 1e-6)
    flow_dir = flow / np.maximum(flow_norm[:, None], 1e-6)
    readiness = 1.0 / (
        1.0
        + np.exp(
            -(args.near_anchor_threshold - anchor_dist)
            / max(args.readiness_temperature, 1e-6)
        )
    )
    blended = (1.0 - readiness[:, None]) * attraction_dir + readiness[:, None] * flow_dir
    field_direction = blended / np.maximum(np.linalg.norm(blended, axis=1, keepdims=True), 1e-6)
    field_valid = (attraction_norm > 1e-6) | (flow_norm > 1e-6)
    field_direction *= field_valid[:, None]
    low_field = field_decomposition(
        low_action,
        field_direction,
        min_positive_parallel=args.min_positive_parallel,
    )
    first_field = field_decomposition(
        first_action,
        field_direction,
        min_positive_parallel=args.min_positive_parallel,
    )
    repeated_direction = np.repeat(field_direction, action_xyz.shape[1], axis=0)
    all_step_field = field_decomposition(
        action_xyz.reshape(-1, 3),
        repeated_direction,
        min_positive_parallel=args.min_positive_parallel,
    )
    attraction_cos = cosine(low_action, attraction)
    flow_cos = cosine(low_action, flow)
    first_attraction_cos = cosine(first_action, attraction)
    first_flow_cos = cosine(first_action, flow)
    action_norm = np.linalg.norm(low_action, axis=1)
    task_names = arrays["task_name"].astype(str)

    report: dict[str, Any] = {
        "num_windows": int(len(task_names)),
        "num_segments": int(len(used_segments)),
        "near_anchor_threshold": float(args.near_anchor_threshold),
        "readiness_temperature": float(args.readiness_temperature),
        "min_positive_parallel": float(args.min_positive_parallel),
        "tasks": {},
    }
    markdown = [
        "# Minimal G Demonstration Alignment",
        "",
        "| Task | Windows | Far A cosine | Near F cosine | Near desired-flow positive | Far action norm | Near action norm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in args.include_tasks:
        task_mask = task_names == task
        far = task_mask & (anchor_dist > args.near_anchor_threshold)
        near = task_mask & (anchor_dist <= args.near_anchor_threshold)
        task_report = {
            "windows": int(np.sum(task_mask)),
            "far_windows": int(np.sum(far)),
            "near_windows": int(np.sum(near)),
            "anchor_distance": stats(anchor_dist[task_mask]),
            "attraction_norm": stats(attraction_norm[task_mask]),
            "flow_norm": stats(flow_norm[task_mask]),
            "far_low_action_attraction_cos": stats(attraction_cos[far]),
            "near_low_action_flow_cos": stats(flow_cos[near]),
            "far_first_action_attraction_cos": stats(first_attraction_cos[far]),
            "near_first_action_flow_cos": stats(first_flow_cos[near]),
            "far_action_norm": stats(action_norm[far]),
            "near_action_norm": stats(action_norm[near]),
            "near_positive_flow_projection_fraction": (
                None if not np.any(near) else float(np.mean(np.sum(low_action[near] * flow[near], axis=1) > 0.0))
            ),
            "active_field": {},
        }
        all_step_task_mask = np.repeat(task_mask, action_xyz.shape[1])
        for label, decomposition, mask in (
            ("first_action", first_field, task_mask),
            ("chunk_mean_action", low_field, task_mask),
            ("all_chunk_steps_static_current_field", all_step_field, all_step_task_mask),
        ):
            task_report["active_field"][label] = {
                "cosine": stats(decomposition["cosine"][mask]),
                "parallel": stats(decomposition["parallel"][mask]),
                "orthogonal_norm": stats(decomposition["orthogonal_norm"][mask]),
                "orthogonal_to_positive_parallel_ratio": stats(
                    decomposition["orthogonal_to_positive_parallel_ratio"][mask]
                ),
                "positive_parallel_fraction": float(np.mean(decomposition["positive_parallel"][mask])),
                "usable_parallel_fraction": float(np.mean(decomposition["usable_parallel"][mask])),
            }
        report["tasks"][task] = task_report
        markdown.append(
            f"| {task} | {task_report['windows']} | "
            f"{task_report['far_low_action_attraction_cos']['p50']:.3f} | "
            f"{task_report['near_low_action_flow_cos']['p50']:.3f} | "
            f"{task_report['near_positive_flow_projection_fraction']:.3f} | "
            f"{task_report['far_action_norm']['p50']:.4f} | "
            f"{task_report['near_action_norm']['p50']:.4f} |"
        )

    markdown.extend(
        [
            "",
            "## Active blended field decomposition",
            "",
            "The all-step row compares every future action in a chunk with the field observed at the chunk start.",
            "",
            "| Task | Sample | Field cosine p50 | Positive parallel | Usable parallel | Orth/parallel p50 | Orth/parallel p90 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task in args.include_tasks:
        for sample in ("first_action", "chunk_mean_action", "all_chunk_steps_static_current_field"):
            rec = report["tasks"][task]["active_field"][sample]
            ratio = rec["orthogonal_to_positive_parallel_ratio"]
            markdown.append(
                f"| {task} | {sample} | {rec['cosine']['p50']:.3f} | "
                f"{rec['positive_parallel_fraction']:.3f} | {rec['usable_parallel_fraction']:.3f} | "
                f"{ratio['p50']:.3f} | {ratio['p90']:.3f} |"
            )

    bins = [0.0, 0.05, 0.10, 0.20, 0.40, float("inf")]
    binned: dict[str, Any] = defaultdict(dict)
    for task in args.include_tasks:
        task_mask = task_names == task
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = task_mask & (anchor_dist >= lo) & (anchor_dist < hi)
            label = f"{lo:.2f}-{hi:.2f}" if np.isfinite(hi) else f"{lo:.2f}+"
            binned[task][label] = {
                "count": int(np.sum(mask)),
                "attraction_cos": stats(attraction_cos[mask]),
                "flow_cos": stats(flow_cos[mask]),
                "action_norm": stats(action_norm[mask]),
            }
    report["anchor_distance_bins"] = dict(binned)

    with open(args.out_dir / "alignment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=json_default)
    with open(args.out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown) + "\n")
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
