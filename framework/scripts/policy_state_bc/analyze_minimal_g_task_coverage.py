#!/usr/bin/env python3
"""Audit compact-G task coverage and primitive geometry before policy training."""

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


DEFAULT_TASKS = [
    "open_drawer",
    "move_slider_left",
    "move_slider_right",
    "place_in_slider",
    "push_into_drawer",
    "push_blue_block_left",
    "push_blue_block_right",
    "push_pink_block_left",
    "push_pink_block_right",
    "push_red_block_left",
    "push_red_block_right",
]


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
    }


def safe_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    result = np.full(len(a), np.nan, dtype=np.float32)
    valid = denom > 1e-8
    result[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return result


def push_into_selection_audit(data_root: Path, primitive_geometry: dict[str, Any]) -> dict[str, Any]:
    split_dir = data_root / "training"
    del primitive_geometry
    records: list[dict[str, Any]] = []
    for segment in base.load_lang_segments(split_dir):
        if str(segment["task"]) != "push_into_drawer":
            continue
        try:
            with np.load(base.episode_file(split_dir, int(segment["start"])), allow_pickle=True) as data:
                start_scene = np.asarray(data["scene_obs"], dtype=np.float32)
                start_robot = np.asarray(data["robot_obs"], dtype=np.float32)
            with np.load(base.episode_file(split_dir, int(segment["end"])), allow_pickle=True) as data:
                end_scene = np.asarray(data["scene_obs"], dtype=np.float32)
        except Exception:
            continue
        starts = chunk._scene_block_positions(start_scene)
        ends = chunk._scene_block_positions(end_scene)
        displacement = np.linalg.norm(ends - starts, axis=1)
        true_idx = int(np.argmax(displacement))
        predicted_idx = int(np.argmin(np.linalg.norm(starts - start_robot[None, :3], axis=1)))
        records.append(
            {
                "start": int(segment["start"]),
                "end": int(segment["end"]),
                "true_active": true_idx,
                "predicted_active": predicted_idx,
                "correct": true_idx == predicted_idx,
                "active_displacement": float(displacement[true_idx]),
                "margin_to_second": float(
                    np.partition(displacement, -2)[-1] - np.partition(displacement, -2)[-2]
                ),
            }
        )
    correct = np.asarray([record["correct"] for record in records], dtype=np.float32)
    confident = np.asarray([record["margin_to_second"] >= 0.03 for record in records], dtype=bool)
    return {
        "num_segments": int(len(records)),
        "accuracy": None if not len(records) else float(np.mean(correct)),
        "confident_segments": int(np.sum(confident)),
        "confident_accuracy": (
            None if not np.any(confident) else float(np.mean(correct[confident]))
        ),
        "records": records,
    }


def scope_report(arrays: dict[str, np.ndarray], tasks: list[str]) -> dict[str, Any]:
    task_names = arrays["task_name"].astype(str)
    minimal = arrays["articulation_minimal_g"]
    attraction = minimal[:, :3]
    flow = minimal[:, 3:]
    attraction_norm = np.linalg.norm(attraction, axis=1)
    flow_norm = np.linalg.norm(flow, axis=1)
    action = arrays["action_chunks"][:, 0, :3]
    anchor_distance = attraction_norm
    result: dict[str, Any] = {"tasks": {}}
    all_unit_flow: list[np.ndarray] = []
    for task in tasks:
        mask = task_names == task
        valid = mask & (flow_norm > 1e-6)
        near = valid & (anchor_distance <= 0.10)
        far = valid & (anchor_distance > 0.10)
        unit_flow = flow[valid] / np.maximum(flow_norm[valid, None], 1e-6)
        if len(unit_flow):
            all_unit_flow.append(unit_flow)
        result["tasks"][task] = {
            "windows": int(np.sum(mask)),
            "nonzero_fraction": None if not np.any(mask) else float(np.mean(flow_norm[mask] > 1e-6)),
            "attraction_norm": stats(attraction_norm[mask]),
            "flow_norm": stats(flow_norm[mask]),
            "mean_unit_flow": None if not len(unit_flow) else np.mean(unit_flow, axis=0).tolist(),
            "far_first_action_attraction_cos": stats(safe_cosine(action[far], attraction[far])),
            "near_first_action_flow_cos": stats(safe_cosine(action[near], flow[near])),
            "near_positive_flow_fraction": (
                None
                if not np.any(near)
                else float(np.mean(np.sum(action[near] * flow[near], axis=1) > 0.0))
            ),
        }
    directions = np.concatenate(all_unit_flow, axis=0) if all_unit_flow else np.zeros((0, 3), np.float32)
    if len(directions):
        singular_values = np.linalg.svd(directions, compute_uv=False)
        normalized = singular_values / max(float(singular_values[0]), 1e-8)
    else:
        singular_values = np.zeros(3, dtype=np.float32)
        normalized = singular_values
    result["direction_coverage"] = {
        "valid_windows": int(len(directions)),
        "singular_values": singular_values.tolist(),
        "normalized_singular_values": normalized.tolist(),
        "effective_rank_0p1": int(np.sum(normalized > 0.1)),
        "axis_sign_support": {
            axis: {
                "positive": int(np.sum(directions[:, index] > 0.5)),
                "negative": int(np.sum(directions[:, index] < -0.5)),
            }
            for index, axis in enumerate(("x", "y", "z"))
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--articulation-geometry-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=0)
    args = parser.parse_args()

    args.calvin_code = args.calvin_code.resolve()
    args.data_root = args.data_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    text_to_task, _ = base.load_eval_annotation_maps(args.calvin_code)
    segments = base.load_lang_segments(args.data_root / "training")
    task_to_id = base.build_task_index(segments, text_to_task)
    geometry = base.load_articulation_geometry(args.articulation_geometry_json.resolve())
    if geometry is None:
        raise RuntimeError("Missing articulation geometry")
    primitive_geometry = chunk.fit_primitive_query_geometry(args.data_root)

    reports: dict[str, Any] = {}
    for scope in ("articulation", "translational"):
        arrays, used_segments = chunk.collect_chunk_records(
            args.data_root,
            task_to_id,
            chunk_size=1,
            max_windows=0,
            max_segments=0,
            stride=args.stride,
            seed=args.data_seed,
            include_tasks=set(args.include_tasks),
            articulation_geometry=geometry,
            minimal_g_scope=scope,
            primitive_geometry=primitive_geometry,
        )
        reports[scope] = scope_report(arrays, args.include_tasks)
        reports[scope]["num_segments"] = int(len(used_segments))
        reports[scope]["num_windows"] = int(len(arrays["task_name"]))

    report = {
        "data_root": args.data_root,
        "stride": int(args.stride),
        "primitive_geometry": primitive_geometry,
        "push_into_active_block_selection": push_into_selection_audit(
            args.data_root, primitive_geometry
        ),
        "scopes": reports,
    }
    with open(args.out_dir / "coverage_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=base.json_default)

    markdown = [
        "# Minimal G Task Coverage",
        "",
        "| Scope | Task | Windows | Nonzero G | Mean unit F | Near F-positive |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for scope, scope_result in reports.items():
        for task in args.include_tasks:
            rec = scope_result["tasks"][task]
            direction = rec["mean_unit_flow"]
            direction_text = "-" if direction is None else ", ".join(f"{value:.3f}" for value in direction)
            nonzero = rec["nonzero_fraction"]
            positive = rec["near_positive_flow_fraction"]
            markdown.append(
                f"| {scope} | {task} | {rec['windows']} | "
                f"{'-' if nonzero is None else f'{nonzero:.3f}'} | {direction_text} | "
                f"{'-' if positive is None else f'{positive:.3f}'} |"
            )
    selection = report["push_into_active_block_selection"]
    markdown.extend(
        [
            "",
            "## Push-into active-block selection",
            "",
            f"- All segments: {selection['accuracy']} over {selection['num_segments']} segments",
            f"- Confident segments: {selection['confident_accuracy']} over {selection['confident_segments']} segments",
            "",
            "## Direction coverage",
            "",
        ]
    )
    for scope, scope_result in reports.items():
        coverage = scope_result["direction_coverage"]
        markdown.append(
            f"- {scope}: effective rank={coverage['effective_rank_0p1']}, "
            f"normalized singular values={coverage['normalized_singular_values']}"
        )
    with open(args.out_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown) + "\n")
    print("\n".join(markdown), flush=True)


if __name__ == "__main__":
    main()
