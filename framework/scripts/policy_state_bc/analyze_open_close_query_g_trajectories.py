#!/usr/bin/env python3
"""Aggregate open/close EEF, action, query-G, and drawer-joint trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def safe_mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), q))


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]


def vector(row: dict[str, Any], prefix: str) -> np.ndarray:
    return np.asarray([row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_z"]], dtype=np.float64)


def summarize_rollout(
    steps: list[dict[str, Any]],
    rollout: dict[str, Any],
    joint_motion_threshold: float,
) -> dict[str, Any]:
    steps = sorted(steps, key=lambda row: int(row["step"]))
    first = steps[0]
    q0 = float(first["q"])
    desired_sign = float(first["desired_sign"])
    joint_range = float(first["joint_range"])
    cumulative_q = [desired_sign * (float(row["q_after"]) - q0) for row in steps]
    effective_idx = next((i for i, q in enumerate(cumulative_q) if q >= joint_motion_threshold), None)
    effective_step = None if effective_idx is None else int(steps[effective_idx]["step"])
    pre = steps if effective_idx is None else steps[:effective_idx]
    post = [] if effective_idx is None else steps[effective_idx:]
    if not pre:
        pre = steps[:1]

    min_part_row = min(steps, key=lambda row: float(row["dist_to_part"]))
    min_anchor_row = min(steps, key=lambda row: float(row["dist_to_contact_anchor"]))
    reference_row = min_anchor_row if effective_idx is None else steps[effective_idx]

    eef0 = vector(first, "eef")
    eef_ref = vector(reference_row, "eef")
    eef_end = vector(steps[-1], "eef")
    desired_axis = vector(first, "desired_axis")
    initial_anchor_dir = vector(first, "contact_anchor") - eef0
    initial_anchor_norm = max(float(np.linalg.norm(initial_anchor_dir)), 1e-8)
    initial_anchor_unit = initial_anchor_dir / initial_anchor_norm

    action_norms = values(steps, "action_norm")
    pre_action_norms = values(pre, "action_norm")
    post_action_norms = values(post, "action_norm")
    late = steps[-min(40, len(steps)) :]
    final_q_progress = cumulative_q[-1]

    return {
        "policy": str(first["policy"]),
        "task": str(first["task"]),
        "rollout_idx": int(first["rollout_idx"]),
        "success": bool(rollout["success"]),
        "success_step": rollout.get("success_step"),
        "desired_sign": desired_sign,
        "desired_axis_x": float(desired_axis[0]),
        "desired_axis_y": float(desired_axis[1]),
        "desired_axis_z": float(desired_axis[2]),
        "g_desired_flow_y_initial": float(first["g_desired_flow_y"]),
        "g_desired_flow_axis_initial": float(first["g_desired_flow_desired_axis"]),
        "effective_contact": effective_idx is not None,
        "effective_contact_step": effective_step,
        "initial_dist_to_part": float(first["dist_to_part"]),
        "min_dist_to_part": float(min_part_row["dist_to_part"]),
        "min_dist_to_contact_anchor": float(min_anchor_row["dist_to_contact_anchor"]),
        "final_dist_to_part": float(steps[-1]["dist_to_part"]),
        "reference_part_to_eef_axis": float(reference_row["part_to_eef_axis"]),
        "reference_part_to_eef_b1": float(reference_row["part_to_eef_b1"]),
        "reference_part_to_eef_b2": float(reference_row["part_to_eef_b2"]),
        "reference_height_offset": float(reference_row["eef_height_minus_part"]),
        "approach_net_toward_initial_anchor": float(np.dot(eef_ref - eef0, initial_anchor_unit)),
        "approach_net_desired_axis": float(np.dot(eef_ref - eef0, desired_axis)),
        "post_contact_net_desired_axis": (
            None if effective_idx is None else float(np.dot(eef_end - eef_ref, desired_axis))
        ),
        "eef_total_net_desired_axis": float(np.dot(eef_end - eef0, desired_axis)),
        "pre_action_norm_mean": safe_mean(pre_action_norms),
        "pre_action_norm_p90": safe_percentile(pre_action_norms, 90),
        "post_action_norm_mean": safe_mean(post_action_norms),
        "post_action_norm_p90": safe_percentile(post_action_norms, 90),
        "action_norm_max": None if not action_norms else float(max(action_norms)),
        "pre_action_desired_axis_mean": safe_mean(values(pre, "action_desired_axis")),
        "post_action_desired_axis_mean": safe_mean(values(post, "action_desired_axis")),
        "late_action_desired_axis_mean": safe_mean(values(late, "action_desired_axis")),
        "late_action_norm_mean": safe_mean(values(late, "action_norm")),
        "pre_eef_delta_desired_axis_mean": safe_mean(values(pre, "eef_delta_desired_axis")),
        "post_eef_delta_desired_axis_mean": safe_mean(values(post, "eef_delta_desired_axis")),
        "pre_action_g_field_cos_mean": safe_mean(values(pre, "action_g_eef_field_cos")),
        "pre_eef_g_field_cos_mean": safe_mean(values(pre, "eef_delta_g_eef_field_cos")),
        "final_q_progress_m": float(final_q_progress),
        "max_q_progress_m": float(max(cumulative_q)),
        "final_q_progress_normalized": float(final_q_progress / max(joint_range, 1e-8)),
        "max_q_progress_normalized": float(max(cumulative_q) / max(joint_range, 1e-8)),
    }


AGGREGATE_KEYS = [
    "min_dist_to_part",
    "min_dist_to_contact_anchor",
    "reference_part_to_eef_axis",
    "approach_net_toward_initial_anchor",
    "approach_net_desired_axis",
    "post_contact_net_desired_axis",
    "pre_action_norm_mean",
    "pre_action_norm_p90",
    "post_action_norm_mean",
    "post_action_norm_p90",
    "action_norm_max",
    "pre_action_desired_axis_mean",
    "post_action_desired_axis_mean",
    "late_action_desired_axis_mean",
    "late_action_norm_mean",
    "pre_eef_delta_desired_axis_mean",
    "post_eef_delta_desired_axis_mean",
    "pre_action_g_field_cos_mean",
    "pre_eef_g_field_cos_mean",
    "final_q_progress_normalized",
    "max_q_progress_normalized",
]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "count": len(rows),
        "success": int(sum(bool(row["success"]) for row in rows)),
        "effective_contact": int(sum(bool(row["effective_contact"]) for row in rows)),
    }
    out["success_rate"] = float(out["success"] / max(len(rows), 1))
    out["effective_contact_rate"] = float(out["effective_contact"] / max(len(rows), 1))
    for key in AGGREGATE_KEYS:
        vals = values(rows, key)
        out[f"{key}_mean"] = safe_mean(vals)
        out[f"{key}_p50"] = safe_percentile(vals, 50)
        out[f"{key}_p10"] = safe_percentile(vals, 10)
        out[f"{key}_p90"] = safe_percentile(vals, 90)
    return out


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def build_markdown(rows: list[dict[str, Any]], aggregates: dict[str, Any]) -> str:
    lines = [
        "# Open/Close Query-G Trajectory Analysis",
        "",
        "Effective contact is the first step where cumulative drawer motion in the task direction reaches 2 mm.",
        "Distances are in meters and action magnitudes are CALVIN relative-action units.",
        "",
        "## Per-Policy Open/Close",
        "",
        "| Policy | Task | Success | Effective contact | Min part dist | Min anchor dist | Pre action norm | Late desired-axis cmd | Max q progress |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(aggregates["by_policy_task"]):
        item = aggregates["by_policy_task"][key]
        lines.append(
            f"| {item['policy']} | {item['task']} | {item['success']}/{item['count']} | "
            f"{item['effective_contact']}/{item['count']} | {fmt(item['min_dist_to_part_p50'])} | "
            f"{fmt(item['min_dist_to_contact_anchor_p50'])} | {fmt(item['pre_action_norm_mean_p50'], 4)} | "
            f"{fmt(item['late_action_desired_axis_mean_p50'], 4)} | "
            f"{fmt(item['max_q_progress_normalized_p50'])} |"
        )

    lines.extend(
        [
            "",
            "## Pooled No-Token-Type Close",
            "",
            "| Outcome | Count | Effective contact | Min part dist | Min anchor dist | Pre action norm | Pre G alignment | Late desired-axis cmd | Max q progress |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for outcome in ["success", "failure_with_contact", "failure_without_contact"]:
        item = aggregates["pooled_no_type_close"][outcome]
        lines.append(
            f"| {outcome} | {item['count']} | {item['effective_contact']}/{item['count']} | "
            f"{fmt(item['min_dist_to_part_p50'])} | {fmt(item['min_dist_to_contact_anchor_p50'])} | "
            f"{fmt(item['pre_action_norm_mean_p50'], 4)} | {fmt(item['pre_eef_g_field_cos_mean_p50'])} | "
            f"{fmt(item['late_action_desired_axis_mean_p50'], 4)} | "
            f"{fmt(item['max_q_progress_normalized_p50'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--joint-motion-threshold", type=float, default=0.002)
    args = parser.parse_args()

    rollout_rows: list[dict[str, Any]] = []
    for input_dir in args.input_dirs:
        steps = json.load(open(input_dir / "axis_contact_steps.json", encoding="utf-8"))
        rollouts = json.load(open(input_dir / "axis_contact_rollouts.json", encoding="utf-8"))
        rollout_map = {
            (str(row["policy"]), str(row["task"]), int(row["rollout_idx"])): row for row in rollouts
        }
        grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for step in steps:
            grouped[(str(step["policy"]), str(step["task"]), int(step["rollout_idx"]))].append(step)
        for key, group_steps in grouped.items():
            rollout_rows.append(
                summarize_rollout(group_steps, rollout_map[key], args.joint_motion_threshold)
            )

    by_policy_task: dict[str, Any] = {}
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        grouped_rows[(row["policy"], row["task"])].append(row)
    for (policy, task), rows in grouped_rows.items():
        rec = aggregate(rows)
        rec.update({"policy": policy, "task": task})
        by_policy_task[f"{policy}::{task}"] = rec

    no_type_close = [
        row for row in rollout_rows if row["policy"].startswith("no_type_") and row["task"] == "close_drawer"
    ]
    pooled = {
        "success": aggregate([row for row in no_type_close if row["success"]]),
        "failure_with_contact": aggregate(
            [row for row in no_type_close if not row["success"] and row["effective_contact"]]
        ),
        "failure_without_contact": aggregate(
            [row for row in no_type_close if not row["success"] and not row["effective_contact"]]
        ),
    }
    aggregates = {"by_policy_task": by_policy_task, "pooled_no_type_close": pooled}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "rollout_phase_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rollout_rows, f, indent=2, default=json_default)
    with open(args.out_dir / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregates, f, indent=2, default=json_default)
    with open(args.out_dir / "rollout_phase_metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rollout_rows[0]))
        writer.writeheader()
        writer.writerows(rollout_rows)
    with open(args.out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(build_markdown(rollout_rows, aggregates))
    print(build_markdown(rollout_rows, aggregates))


if __name__ == "__main__":
    main()
