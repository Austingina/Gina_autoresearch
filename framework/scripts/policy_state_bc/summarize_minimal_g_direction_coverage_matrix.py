#!/usr/bin/env python3
"""Aggregate D0--D4 compact-G direction-coverage results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = {
    "D0": (
        "2026-07-15_D0_sparse_direction_coverage_film_h1_seed",
        "chunk_bc_minimal_g_film",
    ),
    "D1": (
        "2026-07-15_D1_expanded_direction_coverage_film_h1_seed",
        "chunk_bc_minimal_g_film",
    ),
    "D2": (
        "2026-07-15_D2_expanded_direction_coverage_field_coupled_h16_seed",
        "chunk_bc_minimal_g_bounded_projected",
    ),
    "D4": (
        "2026-07-15_D4_expanded_direction_coverage_film_h16_seed",
        "chunk_bc_minimal_g_film",
    ),
    "D3": (
        "2026-07-15_D3_signed_direction_coverage_film_h1_seed",
        "chunk_bc_minimal_g_film",
    ),
}
CONTROLS = ["normal", "zero", "sign_flip", "task_open_drawer"]
ARTICULATION_TASKS = {"move_slider_left", "move_slider_right", "open_drawer"}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"conditions": {}}
    markdown = [
        "# Minimal G Direction-Coverage Matrix",
        "",
        "| Condition | Completed seeds | Normal close | Worst seed | Zero | Sign flip | Wrong open G | Seen health |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, (stem, mode) in CONDITIONS.items():
        condition_report: dict[str, Any] = {"seeds": {}}
        pooled: dict[str, list[int]] = {control: [] for control in CONTROLS}
        completed_controls: dict[str, int] = {control: 0 for control in CONTROLS}
        seen_success = 0
        seen_total = 0
        seen_completed_seeds = 0
        normal_beats_flip = 0
        for seed in args.seeds:
            root = args.experiment_root / f"{stem}{seed}"
            seed_report: dict[str, Any] = {}
            for control in CONTROLS:
                summary = load_json(
                    root
                    / mode
                    / f"eval_close_{control}_n100"
                    / "state_bc_summary.json"
                )
                if summary is None:
                    continue
                values = [int(value) for value in summary["raw_results"]]
                pooled[control].extend(values)
                completed_controls[control] += 1
                seed_report[control] = {
                    "success": int(sum(values)),
                    "total": int(len(values)),
                    "raw_results": values,
                    "selected_sequence_indices": summary.get("selected_sequence_indices", []),
                }
            if "normal" in seed_report and "sign_flip" in seed_report:
                normal_beats_flip += int(
                    seed_report["normal"]["success"] > seed_report["sign_flip"]["success"]
                )
            seen = load_json(
                root / mode / "eval_seen_translational_n25" / "state_bc_summary.json"
            )
            if seen is not None:
                values = [int(value) for value in seen["raw_results"]]
                seen_success += int(sum(values))
                seen_total += int(len(values))
                seen_completed_seeds += 1
                seed_report["seen"] = {
                    "success": int(sum(values)),
                    "total": int(len(values)),
                    "task_info": seen.get("task_info", {}),
                }
            train = load_json(root / mode / "train_metrics.json")
            if train is not None:
                seed_report["train_final"] = train.get("final", {})
            if seed_report:
                condition_report["seeds"][str(seed)] = seed_report

        aggregate: dict[str, Any] = {}
        for control, values in pooled.items():
            aggregate[control] = {
                "success": int(sum(values)),
                "total": int(len(values)),
                "success_rate": None if not values else float(np.mean(values)),
                "completed_seeds": int(completed_controls[control]),
            }
        normal_seed_success = [
            seed_result["normal"]["success"]
            for seed_result in condition_report["seeds"].values()
            if "normal" in seed_result
        ]
        aggregate["normal_worst_seed_success"] = (
            None if not normal_seed_success else int(min(normal_seed_success))
        )
        aggregate["normal_beats_sign_flip_seeds"] = int(normal_beats_flip)
        aggregate["seen"] = {
            "success": int(seen_success),
            "total": int(seen_total),
            "success_rate": None if not seen_total else float(seen_success / seen_total),
            "completed_seeds": int(seen_completed_seeds),
        }

        seen_task_aggregate: dict[str, dict[str, int | float]] = {}
        for seed_result in condition_report["seeds"].values():
            for task, info in seed_result.get("seen", {}).get("task_info", {}).items():
                task_result = seen_task_aggregate.setdefault(task, {"success": 0, "total": 0})
                task_result["success"] += int(info["success"])
                task_result["total"] += int(info["total"])
        for task_result in seen_task_aggregate.values():
            total = int(task_result["total"])
            task_result["success_rate"] = float(task_result["success"] / total)
        aggregate["seen_by_task"] = seen_task_aggregate

        articulation_success = sum(
            int(info["success"])
            for task, info in seen_task_aggregate.items()
            if task in ARTICULATION_TASKS
        )
        articulation_total = sum(
            int(info["total"])
            for task, info in seen_task_aggregate.items()
            if task in ARTICULATION_TASKS
        )
        primitive_success = seen_success - articulation_success
        primitive_total = seen_total - articulation_total
        aggregate["seen_by_family"] = {
            "articulation": {
                "success": articulation_success,
                "total": articulation_total,
                "success_rate": (
                    None
                    if not articulation_total
                    else float(articulation_success / articulation_total)
                ),
            },
            "free_object_push": {
                "success": primitive_success,
                "total": primitive_total,
                "success_rate": (
                    None if not primitive_total else float(primitive_success / primitive_total)
                ),
            },
        }

        paired_controls: dict[str, dict[str, int | bool]] = {}
        for control in CONTROLS[1:]:
            counts = {"both": 0, "normal_only": 0, "control_only": 0, "neither": 0}
            completed_pairs = 0
            matching_indices = True
            normal_beats_control_seeds = 0
            for seed_result in condition_report["seeds"].values():
                if "normal" not in seed_result or control not in seed_result:
                    continue
                normal = seed_result["normal"]
                corrupted = seed_result[control]
                completed_pairs += 1
                matching_indices &= (
                    normal["selected_sequence_indices"]
                    == corrupted["selected_sequence_indices"]
                )
                normal_beats_control_seeds += int(normal["success"] > corrupted["success"])
                for normal_value, control_value in zip(
                    normal["raw_results"], corrupted["raw_results"]
                ):
                    if normal_value and control_value:
                        counts["both"] += 1
                    elif normal_value:
                        counts["normal_only"] += 1
                    elif control_value:
                        counts["control_only"] += 1
                    else:
                        counts["neither"] += 1
            paired_controls[control] = {
                "completed_seed_pairs": completed_pairs,
                "selected_indices_match": matching_indices,
                "normal_beats_control_seeds": normal_beats_control_seeds,
                **counts,
            }
        aggregate["paired_vs_normal"] = paired_controls
        condition_report["aggregate"] = aggregate
        report["conditions"][condition] = condition_report

        def pct(control: str) -> str:
            value = aggregate[control]["success_rate"]
            completed = aggregate[control]["completed_seeds"]
            return "-" if value is None else f"{100.0 * value:.1f}% ({completed}/{len(args.seeds)})"

        seen_rate = aggregate["seen"]["success_rate"]
        seen_text = (
            "-"
            if seen_rate is None
            else f"{100.0 * seen_rate:.1f}% ({aggregate['seen']['completed_seeds']}/{len(args.seeds)})"
        )
        markdown.append(
            f"| {condition} | {len(condition_report['seeds'])}/{len(args.seeds)} | "
            f"{pct('normal')} | {aggregate['normal_worst_seed_success']} | {pct('zero')} | "
            f"{pct('sign_flip')} | {pct('task_open_drawer')} | "
            f"{seen_text} |"
        )

    markdown.extend(
        [
            "",
            "## Seen-Task Families",
            "",
            "| Condition | Articulation | Free-object push | All seen tasks |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for condition, condition_report in report["conditions"].items():
        aggregate = condition_report["aggregate"]

        def fraction(info: dict[str, Any]) -> str:
            total = int(info["total"])
            if not total:
                return "-"
            return f"{info['success']}/{total} ({100.0 * info['success_rate']:.1f}%)"

        families = aggregate["seen_by_family"]
        markdown.append(
            f"| {condition} | {fraction(families['articulation'])} | "
            f"{fraction(families['free_object_push'])} | {fraction(aggregate['seen'])} |"
        )

    markdown.extend(
        [
            "",
            "## Paired Causal Controls",
            "",
            "| Condition | Control | Seed pairs | Both | Normal only | Control only | Neither | Indices match |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for condition, condition_report in report["conditions"].items():
        for control, paired in condition_report["aggregate"]["paired_vs_normal"].items():
            if not paired["completed_seed_pairs"]:
                continue
            markdown.append(
                f"| {condition} | {control} | {paired['completed_seed_pairs']} | "
                f"{paired['both']} | {paired['normal_only']} | {paired['control_only']} | "
                f"{paired['neither']} | {paired['selected_indices_match']} |"
            )

    with open(args.out_dir / "matrix_summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(args.out_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown) + "\n")
    print("\n".join(markdown), flush=True)


if __name__ == "__main__":
    main()
