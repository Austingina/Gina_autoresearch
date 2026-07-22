#!/usr/bin/env python3
"""Aggregate E3 factorial checkerboard results across optimization seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any


HELD_OUT_TASKS = [
    "close_drawer",
    "move_slider_left",
    "push_red_block_right",
    "push_blue_block_left",
]
CONDITIONS = ["with_g_normal", "with_g_sign_flip", "matched_no_g"]
INTERACTION_SOURCE_TASKS = [
    "move_slider_right",
    "open_drawer",
    "push_blue_block_right",
    "push_into_drawer",
    "push_pink_block_left",
    "push_pink_block_right",
    "push_red_block_left",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def exact_mcnemar_p(normal_only: int, control_only: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    discordant = normal_only + control_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(normal_only, control_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    run_roots = [path.resolve() for path in args.run_roots]
    rollout_rows: list[dict[str, Any]] = []
    offline_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    source_language_rows: list[dict[str, Any]] = []

    for seed, run_root in enumerate(run_roots):
        for task in HELD_OUT_TASKS:
            for condition in CONDITIONS:
                summary = read_json(
                    run_root / "eval" / f"{condition}_{task}" / "state_bc_summary.json"
                )
                info = None if summary is None else summary.get("task_info", {}).get(task)
                rollout_rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "condition": condition,
                        "success": None if info is None else int(info["success"]),
                        "total": None if info is None else int(info["total"]),
                        "raw_results": None if summary is None else summary["raw_results"],
                        "sequence_indices": (
                            None if summary is None else summary["selected_sequence_indices"]
                        ),
                    }
                )

        offline = read_json(
            run_root
            / "heldout_condition_sensitivity"
            / "minimal_g_token_condition_sensitivity.json"
        )
        per_task = (
            {}
            if offline is None
            else offline.get("models", {}).get("with_g", {}).get("per_task", {})
        )
        for task in HELD_OUT_TASKS:
            item = per_task.get(task)
            if item is None:
                continue
            offline_rows.append(
                {
                    "seed": seed,
                    "task": task,
                    "normal_mse": item["normal_first_action_mse"],
                    "zero_mse": item["zero_first_action_mse"],
                    "flip_mse": item["flow_flip_first_action_mse"],
                    "normal_f_projection": item["normal_first_xyz_flow_projection"]["mean"],
                    "flip_f_projection": item["flow_flip_first_xyz_flow_projection"]["mean"],
                }
            )

        for condition in ("with_g", "matched_no_g"):
            summary = read_json(
                run_root / "eval" / f"{condition}_source12" / "state_bc_summary.json"
            )
            if summary is None:
                continue
            info = summary.get("task_info", {})
            source_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "success": sum(int(item["success"]) for item in info.values()),
                    "total": sum(int(item["total"]) for item in info.values()),
                    "interaction_success": sum(
                        int(info[task]["success"])
                        for task in INTERACTION_SOURCE_TASKS
                        if task in info
                    ),
                    "interaction_total": sum(
                        int(info[task]["total"])
                        for task in INTERACTION_SOURCE_TASKS
                        if task in info
                    ),
                    "tasks": info,
                }
            )

        language_summary = read_json(
            run_root
            / "eval"
            / "with_g_source12_language_normal"
            / "state_bc_summary.json"
        )
        if language_summary is not None:
            info = language_summary.get("task_info", {})
            source_language_rows.append(
                {
                    "seed": seed,
                    "success": sum(int(item["success"]) for item in info.values()),
                    "total": sum(int(item["total"]) for item in info.values()),
                    "interaction_success": sum(
                        int(info[task]["success"])
                        for task in INTERACTION_SOURCE_TASKS
                        if task in info
                    ),
                    "interaction_total": sum(
                        int(info[task]["total"])
                        for task in INTERACTION_SOURCE_TASKS
                        if task in info
                    ),
                    "tasks": info,
                }
            )

    aggregate: dict[str, dict[str, Any]] = defaultdict(dict)
    for task in HELD_OUT_TASKS:
        for condition in CONDITIONS:
            selected = [
                row
                for row in rollout_rows
                if row["task"] == task
                and row["condition"] == condition
                and row["success"] is not None
            ]
            rates = [row["success"] / row["total"] for row in selected if row["total"]]
            aggregate[task][condition] = {
                "completed_seeds": len(selected),
                "success": sum(row["success"] for row in selected),
                "total": sum(row["total"] for row in selected),
                "per_seed_sr": rates,
                "mean_seed_sr": None if not rates else sum(rates) / len(rates),
                "worst_seed_sr": None if not rates else min(rates),
            }

    paired_comparisons: dict[str, dict[str, Any]] = defaultdict(dict)
    for task in HELD_OUT_TASKS:
        normal_rows = {
            row["seed"]: row
            for row in rollout_rows
            if row["task"] == task and row["condition"] == "with_g_normal"
        }
        for control in ("with_g_sign_flip", "matched_no_g"):
            control_rows = {
                row["seed"]: row
                for row in rollout_rows
                if row["task"] == task and row["condition"] == control
            }
            normal_only = control_only = both = neither = 0
            completed_seeds = 0
            for seed in sorted(normal_rows.keys() & control_rows.keys()):
                normal = normal_rows[seed]
                other = control_rows[seed]
                if normal["raw_results"] is None or other["raw_results"] is None:
                    continue
                if normal["sequence_indices"] != other["sequence_indices"]:
                    raise ValueError(
                        f"Sequence mismatch for {task}, seed {seed}, control {control}"
                    )
                completed_seeds += 1
                for normal_result, control_result in zip(
                    normal["raw_results"], other["raw_results"]
                ):
                    normal_success = normal_result > 0
                    control_success = control_result > 0
                    if normal_success and not control_success:
                        normal_only += 1
                    elif control_success and not normal_success:
                        control_only += 1
                    elif normal_success and control_success:
                        both += 1
                    else:
                        neither += 1
            paired_comparisons[task][control] = {
                "completed_seeds": completed_seeds,
                "normal_only": normal_only,
                "control_only": control_only,
                "both": both,
                "neither": neither,
                "exact_mcnemar_p": exact_mcnemar_p(normal_only, control_only),
            }

    report = {
        "run_roots": [path.as_posix() for path in run_roots],
        "rollout_rows": rollout_rows,
        "rollout_aggregate": aggregate,
        "paired_comparisons": paired_comparisons,
        "offline_rows": offline_rows,
        "source_rows": source_rows,
        "source_language_rows": source_language_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "multiseed_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    lines = [
        "# E3 Factorial Checkerboard Multi-seed Summary",
        "",
        "| Held-out task | G normal | F sign-flip | Matched no-G |",
        "| --- | ---: | ---: | ---: |",
    ]

    def cell(task: str, condition: str) -> str:
        item = aggregate[task][condition]
        if not item["completed_seeds"]:
            return "pending"
        per_seed = ", ".join(f"{value:.0%}" for value in item["per_seed_sr"])
        return f"{item['success']}/{item['total']} ({per_seed})"

    for task in HELD_OUT_TASKS:
        lines.append(
            f"| {task} | {cell(task, 'with_g_normal')} | "
            f"{cell(task, 'with_g_sign_flip')} | {cell(task, 'matched_no_g')} |"
        )
    lines.extend(["", "## Source-task health", ""])
    for row in source_rows:
        lines.append(
            f"- seed{row['seed']} {row['condition']}: {row['success']}/{row['total']} "
            f"overall; {row['interaction_success']}/{row['interaction_total']} "
            "interaction subset"
        )
    lines.extend(["", "### Normal-language source health", ""])
    for row in source_language_rows:
        lines.append(
            f"- seed{row['seed']} with_g: {row['success']}/{row['total']} overall; "
            f"{row['interaction_success']}/{row['interaction_total']} "
            "interaction subset"
        )
    lines.extend(["", "## Paired causal controls", ""])
    for task in HELD_OUT_TASKS:
        for control in ("with_g_sign_flip", "matched_no_g"):
            item = paired_comparisons[task][control]
            lines.append(
                f"- {task}, normal vs {control}: "
                f"normal-only={item['normal_only']}, "
                f"control-only={item['control_only']}, both={item['both']}, "
                f"neither={item['neither']}, exact McNemar "
                f"p={item['exact_mcnemar_p']:.3g}"
            )
    (args.out_dir / "multiseed_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
