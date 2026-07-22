#!/usr/bin/env python3
"""Aggregate paired G-corruption controls for the successful H1 FiLM seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


SEEDS = (3, 4)
CONTROLS = (
    "normal",
    "zero",
    "sign_flip",
    "task_open_drawer",
    "zero_attraction",
    "zero_flow",
    "sign_flip_attraction",
)
NORMAL_STEM = "2026-07-14_learned_g_executor_chunk1_exec1_e50_n100_h160_seed{seed}"
CAUSAL_STEM = "2026-07-14_learned_g_executor_h1_film_causal_n100_h160_seed{seed}"


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def exact_mcnemar(a: list[int], b: list[int]) -> dict[str, Any]:
    if len(a) != len(b):
        raise ValueError("Paired result vectors have different lengths")
    a_only = sum(x == 1 and y == 0 for x, y in zip(a, b))
    b_only = sum(x == 0 and y == 1 for x, y in zip(a, b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(a_only, b_only)
        tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def summary_path(root: Path, seed: int, control: str) -> Path:
    if control == "normal":
        return (
            root
            / NORMAL_STEM.format(seed=seed)
            / "chunk_bc_minimal_g_film"
            / "task_eval"
            / "state_bc_summary.json"
        )
    return root / CAUSAL_STEM.format(seed=seed) / control / "state_bc_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    records: dict[str, dict[str, Any]] = {control: {} for control in CONTROLS}
    missing: list[str] = []
    reference_by_seed: dict[int, list[int]] = {}
    indices_match = True
    for control in CONTROLS:
        for seed in SEEDS:
            path = summary_path(args.experiment_root, seed, control)
            if not path.exists():
                missing.append(path.as_posix())
                continue
            summary = load_json(path)
            raw = [int(value) for value in summary["raw_results"]]
            indices = [int(value) for value in summary["selected_sequence_indices"]]
            if seed not in reference_by_seed:
                reference_by_seed[seed] = indices
            elif reference_by_seed[seed] != indices:
                indices_match = False
            records[control][str(seed)] = {
                "success": int(sum(raw)),
                "total": len(raw),
                "sr": float(mean(raw)),
                "raw_results": raw,
                "selected_sequence_indices": indices,
            }

    aggregate: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    if not missing:
        for control in CONTROLS:
            rows = [records[control][str(seed)] for seed in SEEDS]
            aggregate[control] = {
                "seed_success": [row["success"] for row in rows],
                "seed_sr": [row["sr"] for row in rows],
                "mean_seed_sr": mean(row["sr"] for row in rows),
                "pooled_success": sum(row["success"] for row in rows),
                "pooled_total": sum(row["total"] for row in rows),
            }
        normal = sum((records["normal"][str(seed)]["raw_results"] for seed in SEEDS), [])
        for control in CONTROLS[1:]:
            changed = sum((records[control][str(seed)]["raw_results"] for seed in SEEDS), [])
            per_seed = {
                str(seed): exact_mcnemar(
                    records["normal"][str(seed)]["raw_results"],
                    records[control][str(seed)]["raw_results"],
                )
                for seed in SEEDS
            }
            comparisons.append(
                {
                    "a": "normal",
                    "b": control,
                    "seed_delta_b_minus_a": [
                        records[control][str(seed)]["sr"] - records["normal"][str(seed)]["sr"]
                        for seed in SEEDS
                    ],
                    "per_seed": per_seed,
                    **exact_mcnemar(normal, changed),
                }
            )

    result = {
        "complete": not missing,
        "missing": missing,
        "protocol": {
            "seeds": list(SEEDS),
            "rollouts_per_seed": 100,
            "episode_length": 160,
            "matched_sequence_indices_within_seed": indices_match,
        },
        "aggregate": aggregate,
        "comparisons": comparisons,
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = [
        "# H1 FiLM G Causal Controls",
        "",
        f"Complete: `{result['complete']}`",
        f"Matched sequence indices: `{indices_match}`",
        "",
    ]
    if missing:
        lines.extend(["## Missing", "", *[f"- `{path}`" for path in missing], ""])
    else:
        lines.extend(
            [
                "| G control | Seed successes (/100) | Mean SR | Pooled |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for control in CONTROLS:
            row = aggregate[control]
            lines.append(
                f"| {control} | {row['seed_success']} | {row['mean_seed_sr']:.3f} | "
                f"{row['pooled_success']}/{row['pooled_total']} |"
            )
        lines.extend(
            [
                "",
                "| A | B | Seed delta (B-A) | Per-seed A/B only | Pooled A only | Pooled B only | Exact p |",
                "| --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in comparisons:
            per_seed_label = [
                f"{seed}:{values['a_only']}/{values['b_only']}"
                for seed, values in row["per_seed"].items()
            ]
            lines.append(
                f"| {row['a']} | {row['b']} | {row['seed_delta_b_minus_a']} | "
                f"{per_seed_label} | "
                f"{row['a_only']} | {row['b_only']} | {row['exact_two_sided_p']:.4g} |"
            )
    with open(args.out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"complete": result["complete"], "missing": len(missing)}))


if __name__ == "__main__":
    main()
