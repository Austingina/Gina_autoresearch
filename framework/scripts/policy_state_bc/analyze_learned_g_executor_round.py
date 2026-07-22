#!/usr/bin/env python3
"""Aggregate the matched H1/H16 reusable-G executor experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


MODES = (
    "chunk_bc_lang",
    "chunk_bc_minimal_g_concat",
    "chunk_bc_minimal_g_film",
)
HORIZONS = {
    "h1": "2026-07-14_learned_g_executor_chunk1_exec1_e50_n100_h160_seed{seed}",
    "h16": "2026-07-14_learned_g_executor_chunk16_exec16_e50_n100_h160_seed{seed}",
}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    records: dict[str, dict[str, dict[str, Any]]] = {h: {} for h in HORIZONS}
    missing: list[str] = []
    reference_indices: list[int] | None = None
    indices_match = True

    for horizon, stem in HORIZONS.items():
        for mode in MODES:
            records[horizon][mode] = {}
            for seed in range(5):
                run_dir = args.experiment_root / stem.format(seed=seed)
                summary_path = run_dir / mode / "task_eval" / "state_bc_summary.json"
                metrics_path = run_dir / mode / "train_metrics.json"
                if not summary_path.exists():
                    missing.append(str(summary_path))
                    continue
                summary = load_json(summary_path)
                raw = [int(value) for value in summary["raw_results"]]
                indices = [int(value) for value in summary["selected_sequence_indices"]]
                if reference_indices is None:
                    reference_indices = indices
                elif indices != reference_indices:
                    indices_match = False
                train_final = load_json(metrics_path)["final"] if metrics_path.exists() else {}
                records[horizon][mode][str(seed)] = {
                    "success": int(sum(raw)),
                    "total": len(raw),
                    "sr": float(mean(raw)),
                    "raw_results": raw,
                    "selected_sequence_indices": indices,
                    "val_chunk_mse": train_final.get("val_chunk_mse"),
                    "val_first_action_mse": train_final.get("val_first_action_mse"),
                }

    aggregate: dict[str, dict[str, Any]] = {h: {} for h in HORIZONS}
    comparisons: list[dict[str, Any]] = []
    if not missing:
        for horizon in HORIZONS:
            for mode in MODES:
                rows = [records[horizon][mode][str(seed)] for seed in range(5)]
                rates = [row["sr"] for row in rows]
                aggregate[horizon][mode] = {
                    "seed_success": [row["success"] for row in rows],
                    "seed_sr": rates,
                    "mean_seed_sr": mean(rates),
                    "sample_std_seed_sr": stdev(rates),
                    "median_seed_sr": median(rates),
                    "min_seed_sr": min(rates),
                    "max_seed_sr": max(rates),
                    "pooled_success": sum(row["success"] for row in rows),
                    "pooled_total": sum(row["total"] for row in rows),
                    "mean_val_chunk_mse": mean(float(row["val_chunk_mse"]) for row in rows),
                    "mean_val_first_action_mse": mean(float(row["val_first_action_mse"]) for row in rows),
                }

        for horizon in HORIZONS:
            for mode_a, mode_b in (
                (MODES[0], MODES[1]),
                (MODES[0], MODES[2]),
                (MODES[1], MODES[2]),
            ):
                a = sum((records[horizon][mode_a][str(seed)]["raw_results"] for seed in range(5)), [])
                b = sum((records[horizon][mode_b][str(seed)]["raw_results"] for seed in range(5)), [])
                seed_deltas = [
                    records[horizon][mode_b][str(seed)]["sr"] - records[horizon][mode_a][str(seed)]["sr"]
                    for seed in range(5)
                ]
                comparisons.append(
                    {
                        "scope": horizon,
                        "a": mode_a,
                        "b": mode_b,
                        "seed_delta_b_minus_a": seed_deltas,
                        "mean_seed_delta_b_minus_a": mean(seed_deltas),
                        **exact_mcnemar(a, b),
                    }
                )

        for mode in MODES:
            a = sum((records["h1"][mode][str(seed)]["raw_results"] for seed in range(5)), [])
            b = sum((records["h16"][mode][str(seed)]["raw_results"] for seed in range(5)), [])
            seed_deltas = [
                records["h16"][mode][str(seed)]["sr"] - records["h1"][mode][str(seed)]["sr"]
                for seed in range(5)
            ]
            comparisons.append(
                {
                    "scope": "h16_vs_h1",
                    "a": f"h1/{mode}",
                    "b": f"h16/{mode}",
                    "seed_delta_b_minus_a": seed_deltas,
                    "mean_seed_delta_b_minus_a": mean(seed_deltas),
                    **exact_mcnemar(a, b),
                }
            )

    result = {
        "complete": not missing,
        "missing": missing,
        "protocol": {
            "num_seeds": 5,
            "rollouts_per_seed": 100,
            "episode_length": 160,
            "eval_scan_sequences": 5000,
            "matched_sequence_indices": indices_match,
            "reference_sequence_count": len(reference_indices or []),
            "reference_max_sequence_index": max(reference_indices or [-1]),
        },
        "aggregate": aggregate,
        "comparisons": comparisons,
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = [
        "# Learned G Executor Horizon Comparison",
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
                "## Close Drawer",
                "",
                "| Horizon | Mode | Seed successes (/100) | Mean SR | SD | Pooled |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for horizon in HORIZONS:
            for mode in MODES:
                row = aggregate[horizon][mode]
                lines.append(
                    f"| {horizon} | {mode} | {row['seed_success']} | "
                    f"{row['mean_seed_sr']:.3f} | {row['sample_std_seed_sr']:.3f} | "
                    f"{row['pooled_success']}/{row['pooled_total']} |"
                )
        lines.extend(
            [
                "",
                "## Paired Comparisons",
                "",
                "| Scope | A | B | Mean seed delta (B-A) | A only | B only | Exact p |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in comparisons:
            lines.append(
                f"| {row['scope']} | {row['a']} | {row['b']} | "
                f"{row['mean_seed_delta_b_minus_a']:.3f} | {row['a_only']} | "
                f"{row['b_only']} | {row['exact_two_sided_p']:.4g} |"
            )
    with open(args.out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"complete": not missing, "missing": len(missing), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
