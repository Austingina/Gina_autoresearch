#!/usr/bin/env python3
"""Validate and aggregate disjoint state-policy evaluation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROTOCOL_KEYS = (
    "checkpoint",
    "ep_len",
    "eval_mode",
    "eval_scan_sequences",
    "g_scale",
    "query_g_control",
    "language_control",
    "action_execution_horizon",
    "temporal_ensemble",
    "temporal_ensemble_decay",
    "state_only_observation",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    shards = []
    for directory in args.shard_dirs:
        with open(directory / "eval_progress.json", encoding="utf-8") as handle:
            progress = json.load(handle)
        with open(directory / "state_bc_summary.json", encoding="utf-8") as handle:
            summary = json.load(handle)
        if len(summary["raw_results"]) != int(progress["num_sequences"]):
            raise RuntimeError(f"Incomplete shard: {directory}")
        if summary["selected_sequence_indices"] != progress["selected_sequence_indices"]:
            raise RuntimeError(f"Sequence metadata mismatch: {directory}")
        shards.append((int(progress["eval_sequence_offset"]), directory, progress, summary))
    shards.sort(key=lambda item: item[0])

    reference = shards[0][2]
    for _offset, directory, progress, _summary in shards[1:]:
        for key in PROTOCOL_KEYS:
            if progress.get(key) != reference.get(key):
                raise RuntimeError(
                    f"Protocol mismatch for {key} in {directory}: "
                    f"{progress.get(key)!r} != {reference.get(key)!r}"
                )

    selected_tasks = []
    selected_indices = []
    raw_results = []
    for _offset, _directory, _progress, summary in shards:
        selected_tasks.extend(summary["selected_tasks"])
        selected_indices.extend(int(value) for value in summary["selected_sequence_indices"])
        raw_results.extend(int(value) for value in summary["raw_results"])
    if len(selected_indices) != len(set(selected_indices)):
        raise RuntimeError("Shard sequence indices overlap")

    task_info = {}
    for task in sorted(set(selected_tasks)):
        positions = [index for index, value in enumerate(selected_tasks) if value == task]
        success = int(sum(raw_results[index] for index in positions))
        task_info[task] = {
            "success": success,
            "total": int(len(positions)),
            "sr": float(success / len(positions)),
        }
    total_success = int(sum(raw_results))
    aggregate = {
        "eval_mode": reference["eval_mode"],
        "checkpoint": reference["checkpoint"],
        "num_sequences": int(len(raw_results)),
        "ep_len": int(reference["ep_len"]),
        "g_scale": float(reference["g_scale"]),
        "query_g_control": reference["query_g_control"],
        "language_control": reference["language_control"],
        "action_execution_horizon": int(reference["action_execution_horizon"]),
        "temporal_ensemble": bool(reference["temporal_ensemble"]),
        "temporal_ensemble_decay": float(reference["temporal_ensemble_decay"]),
        "state_only_observation": bool(reference.get("state_only_observation", False)),
        "avg_seq_len": float(total_success / len(raw_results)),
        "chain_sr": {"1": float(total_success / len(raw_results))},
        "raw_results": raw_results,
        "task_info": task_info,
        "selected_tasks": selected_tasks,
        "selected_sequence_indices": selected_indices,
        "shards": [
            {
                "eval_sequence_offset": offset,
                "directory": directory.resolve().as_posix(),
                "num_sequences": int(progress["num_sequences"]),
            }
            for offset, directory, progress, _summary in shards
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "state_bc_summary.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
