#!/usr/bin/env python3
"""Build a matched-task, fixed-reset, multi-seed CALVIN same-G plan."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import build_same_g_full_formal_plan as base


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if len(normalized) < 2:
        raise ValueError("At least two training seeds are required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Training seeds must be unique")
    return normalized


def build_matched_plan(
    repo: Path,
    output_root: Path,
    validation_config: Path,
    seeds: Iterable[int] = (20260814, 20260815, 20260816),
) -> dict[str, Any]:
    repo = repo.resolve()
    output_root = output_root.resolve()
    training_seeds = validate_seeds(seeds)
    base_plan = base.build_plan(repo, output_root, validation_config)

    shared_training = {
        **base_plan["shared_training"],
        "seed_policy": "init_loader_data_split_equal_training_seed",
    }
    shared_training.pop("seed", None)
    training_parity_sha256 = canonical_sha256(shared_training)
    evaluation = {
        "mode": "balanced_first_task",
        "num_sequences_per_task": 100,
        "ep_len": 160,
        "eval_scan_sequences": 50000,
        "policy_sample_seed": 20260814,
        "state_only_observation": True,
        "action_execution_horizon": 1,
        "controls": ["normal", "zero", "sign_flip"],
        "sequence_pool_scope": "shared_by_task_across_all_conditions_and_training_seeds",
    }

    runs: list[dict[str, Any]] = []
    evaluation_cells: list[dict[str, Any]] = []
    groups: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"condition_names": set(), "run_ids": set(), "sequence_pool_keys": set()}
    )
    for condition in base_plan["conditions"]:
        source_tasks = list(condition["source_tasks"])
        heldout_tasks = list(condition["heldout_tasks"])
        if set(source_tasks) & set(heldout_tasks):
            raise RuntimeError(f"Source/held-out task overlap in {condition['name']}")
        for training_seed in training_seeds:
            run_id = f"{condition['name']}__seed{training_seed}"
            run = {
                "run_id": run_id,
                "condition": condition["name"],
                "condition_kind": condition["kind"],
                "training_seed": training_seed,
                "init_seed": training_seed,
                "loader_seed": training_seed,
                "data_seed": training_seed,
                "split_seed": training_seed,
                "view": condition["view"],
                "manifest": condition["manifest"],
                "manifest_sha256": condition["manifest_sha256"],
                "source_segments": condition["source_segments"],
                "heldout_segments": condition["heldout_segments"],
                "source_tasks": source_tasks,
                "heldout_tasks": heldout_tasks,
                "output_dir": (output_root / "runs" / run_id).as_posix(),
                "training_parity_sha256": training_parity_sha256,
            }
            runs.append(run)
            for task in heldout_tasks:
                sequence_pool_key = (
                    f"{task}|seed={evaluation['policy_sample_seed']}|"
                    f"n={evaluation['num_sequences_per_task']}|h={evaluation['ep_len']}"
                )
                evaluation_cells.append(
                    {
                        "run_id": run_id,
                        "condition": condition["name"],
                        "training_seed": training_seed,
                        "task": task,
                        "sequence_cache": (output_root / "sequence_caches" / f"{task}.pkl").as_posix(),
                        "sequence_pool_key": sequence_pool_key,
                    }
                )
                groups[task]["condition_names"].add(str(condition["name"]))
                groups[task]["run_ids"].add(run_id)
                groups[task]["sequence_pool_keys"].add(sequence_pool_key)

    matched_task_groups: dict[str, dict[str, object]] = {}
    for task, values in sorted(groups.items()):
        condition_names = sorted(values["condition_names"])
        run_ids = sorted(values["run_ids"])
        pool_keys = sorted(values["sequence_pool_keys"])
        if len(condition_names) < 2 or len(pool_keys) != 1:
            raise RuntimeError(f"Task {task} is not a valid matched comparison group")
        matched_task_groups[task] = {
            "task": task,
            "condition_names": condition_names,
            "run_ids": run_ids,
            "sequence_pool_keys": pool_keys,
        }

    return {
        "format": "same_g_matched_task_multiseed_plan_v1",
        "validation_config": base_plan["validation_config"],
        "validation_config_sha256": base_plan["validation_config_sha256"],
        "training_seeds": list(training_seeds),
        "shared_training": shared_training,
        "training_parity_sha256": training_parity_sha256,
        "evaluation": evaluation,
        "conditions": base_plan["conditions"],
        "runs": runs,
        "evaluation_cells": evaluation_cells,
        "matched_task_groups": matched_task_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-config", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260814, 20260815, 20260816])
    args = parser.parse_args()
    plan = build_matched_plan(args.repo, args.output_root, args.validation_config, args.seeds)
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    if args.plan_output.exists():
        raise RuntimeError(f"Refusing to overwrite plan: {args.plan_output}")
    temporary = args.plan_output.with_suffix(args.plan_output.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.plan_output)
    print(
        json.dumps(
            {
                "plan": args.plan_output.as_posix(),
                "conditions": len(plan["conditions"]),
                "runs": len(plan["runs"]),
                "evaluation_cells": len(plan["evaluation_cells"]),
                "matched_tasks": len(plan["matched_task_groups"]),
                "training_parity_sha256": plan["training_parity_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
