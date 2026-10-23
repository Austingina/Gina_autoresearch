#!/usr/bin/env python3
"""Contract tests for matched-task, fixed-reset, multi-seed same-G plans."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "framework/scripts/policy_state_bc/build_matched_task_multiseed_plan.py"


def load_module():
    spec = importlib.util.spec_from_file_location("matched_plan", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_seed_runs_and_matched_task_caches() -> None:
    module = load_module()
    validation = (
        REPO
        / "experiments/phase1_calvin/2026-10-20_same_g_full_formal/common/official_validation/.hydra/merged_config.yaml"
    )
    with tempfile.TemporaryDirectory() as tmp:
        plan = module.build_matched_plan(
            REPO,
            Path(tmp),
            validation,
            seeds=(20260814, 20260815, 20260816),
        )

    assert plan["format"] == "same_g_matched_task_multiseed_plan_v1"
    assert plan["training_seeds"] == [20260814, 20260815, 20260816]
    assert len(plan["conditions"]) == 13
    assert len(plan["runs"]) == 39
    assert len({run["run_id"] for run in plan["runs"]}) == 39
    assert all(run["training_parity_sha256"] == plan["training_parity_sha256"] for run in plan["runs"])

    condition_by_name = {row["name"]: row for row in plan["conditions"]}
    for run in plan["runs"]:
        condition = condition_by_name[run["condition"]]
        assert run["heldout_tasks"] == condition["heldout_tasks"]
        assert run["source_tasks"] == condition["source_tasks"]
        assert set(run["heldout_tasks"]).isdisjoint(run["source_tasks"])
        assert run["init_seed"] == run["training_seed"]
        assert run["loader_seed"] == run["training_seed"]
        assert run["data_seed"] == run["training_seed"]
        assert run["split_seed"] == run["training_seed"]

    assert plan["evaluation"]["policy_sample_seed"] == 20260814
    assert plan["evaluation"]["num_sequences_per_task"] == 100
    eval_cells = plan["evaluation_cells"]
    assert len(eval_cells) == sum(len(run["heldout_tasks"]) for run in plan["runs"])
    for cell in eval_cells:
        run = next(run for run in plan["runs"] if run["run_id"] == cell["run_id"])
        assert cell["task"] in run["heldout_tasks"]
        assert cell["sequence_cache"].endswith(f"sequence_caches/{cell['task']}.pkl")
        assert cell["sequence_pool_key"] == f"{cell['task']}|seed=20260814|n=100|h=160"

    for task, group in plan["matched_task_groups"].items():
        assert group["task"] == task
        assert len(group["condition_names"]) >= 3
        assert len(group["run_ids"]) == len(group["condition_names"]) * 3
        assert len(set(group["sequence_pool_keys"])) == 1


if __name__ == "__main__":
    test_three_seed_runs_and_matched_task_caches()
    print("matched-task multiseed plan: OK")
