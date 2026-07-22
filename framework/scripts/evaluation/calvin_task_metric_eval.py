#!/usr/bin/env python3
"""Small wrapper around CALVIN's official long-horizon task evaluator.

This script intentionally keeps the official CALVIN source untouched while
making the two constants that matter for smoke tests configurable:
NUM_SEQUENCES and EP_LEN.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Older CALVIN/tacto/pyrender code can still reference NumPy aliases removed by
# NumPy 1.24. Patch them before importing CALVIN modules.
for _alias, _target in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
}.items():
    if _alias not in np.__dict__:
        setattr(np, _alias, _target)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class ConstantPolicy:
    """A minimal policy for evaluator smoke tests."""

    def __init__(self, action: np.ndarray):
        self.action = action.astype(np.float32)

    def reset(self) -> None:
        return None

    def step(self, obs: Any, goal: Any) -> np.ndarray:
        return self.action.copy()


class RandomPolicy:
    """A bounded random-action policy for evaluator smoke tests."""

    def __init__(self, seed: int, action_dim: int = 7, scale: float = 0.05):
        self.rng = np.random.default_rng(seed)
        self.action_dim = action_dim
        self.scale = scale

    def reset(self) -> None:
        return None

    def step(self, obs: Any, goal: Any) -> np.ndarray:
        action = self.rng.normal(0.0, self.scale, size=self.action_dim).astype(np.float32)
        action[-1] = -1.0
        return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CALVIN task-level metric smoke/eval.")
    parser.add_argument("--calvin-code", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--eval-log-dir", required=True, type=Path)
    parser.add_argument("--policy", choices=("checkpoint", "zero", "random"), default="zero")
    parser.add_argument("--train-folder", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-sequences", type=int, default=2)
    parser.add_argument("--ep-len", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--create-plan-tsne", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launch_cwd = Path.cwd().resolve()

    def resolve_path(path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else (launch_cwd / path).resolve()

    args.calvin_code = resolve_path(args.calvin_code)
    args.dataset_path = resolve_path(args.dataset_path)
    args.eval_log_dir = resolve_path(args.eval_log_dir)
    args.train_folder = resolve_path(args.train_folder)
    args.checkpoint = resolve_path(args.checkpoint)

    calvin_code = args.calvin_code.resolve()
    calvin_models = calvin_code / "calvin_models"
    calvin_agent = calvin_models / "calvin_agent"
    calvin_env = calvin_code / "calvin_env"

    sys.path.insert(0, calvin_models.as_posix())
    sys.path.insert(0, calvin_env.as_posix())

    # CALVIN's hydra.initialize("../../conf/...") calls assume this cwd.
    os.chdir(calvin_agent)

    from calvin_agent.evaluation import evaluate_policy as eval_mod
    from calvin_agent.evaluation.utils import count_success, get_default_model_and_env

    eval_mod.NUM_SEQUENCES = int(args.num_sequences)
    eval_mod.EP_LEN = int(args.ep_len)
    args.eval_log_dir.mkdir(parents=True, exist_ok=True)

    if args.policy == "checkpoint":
        if args.train_folder is None or args.checkpoint is None:
            raise ValueError("--policy checkpoint requires --train-folder and --checkpoint")
        model, env, _ = get_default_model_and_env(
            args.train_folder.as_posix(),
            args.dataset_path.as_posix(),
            args.checkpoint,
            env=None,
            device_id=args.device,
        )
        epoch = args.checkpoint.stem
        create_plan_tsne = args.create_plan_tsne
    elif args.policy == "zero":
        zero_action = np.zeros(7, dtype=np.float32)
        zero_action[-1] = -1.0
        model = ConstantPolicy(zero_action)
        env = eval_mod.make_env(args.dataset_path)
        epoch = "zero_policy"
        create_plan_tsne = False
    else:
        model = RandomPolicy(seed=args.seed)
        env = eval_mod.make_env(args.dataset_path)
        epoch = "random_policy"
        create_plan_tsne = False

    results = eval_mod.evaluate_policy(
        model,
        env,
        epoch=epoch,
        eval_log_dir=args.eval_log_dir,
        debug=False,
        create_plan_tsne=create_plan_tsne,
    )

    summary = {
        "policy": args.policy,
        "dataset_path": args.dataset_path,
        "train_folder": args.train_folder,
        "checkpoint": args.checkpoint,
        "num_sequences": args.num_sequences,
        "ep_len": args.ep_len,
        "avg_seq_len": float(np.mean(results)) if results else 0.0,
        "chain_sr": {str(i + 1): float(sr) for i, sr in enumerate(count_success(results))},
        "raw_results": results,
    }
    with open(args.eval_log_dir / "task_metric_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    sys.stdout.flush()
    sys.stderr.flush()

    # Some CALVIN/PyBullet/EGL combinations keep non-daemon cleanup resources
    # alive after metrics are written. The wrapper is a one-shot CLI process, so
    # terminate cleanly from the run-contract perspective once files are flushed.
    os._exit(0)


if __name__ == "__main__":
    main()
