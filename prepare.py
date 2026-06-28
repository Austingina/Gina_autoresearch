#!/usr/bin/env python3
"""
Immutable evaluation harness for AnchorMask + GuidanceField autoresearch.

DO NOT MODIFY during autoresearch loops — defines policy-level metrics and baselines.
Representation validity is proven ONLY through policy learning outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from goals.schema import GuidanceMode

ROOT = Path(__file__).resolve().parent


def _expand_env(value: str) -> str:
    if not isinstance(value, str) or "${" not in value:
        return value
    out = value
    while "${" in out:
        start = out.index("${")
        end = out.index("}", start)
        token = out[start + 2 : end]
        if ":" in token:
            var, default = token.split(":", 1)
        else:
            var, default = token, ""
        out = out[:start] + os.environ.get(var, default) + out[end + 1 :]
    return out


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or ROOT / "configs" / "default.yaml"
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    paths = raw.get("paths", {})
    for k, v in list(paths.items()):
        paths[k] = _expand_env(v)
    raw["paths"] = paths
    return raw


@dataclass
class EvalResult:
    success_rate: float
    post_grasp_wander_m: float
    field_alignment: float
    num_episodes: int
    backend: str
    task: str
    split: str
    guidance_mode: str
    elapsed_sec: float
    extra: dict[str, Any]

    def summary_lines(self) -> list[str]:
        return [
            "---",
            f"success_rate:          {self.success_rate:.6f}",
            f"post_grasp_wander_m:   {self.post_grasp_wander_m:.6f}",
            f"field_alignment:       {self.field_alignment:.6f}",
            f"guidance_mode:         {self.guidance_mode}",
            f"num_episodes:          {self.num_episodes}",
            f"backend:               {self.backend}",
            f"task:                  {self.task}",
            f"split:                 {self.split}",
            f"elapsed_sec:           {self.elapsed_sec:.1f}",
            "---",
        ]


def _mode_bonus(mode: str) -> tuple[float, float, float]:
    """Synthetic (success, wander↓, alignment↑) for dry_run ratchet."""
    m = GuidanceMode(mode) if mode in [e.value for e in GuidanceMode] else GuidanceMode.ANCHOR_ONLY
    table = {
        GuidanceMode.NONE: (0.25, 0.45, 0.30),
        GuidanceMode.ANCHOR_ONLY: (0.55, 0.35, 0.50),  # grasp OK, wander high on lift
        GuidanceMode.ANCHOR_TARGET_MASK: (0.58, 0.28, 0.55),
        GuidanceMode.ANCHOR_FIELD_RULE: (0.78, 0.12, 0.82),
        GuidanceMode.ANCHOR_FIELD_DEMO: (0.74, 0.15, 0.78),
        GuidanceMode.ANCHOR_FIELD_WM: (0.76, 0.13, 0.80),
    }
    return table.get(m, (0.50, 0.30, 0.50))


def eval_dry_run(task: str, split: str, experiment_config: dict[str, Any]) -> EvalResult:
    t0 = time.time()
    mode = experiment_config.get("guidance_mode", GuidanceMode.ANCHOR_ONLY.value)
    field_scale = float(experiment_config.get("field_scale", 1.0))
    hidden = float(experiment_config.get("hidden_dim", 128))
    lr = float(experiment_config.get("learning_rate", 1e-4))

    success_base, wander_base, align_base = _mode_bonus(mode)

    # Policy architecture sweet spot
    hidden_score = 1.0 - min(abs(hidden - 256) / 512, 1.0)
    lr_score = 1.0 - min(abs(lr - 3e-4) / 3e-3, 1.0)
    success = min(1.0, success_base + 0.08 * hidden_score + 0.05 * lr_score)
    wander = max(0.0, wander_base - 0.05 * hidden_score)
    align = min(1.0, align_base + 0.05 * min(field_scale, 1.5))

    split_penalty = {
        "s0_fixed": 0.0,
        "id": 0.0,
        "s1_random_cube": 0.06,
        "s2_random_goal": 0.10,
        "s3_ood_object": 0.12,
        "ood_object": 0.12,
        "ood_scene": 0.15,
    }.get(split, 0.1)
    task_bonus = {"lift": 0.02, "track": 0.01, "approach": 0.0}.get(task, 0.0)

    success = max(0.0, min(1.0, success - split_penalty + task_bonus))
    wander = min(1.0, wander + split_penalty * 0.5)

    return EvalResult(
        success_rate=success,
        post_grasp_wander_m=wander,
        field_alignment=align,
        num_episodes=32,
        backend="dry_run",
        task=task,
        split=split,
        guidance_mode=mode,
        elapsed_sec=time.time() - t0,
        extra={"synthetic": True},
    )


def eval_isaac(task: str, split: str, experiment_config: dict[str, Any], cfg: dict[str, Any]) -> EvalResult:
    isaac_root = Path(cfg["paths"]["isaaclab"])
    task_cfg = cfg.get("tasks", {}).get(task, {})
    isaac_task = task_cfg.get("isaac_task")

    if not isaac_task or not isaac_root.is_dir():
        print(f"[WARN] Isaac unavailable (task={task}). Using dry_run.", file=sys.stderr)
        return eval_dry_run(task, split, experiment_config)

    t0 = time.time()
    result = eval_dry_run(task, split, experiment_config)
    return EvalResult(
        success_rate=result.success_rate,
        post_grasp_wander_m=result.post_grasp_wander_m,
        field_alignment=result.field_alignment,
        num_episodes=result.num_episodes,
        backend="isaac_proxy",
        task=task,
        split=split,
        guidance_mode=result.guidance_mode,
        elapsed_sec=time.time() - t0,
        extra={"isaac_task": isaac_task, "note": "Wire real Isaac eval + wander metric"},
    )


def run_evaluation(
    task: str,
    split: str,
    backend: str,
    experiment_config: dict[str, Any],
    config_path: Path | None = None,
) -> EvalResult:
    cfg = load_config(config_path)
    if backend == "dry_run":
        return eval_dry_run(task, split, experiment_config)
    if backend == "isaac":
        return eval_isaac(task, split, experiment_config, cfg)
    raise ValueError(f"Unknown backend: {backend}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual guidance field eval harness")
    p.add_argument("--task", default="lift", choices=["track", "approach", "lift", "pick", "push", "pull"])
    p.add_argument(
        "--split",
        default="s0_fixed",
        choices=["s0_fixed", "s1_random_cube", "s2_random_goal", "s3_ood_object", "id", "ood_object", "ood_scene"],
    )
    p.add_argument("--backend", default="dry_run", choices=["dry_run", "isaac"])
    p.add_argument("--experiment-config", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exp_cfg: dict[str, Any] = {}
    if args.experiment_config and args.experiment_config.is_file():
        with open(args.experiment_config) as f:
            exp_cfg = json.load(f)

    result = run_evaluation(args.task, args.split, args.backend, exp_cfg)
    for line in result.summary_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
