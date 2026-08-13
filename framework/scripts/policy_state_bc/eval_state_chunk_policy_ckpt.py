#!/usr/bin/env python3
"""Evaluate an existing CALVIN state-policy checkpoint without loading training data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import train_eval_state_chunk_flow_policy as policy

base = policy.base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--eval-scan-sequences", type=int, default=5000)
    parser.add_argument("--eval-sequence-workers", type=int, default=8)
    parser.add_argument("--eval-task-filter", nargs="+", default=["close_drawer"])
    parser.add_argument("--eval-sequence-offset", type=int, default=0)
    parser.add_argument("--action-execution-horizon", type=int, required=True)
    parser.add_argument("--temporal-ensemble", type=int, choices=[0, 1], default=0)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.25)
    parser.add_argument("--g-scale", type=float, default=1.0)
    parser.add_argument("--query-g-control", default="normal")
    parser.add_argument(
        "--online-minimal-g-mode",
        choices=["generated_query", "behavior_block_center"],
        default="generated_query",
    )
    parser.add_argument(
        "--behavior-contact-radius",
        type=float,
        default=0.01,
        help="Surface radius used to switch behavior block-center G from attraction to flow.",
    )
    parser.add_argument("--language-control", default="normal")
    parser.add_argument("--policy-sample-seed", type=int, default=None)
    parser.add_argument(
        "--fm-reseed-each-episode",
        type=int,
        choices=[0, 1],
        default=1,
        help="Reset FM sampling noise to the same seed at every rollout (1) or keep one deterministic stream (0).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--state-only-observation",
        action="store_true",
        help="Skip unused camera rendering for policies that consume only robot/scene state.",
    )
    parser.add_argument(
        "--profile-eval-timing",
        action="store_true",
        help="Report policy-inference and environment-step timing for each rollout.",
    )
    parser.add_argument(
        "--trace-rollouts",
        type=int,
        default=0,
        help="Record behavior-G, EEF, object, and action traces for the first N rollouts.",
    )
    args = parser.parse_args()

    with open(args.run_meta, encoding="utf-8") as f:
        run_meta = json.load(f)
    if "tasks" not in run_meta:
        raise KeyError(f"Missing task index in {args.run_meta}")
    task_to_id = {str(task): int(index) for task, index in run_meta["tasks"].items()}
    text_to_task, _ = base.load_eval_annotation_maps(args.calvin_code.resolve())

    summary = policy.evaluate_policy_ckpt(
        args.calvin_code.resolve(),
        args.data_root.resolve(),
        args.eval_dir.resolve(),
        args.checkpoint.resolve(),
        task_to_id,
        text_to_task,
        num_sequences=args.num_sequences,
        ep_len=args.ep_len,
        device=args.device,
        eval_mode="balanced_first_task",
        eval_task_filter=set(args.eval_task_filter),
        eval_scan_sequences=args.eval_scan_sequences,
        eval_sequence_workers=args.eval_sequence_workers,
        eval_sequence_offset=args.eval_sequence_offset,
        g_scale=args.g_scale,
        query_g_control=args.query_g_control,
        online_minimal_g_mode=args.online_minimal_g_mode,
        behavior_contact_radius=args.behavior_contact_radius,
        language_control=args.language_control,
        action_execution_horizon=args.action_execution_horizon,
        temporal_ensemble=bool(args.temporal_ensemble),
        temporal_ensemble_decay=args.temporal_ensemble_decay,
        state_only_observation=args.state_only_observation,
        policy_sample_seed=args.policy_sample_seed,
        fm_reseed_each_episode=bool(args.fm_reseed_each_episode),
        profile_eval_timing=args.profile_eval_timing,
        trace_rollouts=args.trace_rollouts,
    )
    print(json.dumps(summary, indent=2, default=base.json_default), flush=True)
    # PyBullet/EGL can hang during interpreter teardown on headless servers.
    os._exit(0)


if __name__ == "__main__":
    main()
