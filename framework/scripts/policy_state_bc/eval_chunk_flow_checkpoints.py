#!/usr/bin/env python3
"""Evaluate existing chunk-BC / chunk-FM checkpoints on selected CALVIN tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

for _alias, _target in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
}.items():
    if _alias not in np.__dict__:
        setattr(np, _alias, _target)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402
import train_eval_state_chunk_flow_policy as chunk_pol  # noqa: E402


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def parse_policy_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.parent.parent.name, path
    alias, path = spec.split("=", 1)
    alias = alias.strip()
    if not alias:
        raise ValueError(f"Empty alias in policy spec: {spec!r}")
    return alias, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", required=True, help="alias=/path/to/policy.pt entries")
    parser.add_argument("--eval-task-filter", nargs="+", required=True)
    parser.add_argument("--num-sequences", type=int, default=25, help="Rollouts per selected first task")
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--eval-scan-sequences", type=int, default=5000)
    parser.add_argument(
        "--eval-sequence-offset",
        type=int,
        default=0,
        help="Skip this many matching CALVIN sequences per task before selecting eval rollouts.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--g-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--query-g-controls",
        nargs="+",
        default=["normal"],
        help=(
            "Eval-time query-G controls: normal, zero, sign_flip, zero_attraction, "
            "zero_flow, sign_flip_attraction, task:TASK_NAME, or mask:FEATURE_GROUP."
        ),
    )
    parser.add_argument(
        "--language-controls",
        nargs="+",
        default=["normal"],
        help="Eval-time language/task controls: normal, zero, or task:TASK_NAME.",
    )
    args = parser.parse_args()

    launch_cwd = Path.cwd().resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (launch_cwd / path).resolve()

    args.calvin_code = resolve(args.calvin_code)
    args.data_root = resolve(args.data_root)
    args.out_dir = resolve(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policies = [(alias, resolve(path)) for alias, path in map(parse_policy_spec, args.policies)]

    text_to_task, _task_to_text = base.load_eval_annotation_maps(args.calvin_code)
    segments = base.load_lang_segments(args.data_root / "training")
    task_to_id = base.build_task_index(segments, text_to_task)

    run_meta = {
        "data_root": args.data_root,
        "policies": [{"alias": alias, "path": path} for alias, path in policies],
        "eval_task_filter": args.eval_task_filter,
        "num_sequences_per_task": args.num_sequences,
        "ep_len": args.ep_len,
        "eval_scan_sequences": args.eval_scan_sequences,
        "eval_sequence_offset": args.eval_sequence_offset,
        "g_scales": args.g_scales,
        "query_g_controls": args.query_g_controls,
        "language_controls": args.language_controls,
    }
    with open(args.out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, default=json_default)

    rows: list[dict[str, Any]] = []
    for alias, ckpt_path in policies:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing policy checkpoint: {ckpt_path}")
        for g_scale in args.g_scales:
            for query_g_control in args.query_g_controls:
                for language_control in args.language_controls:
                    scale_label = f"g{g_scale:g}".replace("-", "m").replace(".", "p")
                    query_label = query_g_control.replace(":", "_")
                    lang_label = language_control.replace(":", "_")
                    eval_dir = args.out_dir / alias / scale_label / query_label / lang_label
                    summary = chunk_pol.evaluate_policy_ckpt(
                        args.calvin_code,
                        args.data_root,
                        eval_dir,
                        ckpt_path,
                        task_to_id,
                        text_to_task,
                        num_sequences=args.num_sequences,
                        ep_len=args.ep_len,
                        device=args.device,
                        eval_mode="balanced_first_task",
                        eval_task_filter=set(args.eval_task_filter),
                        eval_scan_sequences=args.eval_scan_sequences,
                        eval_sequence_offset=args.eval_sequence_offset,
                        g_scale=g_scale,
                        query_g_control=query_g_control,
                        language_control=language_control,
                    )
                    row = {
                        "alias": alias,
                        "checkpoint": ckpt_path,
                        "g_scale": g_scale,
                        "query_g_control": query_g_control,
                        "language_control": language_control,
                        "avg_seq_len": summary["avg_seq_len"],
                        **{f"sr_{k}": v for k, v in summary["chain_sr"].items()},
                    }
                    for task, info in summary.get("task_info", {}).items():
                        row[f"task_{task}_success"] = info["success"]
                        row[f"task_{task}_total"] = info["total"]
                        row[f"task_{task}_sr"] = info["sr"]
                    rows.append(row)
                    print(f"[result] {json.dumps(row, default=json_default)}", flush=True)

    with open(args.out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=json_default)
    with open(args.out_dir / "comparison.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps(rows, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
