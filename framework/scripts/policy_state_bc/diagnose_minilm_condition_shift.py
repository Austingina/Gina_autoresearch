#!/usr/bin/env python3
"""Audit MiniLM normalization shift between CALVIN train and evaluation tasks."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_chunk_flow_policy as policy  # noqa: E402


def training_task_embeddings(data_root: Path) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    for segment in policy.base.load_lang_segments(data_root / "training"):
        if segment.get("emb") is None:
            continue
        rows[str(segment["task"])].append(
            np.asarray(segment["emb"], dtype=np.float32).reshape(-1)
        )
    return {
        task: np.mean(np.stack(task_rows), axis=0).astype(np.float32)
        for task, task_rows in rows.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-data-root", type=Path, required=True)
    parser.add_argument("--eval-data-root", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("executor_language_mode") != "minilm":
        raise ValueError("Checkpoint does not use MiniLM language conditioning")
    state_dim = 15
    if checkpoint.get("executor_state_scope", "full") == "full":
        state_dim += 24
    language_dim = 384
    mean = np.asarray(checkpoint["cond_mean"], dtype=np.float32)[
        state_dim : state_dim + language_dim
    ]
    std = np.asarray(checkpoint["cond_std"], dtype=np.float32)[
        state_dim : state_dim + language_dim
    ]

    train_embeddings = training_task_embeddings(args.train_data_root)
    eval_embeddings = policy.load_task_language_embeddings(args.eval_data_root)
    print(
        f"std min={std.min():.3e} p01={np.quantile(std, 0.01):.3e} "
        f"median={np.median(std):.3e} dims<1e-4={int(np.sum(std < 1e-4))}/{len(std)}"
    )
    for split, embeddings in (("train", train_embeddings), ("eval", eval_embeddings)):
        for task in sorted(embeddings):
            normalized = (embeddings[task] - mean) / std
            print(
                f"{split:5s} {task:32s} l2={np.linalg.norm(normalized):9.2f} "
                f"maxabs={np.max(np.abs(normalized)):9.2f} "
                f"abs>5={np.mean(np.abs(normalized) > 5):.3f} "
                f"abs>20={np.mean(np.abs(normalized) > 20):.3f}"
            )


if __name__ == "__main__":
    main()
