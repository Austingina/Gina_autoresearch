# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Hold-out offline evaluation for a trained CQL-SAC LSTM checkpoint (no Isaac Sim).

Example:

.. code-block:: bash

    python scripts/reinforcement_learning/eval_lift_cube_offline_cql_lstm.py \\
        --checkpoint ./runs/lift_cube_offline_cql_lstm/cql_sac_lstm/checkpoint_epoch_0100.pt \\
        --dataset_file ./datasets/lift_cube_lenient_teacher_maskdepth
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

_OFFLINE_CQL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/teacher_student"
)
if str(_OFFLINE_CQL_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFFLINE_CQL_ROOT))

from offline_cql import (  # noqa: E402
    build_agent_from_checkpoint,
    evaluate_on_holdout,
    load_checkpoint,
    load_maskdepth_hdf5_episodes,
    split_train_holdout_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline hold-out eval for CQL-SAC LSTM checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="./datasets/lift_cube_lenient_teacher_maskdepth",
        help="HDF5 file or directory of *.hdf5 files.",
    )
    parser.add_argument("--obs_mode", type=str, default=None, choices=("vision", "proprio", "vision_proprio"))
    parser.add_argument("--image_size", type=int, default=None, help="Override checkpoint image size (0 = native).")
    parser.add_argument("--lstm_seq_len", type=int, default=None)
    parser.add_argument("--holdout_fraction", type=float, default=0.1)
    parser.add_argument("--eval_sequences", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_demos", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    obs_mode = args.obs_mode or ckpt_args.get("obs_mode", "vision")
    image_size = args.image_size
    if image_size is None:
        image_size = ckpt_args.get("image_size", 64)
    if image_size is not None and image_size <= 0:
        image_size = None

    seq_len = args.lstm_seq_len or ckpt_args.get("lstm_seq_len", 20)
    max_demos = None if args.max_demos <= 0 else args.max_demos

    episodes, action_stats = load_maskdepth_hdf5_episodes(
        args.dataset_file,
        obs_mode=obs_mode,
        image_size=image_size,
        max_demos=max_demos,
    )
    _, holdout_episodes = split_train_holdout_episodes(episodes, args.holdout_fraction, seed=args.seed)
    agent, _, _ = build_agent_from_checkpoint(checkpoint, device=device)

    metrics = evaluate_on_holdout(
        agent,
        action_stats,
        holdout_episodes,
        seq_len=seq_len,
        obs_mode=obs_mode,
        device=device,
        num_sequences=args.eval_sequences,
    )

    result = {
        "checkpoint": args.checkpoint,
        "dataset_file": args.dataset_file,
        "obs_mode": obs_mode,
        "image_size": image_size,
        "lstm_seq_len": seq_len,
        "holdout_fraction": args.holdout_fraction,
        "num_holdout_demos": len(holdout_episodes),
        "action_mse": metrics.action_mse,
        "action_mae": metrics.action_mae,
        "mean_q1": metrics.mean_q1,
        "num_sequences": metrics.num_sequences,
    }

    print("[INFO] Offline hold-out evaluation")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  holdout demos: {len(holdout_episodes)}")
    print(f"  action_mse: {metrics.action_mse:.6f}")
    print(f"  action_mae: {metrics.action_mae:.6f}")
    print(f"  mean_q1: {metrics.mean_q1:.4f}")
    print(f"  sampled sequences: {metrics.num_sequences}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Wrote metrics to {args.output_json}")


if __name__ == "__main__":
    main()
