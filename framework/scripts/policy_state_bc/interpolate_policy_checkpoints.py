#!/usr/bin/env python3
"""Linearly interpolate compatible state-policy checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--alpha-b", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha_b <= 1.0:
        raise ValueError("--alpha-b must be in [0, 1]")

    checkpoint_a = torch.load(args.checkpoint_a, map_location="cpu")
    checkpoint_b = torch.load(args.checkpoint_b, map_location="cpu")
    state_a = checkpoint_a["model_state"]
    state_b = checkpoint_b["model_state"]
    if state_a.keys() != state_b.keys():
        raise ValueError("model_state keys differ")

    compatible_metadata = (
        "mode",
        "chunk_size",
        "cond_dim",
        "action_dim",
        "hidden",
        "layers",
        "base_dim",
        "g_dim",
        "minimal_g_scope",
        "query_g_include_token_type",
        "query_g_token_type_mode",
    )
    for key in compatible_metadata:
        if checkpoint_a.get(key) != checkpoint_b.get(key):
            raise ValueError(f"incompatible checkpoint metadata: {key}")

    alpha = args.alpha_b
    merged_state = {}
    for key, tensor_a in state_a.items():
        tensor_b = state_b[key]
        if tensor_a.shape != tensor_b.shape or tensor_a.dtype != tensor_b.dtype:
            raise ValueError(f"incompatible tensor: {key}")
        if tensor_a.is_floating_point() or tensor_a.is_complex():
            merged_state[key] = torch.lerp(tensor_a, tensor_b, alpha)
        elif torch.equal(tensor_a, tensor_b):
            merged_state[key] = tensor_a.clone()
        else:
            raise ValueError(f"non-floating tensor differs: {key}")

    merged = copy.deepcopy(checkpoint_a)
    merged["model_state"] = merged_state
    merged["checkpoint_interpolation"] = {
        "checkpoint_a": str(args.checkpoint_a.resolve()),
        "checkpoint_b": str(args.checkpoint_b.resolve()),
        "alpha_b": alpha,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out)
    print(json.dumps(merged["checkpoint_interpolation"], indent=2))


if __name__ == "__main__":
    main()
