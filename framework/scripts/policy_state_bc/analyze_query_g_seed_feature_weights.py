#!/usr/bin/env python3
"""Compare effective first-layer query-G weights across policy checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


TOKEN_COUNT = 4
TOKEN_DIM = 19
GROUPS: dict[str, tuple[int, ...]] = {
    "token_type": tuple(range(0, 4)),
    "rel_eef": tuple(range(4, 7)),
    "rel_part": tuple(range(7, 10)),
    "field_vec": tuple(range(10, 13)),
    "desired_flow": tuple(range(13, 16)),
    "dist_norm": (16,),
    "residual_norm": (17,),
    "contact_conf": (18,),
}
LOCAL_NAMES = [
    "type_eef",
    "type_part",
    "type_contact",
    "type_side",
    "rel_eef_x",
    "rel_eef_y",
    "rel_eef_z",
    "rel_part_x",
    "rel_part_y",
    "rel_part_z",
    "field_x",
    "field_y",
    "field_z",
    "flow_x",
    "flow_y",
    "flow_z",
    "dist_norm",
    "residual_norm",
    "contact_conf",
]


def parse_policy(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.parent.parent.name, path
    alias, raw_path = spec.split("=", 1)
    return alias.strip(), Path(raw_path).expanduser().resolve()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", required=True, help="alias=/path/to/policy.pt entries")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    top_features: dict[str, list[dict[str, Any]]] = {}

    for alias, path in map(parse_policy, args.policies):
        ckpt = torch.load(path, map_location="cpu")
        if ckpt["mode"] not in {"chunk_bc_query_g", "chunk_bc_lang_query_g"}:
            raise ValueError(f"{alias}: unsupported mode {ckpt['mode']}")
        weight = ckpt["model_state"]["net.0.weight"].detach().cpu().numpy().astype(np.float64)
        cond_std = np.asarray(ckpt["cond_std"], dtype=np.float64)
        g_dim = TOKEN_COUNT * TOKEN_DIM
        g_start = int(weight.shape[1] - g_dim)
        raw_effective = weight[:, g_start:] / cond_std[g_start:][None, :]
        normalized = weight[:, g_start:]

        for group, local_dims in GROUPS.items():
            group_dims = [token * TOKEN_DIM + dim for token in range(TOKEN_COUNT) for dim in local_dims]
            rows.append(
                {
                    "alias": alias,
                    "policy": path,
                    "scope": "all_tokens",
                    "token": "all",
                    "group": group,
                    "num_features": len(group_dims),
                    "normalized_weight_rms": rms(normalized[:, group_dims]),
                    "raw_effective_weight_rms": rms(raw_effective[:, group_dims]),
                    "raw_effective_fro": float(np.linalg.norm(raw_effective[:, group_dims])),
                }
            )
            for token in range(TOKEN_COUNT):
                token_dims = [token * TOKEN_DIM + dim for dim in local_dims]
                rows.append(
                    {
                        "alias": alias,
                        "policy": path,
                        "scope": "single_token",
                        "token": token,
                        "group": group,
                        "num_features": len(token_dims),
                        "normalized_weight_rms": rms(normalized[:, token_dims]),
                        "raw_effective_weight_rms": rms(raw_effective[:, token_dims]),
                        "raw_effective_fro": float(np.linalg.norm(raw_effective[:, token_dims])),
                    }
                )

        column_norms = np.linalg.norm(raw_effective, axis=0)
        order = np.argsort(-column_norms)[: max(1, args.top_k)]
        top_features[alias] = [
            {
                "rank": rank + 1,
                "token": int(dim // TOKEN_DIM),
                "local_dim": int(dim % TOKEN_DIM),
                "feature": LOCAL_NAMES[int(dim % TOKEN_DIM)],
                "raw_effective_column_norm": float(column_norms[dim]),
            }
            for rank, dim in enumerate(order)
        ]

    with open(args.out_dir / "feature_weight_groups.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=json_default)
    with open(args.out_dir / "feature_weight_groups.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out_dir / "top_effective_g_features.json", "w", encoding="utf-8") as f:
        json.dump(top_features, f, indent=2, default=json_default)
    print(json.dumps({"rows": len(rows), "top_features": top_features}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
