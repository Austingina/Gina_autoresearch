#!/usr/bin/env python3
"""Validate an Interaction Geometry HDF5 demo file against the shared schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goals.demo_schema import required_demo_keys, task_schema  # noqa: E402


def _has_key(group: h5py.Group, key: str) -> bool:
    current: h5py.Group | h5py.Dataset = group
    for part in key.split("/"):
        if not isinstance(current, h5py.Group) or part not in current:
            return False
        current = current[part]
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate shared demo schema for an HDF5 dataset.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--task", required=True, choices=["lift_cube", "open_drawer", "open_side_door"])
    parser.add_argument("--include-rgb", action="store_true")
    parser.add_argument("--max-demos", type=int, default=10)
    args = parser.parse_args()

    schema = task_schema(args.task)
    keys = required_demo_keys(include_rgb=args.include_rgb)
    missing: list[str] = []
    checked = 0

    with h5py.File(args.dataset, "r") as f:
        if "data" not in f:
            print("[ERROR] missing top-level group: data", file=sys.stderr)
            return 1
        for demo_name in sorted(f["data"].keys())[: args.max_demos]:
            demo = f["data"][demo_name]
            checked += 1
            for key in keys:
                if not _has_key(demo, key):
                    missing.append(f"{demo_name}:{key}")

    if missing:
        print(f"[ERROR] {args.dataset} failed schema for task={schema.task.value}", file=sys.stderr)
        for item in missing[:50]:
            print(f"  missing {item}", file=sys.stderr)
        if len(missing) > 50:
            print(f"  ... {len(missing) - 50} more", file=sys.stderr)
        return 1

    print(
        f"[OK] {args.dataset} matches schema for task={schema.task.value} "
        f"kappa={schema.kappa.value} checked_demos={checked}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
