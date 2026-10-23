#!/usr/bin/env python3
"""Audit distribution and read-back integrity of a CALVIN push collection."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def load_collector(repo: Path):
    path = repo / "framework/scripts/data/collect_calvin_parametric_push_expert.py"
    spec = importlib.util.spec_from_file_location("collect_calvin_parametric_push_expert", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load collector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bounds(values: list[float]) -> list[float] | None:
    return [float(min(values)), float(max(values))] if values else None


def summarize(root: Path, collector: Any) -> dict[str, Any]:
    verification = collector.validate_collection(root)
    with open(root / "collection_manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    cells: dict[str, dict[str, Any]] = {}
    all_runtime_scenes: set[str] = set()
    all_source_kinds: set[str] = set()
    camera_shapes: dict[str, set[tuple[int, ...]]] = {"rgb_static": set(), "rgb_gripper": set()}
    collection_tasks = tuple(manifest.get("tasks", collector.PUSH_TASKS))
    for task in collection_tasks:
        rows = [row for row in manifest["results"] if row["task"] == task]
        accepted = [row for row in rows if bool(row["oracle_success"])]
        initial = [row["initial_object_pos"] for row in accepted]
        reset_ids = {
            str(row["reset"].get("reset_id", row["reset"].get("start", "unknown"))) for row in accepted
        }
        for row in accepted:
            path = root / row["path"]
            with np.load(path, allow_pickle=False) as payload:
                all_runtime_scenes.add(str(np.asarray(payload["runtime_scene"]).item()))
                all_source_kinds.add(str(np.asarray(payload["reset_source_kind"]).item()))
                camera_shapes["rgb_static"].add(tuple(payload["rgb_static"].shape[1:]))
                camera_shapes["rgb_gripper"].add(tuple(payload["rgb_gripper"].shape[1:]))
                if payload["g"].shape[-1] != 6 or payload["actions"].shape[-1] != 7:
                    raise ValueError(f"Invalid G/action dimensions: {path}")
        lengths = [int(row["executed_steps"]) for row in accepted]
        cells[task] = {
            "attempted": len(rows),
            "accepted": len(accepted),
            "failed": len(rows) - len(accepted),
            "success_rate": len(accepted) / len(rows) if rows else 0.0,
            "frames": sum(lengths),
            "trajectory_length": {
                "min": min(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
                "mean": float(np.mean(lengths)) if lengths else None,
            },
            "distinct_successful_reset_ids": len(reset_ids),
            "initial_x_range": bounds([float(pos[0]) for pos in initial]),
            "initial_y_range": bounds([float(pos[1]) for pos in initial]),
        }
    accepted_counts = [cell["accepted"] for cell in cells.values()]
    return {
        "format": "calvin_push_distribution_audit_v1",
        "scope": "data_expansion_only_no_evaluation",
        "collection": root.as_posix(),
        "verification": verification,
        "runtime_scenes": sorted(all_runtime_scenes),
        "reset_source_kinds": sorted(all_source_kinds),
        "evaluation_folds_present": bool(manifest.get("folds")),
        "balanced_accepted_cells": len(set(accepted_counts)) == 1,
        "accepted_total": sum(accepted_counts),
        "failed_total": sum(cell["failed"] for cell in cells.values()),
        "frames_total": sum(cell["frames"] for cell in cells.values()),
        "camera_frame_shapes": {name: [list(shape) for shape in sorted(shapes)] for name, shapes in camera_shapes.items()},
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    payload = summarize(args.collection.resolve(), load_collector(repo))
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.parent / f".{args.out.name}.tmp"
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temp, args.out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
