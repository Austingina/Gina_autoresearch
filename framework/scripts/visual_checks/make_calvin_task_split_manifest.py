#!/usr/bin/env python3
"""Build reproducible CALVIN source/target task split manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from calvin_vis_utils import (
    annotation_to_dict,
    default_dataset_root,
    ensure_dir,
    load_annotations,
    task_family,
    task_object_color,
    timestamp,
    write_json,
)


def select_recipe(entries, recipe: str):
    if recipe == "object_transfer_red_blue_to_pink":
        source = [e for e in entries if task_object_color(e.task) in {"red", "blue"}]
        target = [e for e in entries if task_object_color(e.task) == "pink"]
        holdout = [e for e in entries if task_object_color(e.task) == "none"]
    elif recipe == "primitive_transfer_push_lift_rotate_to_place_stack":
        source_families = {"push", "lift", "rotate"}
        target_families = {"place", "stack", "push_into"}
        source = [e for e in entries if task_family(e.task) in source_families]
        target = [e for e in entries if task_family(e.task) in target_families]
        holdout = [e for e in entries if e not in source and e not in target]
    elif recipe == "articulation_transfer_slider_to_drawer":
        source = [e for e in entries if task_family(e.task) == "slider"]
        target = [e for e in entries if task_family(e.task) == "drawer"]
        holdout = [e for e in entries if e not in source and e not in target]
    elif recipe == "articulation_transfer_drawer_to_slider":
        source = [e for e in entries if task_family(e.task) == "drawer"]
        target = [e for e in entries if task_family(e.task) == "slider"]
        holdout = [e for e in entries if e not in source and e not in target]
    elif recipe == "low_data_all_tasks":
        source = entries
        target = entries
        holdout = []
    else:
        raise ValueError(f"Unknown recipe: {recipe}")
    return source, target, holdout


def low_data_subsets(target, fractions=(0.01, 0.05, 0.10, 1.0)):
    by_task = {}
    for e in target:
        by_task.setdefault(e.task, []).append(e)
    result = {}
    for frac in fractions:
        key = f"{int(frac * 100):03d}pct"
        subset = []
        for task, rows in sorted(by_task.items()):
            n = max(1, int(round(len(rows) * frac)))
            subset.extend(rows[:n])
        result[key] = [annotation_to_dict(e) for e in subset]
    return result


def counts(entries):
    return {
        "num_sequences": len(entries),
        "num_frames": int(sum(e.length for e in entries)),
        "by_task": dict(Counter(e.task for e in entries)),
        "by_family": dict(Counter(task_family(e.task) for e in entries)),
        "by_object_color": dict(Counter(task_object_color(e.task) for e in entries)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    ap.add_argument("--split", default="training")
    ap.add_argument(
        "--recipe",
        default="object_transfer_red_blue_to_pink",
        choices=[
            "object_transfer_red_blue_to_pink",
            "primitive_transfer_push_lift_rotate_to_place_stack",
            "articulation_transfer_slider_to_drawer",
            "articulation_transfer_drawer_to_slider",
            "low_data_all_tasks",
        ],
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    entries = load_annotations(args.dataset_root, args.split)
    source, target, holdout = select_recipe(entries, args.recipe)
    out = args.out or (
        Path("/mnt/nvme33n1/wuk_code/AutoResearch")
        / "artifacts"
        / "task_splits"
        / f"{args.recipe}.json"
    )
    ensure_dir(out.parent)
    manifest = {
        "created_at": timestamp(),
        "script": "make_calvin_task_split_manifest.py",
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "recipe": args.recipe,
        "counts": {
            "all": counts(entries),
            "source": counts(source),
            "target": counts(target),
            "holdout": counts(holdout),
        },
        "source": [annotation_to_dict(e) for e in source],
        "target": [annotation_to_dict(e) for e in target],
        "holdout": [annotation_to_dict(e) for e in holdout],
        "low_data_target_subsets": low_data_subsets(target),
    }
    write_json(out, manifest)
    print(out)
    print("source", manifest["counts"]["source"]["num_sequences"], "target", manifest["counts"]["target"]["num_sequences"])


if __name__ == "__main__":
    main()
