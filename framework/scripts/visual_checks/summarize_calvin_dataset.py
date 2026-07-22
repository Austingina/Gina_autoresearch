#!/usr/bin/env python3
"""Summarize CALVIN dataset schema, annotations, and state statistics."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from calvin_vis_utils import (
    default_dataset_root,
    ensure_dir,
    list_frame_ids,
    load_annotations,
    load_frame,
    scene_schema_from_dataset,
    task_family,
    task_object_color,
    timestamp,
    write_json,
)


def stats_array(values: np.ndarray) -> dict:
    return {
        "shape": list(values.shape),
        "mean": np.nanmean(values, axis=0).round(6).tolist(),
        "std": np.nanstd(values, axis=0).round(6).tolist(),
        "min": np.nanmin(values, axis=0).round(6).tolist(),
        "max": np.nanmax(values, axis=0).round(6).tolist(),
    }


def summarize_split(dataset_root: Path, split: str, max_frames: int) -> dict:
    frame_ids = list_frame_ids(dataset_root, split)
    sample_ids = frame_ids
    if max_frames and len(sample_ids) > max_frames:
        keep = np.linspace(0, len(sample_ids) - 1, max_frames)
        sample_ids = [sample_ids[int(round(x))] for x in keep]
    sample_frames = [load_frame(dataset_root, split, fid) for fid in sample_ids]
    first = sample_frames[0] if sample_frames else {}
    annotations = load_annotations(dataset_root, split)
    robot = np.asarray([f["robot_obs"] for f in sample_frames]) if sample_frames else np.empty((0,))
    scene = np.asarray([f["scene_obs"] for f in sample_frames]) if sample_frames else np.empty((0,))
    return {
        "num_episode_files": len(frame_ids),
        "first_frame_id": frame_ids[0] if frame_ids else None,
        "last_frame_id": frame_ids[-1] if frame_ids else None,
        "sampled_frames_for_stats": len(sample_ids),
        "keys": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in first.items()},
        "num_annotations": len(annotations),
        "annotation_frames": int(sum(e.length for e in annotations)),
        "tasks": dict(Counter(e.task for e in annotations)),
        "families": dict(Counter(task_family(e.task) for e in annotations)),
        "object_colors": dict(Counter(task_object_color(e.task) for e in annotations)),
        "robot_obs_stats": stats_array(robot) if robot.size else {},
        "scene_obs_stats": stats_array(scene) if scene.size else {},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    ap.add_argument("--splits", default="training,validation")
    ap.add_argument("--max-frames", type=int, default=1000)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (
        Path("/mnt/nvme33n1/wuk_code/AutoResearch")
        / "artifacts"
        / "dataset_summaries"
        / f"{timestamp()}_calvin_dataset_summary"
    )
    ensure_dir(out_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    result = {
        "created_at": timestamp(),
        "script": "summarize_calvin_dataset.py",
        "dataset_root": str(args.dataset_root),
        "splits": {},
        "scene_schemas": {},
    }
    for split in splits:
        result["scene_schemas"][split] = scene_schema_from_dataset(args.dataset_root, split)
        result["splits"][split] = summarize_split(args.dataset_root, split, args.max_frames)
    write_json(out_dir / "summary.json", result)

    lines = [
        "# CALVIN Dataset Summary",
        "",
        f"- dataset_root: `{args.dataset_root}`",
        f"- max_frames_for_stats: {args.max_frames}",
        "",
    ]
    for split in splits:
        schema = result["scene_schemas"][split]
        data = result["splits"][split]
        lines.extend(
            [
                f"## {split}",
                "",
                f"- scene: {schema['scene_name']} ({schema['source']})",
                f"- scene scalars: {schema['scalar_names']}",
                f"- object order: {schema['object_names']}",
                f"- episode files: {data['num_episode_files']}",
                f"- annotations: {data['num_annotations']}",
                f"- annotation frames: {data['annotation_frames']}",
                f"- families: {data['families']}",
                f"- object colors: {data['object_colors']}",
                "",
                "### Keys",
                "",
            ]
        )
        for key, meta in data["keys"].items():
            lines.append(f"- `{key}`: shape={meta['shape']} dtype={meta['dtype']}")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
