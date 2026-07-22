#!/usr/bin/env python3
"""Create manual visual checks for raw CALVIN episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio

from calvin_vis_utils import (
    annotation_to_dict,
    default_dataset_root,
    ensure_dir,
    load_annotations,
    load_frame,
    make_raw_frame_panel,
    sample_frame_ids,
    select_annotation_entries,
    timestamp,
    write_json,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    ap.add_argument("--split", default="training")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--task", default=None, help="Substring filter over CALVIN task name")
    ap.add_argument("--indices", default="", help="Comma-separated annotation indices")
    ap.add_argument("--num-sequences", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mp4", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or (
        Path("/mnt/nvme33n1/wuk_code/AutoResearch")
        / "artifacts"
        / "visual_checks"
        / f"{timestamp()}_visualize_calvin_episode"
    )
    ensure_dir(out_dir)

    indices = [int(x) for x in args.indices.split(",") if x.strip()] or None
    entries = load_annotations(args.dataset_root, args.split)
    selected = select_annotation_entries(
        entries,
        task=args.task,
        indices=indices,
        n=args.num_sequences,
        seed=args.seed,
    )
    manifest = {
        "script": "visualize_calvin_episode.py",
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "task_filter": args.task,
        "selected": [annotation_to_dict(e) for e in selected],
    }
    write_json(out_dir / "manifest.json", manifest)

    summary_lines = ["# CALVIN Raw Episode Visual Check", ""]
    for entry in selected:
        frame_ids = sample_frame_ids(entry.start, entry.end, args.max_frames, args.stride)
        panels = []
        video_frames = []
        for frame_id in frame_ids:
            frame = load_frame(args.dataset_root, args.split, frame_id)
            panel = make_raw_frame_panel(
                frame,
                title=f"ann_idx={entry.index} frame={frame_id}",
                text_lines=[
                    f"task: {entry.task}",
                    f"ann: {entry.ann}",
                    f"range: {entry.start}-{entry.end}",
                    f"rel_action: {frame['rel_actions'].round(3).tolist()}",
                    f"robot_xyz: {frame['robot_obs'][:3].round(3).tolist()}",
                    f"scene0: {frame['scene_obs'][:6].round(3).tolist()}",
                ],
            )
            png = out_dir / f"ann_{entry.index:04d}_frame_{frame_id:07d}.png"
            panel.save(png)
            panels.append(str(png.name))
            video_frames.append(panel)
        if args.mp4 and video_frames:
            mp4 = out_dir / f"ann_{entry.index:04d}_{entry.task}.mp4"
            imageio.mimsave(mp4, [p for p in video_frames], fps=4)
        summary_lines.extend(
            [
                f"## {entry.index}: {entry.task}",
                "",
                f"- range: `{entry.start}-{entry.end}`",
                f"- language: {entry.ann}",
                f"- frames: {', '.join(panels)}",
                "",
            ]
        )
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
