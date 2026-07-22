#!/usr/bin/env python3
"""Visualize candidate interaction-geometry labels against CALVIN frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calvin_vis_utils import (
    add_label,
    default_dataset_root,
    ensure_dir,
    hstack,
    load_frame,
    make_raw_frame_panel,
    text_panel,
    timestamp,
    vstack,
    write_json,
)


def read_records(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fmt_vec(x, ndigits: int = 3) -> str:
    arr = np.asarray(x).reshape(-1)
    return "[" + ", ".join(f"{float(v):.{ndigits}f}" for v in arr) + "]"


def label_object_names(labels) -> list[str]:
    if "object_names" in labels.files:
        return [str(x) for x in labels["object_names"].tolist()]
    return ["block_red", "block_blue", "block_pink"]


def geometry_lines(labels, i: int) -> list[str]:
    object_names = label_object_names(labels)
    dists = labels["dist_object_to_eef"][i]
    near = labels["near_contact"][i]
    rel = labels["rel_object_to_eef"][i]
    lines = [
        f"eef_pos: {fmt_vec(labels['eef_pos'][i])}",
        f"gripper: {float(labels['gripper'][i][0]):.3f}",
        f"scene0: {fmt_vec(labels['scene_scalars'][i])}",
    ]
    for j, name in enumerate(object_names):
        lines.append(
            f"{name}: d={float(dists[j]):.3f} near={int(near[j])} rel={fmt_vec(rel[j])}"
        )
    return lines


def make_distance_strip(labels, indices: list[int], width: int = 760, height: int = 180):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((8, 6), "object-to-EEF distance over selected frames", fill=(20, 20, 20), font=font)
    if len(indices) < 2:
        return img
    left, top, right, bottom = 45, 28, width - 15, height - 28
    draw.rectangle((left, top, right, bottom), outline=(170, 170, 170))
    d = labels["dist_object_to_eef"][indices]
    max_d = max(float(np.nanmax(d)), 1e-6)
    colors = [(220, 60, 60), (55, 110, 230), (225, 80, 190)]
    object_names = label_object_names(labels)
    for obj_i, color in enumerate(colors):
        pts = []
        for t, idx in enumerate(indices):
            x = left + (right - left) * t / max(len(indices) - 1, 1)
            y = bottom - (bottom - top) * float(labels["dist_object_to_eef"][idx, obj_i]) / max_d
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)
        name = object_names[obj_i] if obj_i < len(object_names) else f"object_{obj_i}"
        draw.text((left + obj_i * 130, height - 18), name, fill=color, font=font)
    draw.text((4, top), f"{max_d:.2f}", fill=(80, 80, 80), font=font)
    draw.text((14, bottom - 8), "0", fill=(80, 80, 80), font=font)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    ap.add_argument("--g-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--num-sequences", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=8)
    args = ap.parse_args()

    labels = np.load(args.g_dir / "candidate_g_labels.npz", allow_pickle=True)
    records = read_records(args.g_dir / "records.jsonl")
    out_dir = args.out_dir or (
        Path("/mnt/nvme33n1/wuk_code/AutoResearch")
        / "artifacts"
        / "visual_checks"
        / f"{timestamp()}_visualize_interaction_geometry"
    )
    ensure_dir(out_dir)

    by_seq: dict[int, list[int]] = {}
    for i, row in enumerate(records):
        by_seq.setdefault(int(row["sequence_index"]), []).append(i)
    selected_seq = list(by_seq.keys())[: args.num_sequences]
    write_json(
        out_dir / "manifest.json",
        {
            "script": "visualize_interaction_geometry.py",
            "dataset_root": str(args.dataset_root),
            "g_dir": str(args.g_dir),
            "selected_sequence_indices": selected_seq,
        },
    )

    summary = ["# Interaction Geometry Visual Check", ""]
    for seq in selected_seq:
        idxs = by_seq[seq]
        if len(idxs) > args.max_frames:
            keep = np.linspace(0, len(idxs) - 1, args.max_frames)
            idxs = [idxs[int(round(x))] for x in keep]
        panels = []
        first = records[idxs[0]]
        for i in idxs:
            row = records[i]
            frame = load_frame(args.dataset_root, row["split"], int(row["frame_id"]))
            raw = make_raw_frame_panel(
                frame,
                title=f"seq={seq} frame={row['frame_id']}",
                text_lines=[
                    f"task: {row['task']}",
                    f"ann: {row['ann']}",
                    f"nearest: {row['nearest_object_name']}",
                ],
            )
            geom = add_label(text_panel(geometry_lines(labels, i), (720, 200)), "candidate G")
            panels.append(hstack([raw, geom]))
        dist = make_distance_strip(labels, idxs)
        grid = vstack([dist] + panels)
        png = out_dir / f"seq_{seq:04d}_{first['task']}.png"
        grid.save(png)
        summary.extend(
            [
                f"## sequence {seq}: {first['task']}",
                "",
                f"- language: {first['ann']}",
                f"- png: `{png.name}`",
                "",
            ]
        )
    (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
