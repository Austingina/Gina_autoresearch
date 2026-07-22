#!/usr/bin/env python3
"""Extract first-pass interaction-geometry labels from CALVIN frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from calvin_vis_utils import (
    compute_g_features,
    default_dataset_root,
    ensure_dir,
    load_frame,
    read_json,
    sample_frame_ids,
    scene_schema_from_dataset,
    timestamp,
    write_json,
    write_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--partition", default="source", choices=["source", "target", "holdout"])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-sequences", type=int, default=0, help="0 means all")
    ap.add_argument("--max-frames-per-sequence", type=int, default=0, help="0 means all")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--contact-threshold", type=float, default=0.10)
    ap.add_argument("--future-horizons", default="1,4,8,16")
    args = ap.parse_args()

    manifest = read_json(args.split_manifest)
    rows = list(manifest[args.partition])
    if args.max_sequences:
        rows = rows[: args.max_sequences]
    horizons = [int(x) for x in args.future_horizons.split(",") if x.strip()]
    schema = scene_schema_from_dataset(args.dataset_root, rows[0]["split"] if rows else "training")
    object_names = list(schema["object_names"])
    scalar_names = list(schema["scalar_names"])
    out_dir = args.out_dir or (
        Path("/mnt/nvme33n1/wuk_code/AutoResearch")
        / "artifacts"
        / "g_labels"
        / f"{timestamp()}_{manifest['recipe']}_{args.partition}"
    )
    ensure_dir(out_dir)

    records = []
    arrays = {
        "frame_ids": [],
        "sequence_index": [],
        "eef_pos": [],
        "eef_orn": [],
        "gripper": [],
        "scene_scalars": [],
        "object_pos": [],
        "object_orn": [],
        "rel_object_to_eef": [],
        "dist_object_to_eef": [],
        "near_contact": [],
        "nearest_object_index": [],
    }
    future_delta_eef = {h: [] for h in horizons}
    future_delta_objects = {h: [] for h in horizons}
    future_delta_scene_scalars = {h: [] for h in horizons}

    for seq_i, row in enumerate(rows):
        frame_ids = sample_frame_ids(
            row["start"],
            row["end"],
            args.max_frames_per_sequence,
            args.stride,
        )
        features = {}
        for frame_id in frame_ids:
            frame = load_frame(args.dataset_root, row["split"], frame_id)
            features[frame_id] = compute_g_features(
                frame,
                contact_threshold=args.contact_threshold,
                scalar_names=scalar_names,
                object_names=object_names,
            )
        for frame_id in frame_ids:
            g = features[frame_id]
            records.append(
                {
                    "record_index": len(records),
                    "split": row["split"],
                    "frame_id": frame_id,
                    "sequence_index": row["index"],
                    "sequence_start": row["start"],
                    "sequence_end": row["end"],
                    "task": row["task"],
                    "ann": row["ann"],
                    "nearest_object_name": g["nearest_object_name"],
                }
            )
            arrays["frame_ids"].append(frame_id)
            arrays["sequence_index"].append(row["index"])
            for key in (
                "eef_pos",
                "eef_orn",
                "gripper",
                "scene_scalars",
                "object_pos",
                "object_orn",
                "rel_object_to_eef",
                "dist_object_to_eef",
                "near_contact",
                "nearest_object_index",
            ):
                arrays[key].append(g[key])
            for h in horizons:
                future_id = min(frame_id + h, row["end"])
                if future_id not in features:
                    future_frame = load_frame(args.dataset_root, row["split"], future_id)
                    features[future_id] = compute_g_features(
                        future_frame,
                        contact_threshold=args.contact_threshold,
                        scalar_names=scalar_names,
                        object_names=object_names,
                    )
                gf = features[future_id]
                future_delta_eef[h].append(gf["eef_pos"] - g["eef_pos"])
                future_delta_objects[h].append(gf["object_pos"] - g["object_pos"])
                future_delta_scene_scalars[h].append(gf["scene_scalars"] - g["scene_scalars"])

    npz_data = {k: np.asarray(v) for k, v in arrays.items()}
    for h in horizons:
        npz_data[f"future_delta_eef_h{h}"] = np.asarray(future_delta_eef[h], dtype=np.float32)
        npz_data[f"future_delta_object_pos_h{h}"] = np.asarray(future_delta_objects[h], dtype=np.float32)
        npz_data[f"future_delta_scene_scalars_h{h}"] = np.asarray(
            future_delta_scene_scalars[h],
            dtype=np.float32,
        )
    npz_data["object_names"] = np.asarray(object_names)
    npz_data["scene_scalar_names"] = np.asarray(scalar_names)
    np.savez_compressed(out_dir / "candidate_g_labels.npz", **npz_data)
    write_jsonl(out_dir / "records.jsonl", records)
    write_json(
        out_dir / "g_schema.json",
        {
            "created_at": timestamp(),
            "script": "extract_candidate_g_labels.py",
            "dataset_root": str(args.dataset_root),
            "split_manifest": str(args.split_manifest),
            "partition": args.partition,
            "contact_threshold": args.contact_threshold,
            "future_horizons": horizons,
            "scene_name": schema["scene_name"],
            "schema_source": schema["source"],
            "scene_scalar_names": scalar_names,
            "object_names": object_names,
            "scene_obs_schema": {
                "scalars": f"scene_obs[0:{len(scalar_names)}] = {scalar_names}",
                "objects": [
                    {
                        "name": name,
                        "slice": [
                            len(scalar_names) + i * 6,
                            len(scalar_names) + (i + 1) * 6,
                        ],
                        "layout": "xyz + euler orientation",
                    }
                    for i, name in enumerate(object_names)
                ],
            },
            "arrays": {k: list(v.shape) for k, v in npz_data.items()},
        },
    )
    (out_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Candidate G Labels",
                "",
                f"- manifest: `{args.split_manifest}`",
                f"- partition: `{args.partition}`",
                f"- sequences: {len(rows)}",
                f"- records: {len(records)}",
                f"- labels: `candidate_g_labels.npz`",
                f"- records: `records.jsonl`",
                f"- schema: `g_schema.json`",
            ]
        ),
        encoding="utf-8",
    )
    print(out_dir)
    print("records", len(records))


if __name__ == "__main__":
    main()
