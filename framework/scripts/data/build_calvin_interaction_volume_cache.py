#!/usr/bin/env python3
"""Build a compact CALVIN state cache with matched interaction-volume G labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
POLICY_DIR = SCRIPT_DIR.parent / "policy_state_bc"
sys.path.insert(0, SCRIPT_DIR.as_posix())
sys.path.insert(0, POLICY_DIR.as_posix())

from behavior_surface_geometry import oriented_box_volume_delta  # noqa: E402
from calibrate_calvin_block_surface_g import (  # noqa: E402
    DEFAULT_SOURCE_TASKS,
    TASKS,
    geometry_for_segment,
    load_scene_geometries,
    load_segments,
)


PACKED_FIELDS = ("robot_obs", "scene_obs", "rel_actions", "minimal_g")


def parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown block task(s): {sorted(unknown)}")
    if not tasks:
        raise argparse.ArgumentTypeError("At least one task is required")
    return tasks


def load_clearance(report_path: Path) -> float:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    value = float(report["recommended_interaction_volume_clearance"])
    if value < 0.0:
        raise RuntimeError(f"Invalid interaction-volume clearance in {report_path}: {value}")
    protocol = report.get("protocol", {})
    if protocol.get("heldout_used_for_fitting") is not False:
        raise RuntimeError(
            "Calibration report does not prove that held-out tasks were excluded from fitting"
        )
    return value


def relabel_segment(
    split_dir: Path,
    task: str,
    start: int,
    end: int,
    scene_geometry: dict[str, object],
    tool_clearance: float,
    motion_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    color, desired_flow = TASKS[task]
    position_slice = scene_geometry["slices"][color]
    orientation_slice = slice(position_slice.start + 3, position_slice.start + 6)
    frame_indices = np.arange(start, end + 1, dtype=np.int64)
    values: dict[str, list[np.ndarray]] = {
        "robot_obs": [],
        "scene_obs": [],
        "rel_actions": [],
    }
    for frame_index in frame_indices:
        frame_path = split_dir / f"episode_{int(frame_index):07d}.npz"
        with np.load(frame_path, allow_pickle=False) as frame:
            for field in values:
                values[field].append(np.asarray(frame[field], dtype=np.float32))

    arrays = {field: np.stack(rows).astype(np.float32) for field, rows in values.items()}
    centers = arrays["scene_obs"][:, position_slice]
    progress = (centers - centers[0]) @ desired_flow
    onset_candidates = np.flatnonzero(progress >= motion_threshold)
    if not len(onset_candidates):
        raise RuntimeError(
            f"{task} [{start}, {end}] never reaches motion threshold {motion_threshold}"
        )
    onset = int(onset_candidates[0])

    half_extents = scene_geometry["half_extents"][color]
    minimal_g = np.zeros((len(frame_indices), 6), dtype=np.float32)
    minimal_g[:, 3:6] = desired_flow
    for local_index in range(onset):
        minimal_g[local_index, :3] = oriented_box_volume_delta(
            arrays["robot_obs"][local_index, :3],
            centers[local_index],
            arrays["scene_obs"][local_index, orientation_slice],
            half_extents,
            clearance=tool_clearance,
        )
    arrays["minimal_g"] = minimal_g
    arrays["frame_index"] = frame_indices
    metadata = {
        "task": task,
        "scene": str(scene_geometry["name"]),
        "start": start,
        "end": end,
        "frames": len(frame_indices),
        "motion_onset_local_index": onset,
        "motion_onset_frame": start + onset,
        "pre_motion_frames": onset,
        "post_motion_frames": len(frame_indices) - onset,
        "max_attraction_norm": float(np.linalg.norm(minimal_g[:, :3], axis=1).max()),
    }
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", type=parse_tasks, default=DEFAULT_SOURCE_TASKS)
    parser.add_argument("--motion-threshold", type=float, default=0.002)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    split_dir = data_root / "training"
    output = (
        args.output.resolve()
        if args.output is not None
        else split_dir / "state_only_arrays.npz"
    )
    if output.exists() and not args.force:
        raise RuntimeError(f"Output already exists (pass --force to replace): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    tool_clearance = load_clearance(args.calibration_report.resolve())
    scene_geometries = load_scene_geometries(
        split_dir, args.calvin_code.resolve() / "calvin_env"
    )
    selected = [
        (task, start, end)
        for task, start, end in load_segments(split_dir)
        if task in args.tasks
    ]
    if not selected:
        raise RuntimeError(f"No selected task segments under {split_dir}")

    packed_rows: dict[str, list[np.ndarray]] = {
        "frame_index": [],
        **{field: [] for field in PACKED_FIELDS},
    }
    segment_rows: list[dict[str, object]] = []
    for index, (task, start, end) in enumerate(selected, start=1):
        arrays, metadata = relabel_segment(
            split_dir,
            task,
            start,
            end,
            geometry_for_segment(scene_geometries, start, end),
            tool_clearance,
            args.motion_threshold,
        )
        for field in packed_rows:
            packed_rows[field].append(arrays[field])
        segment_rows.append(metadata)
        print(
            f"[volume-cache] {index}/{len(selected)} {metadata['scene']} {task} "
            f"frames={metadata['frames']} onset={metadata['motion_onset_local_index']}",
            flush=True,
        )

    packed = {
        field: np.concatenate(rows, axis=0)
        for field, rows in packed_rows.items()
    }
    frame_indices = np.asarray(packed["frame_index"], dtype=np.int64)
    if len(np.unique(frame_indices)) != len(frame_indices):
        raise RuntimeError("Selected source segments contain overlapping frame indices")
    order = np.argsort(frame_indices)
    packed = {field: np.asarray(value[order]) for field, value in packed.items()}
    packed["frame_index"] = packed["frame_index"].astype(np.int64)
    for field in PACKED_FIELDS:
        packed[field] = packed[field].astype(np.float32)

    tmp = output.with_suffix(output.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **packed)
    os.replace(tmp, output)
    with np.load(output, allow_pickle=False) as check:
        for field, expected in packed.items():
            if check[field].shape != expected.shape or check[field].dtype != expected.dtype:
                raise RuntimeError(f"Packed cache verification failed for {field}")
            if not np.array_equal(check[field], expected):
                raise RuntimeError(f"Packed cache value verification failed for {field}")

    manifest = {
        "data_root": data_root.as_posix(),
        "output": output.as_posix(),
        "calibration_report": args.calibration_report.resolve().as_posix(),
        "label_type": "interaction_volume_minimal_g_v1",
        "minimal_g_definition": [
            "delta_to_expanded_oriented_object_box_xyz",
            "desired_object_flow_xyz",
        ],
        "tool_clearance": tool_clearance,
        "motion_threshold": args.motion_threshold,
        "post_motion_rule": "sticky_zero_attraction_after_first_directed_2mm_object_motion",
        "tasks": list(args.tasks),
        "heldout_tasks_written": False,
        "segments": len(segment_rows),
        "frames": len(frame_indices),
        "task_counts": dict(sorted(Counter(row["task"] for row in segment_rows).items())),
        "scene_counts": dict(sorted(Counter(row["scene"] for row in segment_rows).items())),
        "field_shapes": {field: list(value.shape) for field, value in packed.items()},
        "bytes": output.stat().st_size,
        "segments_detail": segment_rows,
        "complete": True,
    }
    manifest_path = output.with_name("interaction_volume_cache_manifest.json")
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "segments_detail"},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
