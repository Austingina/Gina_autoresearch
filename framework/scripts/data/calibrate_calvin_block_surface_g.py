#!/usr/bin/env python3
"""Calibrate one embodiment clearance for asset-aware CALVIN block surface G."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml


POLICY_DIR = Path(__file__).resolve().parents[1] / "policy_state_bc"
sys.path.insert(0, POLICY_DIR.as_posix())

from behavior_surface_geometry import (  # noqa: E402
    oriented_box_ray_distance,
    oriented_box_required_clearance,
    oriented_box_volume_delta,
    safe_unit,
)


TASKS = {
    f"push_{color}_block_{side}": (color, np.asarray([sign, 0.0, 0.0], dtype=np.float32))
    for color in ("red", "blue", "pink")
    for side, sign in (("left", -1.0), ("right", 1.0))
}

DEFAULT_SOURCE_TASKS = (
    "push_red_block_left",
    "push_blue_block_right",
    "push_pink_block_left",
    "push_pink_block_right",
)
DEFAULT_HELDOUT_TASKS = (
    "push_blue_block_left",
    "push_red_block_right",
)


def scalar_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown block task(s): {sorted(unknown)}")
    if not tasks:
        raise argparse.ArgumentTypeError("At least one block task is required")
    return tasks


def grouped_summary(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, object]]:
    return {
        task: scalar_summary(
            np.asarray([float(row[key]) for row in rows if row["task"] == task])
        )
        for task in TASKS
    }


def grouped_scene_task_summary(
    rows: list[dict[str, object]], key: str
) -> dict[str, dict[str, dict[str, object]]]:
    return {
        scene: {
            task: scalar_summary(
                np.asarray(
                    [
                        float(row[key])
                        for row in rows
                        if row["scene"] == scene and row["task"] == task
                    ]
                )
            )
            for task in TASKS
        }
        for scene in sorted({str(row["scene"]) for row in rows})
    }


def parse_scene_geometry(
    scene: dict[str, object], calvin_env_root: Path
) -> tuple[dict[str, slice], dict[str, np.ndarray], float]:
    objects = scene["objects"]
    scalar_dim = sum(
        len(fixed_object.get(field, {}) or {})
        for fixed_object in objects.get("fixed_objects", {}).values()
        for field in ("joints", "buttons", "switches", "lights")
    )
    scaling = float(scene.get("global_scaling", 1.0))
    slices: dict[str, slice] = {}
    half_extents: dict[str, np.ndarray] = {}
    for object_index, (name, item) in enumerate(objects["movable_objects"].items()):
        if not name.startswith("block_"):
            continue
        color = name.split("_", 1)[1]
        start = scalar_dim + 6 * object_index
        slices[color] = slice(start, start + 3)
        urdf_path = calvin_env_root / "data" / str(item["file"])
        box = ET.parse(urdf_path).getroot().find(".//collision/geometry/box")
        if box is None or not box.attrib.get("size"):
            raise RuntimeError(f"Missing collision box in {urdf_path}")
        size = np.asarray([float(value) for value in box.attrib["size"].split()], dtype=np.float32)
        half_extents[color] = size * np.float32(0.5 * scaling)
    missing = {"red", "blue", "pink"} - set(slices)
    if missing:
        raise RuntimeError(f"Scene config is missing block colors: {sorted(missing)}")
    return slices, half_extents, scaling


def load_scene_geometries(
    split_dir: Path, calvin_env_root: Path
) -> list[dict[str, object]]:
    scene_info_path = split_dir / "scene_info.npy"
    if scene_info_path.exists():
        scene_info = np.load(scene_info_path, allow_pickle=True).item()
        geometries = []
        for scene_name, bounds in scene_info.items():
            config_path = calvin_env_root / "conf" / "scene" / f"{scene_name}.yaml"
            scene = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            slices, half_extents, scaling = parse_scene_geometry(scene, calvin_env_root)
            geometries.append(
                {
                    "name": str(scene_name),
                    "start": int(bounds[0]),
                    "end": int(bounds[1]),
                    "slices": slices,
                    "half_extents": half_extents,
                    "scaling": scaling,
                }
            )
        return sorted(geometries, key=lambda item: int(item["start"]))

    config_path = split_dir / ".hydra" / "merged_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scene = config["scene"]
    slices, half_extents, scaling = parse_scene_geometry(scene, calvin_env_root)
    return [
        {
            "name": str(scene.get("name", "merged_scene")),
            "start": -1,
            "end": np.iinfo(np.int64).max,
            "slices": slices,
            "half_extents": half_extents,
            "scaling": scaling,
        }
    ]


def geometry_for_segment(
    geometries: list[dict[str, object]], start: int, end: int
) -> dict[str, object]:
    matches = [
        item
        for item in geometries
        if int(item["start"]) <= start <= end <= int(item["end"])
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Segment [{start}, {end}] is not contained in exactly one scene range"
        )
    return matches[0]


def load_segments(split_dir: Path) -> list[tuple[str, int, int]]:
    annotation = np.load(
        split_dir / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True
    ).item()
    return [
        (str(task), int(interval[0]), int(interval[1]))
        for task, interval in zip(
            annotation["language"]["task"], annotation["info"]["indx"]
        )
        if str(task) in TASKS
    ]


def load_frame(split_dir: Path, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    path = split_dir / f"episode_{frame_index:07d}.npz"
    with np.load(path, allow_pickle=False) as frame:
        return (
            np.asarray(frame["robot_obs"], dtype=np.float32),
            np.asarray(frame["scene_obs"], dtype=np.float32),
        )


def segment_contact_row(
    split_dir: Path,
    task: str,
    start: int,
    end: int,
    scene_geometry: dict[str, object],
    motion_threshold: float,
    contact_window: int,
) -> dict[str, object] | None:
    color, flow = TASKS[task]
    scene_slices = scene_geometry["slices"]
    half_extents = scene_geometry["half_extents"]
    position_slice = scene_slices[color]
    orientation_slice = slice(position_slice.start + 3, position_slice.start + 6)
    robots: list[np.ndarray] = []
    scenes: list[np.ndarray] = []
    for frame_index in range(start, end + 1):
        robot, scene = load_frame(split_dir, frame_index)
        robots.append(robot)
        scenes.append(scene)
    robot_array = np.stack(robots)
    scene_array = np.stack(scenes)
    centers = scene_array[:, position_slice]
    progress = (centers - centers[0]) @ flow
    candidates = np.flatnonzero(progress >= motion_threshold)
    if not len(candidates):
        return None
    onset = int(candidates[0])
    stop = min(len(robot_array), onset + max(1, contact_window))
    eef = np.median(robot_array[onset:stop, :3], axis=0).astype(np.float32)
    center = np.median(centers[onset:stop], axis=0).astype(np.float32)
    euler = np.asarray(scene_array[onset, orientation_slice], dtype=np.float32)
    upstream = -safe_unit(flow)
    support = oriented_box_ray_distance(upstream, euler, half_extents[color])
    surface = center + upstream * np.float32(support)
    offset = eef - surface
    clearance = float(np.dot(offset, upstream))
    orthogonal = offset - upstream * np.float32(clearance)
    volume_clearance = oriented_box_required_clearance(
        eef, center, euler, half_extents[color]
    )
    volume_clearance_trace = [
        oriented_box_required_clearance(
            robot_obs[:3],
            scene_obs[position_slice],
            scene_obs[orientation_slice],
            half_extents[color],
        )
        for robot_obs, scene_obs in zip(robot_array, scene_array)
    ]
    return {
        "task": task,
        "color": color,
        "scene": str(scene_geometry["name"]),
        "start": start,
        "end": end,
        "onset_local_index": onset,
        "onset_frame": start + onset,
        "eef": eef,
        "center": center,
        "euler": euler,
        "support_distance": support,
        "clearance": clearance,
        "volume_clearance": volume_clearance,
        "volume_clearance_trace": volume_clearance_trace,
        "orthogonal_error": float(np.linalg.norm(orthogonal)),
        "progress_at_onset": float(progress[onset]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-threshold", type=float, default=0.002)
    parser.add_argument("--contact-window", type=int, default=3)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument(
        "--source-tasks",
        type=parse_tasks,
        default=DEFAULT_SOURCE_TASKS,
        help="Comma-separated tasks allowed to fit calibration parameters.",
    )
    parser.add_argument(
        "--heldout-tasks",
        type=parse_tasks,
        default=DEFAULT_HELDOUT_TASKS,
        help="Comma-separated tasks used only for frozen-parameter auditing.",
    )
    parser.add_argument("--min-source-segments-per-task", type=int, default=20)
    parser.add_argument("--min-source-validation-per-task", type=int, default=3)
    args = parser.parse_args()

    overlap = set(args.source_tasks) & set(args.heldout_tasks)
    if overlap:
        raise RuntimeError(f"Source and held-out tasks overlap: {sorted(overlap)}")

    split_dir = args.data_root.resolve() / "training"
    calvin_env_root = args.calvin_code.resolve() / "calvin_env"
    scene_geometries = load_scene_geometries(split_dir, calvin_env_root)
    rows = [
        row
        for task, start, end in load_segments(split_dir)
        if (
            row := segment_contact_row(
                split_dir,
                task,
                start,
                end,
                geometry_for_segment(scene_geometries, start, end),
                args.motion_threshold,
                args.contact_window,
            )
        )
        is not None
    ]
    source_rows = [row for row in rows if row["task"] in args.source_tasks]
    heldout_rows = [row for row in rows if row["task"] in args.heldout_tasks]
    source_counts = {
        task: sum(row["task"] == task for row in source_rows) for task in args.source_tasks
    }
    insufficient_source = {
        task: count
        for task, count in source_counts.items()
        if count < args.min_source_segments_per_task
    }
    if insufficient_source:
        raise RuntimeError(
            "Insufficient source-task contact coverage; refusing calibration: "
            f"{insufficient_source}"
        )

    train_rows = [
        row
        for row in source_rows
        if int(row["start"]) % args.validation_modulus != 0
    ]
    val_rows = [
        row
        for row in source_rows
        if int(row["start"]) % args.validation_modulus == 0
    ]
    if not train_rows or not val_rows:
        raise RuntimeError("Calibration split produced an empty train or validation set")
    val_counts = {
        task: sum(row["task"] == task for row in val_rows) for task in args.source_tasks
    }
    insufficient_val = {
        task: count
        for task, count in val_counts.items()
        if count < args.min_source_validation_per_task
    }
    if insufficient_val:
        raise RuntimeError(
            "Insufficient source-task validation coverage; refusing calibration: "
            f"{insufficient_val}"
        )
    train_clearance = np.asarray([row["clearance"] for row in train_rows], dtype=np.float32)
    recommended_clearance = float(np.median(train_clearance))
    train_volume_clearance = np.asarray(
        [row["volume_clearance"] for row in train_rows], dtype=np.float32
    )
    recommended_volume_clearance = float(np.quantile(train_volume_clearance, 0.90))

    for row in rows:
        upstream = -safe_unit(TASKS[str(row["task"])][1])
        prediction = (
            np.asarray(row["center"], dtype=np.float32)
            + upstream * np.float32(float(row["support_distance"]) + recommended_clearance)
        )
        row["calibrated_anchor_error"] = float(
            np.linalg.norm(prediction - np.asarray(row["eef"], dtype=np.float32))
        )
        row["volume_anchor_error"] = float(
            np.linalg.norm(
                oriented_box_volume_delta(
                    np.asarray(row["eef"], dtype=np.float32),
                    np.asarray(row["center"], dtype=np.float32),
                    np.asarray(row["euler"], dtype=np.float32),
                    half_extents_local=next(
                        item["half_extents"][str(row["color"])]
                        for item in scene_geometries
                        if item["name"] == row["scene"]
                    ),
                    clearance=recommended_volume_clearance,
                )
            )
        )
        entry_candidates = np.flatnonzero(
            np.asarray(row["volume_clearance_trace"], dtype=np.float32)
            <= recommended_volume_clearance
        )
        row["volume_entry_local_index"] = (
            int(entry_candidates[0]) if len(entry_candidates) else None
        )
        row["volume_entry_lead_frames"] = (
            int(row["onset_local_index"]) - int(entry_candidates[0])
            if len(entry_candidates)
            else None
        )
    val_errors = np.asarray(
        [row["calibrated_anchor_error"] for row in val_rows], dtype=np.float32
    )
    heldout_errors = np.asarray(
        [row["calibrated_anchor_error"] for row in heldout_rows], dtype=np.float32
    )
    val_volume_errors = np.asarray(
        [row["volume_anchor_error"] for row in val_rows], dtype=np.float32
    )
    heldout_volume_errors = np.asarray(
        [row["volume_anchor_error"] for row in heldout_rows], dtype=np.float32
    )
    val_volume_leads = np.asarray(
        [
            row["volume_entry_lead_frames"]
            for row in val_rows
            if row["volume_entry_lead_frames"] is not None
        ],
        dtype=np.float32,
    )
    heldout_volume_leads = np.asarray(
        [
            row["volume_entry_lead_frames"]
            for row in heldout_rows
            if row["volume_entry_lead_frames"] is not None
        ],
        dtype=np.float32,
    )
    val_volume_coverage = float(np.mean(val_volume_errors <= 1e-6))
    heldout_volume_coverage = float(np.mean(heldout_volume_errors <= 1e-6))
    source_task_clearance_medians = {
        task: float(
            np.median([row["clearance"] for row in source_rows if row["task"] == task])
        )
        for task in args.source_tasks
    }
    heldout_task_clearance_medians = {
        task: float(
            np.median([row["clearance"] for row in heldout_rows if row["task"] == task])
        )
        for task in args.heldout_tasks
        if any(row["task"] == task for row in heldout_rows)
    }
    source_median_spread = max(source_task_clearance_medians.values()) - min(
        source_task_clearance_medians.values()
    )
    heldout_median_gap = max(
        (
            abs(value - recommended_clearance)
            for value in heldout_task_clearance_medians.values()
        ),
        default=float("nan"),
    )
    source_scene_task_medians = {
        f"{scene}/{task}": float(np.median(values))
        for scene in sorted({str(row["scene"]) for row in source_rows})
        for task in args.source_tasks
        if (
            values := [
                row["clearance"]
                for row in source_rows
                if row["scene"] == scene and row["task"] == task
            ]
        )
    }
    source_scene_task_max_gap = max(
        abs(value - recommended_clearance)
        for value in source_scene_task_medians.values()
    )
    report = {
        "data_root": split_dir.parent.as_posix(),
        "calvin_code": args.calvin_code.resolve().as_posix(),
        "motion_threshold": args.motion_threshold,
        "contact_window": args.contact_window,
        "scene_geometries": [
            {
                "name": item["name"],
                "start": item["start"],
                "end": item["end"],
                "global_scaling": item["scaling"],
                "scene_slices": {
                    color: [value.start, value.stop]
                    for color, value in item["slices"].items()
                },
                "block_half_extents": {
                    color: value.tolist()
                    for color, value in item["half_extents"].items()
                },
            }
            for item in scene_geometries
        ],
        "protocol": {
            "source_tasks": list(args.source_tasks),
            "heldout_tasks": list(args.heldout_tasks),
            "heldout_used_for_fitting": False,
            "min_source_segments_per_task": args.min_source_segments_per_task,
            "min_source_validation_per_task": args.min_source_validation_per_task,
        },
        "segments": len(rows),
        "source_segments": len(source_rows),
        "heldout_audit_segments": len(heldout_rows),
        "source_train_segments": len(train_rows),
        "source_validation_segments": len(val_rows),
        "source_counts": source_counts,
        "source_validation_counts": val_counts,
        "clearance_all": scalar_summary(
            np.asarray([row["clearance"] for row in rows], dtype=np.float32)
        ),
        "clearance_by_task": grouped_summary(rows, "clearance"),
        "clearance_by_scene_task": grouped_scene_task_summary(rows, "clearance"),
        "volume_clearance_all": scalar_summary(
            np.asarray([row["volume_clearance"] for row in rows], dtype=np.float32)
        ),
        "volume_clearance_by_task": grouped_summary(rows, "volume_clearance"),
        "volume_clearance_by_scene_task": grouped_scene_task_summary(
            rows, "volume_clearance"
        ),
        "orthogonal_error_all": scalar_summary(
            np.asarray([row["orthogonal_error"] for row in rows], dtype=np.float32)
        ),
        "orthogonal_error_by_task": grouped_summary(rows, "orthogonal_error"),
        "validation_anchor_error": scalar_summary(val_errors),
        "source_validation_anchor_error_by_task": grouped_summary(
            val_rows, "calibrated_anchor_error"
        ),
        "heldout_anchor_error": scalar_summary(heldout_errors),
        "heldout_anchor_error_by_task": grouped_summary(
            heldout_rows, "calibrated_anchor_error"
        ),
        "interface_audit": {
            "source_task_clearance_medians": source_task_clearance_medians,
            "heldout_task_clearance_medians": heldout_task_clearance_medians,
            "source_scene_task_clearance_medians": source_scene_task_medians,
            "source_task_median_spread": source_median_spread,
            "source_scene_task_median_max_gap_from_source_fit": (
                source_scene_task_max_gap
            ),
            "heldout_median_max_gap_from_source_fit": heldout_median_gap,
            "clearance_reuse_gate_max_spread": 0.005,
            "clearance_reuse_gate_pass": bool(
                source_median_spread <= 0.005
                and source_scene_task_max_gap <= 0.005
                and np.isfinite(heldout_median_gap)
                and heldout_median_gap <= 0.005
            ),
            "canonical_anchor_validation_error_p90": float(np.quantile(val_errors, 0.90)),
            "canonical_anchor_heldout_error_p90": (
                float(np.quantile(heldout_errors, 0.90)) if len(heldout_errors) else None
            ),
            "interaction_volume_validation_error": scalar_summary(val_volume_errors),
            "interaction_volume_heldout_error": scalar_summary(heldout_volume_errors),
            "interaction_volume_validation_entry_lead_frames": scalar_summary(
                val_volume_leads
            ),
            "interaction_volume_heldout_entry_lead_frames": scalar_summary(
                heldout_volume_leads
            ),
            "interaction_volume_validation_early_gt5_fraction": float(
                np.mean(val_volume_leads > 5)
            ),
            "interaction_volume_heldout_early_gt5_fraction": float(
                np.mean(heldout_volume_leads > 5)
            ),
            "interaction_volume_contact_coverage_gate_min": 0.90,
            "interaction_volume_contact_coverage_gate_pass": bool(
                val_volume_coverage >= 0.90 and heldout_volume_coverage >= 0.90
            ),
            "interaction_volume_matched_to_existing_contact_anchor_policy": False,
            "interaction_volume_requires_relabel_and_retrain": True,
        },
        "recommended_tool_clearance": recommended_clearance,
        "recommended_interaction_volume_clearance": recommended_volume_clearance,
        "recommended_motion_threshold": args.motion_threshold,
        "recommended_geometric_contact_fallback": 0.0,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=lambda value: value.tolist()) + "\n")
    tmp.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
