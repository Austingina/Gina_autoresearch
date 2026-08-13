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

from behavior_surface_geometry import oriented_box_ray_distance, safe_unit  # noqa: E402


TASKS = {
    f"push_{color}_block_{side}": (color, np.asarray([sign, 0.0, 0.0], dtype=np.float32))
    for color in ("red", "blue", "pink")
    for side, sign in (("left", -1.0), ("right", 1.0))
}


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


def load_scene_geometry(
    split_dir: Path, calvin_env_root: Path
) -> tuple[dict[str, slice], dict[str, np.ndarray], float]:
    config_path = split_dir / ".hydra" / "merged_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scene = config["scene"]
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
    scene_slices: dict[str, slice],
    half_extents: dict[str, np.ndarray],
    motion_threshold: float,
    contact_window: int,
) -> dict[str, object] | None:
    color, flow = TASKS[task]
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
    return {
        "task": task,
        "color": color,
        "start": start,
        "end": end,
        "onset_local_index": onset,
        "onset_frame": start + onset,
        "eef": eef,
        "center": center,
        "euler": euler,
        "support_distance": support,
        "clearance": clearance,
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
    args = parser.parse_args()

    split_dir = args.data_root.resolve() / "training"
    calvin_env_root = args.calvin_code.resolve() / "calvin_env"
    scene_slices, half_extents, scaling = load_scene_geometry(split_dir, calvin_env_root)
    rows = [
        row
        for task, start, end in load_segments(split_dir)
        if (
            row := segment_contact_row(
                split_dir,
                task,
                start,
                end,
                scene_slices,
                half_extents,
                args.motion_threshold,
                args.contact_window,
            )
        )
        is not None
    ]
    if len(rows) < 10:
        raise RuntimeError(f"Only {len(rows)} block segments have a motion onset")

    train_rows = [row for row in rows if int(row["start"]) % args.validation_modulus != 0]
    val_rows = [row for row in rows if int(row["start"]) % args.validation_modulus == 0]
    if not train_rows or not val_rows:
        raise RuntimeError("Calibration split produced an empty train or validation set")
    train_clearance = np.asarray([row["clearance"] for row in train_rows], dtype=np.float32)
    recommended_clearance = float(np.median(train_clearance))

    for row in rows:
        upstream = -safe_unit(TASKS[str(row["task"])][1])
        prediction = (
            np.asarray(row["center"], dtype=np.float32)
            + upstream * np.float32(float(row["support_distance"]) + recommended_clearance)
        )
        row["calibrated_anchor_error"] = float(
            np.linalg.norm(prediction - np.asarray(row["eef"], dtype=np.float32))
        )
    val_errors = np.asarray(
        [row["calibrated_anchor_error"] for row in val_rows], dtype=np.float32
    )
    recommended_tolerance = float(np.quantile(val_errors, 0.90))
    report = {
        "data_root": split_dir.parent.as_posix(),
        "calvin_code": args.calvin_code.resolve().as_posix(),
        "motion_threshold": args.motion_threshold,
        "contact_window": args.contact_window,
        "scene_global_scaling": scaling,
        "scene_slices": {
            color: [value.start, value.stop] for color, value in scene_slices.items()
        },
        "block_half_extents": {
            color: value.tolist() for color, value in half_extents.items()
        },
        "segments": len(rows),
        "train_segments": len(train_rows),
        "validation_segments": len(val_rows),
        "clearance_all": scalar_summary(
            np.asarray([row["clearance"] for row in rows], dtype=np.float32)
        ),
        "orthogonal_error_all": scalar_summary(
            np.asarray([row["orthogonal_error"] for row in rows], dtype=np.float32)
        ),
        "validation_anchor_error": scalar_summary(val_errors),
        "recommended_tool_clearance": recommended_clearance,
        "recommended_contact_tolerance": recommended_tolerance,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, default=lambda value: value.tolist()) + "\n")
    tmp.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
