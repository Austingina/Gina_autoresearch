#!/usr/bin/env python3
"""Measure whether primitive minimal-G anchors match demonstrated contact transitions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402
import train_eval_state_chunk_flow_policy as chunk  # noqa: E402


DEFAULT_TASKS = sorted(chunk.BLOCK_PUSH_TASKS) + ["push_into_drawer"]


def stats(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def named_block_index(color: str) -> int:
    return list(chunk.BLOCK_SCENE_SLICES).index(color)


def load_frame(split_dir: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(base.episode_file(split_dir, index), allow_pickle=True) as data:
        return (
            np.asarray(data["robot_obs"], dtype=np.float32),
            np.asarray(data["scene_obs"], dtype=np.float32),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--motion-threshold", type=float, default=0.005)
    parser.add_argument("--anchor-offset", type=float, default=0.05)
    parser.add_argument("--readiness-distance", type=float, default=0.10)
    parser.add_argument("--readiness-temperature", type=float, default=0.025)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    split_dir = data_root / "training"
    primitive_geometry = chunk.fit_primitive_query_geometry(data_root)
    task_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for segment in base.load_lang_segments(split_dir):
        task = str(segment["task"])
        if task not in args.tasks:
            continue
        start, end = int(segment["start"]), int(segment["end"])
        try:
            start_robot, start_scene = load_frame(split_dir, start)
            _end_robot, end_scene = load_frame(split_dir, end)
        except Exception:
            continue
        start_blocks = chunk._scene_block_positions(start_scene)
        end_blocks = chunk._scene_block_positions(end_scene)

        if task in chunk.BLOCK_PUSH_TASKS:
            color, desired_flow = chunk.BLOCK_PUSH_TASKS[task]
            active_idx = named_block_index(color)
            flow_dir = base._safe_unit(np.asarray(desired_flow, dtype=np.float32))
        else:
            block_delta = end_blocks - start_blocks
            active_idx = int(np.argmax(np.linalg.norm(block_delta, axis=1)))
            flow_dir = base._safe_unit(
                np.asarray(primitive_geometry[task]["flow_dir"], dtype=np.float32)
            )

        active_displacement = float(np.linalg.norm(end_blocks[active_idx] - start_blocks[active_idx]))
        if active_displacement < max(0.03, args.motion_threshold):
            continue

        frame_rows: list[dict[str, Any]] = []
        first_motion_order: int | None = None
        for order, frame_index in enumerate(range(start, end + 1)):
            try:
                robot, scene = load_frame(split_dir, frame_index)
            except Exception:
                frame_rows = []
                break
            blocks = chunk._scene_block_positions(scene)
            center = blocks[active_idx]
            eef = robot[:3]
            object_motion = float(np.linalg.norm(center - start_blocks[active_idx]))
            fixed_anchor = center - flow_dir * np.float32(args.anchor_offset)
            anchor_distance = float(np.linalg.norm(eef - fixed_anchor))
            relative = eef - center
            along_flow = float(np.dot(relative, flow_dir))
            orthogonal = relative - along_flow * flow_dir
            readiness = float(
                1.0
                / (
                    1.0
                    + np.exp(
                        -(
                            args.readiness_distance - anchor_distance
                        )
                        / max(args.readiness_temperature, 1e-6)
                    )
                )
            )
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "object_motion": object_motion,
                    "anchor_distance": anchor_distance,
                    "eef_center_along_flow": along_flow,
                    "eef_center_orthogonal": float(np.linalg.norm(orthogonal)),
                    "eef_center_z": float(relative[2]),
                    "readiness": readiness,
                }
            )
            if first_motion_order is None and object_motion >= args.motion_threshold:
                first_motion_order = order

        if not frame_rows or first_motion_order is None:
            continue
        contact_search = frame_rows[: first_motion_order + 1]
        contact_order = int(np.argmin([row["anchor_distance"] for row in contact_search]))
        first_motion = frame_rows[first_motion_order]
        contact = frame_rows[contact_order]
        readiness_orders = [
            order for order, row in enumerate(frame_rows[: first_motion_order + 1]) if row["readiness"] >= 0.5
        ]
        first_ready_order = readiness_orders[0] if readiness_orders else first_motion_order
        task_records[task].append(
            {
                "start": start,
                "end": end,
                "active_block_index": active_idx,
                "active_displacement": active_displacement,
                "flow_dir": flow_dir.tolist(),
                "first_motion_frame": int(first_motion["frame_index"]),
                "contact_proxy_frame": int(contact["frame_index"]),
                "frames_ready_before_motion": int(first_motion_order - first_ready_order),
                "first_motion": first_motion,
                "contact_proxy": contact,
            }
        )

    report: dict[str, Any] = {
        "motion_threshold": float(args.motion_threshold),
        "fixed_anchor_offset": float(args.anchor_offset),
        "readiness_distance": float(args.readiness_distance),
        "readiness_temperature": float(args.readiness_temperature),
        "tasks": {},
    }
    for task in args.tasks:
        records = task_records.get(task, [])
        report["tasks"][task] = {
            "segments": int(len(records)),
            "contact_proxy_eef_center_along_flow": stats(
                [record["contact_proxy"]["eef_center_along_flow"] for record in records]
            ),
            "contact_proxy_anchor_distance": stats(
                [record["contact_proxy"]["anchor_distance"] for record in records]
            ),
            "first_motion_anchor_distance": stats(
                [record["first_motion"]["anchor_distance"] for record in records]
            ),
            "first_motion_readiness": stats(
                [record["first_motion"]["readiness"] for record in records]
            ),
            "frames_ready_before_motion": stats(
                [record["frames_ready_before_motion"] for record in records]
            ),
            "records": records,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "primitive_contact_transition.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    markdown = [
        "# Primitive Contact-Transition Audit",
        "",
        "| Task | Segments | Contact along F p50 | Contact-to-fixed-anchor p50 | First-motion anchor dist p50 | First-motion readiness p50 | Ready frames before motion p50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in args.tasks:
        rec = report["tasks"][task]

        def p50(name: str) -> str:
            value = rec[name]["p50"]
            return "-" if value is None else f"{value:.4f}"

        markdown.append(
            f"| {task} | {rec['segments']} | {p50('contact_proxy_eef_center_along_flow')} | "
            f"{p50('contact_proxy_anchor_distance')} | {p50('first_motion_anchor_distance')} | "
            f"{p50('first_motion_readiness')} | {p50('frames_ready_before_motion')} |"
        )
    with open(args.out_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown) + "\n")
    print("\n".join(markdown), flush=True)


if __name__ == "__main__":
    main()
