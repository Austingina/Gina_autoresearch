#!/usr/bin/env python3
"""Build a multi-fold CALVIN dataset for intent-causal coarse-G discovery."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any
from zipfile import ZipFile

import numpy as np
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())
import extract_calvin_intent_factorial_subset as base
import extract_calvin_remote_calibration_subset as remote


BLOCK_TASKS = (
    "push_red_block_left",
    "push_red_block_right",
    "push_blue_block_left",
    "push_blue_block_right",
    "push_pink_block_left",
    "push_pink_block_right",
)
ARTICULATION_TASKS = (
    "open_drawer",
    "close_drawer",
    "move_slider_left",
    "move_slider_right",
)
CONTAINER_TASKS = ("push_into_drawer", "place_in_slider")
ALL_INTENT_TASKS = BLOCK_TASKS + ARTICULATION_TASKS + CONTAINER_TASKS

BLOCK_FOLDS = (
    {
        "fold_id": "block_bridge_pink",
        "bridge_entity": "pink_block",
        "source_tasks": (
            "push_red_block_left",
            "push_blue_block_right",
            "push_pink_block_left",
            "push_pink_block_right",
        ),
        "heldout_tasks": ("push_red_block_right", "push_blue_block_left"),
    },
    {
        "fold_id": "block_bridge_red",
        "bridge_entity": "red_block",
        "source_tasks": (
            "push_red_block_left",
            "push_red_block_right",
            "push_blue_block_left",
            "push_pink_block_right",
        ),
        "heldout_tasks": ("push_blue_block_right", "push_pink_block_left"),
    },
    {
        "fold_id": "block_bridge_blue",
        "bridge_entity": "blue_block",
        "source_tasks": (
            "push_blue_block_left",
            "push_blue_block_right",
            "push_red_block_right",
            "push_pink_block_left",
        ),
        "heldout_tasks": ("push_red_block_left", "push_pink_block_right"),
    },
)

ARTICULATION_FOLDS = (
    {
        "fold_id": "art_hold_slider_left",
        "bridge_mechanism": "drawer",
        "source_tasks": ("open_drawer", "close_drawer", "move_slider_right"),
        "heldout_tasks": ("move_slider_left",),
    },
    {
        "fold_id": "art_hold_slider_right",
        "bridge_mechanism": "drawer",
        "source_tasks": ("open_drawer", "close_drawer", "move_slider_left"),
        "heldout_tasks": ("move_slider_right",),
    },
    {
        "fold_id": "art_hold_drawer_close",
        "bridge_mechanism": "slider",
        "source_tasks": ("move_slider_left", "move_slider_right", "open_drawer"),
        "heldout_tasks": ("close_drawer",),
    },
    {
        "fold_id": "art_hold_drawer_open",
        "bridge_mechanism": "slider",
        "source_tasks": ("move_slider_left", "move_slider_right", "close_drawer"),
        "heldout_tasks": ("open_drawer",),
    },
)


def build_g_contract() -> dict[str, object]:
    return {
        "representation_scope": "task_frame_low_dimensional_guidance",
        "components": {
            "A": "delta_or_relation_to_interaction_region",
            "F": "desired_entity_motion_in_task_frame",
            "phase": "approach_align_contact_transport_release_or_stop",
        },
        "causal_axes": ["interaction_region", "entity_motion", "goal_relation", "phase"],
        "nuisance_axes": [
            "visual_color",
            "texture",
            "scene_id",
            "camera_pixel_location",
            "trajectory_timing",
        ],
        "adapter_only_axes": ["robot_joint_ids", "action_scale", "controller_rate", "simulator_state_layout"],
        "current_evidence_scope": "CALVIN translational task families only",
    }


def build_factor_folds() -> list[dict[str, object]]:
    folds: list[dict[str, object]] = []
    for block in BLOCK_FOLDS:
        for articulation in ARTICULATION_FOLDS:
            source = tuple(block["source_tasks"]) + tuple(articulation["source_tasks"]) + CONTAINER_TASKS
            heldout = tuple(block["heldout_tasks"]) + tuple(articulation["heldout_tasks"])
            if set(source) & set(heldout):
                raise RuntimeError("fold source/heldout overlap")
            if set(source) | set(heldout) != set(ALL_INTENT_TASKS):
                raise RuntimeError("fold does not cover all intent tasks")
            folds.append(
                {
                    "fold_id": f"{block['fold_id']}__{articulation['fold_id']}",
                    "block_fold_id": block["fold_id"],
                    "block_bridge_entity": block["bridge_entity"],
                    "articulation_fold_id": articulation["fold_id"],
                    "articulation_bridge_mechanism": articulation["bridge_mechanism"],
                    "source_tasks": list(source),
                    "heldout_tasks": list(heldout),
                }
            )
    return folds


def segment_key(row: dict[str, object]) -> tuple[str, str, int, int]:
    return str(row["task"]), str(row["scene"]), int(row["start"]), int(row["end"])


def interval_key(row: dict[str, object]) -> tuple[int, int]:
    return int(row["start"]), int(row["end"])


def _rng(seed: int, *parts: str) -> random.Random:
    payload = "|".join((str(seed), *parts)).encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def _decorate(row: dict[str, object], retained: bool) -> dict[str, object]:
    intent = base.INTENT_SCHEMA[str(row["task"])]
    return {
        **row,
        "intent_family": intent["family"],
        "primitive": intent["primitive"],
        "entity": intent["entity"],
        "direction": intent["direction"],
        "g_semantics": intent["g_semantics"],
        "retained_from_previous_intent_dataset": retained,
    }


def select_intent_superset(
    rows: list[dict[str, object]],
    baseline_segments: list[dict[str, object]],
    per_task_scene: int,
    seed: int,
) -> list[dict[str, object]]:
    if per_task_scene < 1:
        raise ValueError("per_task_scene must be positive")
    rows = [row for row in rows if str(row["task"]) in set(ALL_INTENT_TASKS)]
    lookup = {segment_key(row): row for row in rows}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["scene"]))].append(row)
    scenes = sorted({str(row["scene"]) for row in rows})
    if not scenes:
        raise ValueError("no candidate scenes")

    selected: list[dict[str, object]] = []
    selected_keys: set[tuple[str, str, int, int]] = set()
    intervals: list[tuple[int, int]] = []
    for raw in baseline_segments:
        task = str(raw["task"])
        if task not in ALL_INTENT_TASKS:
            raise ValueError(f"baseline task outside intent contract: {task}")
        key = segment_key(raw)
        if key not in lookup:
            raise ValueError(f"baseline segment absent from full annotation: {key}")
        if key in selected_keys:
            continue
        row = lookup[key]
        if remote.overlaps(interval_key(row), intervals):
            raise ValueError(f"baseline intervals overlap at {key}")
        selected.append(_decorate(row, retained=True))
        selected_keys.add(key)
        intervals.append(interval_key(row))

    for task in ALL_INTENT_TASKS:
        for scene in scenes:
            existing = [row for row in selected if row["task"] == task and row["scene"] == scene]
            if len(existing) > per_task_scene:
                raise ValueError(f"baseline exceeds quota for {task}/{scene}")
            candidates = [
                row
                for row in sorted(grouped[(task, scene)], key=segment_key)
                if segment_key(row) not in selected_keys
            ]
            _rng(seed, "superset", task, scene).shuffle(candidates)
            for row in candidates:
                if len(existing) == per_task_scene:
                    break
                if remote.overlaps(interval_key(row), intervals):
                    continue
                decorated = _decorate(row, retained=False)
                selected.append(decorated)
                existing.append(decorated)
                selected_keys.add(segment_key(row))
                intervals.append(interval_key(row))
            if len(existing) != per_task_scene:
                raise RuntimeError(f"Only selected {len(existing)}/{per_task_scene} for {task}/{scene}")

    result = sorted(selected, key=lambda row: (int(row["start"]), str(row["task"])))
    expected = len(ALL_INTENT_TASKS) * len(scenes) * per_task_scene
    if len(result) != expected:
        raise RuntimeError(f"superset size {len(result)} != expected {expected}")
    return result


def materialize_fold(
    superset: list[dict[str, object]],
    fold_spec: dict[str, object],
    source_per_task_scene: int,
    heldout_per_task_scene: int,
    seed: int,
) -> dict[str, object]:
    source_tasks = tuple(str(task) for task in fold_spec["source_tasks"])
    heldout_tasks = tuple(str(task) for task in fold_spec["heldout_tasks"])
    if set(source_tasks) & set(heldout_tasks):
        raise ValueError("fold source/heldout overlap")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in superset:
        grouped[(str(row["task"]), str(row["scene"]))].append(row)
    scenes = sorted({str(row["scene"]) for row in superset})
    segments: list[dict[str, object]] = []
    for task in source_tasks:
        for scene in scenes:
            cell = sorted(grouped[(task, scene)], key=segment_key)
            if len(cell) < source_per_task_scene:
                raise RuntimeError(f"insufficient source cell {task}/{scene}")
            segments.extend({**row, "split": "source", "fold_id": fold_spec["fold_id"]} for row in cell[:source_per_task_scene])
    for task in heldout_tasks:
        for scene in scenes:
            cell = sorted(grouped[(task, scene)], key=segment_key)
            chooser = _rng(seed, "heldout", task, scene)
            chooser.shuffle(cell)
            chosen = sorted(cell[:heldout_per_task_scene], key=segment_key)
            if len(chosen) != heldout_per_task_scene:
                raise RuntimeError(f"insufficient heldout cell {task}/{scene}")
            segments.extend({**row, "split": "heldout", "fold_id": fold_spec["fold_id"]} for row in chosen)
    segments.sort(key=lambda row: (int(row["start"]), str(row["task"])))
    interval_counts = Counter(interval_key(row) for row in segments)
    duplicate_intervals = [interval for interval, count in interval_counts.items() if count != 1]
    if duplicate_intervals:
        raise RuntimeError(f"fold interval leakage: {duplicate_intervals[:3]}")
    return {
        **fold_spec,
        "source_tasks": list(source_tasks),
        "heldout_tasks": list(heldout_tasks),
        "source_per_task_scene": source_per_task_scene,
        "heldout_per_task_scene": heldout_per_task_scene,
        "segments": len(segments),
        "frames": sum(int(row["end"]) - int(row["start"]) + 1 for row in segments),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in segments).items())),
        "g_contract": build_g_contract(),
        "counterfactual_policy": {
            "usage": "audit_only",
            "expert_action_label": False,
            "requires_real_rollout_for_training": True,
        },
        "segments_detail": segments,
    }


def _load_rows(annotation: dict[str, object], scene_info: dict[str, list[int]]) -> list[dict[str, object]]:
    rows = base.build_segment_rows(annotation, scene_info)
    return [row for row in rows if str(row["task"]) in ALL_INTENT_TASKS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip")
    parser.add_argument("--zip-index", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--full-annotation", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--reuse-data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--per-task-scene", type=int, default=12)
    parser.add_argument("--heldout-per-task-scene", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-range-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_range_bytes < 1:
        parser.error("max-range-bytes must be positive")

    root = args.output_dir.resolve()
    split = root / "training"
    selection_dir = args.selection_dir.resolve()
    folds_dir = selection_dir / "folds"
    selection_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)
    metadata = base.load_local_metadata(args.full_annotation, args.metadata_root)
    names = base.METADATA_NAMES
    annotation = np.load(io.BytesIO(metadata[names[0]]), allow_pickle=True).item()
    scene_info = np.load(io.BytesIO(metadata[names[2]]), allow_pickle=True).item()
    previous = json.loads(args.previous_manifest.read_text(encoding="utf-8"))
    rows = _load_rows(annotation, scene_info)
    superset = select_intent_superset(rows, previous["segments_detail"], args.per_task_scene, args.seed)

    with ZipFile(args.zip_index.resolve()) as archive:
        frame_infos: dict[int, Any] = {}
        for segment in superset:
            for frame in range(int(segment["start"]), int(segment["end"]) + 1):
                frame_infos[frame] = archive.getinfo(f"task_ABC_D/training/episode_{frame:07d}.npz")
        selection = {
            "format": "calvin_g_intent_multifold_v1",
            "selection_seed": args.seed,
            "source_url": args.url,
            "g_contract": build_g_contract(),
            "all_intent_tasks": list(ALL_INTENT_TASKS),
            "per_task_scene": args.per_task_scene,
            "segments": len(superset),
            "frames": len(frame_infos),
            "compressed_bytes": sum(int(info.compress_size) for info in frame_infos.values()),
            "retained_segments": sum(bool(row["retained_from_previous_intent_dataset"]) for row in superset),
            "new_segments": sum(not bool(row["retained_from_previous_intent_dataset"]) for row in superset),
            "fold_count": len(build_factor_folds()),
            "container_factorial_status": "coverage_only_missing_crossed_primitive_container_tasks",
            "phase_label_status": "not_generated_requires_state_grounded_phase_builder",
            "negative_trajectory_status": "not_present_requires_simulator_rollout",
            "segments_detail": superset,
        }
        (selection_dir / "superset_selection_s12.json").write_text(json.dumps(selection, indent=2) + "\n")
        fold_manifests = []
        for fold_spec in build_factor_folds():
            fold = materialize_fold(
                superset,
                fold_spec,
                args.per_task_scene,
                args.heldout_per_task_scene,
                args.seed,
            )
            fold_path = folds_dir / f"{fold['fold_id']}.json"
            fold_path.write_text(json.dumps(fold, indent=2) + "\n")
            fold_manifests.append({k: v for k, v in fold.items() if k != "segments_detail"})
        (selection_dir / "fold_index.json").write_text(json.dumps(fold_manifests, indent=2) + "\n")
        if args.dry_run:
            print(json.dumps({k: v for k, v in selection.items() if k != "segments_detail"}, indent=2))
            return

        prefix = "task_ABC_D/"
        linked_files = 0
        linked_bytes = 0
        if args.reuse_data_root:
            reuse_split = args.reuse_data_root.resolve() / "training"
            for frame, info in frame_infos.items():
                source = reuse_split / f"episode_{frame:07d}.npz"
                destination = split / f"episode_{frame:07d}.npz"
                if not source.exists() or source.stat().st_size != int(info.file_size):
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    try:
                        os.link(source, destination)
                    except OSError:
                        shutil.copy2(source, destination)
                linked_files += 1
                linked_bytes += int(info.file_size)

        def destination_for(info: Any) -> Path:
            if not info.filename.startswith(prefix):
                raise RuntimeError(f"archive member outside prefix: {info.filename}")
            return root / info.filename[len(prefix) :]

        def extract_segment(item: tuple[int, dict[str, object]]) -> tuple[int, int, int, int]:
            index, segment = item
            infos = [frame_infos[frame] for frame in range(int(segment["start"]), int(segment["end"]) + 1)]
            missing = [
                info
                for info in infos
                if not destination_for(info).exists()
                or destination_for(info).stat().st_size != int(info.file_size)
            ]
            groups = remote.physical_groups(
                missing,
                max_range_bytes=args.max_range_bytes,
            )
            downloaded = 0
            worker = requests.Session()
            for group in groups:
                range_start = int(group[0].header_offset)
                range_end = remote.member_end(group[-1])
                response = None
                for attempt in range(1, args.retries + 1):
                    try:
                        response = worker.get(
                            args.url,
                            headers={"Range": f"bytes={range_start}-{range_end}"},
                            timeout=args.timeout,
                        )
                        response.raise_for_status()
                        if response.status_code != 206:
                            raise RuntimeError(f"range ignored: {response.status_code}")
                        break
                    except (requests.RequestException, RuntimeError):
                        if attempt == args.retries:
                            raise
                        time.sleep(min(2 ** (attempt - 1), 30))
                assert response is not None
                downloaded += len(response.content)
                for info in group:
                    remote.extract_member_from_range(response.content, range_start, info, destination_for(info))
            return index, len(infos), len(groups), downloaded

        downloaded_bytes = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, frames, groups, downloaded in executor.map(extract_segment, enumerate(superset, start=1)):
                downloaded_bytes += downloaded
                print(
                    f"[extract] {index}/{len(superset)} frames={frames} ranges={groups} "
                    f"downloaded_mib={downloaded_bytes / 2**20:.1f}",
                    flush=True,
                )

    (split / ".hydra").mkdir(parents=True, exist_ok=True)
    (split / ".hydra/merged_config.yaml").write_bytes(metadata[names[1]])
    np.save(split / "scene_info.npy", scene_info, allow_pickle=True)
    language_dir = split / "lang_annotations"
    language_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        language_dir / "auto_lang_ann.npy",
        {
            "language": {
                "ann": [str(annotation["language"]["ann"][int(row["annotation_index"])]) for row in superset],
                "task": [str(row["task"]) for row in superset],
                "emb": np.stack(
                    [np.asarray(annotation["language"]["emb"][int(row["annotation_index"])], dtype=np.float32) for row in superset]
                ),
            },
            "info": {
                "indx": np.asarray([[int(row["start"]), int(row["end"])] for row in superset], dtype=np.int64)
            },
        },
        allow_pickle=True,
    )
    manifest = {
        **selection,
        "linked_files": linked_files,
        "linked_bytes": linked_bytes,
        "downloaded_bytes_this_run": downloaded_bytes,
        "fold_manifest_dir": folds_dir.as_posix(),
        "complete": True,
    }
    (root / "g_intent_multifold_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "segments_detail"}, indent=2))


if __name__ == "__main__":
    main()
