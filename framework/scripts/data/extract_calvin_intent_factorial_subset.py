#!/usr/bin/env python3
"""Build a leakage-safe CALVIN subset for coarse-G intent factorization."""

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
import extract_calvin_remote_calibration_subset as remote


SOURCE_TASKS = (
    "push_red_block_left",
    "push_blue_block_right",
    "push_pink_block_left",
    "push_pink_block_right",
    "open_drawer",
    "move_slider_right",
    "push_into_drawer",
    "place_in_slider",
)
HELDOUT_TASKS = (
    "push_blue_block_left",
    "push_red_block_right",
    "close_drawer",
    "move_slider_left",
)
METADATA_NAMES = (
    "task_ABC_D/training/lang_annotations/auto_lang_ann.npy",
    "task_ABC_D/training/.hydra/merged_config.yaml",
    "task_ABC_D/training/scene_info.npy",
)


def _intent(
    family: str,
    primitive: str,
    entity: str,
    direction: str,
    split: str,
) -> dict[str, str]:
    return {
        "family": family,
        "primitive": primitive,
        "entity": entity,
        "direction": direction,
        "split": split,
        "g_semantics": "A=delta_to_interaction_region;F=desired_entity_motion",
    }


INTENT_SCHEMA = {
    "push_red_block_left": _intent("block_push", "push", "red_block", "x-", "source"),
    "push_red_block_right": _intent("block_push", "push", "red_block", "x+", "heldout"),
    "push_blue_block_left": _intent("block_push", "push", "blue_block", "x-", "heldout"),
    "push_blue_block_right": _intent("block_push", "push", "blue_block", "x+", "source"),
    "push_pink_block_left": _intent("block_push", "push", "pink_block", "x-", "source"),
    "push_pink_block_right": _intent("block_push", "push", "pink_block", "x+", "source"),
    "open_drawer": _intent("articulation", "translate_joint", "drawer", "open", "source"),
    "close_drawer": _intent("articulation", "translate_joint", "drawer", "close", "heldout"),
    "move_slider_right": _intent("articulation", "translate_joint", "slider", "right", "source"),
    "move_slider_left": _intent("articulation", "translate_joint", "slider", "left", "heldout"),
    "push_into_drawer": _intent(
        "container_transport", "push_into", "nearest_block_to_drawer", "to_drawer", "source"
    ),
    "place_in_slider": _intent(
        "container_transport", "place_into", "nearest_block_to_slider", "to_slider", "source"
    ),
}


def build_counterfactual_audit_plan() -> list[dict[str, object]]:
    """Describe diagnostics that must never be treated as expert demonstrations."""

    common = {"usage": "audit_only", "expert_action_label": False}
    return [
        {**common, "condition": "zero_g", "scope": "all_source_tasks"},
        {**common, "condition": "sign_flip_f", "scope": "all_source_tasks"},
        {
            **common,
            "condition": "inverse_intent",
            "source_task": "open_drawer",
            "target_task": "close_drawer",
        },
        {
            **common,
            "condition": "inverse_intent",
            "source_task": "move_slider_right",
            "target_task": "move_slider_left",
        },
        {
            **common,
            "condition": "wrong_entity",
            "scope": "block_push",
            "rule": "preserve_direction_replace_block_identity",
        },
    ]


def segment_key(row: dict[str, object]) -> tuple[str, str, int, int]:
    return (
        str(row["task"]),
        str(row["scene"]),
        int(row["start"]),
        int(row["end"]),
    )


def interval_key(row: dict[str, object]) -> tuple[int, int]:
    return int(row["start"]), int(row["end"])


def build_segment_rows(
    annotation: dict[str, object], scene_info: dict[str, list[int]]
) -> list[dict[str, object]]:
    wanted = set(SOURCE_TASKS) | set(HELDOUT_TASKS)
    rows: list[dict[str, object]] = []
    for annotation_index, (task_value, interval) in enumerate(
        zip(annotation["language"]["task"], annotation["info"]["indx"])
    ):
        task = str(task_value)
        if task not in wanted:
            continue
        start, end = int(interval[0]), int(interval[1])
        rows.append(
            {
                "task": task,
                "scene": remote.scene_for_interval(scene_info, start, end),
                "start": start,
                "end": end,
                "annotation_index": annotation_index,
            }
        )
    return rows


def load_local_metadata(full_annotation: Path, metadata_root: Path) -> dict[str, bytes]:
    """Load cached full annotations plus immutable metadata from an existing subset."""

    split = metadata_root.resolve() / "training"
    paths = (
        full_annotation.resolve(),
        split / ".hydra/merged_config.yaml",
        split / "scene_info.npy",
    )
    missing = [path.as_posix() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local metadata: {missing}")
    return {name: path.read_bytes() for name, path in zip(METADATA_NAMES, paths)}


def _stable_rng(seed: int, task: str, scene: str, bucket: str) -> random.Random:
    payload = f"{seed}|{task}|{scene}|{bucket}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(derived)


def _choose(
    candidates: list[dict[str, object]],
    count: int,
    rng: random.Random,
    selected_intervals: list[tuple[int, int]],
) -> list[dict[str, object]]:
    if count == 0:
        return []
    ordered = sorted(candidates, key=segment_key)
    rng.shuffle(ordered)
    chosen: list[dict[str, object]] = []
    for row in ordered:
        interval = interval_key(row)
        if remote.overlaps(interval, selected_intervals):
            continue
        chosen.append(row)
        selected_intervals.append(interval)
        if len(chosen) == count:
            return chosen
    if count:
        raise RuntimeError(f"Only selected {len(chosen)}/{count} non-overlapping candidates")
    return chosen


def _decorate(row: dict[str, object], retained: bool) -> dict[str, object]:
    task = str(row["task"])
    return {
        **row,
        "split": INTENT_SCHEMA[task]["split"],
        "intent_family": INTENT_SCHEMA[task]["family"],
        "primitive": INTENT_SCHEMA[task]["primitive"],
        "entity": INTENT_SCHEMA[task]["entity"],
        "direction": INTENT_SCHEMA[task]["direction"],
        "retained_from_baseline": retained,
    }


def select_intent_factorial_segments(
    rows: list[dict[str, object]],
    baseline_segments: list[dict[str, object]],
    source_per_task_scene: int,
    heldout_per_task_scene: int,
    validation_modulus: int,
    seed: int,
) -> list[dict[str, object]]:
    if source_per_task_scene < 2:
        raise ValueError("source_per_task_scene must be at least 2")
    if heldout_per_task_scene < 1:
        raise ValueError("heldout_per_task_scene must be positive")
    if set(SOURCE_TASKS) & set(HELDOUT_TASKS):
        raise ValueError("source and heldout task sets overlap")

    lookup = {segment_key(row): row for row in rows}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["scene"]))].append(row)
    scenes = sorted({str(row["scene"]) for row in rows})
    if not scenes:
        raise ValueError("No candidate scenes")

    selected: list[dict[str, object]] = []
    selected_keys: set[tuple[str, str, int, int]] = set()
    selected_intervals: list[tuple[int, int]] = []
    for raw in baseline_segments:
        task = str(raw["task"])
        expected_split = INTENT_SCHEMA.get(task, {}).get("split")
        stated_split = raw.get("split", expected_split)
        if expected_split is None or stated_split != expected_split:
            raise ValueError(f"baseline split mismatch for {task}")
        key = segment_key(raw)
        if key not in lookup:
            raise ValueError(f"baseline segment absent from full annotation: {key}")
        row = lookup[key]
        interval = interval_key(row)
        if remote.overlaps(interval, selected_intervals):
            raise ValueError(f"baseline intervals overlap at {key}")
        selected.append(_decorate(row, retained=True))
        selected_keys.add(key)
        selected_intervals.append(interval)

    def add_rows(chosen: list[dict[str, object]]) -> None:
        for row in chosen:
            key = segment_key(row)
            if key in selected_keys:
                continue
            selected.append(_decorate(row, retained=False))
            selected_keys.add(key)

    for task in SOURCE_TASKS:
        for scene in scenes:
            cell = grouped[(task, scene)]
            existing = [r for r in selected if r["task"] == task and r["scene"] == scene]
            if len(existing) > source_per_task_scene:
                raise ValueError(f"baseline exceeds source quota for {task}/{scene}")
            missing = source_per_task_scene - len(existing)
            available = [r for r in cell if segment_key(r) not in selected_keys]
            has_validation = any(int(r["start"]) % validation_modulus == 0 for r in existing)
            if not has_validation and missing:
                validation = [r for r in available if int(r["start"]) % validation_modulus == 0]
                chosen = _choose(
                    validation,
                    1,
                    _stable_rng(seed, task, scene, "validation"),
                    selected_intervals,
                )
                add_rows(chosen)
                missing -= 1
                available = [r for r in available if segment_key(r) not in selected_keys]
            train = [r for r in available if int(r["start"]) % validation_modulus != 0]
            add_rows(
                _choose(
                    train,
                    missing,
                    _stable_rng(seed, task, scene, "train"),
                    selected_intervals,
                )
            )

    for task in HELDOUT_TASKS:
        for scene in scenes:
            existing = [r for r in selected if r["task"] == task and r["scene"] == scene]
            if len(existing) > heldout_per_task_scene:
                raise ValueError(f"baseline exceeds heldout quota for {task}/{scene}")
            candidates = [
                r
                for r in grouped[(task, scene)]
                if segment_key(r) not in selected_keys
            ]
            add_rows(
                _choose(
                    candidates,
                    heldout_per_task_scene - len(existing),
                    _stable_rng(seed, task, scene, "heldout"),
                    selected_intervals,
                )
            )

    result = sorted(selected, key=lambda row: (int(row["start"]), str(row["task"])))
    expected = len(SOURCE_TASKS) * len(scenes) * source_per_task_scene + len(
        HELDOUT_TASKS
    ) * len(scenes) * heldout_per_task_scene
    if len(result) != expected:
        raise RuntimeError(f"Selection size {len(result)} != expected {expected}")
    return result


def summarize(segments: list[dict[str, object]], compressed_bytes: int) -> dict[str, object]:
    frames = sum(int(r["end"]) - int(r["start"]) + 1 for r in segments)
    split_counts = Counter(str(r["split"]) for r in segments)
    family_counts = Counter(str(r["intent_family"]) for r in segments)
    return {
        "segments": len(segments),
        "frames": frames,
        "compressed_bytes": compressed_bytes,
        "retained_segments": sum(bool(r["retained_from_baseline"]) for r in segments),
        "new_segments": sum(not bool(r["retained_from_baseline"]) for r in segments),
        "split_counts": dict(sorted(split_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "task_counts": dict(sorted(Counter(str(r["task"]) for r in segments).items())),
        "scene_counts": dict(sorted(Counter(str(r["scene"]) for r in segments).items())),
        "direction_counts": dict(sorted(Counter(str(r["direction"]) for r in segments).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip")
    parser.add_argument("--zip-index", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--full-annotation", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--reuse-data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--source-per-task-scene", type=int, default=12)
    parser.add_argument("--heldout-per-task-scene", type=int, default=3)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if bool(args.full_annotation) != bool(args.metadata_root):
        parser.error("--full-annotation and --metadata-root must be supplied together")

    root = args.output_dir.resolve()
    split = root / "training"
    prefix = "task_ABC_D/"
    metadata_names = METADATA_NAMES
    baseline_manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    baseline_segments = baseline_manifest["segments_detail"]

    with ZipFile(args.zip_index.resolve()) as archive, requests.Session() as session:
        if args.full_annotation:
            metadata = load_local_metadata(args.full_annotation, args.metadata_root)
        else:
            metadata = {
                name: remote.fetch_member(session, args.url, archive.getinfo(name), args.timeout)
                for name in metadata_names
            }
        annotation = np.load(io.BytesIO(metadata[metadata_names[0]]), allow_pickle=True).item()
        scene_info = np.load(io.BytesIO(metadata[metadata_names[2]]), allow_pickle=True).item()
        rows = build_segment_rows(annotation, scene_info)
        segments = select_intent_factorial_segments(
            rows,
            baseline_segments,
            args.source_per_task_scene,
            args.heldout_per_task_scene,
            args.validation_modulus,
            args.seed,
        )
        frame_infos: dict[int, Any] = {}
        for segment in segments:
            for frame in range(int(segment["start"]), int(segment["end"]) + 1):
                frame_infos[frame] = archive.getinfo(
                    f"task_ABC_D/training/episode_{frame:07d}.npz"
                )
        compressed_bytes = sum(int(info.compress_size) for info in frame_infos.values())
        summary = summarize(segments, compressed_bytes)
        selection = {
            **summary,
            "source_url": args.url,
            "selection_seed": args.seed,
            "source_tasks": list(SOURCE_TASKS),
            "heldout_tasks": list(HELDOUT_TASKS),
            "source_per_task_scene": args.source_per_task_scene,
            "heldout_per_task_scene": args.heldout_per_task_scene,
            "validation_modulus": args.validation_modulus,
            "intent_schema": INTENT_SCHEMA,
            "counterfactual_audit_plan": build_counterfactual_audit_plan(),
            "negative_trajectory_status": "not_present_in_official_success_demonstrations; requires simulator rollout collection",
            "excluded_scope": {
                "binary_switches": "empty translational G",
                "lift_rotate_stack": "coarse G representation not yet validated",
            },
            "segments_detail": segments,
        }
        if args.selection_manifest:
            args.selection_manifest.parent.mkdir(parents=True, exist_ok=True)
            args.selection_manifest.write_text(json.dumps(selection, indent=2) + "\n")
        if args.dry_run:
            print(json.dumps(summary, indent=2))
            return

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
                raise RuntimeError(f"Archive member outside prefix: {info.filename}")
            return root / info.filename[len(prefix) :]

        def extract_segment(item: tuple[int, dict[str, object]]) -> tuple[int, int, int, int]:
            index, segment = item
            infos = [
                frame_infos[frame]
                for frame in range(int(segment["start"]), int(segment["end"]) + 1)
            ]
            missing = [
                info
                for info in infos
                if not destination_for(info).exists()
                or destination_for(info).stat().st_size != int(info.file_size)
            ]
            groups = remote.physical_groups(missing)
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
                            raise RuntimeError(f"Range ignored: {response.status_code}")
                        break
                    except (requests.RequestException, RuntimeError):
                        if attempt == args.retries:
                            raise
                        time.sleep(min(2 ** (attempt - 1), 30))
                assert response is not None
                downloaded += len(response.content)
                for info in group:
                    remote.extract_member_from_range(
                        response.content, range_start, info, destination_for(info)
                    )
            return index, len(infos), len(groups), downloaded

        downloaded_bytes = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, frames, groups, downloaded in executor.map(
                extract_segment, enumerate(segments, start=1)
            ):
                downloaded_bytes += downloaded
                print(
                    f"[extract] {index}/{len(segments)} frames={frames} ranges={groups} "
                    f"downloaded_mib={downloaded_bytes / 2**20:.1f}",
                    flush=True,
                )

    (split / ".hydra").mkdir(parents=True, exist_ok=True)
    (split / ".hydra" / "merged_config.yaml").write_bytes(metadata[metadata_names[1]])
    np.save(split / "scene_info.npy", scene_info, allow_pickle=True)
    language_dir = split / "lang_annotations"
    language_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        language_dir / "auto_lang_ann.npy",
        {
            "language": {
                "ann": [str(annotation["language"]["ann"][int(r["annotation_index"])]) for r in segments],
                "task": [str(r["task"]) for r in segments],
                "emb": np.stack(
                    [np.asarray(annotation["language"]["emb"][int(r["annotation_index"])], dtype=np.float32) for r in segments]
                ),
            },
            "info": {
                "indx": np.asarray([[int(r["start"]), int(r["end"])] for r in segments], dtype=np.int64)
            },
        },
        allow_pickle=True,
    )
    manifest = {
        **selection,
        "linked_files": linked_files,
        "linked_bytes": linked_bytes,
        "downloaded_bytes_this_run": downloaded_bytes,
        "complete": True,
    }
    (root / "intent_factorial_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"segments_detail", "intent_schema"}}, indent=2))


if __name__ == "__main__":
    main()
