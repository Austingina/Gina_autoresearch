#!/usr/bin/env python3
"""Build a semantic sidecar over an extracted official VIMA trajectory subset."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickle
import pickletools
from typing import Any

import numpy as np
from PIL import Image

SEMANTICS: dict[str, dict[str, Any]] = {
    "rotate": {
        "task_family": "orientation_manipulation",
        "task_goal": "rotate_target_object_to_prompted_orientation",
        "entities": ["target_object"],
        "relations": ["prompted_planar_rotation"],
    },
    "twist": {
        "task_family": "orientation_manipulation",
        "task_goal": "twist_target_object_to_prompted_orientation",
        "entities": ["target_object"],
        "relations": ["prompted_twist"],
    },
    "sweep_without_exceeding": {
        "task_family": "constraint_aware_tool_interaction",
        "task_goal": "sweep_target_without_crossing_boundary",
        "entities": ["target_object", "tool", "boundary"],
        "relations": ["sweep", "do_not_exceed"],
    },
    "follow_order": {
        "task_family": "ordered_imitation",
        "task_goal": "execute_prompted_object_order",
        "entities": ["ordered_objects", "prompt_demonstration"],
        "relations": ["temporal_order"],
    },
    "pick_in_order_then_restore": {
        "task_family": "long_horizon_memory_manipulation",
        "task_goal": "pick_objects_in_order_then_restore_scene",
        "entities": ["ordered_objects", "original_scene"],
        "relations": ["temporal_order", "restore"],
    },
    "rearrange_then_restore": {
        "task_family": "long_horizon_rearrangement",
        "task_goal": "rearrange_scene_then_restore_original_configuration",
        "entities": ["scene_objects", "goal_scene", "original_scene"],
        "relations": ["rearrange", "restore"],
    },
}
PHASE_NAMES = (
    "approach_proxy",
    "contact_transition_proxy",
    "sustained_interaction_proxy",
    "completion_proxy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pickle_scalar(path: Path, key: str) -> Any:
    operations = list(pickletools.genops(path.read_bytes()))
    for index, (operation, argument, _position) in enumerate(operations):
        if operation.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"} and argument == key:
            for value_operation, value_argument, _ in operations[index + 1 : index + 6]:
                if value_operation.name == "MEMOIZE":
                    continue
                if value_operation.name in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT"}:
                    return int(value_argument)
                if value_operation.name == "NEWTRUE":
                    return True
                if value_operation.name == "NEWFALSE":
                    return False
                if value_operation.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
                    return str(value_argument)
                break
    raise ValueError(f"pickle key {key!r} not found as a direct scalar in {path}")


def phase_views(transitions: int) -> list[dict[str, Any]]:
    if transitions < 1:
        raise ValueError("parent must contain at least one transition")
    count = min(len(PHASE_NAMES), transitions)
    boundaries = [(index * transitions) // count for index in range(count + 1)]
    phases = []
    for index in range(count):
        start, end = boundaries[index], boundaries[index + 1]
        if end <= start:
            raise AssertionError("phase partition produced an empty interval")
        phases.append(
            {
                "phase": PHASE_NAMES[index if count == len(PHASE_NAMES) else len(PHASE_NAMES) - count + index],
                "start": start,
                "end": end,
                "start_transition": start,
                "end_transition_exclusive": end,
                "evidence_method": "temporal_proxy_over_native_oracle_actions",
                "independent_success": False,
            }
        )
    return phases


def _load_numpy_pickle(path: Path) -> dict[str, Any]:
    value = pickle.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected dict pickle: {path}")
    return value


def _leading_lengths(mapping: dict[str, Any], prefix: str = "") -> dict[str, int]:
    lengths: dict[str, int] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            lengths.update(_leading_lengths(value, name))
        else:
            array = np.asarray(value)
            if array.ndim < 1:
                raise ValueError(f"native array is scalar: {name}")
            lengths[name] = int(array.shape[0])
    return lengths


def inspect_parent(
    parent: Path,
    *,
    task: str,
    trajectory_id: str,
    source_revision: str,
    payload_root: Path,
    member_receipts: list[dict[str, Any]] | None = None,
    archive_lfs_sha256: str | None = None,
) -> dict[str, Any]:
    if task not in SEMANTICS:
        raise ValueError(f"unsupported selected VIMA task: {task}")
    required = [parent / name for name in ("action.pkl", "obs.pkl", "trajectory.pkl")]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing native payload in {parent}: {missing}")
    action = _load_numpy_pickle(parent / "action.pkl")
    obs = _load_numpy_pickle(parent / "obs.pkl")
    action_lengths = _leading_lengths(action)
    if len(set(action_lengths.values())) != 1:
        raise ValueError(f"action arrays have inconsistent lengths: {action_lengths}")
    transitions = next(iter(action_lengths.values()))
    obs_lengths = _leading_lengths(obs)
    if set(obs_lengths.values()) != {transitions + 1}:
        raise ValueError(f"observation alignment is not T+1 to T: actions={transitions} obs={obs_lengths}")

    metadata_steps = _pickle_scalar(parent / "trajectory.pkl", "steps")
    success = _pickle_scalar(parent / "trajectory.pkl", "success")
    failure = _pickle_scalar(parent / "trajectory.pkl", "failure")
    prompt_template = _pickle_scalar(parent / "trajectory.pkl", "prompt")
    if metadata_steps != transitions:
        raise ValueError(f"trajectory steps disagree with action length: {metadata_steps} != {transitions}")
    if success is not True or failure is not False:
        raise ValueError(f"native success evidence failed: success={success} failure={failure}")

    rgb_frames: dict[str, list[Path]] = {}
    dimensions: dict[str, list[int]] = {}
    for view in ("front", "top"):
        files = sorted((parent / f"rgb_{view}").glob("*.jpg"), key=lambda path: int(path.stem))
        ids = [int(path.stem) for path in files]
        if ids != list(range(transitions + 1)):
            raise ValueError(f"rgb_{view} frame IDs do not cover T+1: {ids[:3]}...{ids[-3:] if ids else []}")
        with Image.open(files[0]) as image:
            image.verify()
        with Image.open(files[-1]) as image:
            image.verify()
        with Image.open(files[0]) as image:
            dimensions[view] = [int(image.height), int(image.width), len(image.getbands())]
        rgb_frames[view] = files

    phases = phase_views(transitions)
    semantic = SEMANTICS[task]
    return {
        "schema_version": "official_trajectory_semantic_index_v1",
        "simulator": "vima_bench",
        "dataset_version": source_revision,
        "source_dataset": "VIMA/VIMA-Data",
        "source_revision": source_revision,
        "provenance_class": "official_demonstration_reorganization_and_label_enrichment",
        "native_task": task,
        "native_task_id": task,
        "scene": f"VIMA:{task}",
        "scene_or_suite": "vima_official_multimodal_prompted_manipulation",
        "parent_episode_id": f"vima:{task}:{trajectory_id}",
        "native_transition_count": transitions,
        "language_instruction_template": prompt_template,
        **semantic,
        "canonical_task": {
            "family": semantic["task_family"],
            "primitive": semantic["task_goal"],
            "entity": semantic["entities"],
            "direction_or_relation": semantic["relations"],
            "semantic_atoms": [*semantic["entities"], *semantic["relations"]],
            "semantic_label_evidence": "reviewed_native_vima_task_mapping_v1",
        },
        "native_payload": {
            "local_parent": parent.relative_to(payload_root).as_posix(),
            "trajectory_id": trajectory_id,
            "transitions": transitions,
            "observation_frames": transitions + 1,
            "temporal_alignment": "T_plus_1_observations_to_T_actions",
            "action_fields": sorted(action_lengths),
            "observation_fields": sorted(obs_lengths),
            "rgb_views": sorted(rgb_frames),
            "rgb_dimensions_hwc": dimensions,
            "segmentation_views": sorted(obs.get("segm", {})),
            "archive_lfs_sha256": archive_lfs_sha256,
            "members": member_receipts or [],
        },
        "parent_success_evidence": {
            "success": True,
            "failure": False,
            "method": "official_trajectory_pickle_success_and_failure_fields",
            "source_file": (parent / "trajectory.pkl").relative_to(payload_root).as_posix(),
        },
        "derived_phase_views": phases,
        "phase_views": phases,
        "provenance": {
            "class": "official_trajectory_reorganization_and_label_enrichment",
            "source": "https://huggingface.co/datasets/VIMA/VIMA-Data",
            "source_revision": source_revision,
            "source_archive_lfs_sha256": archive_lfs_sha256,
            "native_payload_rewritten": False,
            "selection_method": "uniform_midpoint_quantiles_over_sorted_native_trajectory_ids",
        },
        "candidate_g_policy": "absent_not_used_for_selection_or_admission",
        "heldout_policy": "none_inventory_expansion_only",
    }


def inspect_candidates(
    candidates: list[dict[str, Any]],
    *,
    source_revision: str,
    payload_root: Path,
    archive_lfs_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates, start=1):
        try:
            records.append(
                inspect_parent(
                    candidate["parent"],
                    task=candidate["task"],
                    trajectory_id=candidate["trajectory_id"],
                    source_revision=source_revision,
                    payload_root=payload_root,
                    member_receipts=candidate.get("member_receipts"),
                    archive_lfs_sha256=archive_lfs_sha256,
                )
            )
        except ValueError as error:
            exclusions.append(
                {
                    "task": candidate["task"],
                    "trajectory_id": candidate["trajectory_id"],
                    "reason": str(error),
                }
            )
        if index % 500 == 0 or index == len(candidates):
            print(f"[inspect] {index}/{len(candidates)} accepted={len(records)} excluded={len(exclusions)}", flush=True)
    if not records:
        raise ValueError("no valid official VIMA parents remain after native validation")
    tasks_with_records = {record["native_task"] for record in records}
    missing_tasks = sorted({candidate["task"] for candidate in candidates} - tasks_with_records)
    if missing_tasks:
        raise ValueError(f"all candidates were excluded for tasks: {missing_tasks}")
    return records, exclusions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--extraction-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"immutable-style output already exists: {args.output_dir}")

    selection = json.loads(args.selection_manifest.read_text())
    receipt = json.loads(args.extraction_receipt.read_text())
    if receipt.get("status") != "COMPLETE":
        raise ValueError("extraction receipt is not COMPLETE")
    receipt_map = {
        (row["task"], row["trajectory_id"]): row for row in receipt["trajectories"]
    }
    expected = {(row["task"], row["trajectory_id"]) for row in selection["selection"]}
    if set(receipt_map) != expected:
        raise ValueError("extraction receipt does not exactly match frozen selection")

    candidates = []
    for task, trajectory_id in sorted(expected):
        row = receipt_map[(task, trajectory_id)]
        candidates.append(
            {
                "parent": args.payload_root / task / trajectory_id,
                "task": task,
                "trajectory_id": trajectory_id,
                "member_receipts": row["members"],
            }
        )
    records, exclusions = inspect_candidates(
        candidates,
        source_revision=selection["source_revision"],
        payload_root=args.payload_root,
        archive_lfs_sha256=selection["archive_lfs_sha256"],
    )

    parent_ids = [record["parent_episode_id"] for record in records]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("duplicate VIMA parent IDs")
    args.output_dir.mkdir(parents=True)
    records_path = args.output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    exclusions_path = args.output_dir / "excluded_candidates.jsonl"
    with exclusions_path.open("w", encoding="utf-8") as handle:
        for exclusion in exclusions:
            handle.write(json.dumps(exclusion, sort_keys=True) + "\n")
    by_task = Counter(record["native_task"] for record in records)
    by_family = Counter(record["task_family"] for record in records)
    by_scene = Counter(record["scene"] for record in records)
    transitions = sum(record["native_payload"]["transitions"] for record in records)
    views = sum(len(record["derived_phase_views"]) for record in records)
    summary = {
        "schema_version": "vima_official_semantic_index_v1",
        "index_name": args.output_dir.name,
        "provenance_class": "official_demonstration_reorganization_and_label_enrichment",
        "source_dataset": selection["source_dataset"],
        "source_revision": selection["source_revision"],
        "archive_lfs_sha256": selection["archive_lfs_sha256"],
        "candidate_trajectories": len(candidates),
        "excluded_candidates": len(exclusions),
        "parent_trajectories": len(records),
        "native_frames_or_transitions": transitions,
        "native_observation_frames": transitions + len(records),
        "derived_phase_views": views,
        "parents_with_rgb_front_top": len(records),
        "by_simulator": {"vima_bench": len(records)},
        "by_task": dict(sorted(by_task.items())),
        "by_task_family": dict(sorted(by_family.items())),
        "by_family": dict(sorted(by_family.items())),
        "by_scene": dict(sorted(by_scene.items())),
        "by_scene_or_suite": {"vima_bench:vima_official_multimodal_prompted_manipulation": len(records)},
        "records_sha256": _sha256(records_path),
        "records_jsonl_sha256": _sha256(records_path),
        "excluded_candidates_sha256": _sha256(exclusions_path),
        "selection_manifest_sha256": _sha256(args.selection_manifest),
        "extraction_receipt_sha256": _sha256(args.extraction_receipt),
        "new_physical_trajectories": 0,
        "derived_views_are_new_physical_trajectories": False,
        "native_payload_rewritten": False,
        "candidate_g_used_for_admission": False,
        "candidate_g_policy": "absent_not_used_for_selection_or_admission",
        "heldout_policy": "none_inventory_expansion_only",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "distribution.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["native_task", "task_family", "parent_trajectories", "native_transitions"])
        for task in sorted(by_task):
            rows = [record for record in records if record["native_task"] == task]
            writer.writerow([task, rows[0]["task_family"], len(rows), sum(row["native_payload"]["transitions"] for row in rows)])
    report = [
        "# VIMA Official Trajectory Semantic Index",
        "",
        f"- Official successful parents: **{len(records)}**",
        f"- Excluded native candidates: **{len(exclusions)}**",
        f"- Native action transitions: **{transitions}**",
        f"- Native RGB views: front + top for **{len(records)}** parents",
        f"- Derived phase views: **{views}** (not independent demonstrations)",
        "- Candidate G: absent and not used for selection or admission",
        "- Held-out split: none; this is inventory expansion only",
        "- New physical trajectories: **0**",
        "- Live VIMA/PyBullet replay: not performed by this builder",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
