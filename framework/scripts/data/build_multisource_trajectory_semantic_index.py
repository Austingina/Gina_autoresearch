#!/usr/bin/env python3
"""Deprecated audit-only builder for the former mixed-provenance index.

Do not use this output for training, distribution counts, or official-data
expansion. Official trajectories and online native-expert collections now have
separate authoritative indexes. This file is retained only to reproduce the
paused historical artifact and requires an explicit acknowledgement flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_views(length: int) -> list[dict[str, Any]]:
    names = ["approach", "contact_transition_proxy", "sustained_interaction", "completion"]
    boundaries = [round(i * length / 4) for i in range(5)]
    views = []
    for index, name in enumerate(names):
        start = boundaries[index]
        end_exclusive = boundaries[index + 1]
        if end_exclusive <= start:
            end_exclusive = start + 1
        views.append(
            {
                "phase": name,
                "start": start,
                "end": end_exclusive - 1,
                "native_start": start,
                "native_end": end_exclusive - 1,
                "boundary_evidence": "normalized_time_quartile_v1_not_contact_measurement",
                "independent_success": False,
                "success_scope": "inherited_parent_context_only",
            }
        )
    return views


def build_robotwin_record(final_path: Path) -> dict[str, Any]:
    final = json.loads(final_path.read_text(encoding="utf-8"))
    required_true = (
        "new_physical_trajectory",
        "native_oracle_success",
        "quality_passed",
        "independent_hdf5_validation_passed",
        "training_manifest_eligible",
    )
    if final.get("status") != "ACCEPTED" or not all(final.get(key) is True for key in required_true):
        raise ValueError(f"not a fully accepted trajectory: {final_path}")
    artifact = Path(final["artifact"]["path"]).resolve(strict=True)
    if sha256_file(artifact) != final["artifact"]["sha256"]:
        raise ValueError(f"artifact hash mismatch: {artifact}")

    import h5py

    with h5py.File(artifact, "r") as handle:
        lengths = [len(dataset) for dataset in handle["action"].values()]
        if not lengths or len(set(lengths)) != 1:
            raise ValueError(f"unaligned action lengths: {artifact}")
        transition_count = lengths[0]
        cameras = sorted(handle["vision"].keys())
        if transition_count != int(final["transition_count"]):
            raise ValueError(f"transition count mismatch: {artifact}")

    acceptance_sha = sha256_file(final_path)
    seed = int(final["seed"])
    artifact_sha = str(final["artifact"]["sha256"])
    return {
        "schema_version": "multisource_trajectory_semantic_index_v2",
        "simulator": "robotwin",
        "dataset_version": "robotwin2_native_expert_place_empty_cup_v1",
        "parent_episode_id": f"robotwin:place_empty_cup:seed-{seed}:{artifact_sha[:16]}",
        "native_task_id": "place_empty_cup",
        "scene_or_suite": "robotwin2_clean",
        "native_frame_start": 0,
        "native_frame_end": transition_count - 1,
        "native_frame_count": transition_count,
        "canonical_task": {
            "family": "object_placement",
            "primitive": "place",
            "entity": "empty_cup",
            "direction_or_relation": "on_coaster",
        },
        "native_payload": {
            "root": str(artifact.parent.parent),
            "path": str(artifact),
            "storage": "hdf5",
            "action_field": "action/*",
            "observation_fields": ["state/*", "vision/*/colors"],
            "cameras": cameras,
            "sha256": artifact_sha,
        },
        "parent_success_evidence": {
            "kind": "robotwin_native_oracle_plus_quality_plus_independent_readback",
            "scope": "complete_parent_trajectory",
            "success": True,
            "final_acceptance": str(final_path.resolve()),
            "final_acceptance_sha256": acceptance_sha,
            "quality": final["quality"],
            "independent_validation": final["independent_validation"],
        },
        "phase_views": phase_views(transition_count),
        "provenance": {
            "class": "online_native_expert_collection",
            "native_payload_rewritten": False,
            "new_physical_trajectory": True,
            "robotwin_head": final["robotwin_head"],
            "gina_head": final["gina_head"],
            "seed": seed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-index", type=Path, required=True)
    parser.add_argument("--robotwin-final", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-deprecated-merge",
        action="store_true",
        help="Reproduce the paused mixed-provenance artifact for audit only.",
    )
    args = parser.parse_args()
    if not args.acknowledge_deprecated_merge:
        parser.error(
            "mixed-provenance merge is paused; use separate official_reorg and "
            "online_native_expert indexes (or pass --acknowledge-deprecated-merge "
            "only to reproduce historical audit evidence)"
        )

    official_records_path = args.official_index / "records.jsonl"
    official_summary_path = args.official_index / "summary.json"
    official_text = official_records_path.read_text(encoding="utf-8")
    if official_text and not official_text.endswith("\n"):
        official_text += "\n"
    official_records = [json.loads(line) for line in official_text.splitlines() if line]
    official_summary = json.loads(official_summary_path.read_text(encoding="utf-8"))
    robotwin_records = [build_robotwin_record(path.resolve(strict=True)) for path in args.robotwin_final]
    ids = [row["parent_episode_id"] for row in official_records + robotwin_records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate parent_episode_id")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    appended = "".join(json.dumps(row, sort_keys=True) + "\n" for row in robotwin_records)
    records_text = official_text + appended
    (output / "records.jsonl").write_text(records_text, encoding="utf-8")

    all_records = official_records + robotwin_records
    simulator_counts = Counter(row["simulator"] for row in all_records)
    provenance_counts = Counter(row["provenance"]["class"] for row in all_records)
    summary = {
        "schema_version": "multisource_trajectory_semantic_index_v2",
        "parent_trajectory_count": len(all_records),
        "phase_view_count": sum(len(row["phase_views"]) for row in all_records),
        "native_frame_or_transition_count": sum(int(row["native_frame_count"]) for row in all_records),
        "simulator_parent_counts": dict(sorted(simulator_counts.items())),
        "provenance_parent_counts": dict(sorted(provenance_counts.items())),
        "official_base": {
            "path": str(args.official_index.resolve()),
            "records_sha256": sha256_file(official_records_path),
            "summary_sha256": sha256_file(official_summary_path),
            "parent_trajectory_count": len(official_records),
            "declared_summary": official_summary,
        },
        "robotwin_native_expert": {
            "parent_trajectory_count": len(robotwin_records),
            "transition_count": sum(row["native_frame_count"] for row in robotwin_records),
            "new_physical_trajectory_count": len(robotwin_records),
            "task_ids": sorted({row["native_task_id"] for row in robotwin_records}),
        },
        "records_jsonl_sha256": sha256_file(output / "records.jsonl"),
        "provenance_classes_are_not_interchangeable": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
