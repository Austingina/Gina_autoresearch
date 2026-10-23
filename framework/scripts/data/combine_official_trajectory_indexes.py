#!/usr/bin/env python3
"""Combine separately verified official-only indexes into one semantic view.

This never rewrites or concatenates native state/action vectors. It refuses any
record outside the official trajectory reorganization provenance class.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

PROVENANCE = "official_trajectory_reorganization_and_label_enrichment"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-index", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    records: list[dict] = []
    sources = []
    parent_ids: set[str] = set()
    for index in args.input_index:
        records_path = index / "records.jsonl"
        summary_path = index / "summary.json"
        if not records_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"incomplete source index: {index}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records_hash = sha256_file(records_path)
        if summary.get("records_jsonl_sha256") != records_hash:
            raise ValueError(f"source index hash mismatch: {index}")
        source_count = 0
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            record = json.loads(line)
            provenance = record.get("provenance", {})
            if provenance.get("class") != PROVENANCE:
                raise ValueError(f"non-official provenance in {index}: {provenance.get('class')}")
            parent_id = str(record["parent_episode_id"])
            if parent_id in parent_ids:
                raise ValueError(f"duplicate parent episode id: {parent_id}")
            parent_ids.add(parent_id)
            records.append(record)
            source_count += 1
        if source_count != int(summary["parent_trajectories"]):
            raise ValueError(f"source record count mismatch: {index}")
        sources.append({
            "index": index.resolve().as_posix(),
            "records_sha256": records_hash,
            "summary_sha256": sha256_file(summary_path),
            "parent_trajectories": source_count,
        })

    args.output.mkdir(parents=True, exist_ok=False)
    records_path = args.output / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    by_simulator = Counter(str(row["simulator"]) for row in records)
    by_task = Counter(f"{row['simulator']}:{row['native_task_id']}" for row in records)
    by_family = Counter(f"{row['simulator']}:{row['canonical_task']['family']}" for row in records)
    by_domain = Counter(f"{row['simulator']}:{row['scene_or_suite']}" for row in records)
    summary = {
        "schema_version": "official_trajectory_unified_semantic_view_v1",
        "provenance_class": PROVENANCE,
        "index_mode": "unified_semantic_view_not_native_vector_merge",
        "parent_trajectories": len(records),
        "native_frames_or_transitions": sum(int(row.get("native_frame_count", row.get("native_transition_count", 0))) for row in records),
        "derived_phase_views": sum(len(row.get("phase_views", [])) for row in records),
        "derived_views_are_new_physical_trajectories": False,
        "native_action_state_spaces_remain_separate": True,
        "candidate_g_used_for_admission": False,
        "by_simulator": dict(sorted(by_simulator.items())),
        "by_task": dict(sorted(by_task.items())),
        "by_family": dict(sorted(by_family.items())),
        "by_scene_or_suite": dict(sorted(by_domain.items())),
        "source_indexes": sources,
        "records_jsonl_sha256": sha256_file(records_path),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.output / "distribution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["dimension", "key", "parent_trajectories"])
        for dimension, values in (("simulator", by_simulator), ("family", by_family), ("task", by_task), ("scene_or_suite", by_domain)):
            for key, value in sorted(values.items()):
                writer.writerow([dimension, key, value])
    (args.output / "REPORT.md").write_text(
        "# Unified official trajectory semantic view\n\n"
        "> This is an official-only semantic index, not a native action/state vector merge.\n\n"
        f"- Parent official trajectories: **{len(records)}**\n"
        f"- Native frames/transitions: **{summary['native_frames_or_transitions']}**\n"
        f"- Derived phase views: **{summary['derived_phase_views']}**\n"
        f"- By simulator: `{json.dumps(summary['by_simulator'], sort_keys=True)}`\n"
        "- New physical trajectories generated: **no**\n"
        "- Candidate G used for admission: **no**\n"
        "- Native image/state/action/frequency and success evidence remain simulator-specific in every parent record.\n"
        "- CALVIN phases: task-native state-motion proxies.\n"
        "- LIBERO phases: gripper/action-motion proxies.\n"
        "- RoboTwin phases: gripper/end-effector-motion proxies.\n"
        "- ManiSkill phases: native action-motion quantile proxies.\n"
        "- VIMA phases: temporal proxies over native oracle actions; official front/top RGB remains native payload.\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
