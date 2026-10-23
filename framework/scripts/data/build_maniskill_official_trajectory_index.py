#!/usr/bin/env python3
"""Build a semantic sidecar over pinned official ManiSkill demonstration ZIPs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import h5py
import numpy as np

PROVENANCE = "official_trajectory_reorganization_and_label_enrichment"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_semantics(task: str) -> dict[str, object]:
    mapping = {
        "LiftPegUpright-v1": ("orientation_manipulation", "grasp_reorient_upright"),
        "PegInsertionSide-v1": ("insertion_assembly", "grasp_align_insert"),
        "PlugCharger-v1": ("insertion_assembly", "grasp_align_insert"),
        "PokeCube-v1": ("tool_or_contact_operation", "tool_contact_push"),
        "PullCubeTool-v1": ("tool_or_contact_operation", "tool_hook_pull"),
        "PushT-v1": ("planar_push", "nonprehensile_planar_push"),
        "RollBall-v1": ("rolling_contact", "nonprehensile_roll"),
        "DrawTriangle-v1": ("tool_or_contact_operation", "sustained_contact_trace"),
        "StackPyramid-v1": ("multi_object_arrangement", "multi_stage_stack"),
    }
    if task not in mapping:
        raise ValueError(f"unreviewed ManiSkill task semantics: {task}")
    family, primitive = mapping[task]
    atoms = task.removesuffix("-v1").replace("-", "_")
    return {
        "family": family,
        "primitive": primitive,
        "entity": atoms,
        "direction_or_relation": "native_task_relation",
        "semantic_atoms": [atoms],
        "semantic_label_evidence": "reviewed_native_task_id_mapping_v1",
    }


def _phase(name: str, start: int, end: int, evidence: str) -> dict[str, object]:
    return {
        "phase": name,
        "start": int(start),
        "end": int(end),
        "native_start": int(start),
        "native_end": int(end),
        "boundary_evidence": evidence,
        "independent_success": False,
        "success_scope": "inherited_parent_context_only",
    }


def validate_phase_partition(phases: list[dict[str, object]], n: int) -> None:
    cursor = 0
    for phase in phases:
        start, end = int(phase["start"]), int(phase["end"])
        if start != cursor or end < start or end >= n:
            raise ValueError(f"invalid phase partition at {phase}: cursor={cursor}, n={n}")
        cursor = end + 1
    if cursor != n:
        raise ValueError(f"invalid phase partition coverage: covered={cursor}, n={n}")


def partition_action_phases(actions: np.ndarray) -> list[dict[str, object]]:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("ManiSkill actions must be finite [T>=4, A]")
    n = len(values)
    motion = np.linalg.norm(values, axis=1)
    cumulative = np.cumsum(motion)
    if cumulative[-1] > 1e-12:
        raw = [int(np.searchsorted(cumulative, cumulative[-1] * q)) for q in (0.15, 0.75, 0.90)]
    else:
        raw = [round(n * q) for q in (0.25, 0.65, 0.85)]
    bounds: list[int] = []
    for i, value in enumerate(raw):
        low = bounds[-1] + 1 if bounds else 1
        high = n - (3 - i)
        bounds.append(min(max(value, low), high))
    b1, b2, b3 = bounds
    phases = [
        _phase("approach", 0, b1 - 1, "action_motion_quantile_proxy_v1"),
        _phase("contact_or_grasp_transition_proxy", b1, b2 - 1, "action_motion_quantile_proxy_v1"),
        _phase("sustained_interaction", b2, b3 - 1, "action_motion_quantile_proxy_v1"),
        _phase("completion", b3, n - 1, "action_motion_quantile_proxy_v1"),
    ]
    validate_phase_partition(phases, n)
    return phases


def _leaf_lengths(group: h5py.Group) -> dict[str, int]:
    out: dict[str, int] = {}
    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            if obj.ndim < 1:
                raise ValueError(f"scalar state leaf is not temporally aligned: {name}")
            out[name] = int(obj.shape[0])
    group.visititems(visit)
    return out


def _metadata_success(episode: dict[str, object]) -> bool:
    direct = episode.get("success")
    info = episode.get("info")
    info_success = info.get("success") if isinstance(info, dict) else None
    return bool(direct) or bool(info_success)


def parent_episode_id(task: str, hdf5_member: str, group_name: str) -> str:
    native_key = str(PurePosixPath(hdf5_member).with_suffix(""))
    return f"maniskill:{task}:{native_key}:{group_name}"


def _safe_members(bundle: zipfile.ZipFile) -> list[str]:
    names = bundle.namelist()
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe ZIP member path: {name}")
    return names


def inspect_archive(archive: Path, inventory_row: dict[str, object], *, source_revision: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    archive = Path(archive).resolve()
    expected_size = int(inventory_row["bytes"])
    expected_sha = str(inventory_row["sha256"])
    actual_sha = sha256_file(archive)
    if archive.stat().st_size != expected_size or actual_sha != expected_sha:
        raise ValueError(f"archive integrity mismatch: {archive}")
    task = str(inventory_row["task"])
    semantics = infer_semantics(task)
    records: list[dict[str, object]] = []
    native_episode_candidates = 0
    excluded_without_success_evidence = 0
    with zipfile.ZipFile(archive) as bundle:
        names = _safe_members(bundle)
        h5_members = sorted(name for name in names if name.endswith((".h5", ".hdf5")))
        if not h5_members:
            raise ValueError(f"no HDF5 payload in {archive}")
        for h5_member in h5_members:
            json_member = str(PurePosixPath(h5_member).with_suffix(".json"))
            if json_member not in names:
                raise ValueError(f"missing JSON metadata pair for {h5_member}")
            metadata = json.loads(bundle.read(json_member))
            env_info = metadata.get("env_info")
            episodes = metadata.get("episodes")
            if not isinstance(env_info, dict) or str(env_info.get("env_id")) != task:
                raise ValueError(f"metadata env_id mismatch: {json_member}")
            if not isinstance(episodes, list) or not episodes:
                raise ValueError(f"metadata has no episodes: {json_member}")
            episode_by_id = {int(row["episode_id"]): row for row in episodes}
            payload = bundle.read(h5_member)
            payload_sha = hashlib.sha256(payload).hexdigest()
            with h5py.File(io.BytesIO(payload), "r") as handle:
                groups = sorted(handle.keys(), key=lambda name: int(name.removeprefix("traj_")))
                if len(groups) != len(episodes):
                    raise ValueError(f"episode/HDF5 group count mismatch: {h5_member}")
                for group_name in groups:
                    if not group_name.startswith("traj_"):
                        raise ValueError(f"unexpected HDF5 root group: {group_name}")
                    episode_id = int(group_name.removeprefix("traj_"))
                    if episode_id not in episode_by_id:
                        raise ValueError(f"missing episode metadata: {task}/{episode_id}")
                    episode = episode_by_id[episode_id]
                    native_episode_candidates += 1
                    traj = handle[group_name]
                    required = {"actions", "terminated", "truncated", "env_states"}
                    absent = required - set(traj.keys())
                    if absent:
                        raise ValueError(f"{h5_member}:{group_name} missing fields {sorted(absent)}")
                    actions = np.asarray(traj["actions"])
                    n = len(actions)
                    if n < 4 or actions.ndim != 2 or not np.isfinite(actions).all():
                        raise ValueError(f"invalid actions: {h5_member}:{group_name}")
                    if len(traj["terminated"]) != n or len(traj["truncated"]) != n:
                        raise ValueError(f"T-alignment failure: {h5_member}:{group_name}")
                    state_lengths = _leaf_lengths(traj["env_states"])
                    if not state_lengths or set(state_lengths.values()) != {n + 1}:
                        raise ValueError(f"T+1 env-state alignment failure: {h5_member}:{group_name}: {state_lengths}")
                    stream_success = bool(np.any(np.asarray(traj["success"]))) if "success" in traj else False
                    metadata_success = _metadata_success(episode)
                    if not (metadata_success or stream_success):
                        excluded_without_success_evidence += 1
                        continue
                    if int(episode.get("elapsed_steps", n)) != n:
                        raise ValueError(f"elapsed_steps/action mismatch: {h5_member}:{group_name}")
                    phases = partition_action_phases(actions)
                    record = {
                        "schema_version": "official_trajectory_semantic_index_v1",
                        "simulator": "maniskill",
                        "dataset_version": source_revision,
                        "parent_episode_id": parent_episode_id(task, h5_member, group_name),
                        "native_task_id": task,
                        "scene_or_suite": "maniskill_official_motionplanning",
                        "native_transition_count": n,
                        "native_payload": {
                            "storage": "zip_member_hdf5_group",
                            "archive": archive.as_posix(),
                            "archive_sha256": actual_sha,
                            "hdf5_member": h5_member,
                            "hdf5_member_sha256": payload_sha,
                            "group": group_name,
                            "json_member": json_member,
                            "action_field": "actions",
                            "state_group": "env_states",
                            "state_leaf_count": len(state_lengths),
                            "temporal_alignment": "T_plus_1_env_states_to_T_actions",
                            "observation_group_present": "obs" in traj,
                            "obs_mode": env_info.get("env_kwargs", {}).get("obs_mode"),
                            "env_kwargs": env_info.get("env_kwargs", {}),
                            "control_mode": episode.get("control_mode") or env_info.get("env_kwargs", {}).get("control_mode"),
                            "sim_backend": env_info.get("env_kwargs", {}).get("sim_backend"),
                        },
                        "native_reset_kwargs": episode.get("reset_kwargs", {}),
                        "canonical_task": semantics,
                        "parent_success_evidence": {
                            "kind": "official_episode_metadata_or_native_success_stream",
                            "success": True,
                            "metadata_success": metadata_success,
                            "native_success_stream_present": "success" in traj,
                            "native_success_stream_any": stream_success,
                            "scope": "complete_parent_trajectory",
                        },
                        "phase_views": phases,
                        "provenance": {
                            "class": PROVENANCE,
                            "source": "https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations",
                            "source_revision": source_revision,
                            "source_archive_path": inventory_row["path"],
                            "source_archive_sha256": actual_sha,
                            "native_payload_rewritten": False,
                        },
                    }
                    records.append(record)
    if not records:
        raise ValueError(f"official archive has no parent with success evidence: {archive}")
    receipt = {
        "task": task,
        "archive": archive.as_posix(),
        "archive_sha256": actual_sha,
        "native_episode_candidates": native_episode_candidates,
        "excluded_without_success_evidence": excluded_without_success_evidence,
        "parents": len(records),
        "transitions": sum(int(row["native_transition_count"]) for row in records),
        "hdf5_payloads": len({row["native_payload"]["hdf5_member"] for row in records}),
        "parents_with_observations": sum(bool(row["native_payload"]["observation_group_present"]) for row in records),
    }
    return records, receipt


def write_index(records: list[dict[str, object]], receipts: list[dict[str, object]], output: Path, inventory: dict[str, object], inventory_path: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    records_path = output / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    by_task = Counter(str(row["native_task_id"]) for row in records)
    by_family = Counter(str(row["canonical_task"]["family"]) for row in records)
    summary = {
        "schema_version": "maniskill_official_semantic_index_v1",
        "provenance_class": PROVENANCE,
        "source_revision": inventory["source_revision"],
        "parent_trajectories": len(records),
        "native_frames_or_transitions": sum(int(row["native_transition_count"]) for row in records),
        "derived_phase_views": sum(len(row["phase_views"]) for row in records),
        "derived_views_are_new_physical_trajectories": False,
        "native_payload_rewritten": False,
        "candidate_g_used_for_admission": False,
        "by_simulator": {"maniskill": len(records)},
        "by_task": dict(sorted(by_task.items())),
        "by_family": dict(sorted(by_family.items())),
        "by_scene_or_suite": {"maniskill:maniskill_official_motionplanning": len(records)},
        "archives": receipts,
        "source_inventory": inventory_path.resolve().as_posix(),
        "source_inventory_sha256": sha256_file(inventory_path),
        "records_jsonl_sha256": sha256_file(records_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (output / "distribution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["dimension", "key", "parent_trajectories"])
        for dimension, values in (("task", by_task), ("family", by_family)):
            for key, count in sorted(values.items()): writer.writerow([dimension, key, count])
    (output / "REPORT.md").write_text(
        "# ManiSkill official trajectory reorganization + label enrichment\n\n"
        "> No new simulator trajectory was generated. Native ZIP/HDF5 payloads remain unchanged.\n\n"
        f"- Official parents: **{len(records)}**\n"
        f"- Native transitions: **{summary['native_frames_or_transitions']}**\n"
        f"- Derived phase views: **{summary['derived_phase_views']}** (not independent successes)\n"
        f"- Tasks: **{len(by_task)}**\n"
        "- Native images: preserved when present; absent compressed demos remain explicitly absent.\n"
        "- Candidate G used for admission: **no**\n"
        "- Phase evidence: action-motion proxy, not force/contact ground truth.\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    records: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    for row in inventory["files"]:
        archive = args.payload_root / "archives" / Path(str(row["path"])).name
        archive_records, receipt = inspect_archive(archive, row, source_revision=str(inventory["source_revision"]))
        records.extend(archive_records); receipts.append(receipt)
    if len({str(row["parent_episode_id"]) for row in records}) != len(records):
        raise ValueError("duplicate parent episode IDs")
    summary = write_index(records, receipts, args.output, inventory, args.inventory)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
