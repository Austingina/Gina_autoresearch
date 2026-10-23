#!/usr/bin/env python3
"""Audit and index pinned RoboTwin official demo_clean trajectory archives.

Native ZIP/HDF5 payloads are never rewritten. Every HDF5 parent is hash-read,
all state/action streams are checked for alignment and finite values, and every
encoded RGB stream is decoded with RoboTwin's pinned codec. Phase views are
kinematic/gripper proxies, not force/contact ground truth or independent
success evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

PROVENANCE = "official_trajectory_reorganization_and_label_enrichment"
EPISODE_RE = re.compile(r"episode_(\d+)\.hdf5$")
REQUIRED_VECTOR_FIELDS = (
    "action/left_arm_joint_states", "action/left_ee_joint_states", "action/left_ee_poses",
    "action/right_arm_joint_states", "action/right_ee_joint_states", "action/right_ee_poses",
    "state/left_arm_joint_states", "state/left_ee_joint_states", "state/left_ee_poses",
    "state/right_arm_joint_states", "state/right_ee_joint_states", "state/right_ee_poses",
)
REQUIRED_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist", "cam_third_view")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_decoder(robotwin_root: Path) -> Callable[[object], np.ndarray]:
    path = robotwin_root / "data/decode_image_bit.py"
    spec = importlib.util.spec_from_file_location("pinned_robotwin_image_codec", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load RoboTwin codec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_image_bit


def infer_semantics(task: str) -> dict[str, object]:
    tokens = task.split("_")
    verb = tokens[0]
    if verb in {"place", "put", "dump"}:
        family, primitive = "pick_place", "grasp_transport_place"
    elif verb in {"stack"} or task.startswith("blocks_ranking"):
        family, primitive = "multi_object_arrangement", "grasp_transport_arrange"
    elif verb == "handover":
        family, primitive = "handover", "grasp_transport_handover"
    elif verb in {"grab", "pick", "lift"}:
        family, primitive = "object_acquisition", "approach_grasp_lift"
    elif verb == "open":
        family, primitive = "articulation", "contact_articulate"
    elif verb in {"click", "press", "turn"}:
        family, primitive = "switch_activation", "approach_contact_activate"
    elif verb in {"beat", "stamp", "scan"}:
        family, primitive = "tool_or_contact_operation", "approach_contact_operate"
    elif verb in {"move", "adjust", "rotate", "shake", "hanging"}:
        family, primitive = "object_manipulation", "grasp_or_contact_manipulate"
    else:
        family, primitive = "other_manipulation", "native_task"
    direction = "left" if task.endswith("_left") else "right" if task.endswith("_right") else "unspecified_native_relation"
    return {
        "family": family,
        "primitive": primitive,
        "entity": task,
        "direction_or_relation": direction,
        "semantic_atoms": tokens,
        "semantic_label_evidence": "native_task_name_parse_v1",
        "manual_relation_review_required": direction == "unspecified_native_relation",
    }


def _phase(name: str, start: int, end: int, evidence: str) -> dict[str, object]:
    return {
        "phase": name, "start": int(start), "end": int(end),
        "native_start": int(start), "native_end": int(end),
        "boundary_evidence": evidence,
        "independent_success": False,
        "success_scope": "inherited_parent_context_only",
    }


def validate_phase_partition(phases: list[dict[str, object]], n: int) -> None:
    cursor = 0
    for phase in phases:
        start = int(phase["start"])
        end = int(phase["end"])
        if start != cursor or end < start or end >= n:
            raise ValueError(f"invalid phase partition at {phase}: cursor={cursor}, n={n}")
        cursor = end + 1
    if cursor != n:
        raise ValueError(f"invalid phase partition coverage: covered={cursor}, n={n}")


def _motion_quantile_boundaries(poses: list[np.ndarray], n: int) -> list[int]:
    motion = np.zeros(n, dtype=np.float64)
    for pose in poses:
        xyz = np.asarray(pose, dtype=np.float64)[:, :3]
        motion[1:] += np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cumulative = np.cumsum(motion)
    if cumulative[-1] <= 1e-9:
        return [max(1, min(n - 4, round(n * q))) for q in (0.15, 0.30, 0.75, 0.90)]
    raw = [int(np.searchsorted(cumulative, cumulative[-1] * q)) for q in (0.15, 0.30, 0.75, 0.90)]
    bounds: list[int] = []
    for i, value in enumerate(raw):
        low = bounds[-1] + 1 if bounds else 1
        high = n - (4 - i)
        bounds.append(min(max(value, low), high))
    return bounds


def partition_phases(handle: h5py.File, family: str, n: int) -> list[dict[str, object]]:
    poses = [np.asarray(handle[f"action/{arm}_ee_poses"]) for arm in ("left", "right")]
    grippers = [np.asarray(handle[f"action/{arm}_ee_joint_states"]).reshape(n, -1)[:, 0] for arm in ("left", "right")]
    grasp_families = {"pick_place", "multi_object_arrangement", "handover", "object_acquisition", "object_manipulation"}
    if family in grasp_families:
        ranges = [float(np.ptp(values)) for values in grippers]
        arm = int(np.argmax(ranges))
        values = grippers[arm]
        amplitude = ranges[arm]
        if amplitude > 0.25:
            high = float(np.max(values)); low = float(np.min(values))
            close_hi = low + 0.8 * amplitude; close_lo = low + 0.2 * amplitude
            descending = np.flatnonzero((values[:-1] >= close_hi) & (values[1:] < close_hi))
            close_start = int(descending[0] + 1) if len(descending) else max(1, int(np.argmin(values)) - 1)
            closed_candidates = np.flatnonzero((np.arange(n) >= close_start) & (values <= close_lo))
            closed = int(closed_candidates[0]) if len(closed_candidates) else min(close_start + 1, n - 3)
            rising = np.flatnonzero((np.arange(n - 1) >= closed) & (values[1:] > values[:-1] + 1e-6))
            if len(rising):
                release_start = int(rising[0] + 1)
                opened_candidates = np.flatnonzero((np.arange(n) >= release_start) & (values >= close_hi))
                opened = int(opened_candidates[0]) if len(opened_candidates) else min(release_start + 1, n - 1)
                # Every emitted phase must contain at least one native transition.
                # An immediate reopen can make the transport interval empty, and an
                # opening on the final step can make completion empty; in either case
                # fall through to the coarser nonempty partition below.
                if 1 <= close_start <= closed and closed + 2 <= release_start <= opened < n - 1:
                    return [
                        _phase("approach", 0, close_start - 1, "gripper_transition_proxy_v1"),
                        _phase("grasp_or_contact_proxy", close_start, closed, "gripper_transition_proxy_v1"),
                        _phase("transport_or_manipulation", closed + 1, release_start - 1, "gripper_transition_proxy_v1"),
                        _phase("place_or_release_proxy", release_start, opened, "gripper_transition_proxy_v1"),
                        _phase("completion", opened + 1, n - 1, "gripper_transition_proxy_v1"),
                    ]
            completion = max(closed + 1, n - max(1, n // 10))
            if 1 <= close_start <= closed and closed + 2 <= completion < n:
                return [
                    _phase("approach", 0, close_start - 1, "gripper_transition_proxy_v1"),
                    _phase("grasp_or_contact_proxy", close_start, closed, "gripper_transition_proxy_v1"),
                    _phase("transport_or_manipulation", closed + 1, completion - 1, "gripper_transition_proxy_v1"),
                    _phase("completion", completion, n - 1, "gripper_transition_proxy_v1"),
                ]
    b1, b2, b3, b4 = _motion_quantile_boundaries(poses, n)
    return [
        _phase("approach", 0, b1 - 1, "ee_motion_quantile_proxy_v1"),
        _phase("contact_transition_proxy", b1, b2 - 1, "ee_motion_quantile_proxy_v1"),
        _phase("sustained_interaction", b2, b3 - 1, "ee_motion_quantile_proxy_v1"),
        _phase("disengage_or_release_proxy", b3, b4 - 1, "ee_motion_quantile_proxy_v1"),
        _phase("completion", b4, n - 1, "ee_motion_quantile_proxy_v1"),
    ]


def inspect_episode(data: bytes, decoder: Callable[[object], np.ndarray]) -> dict[str, object]:
    with h5py.File(io.BytesIO(data), "r") as handle:
        absent = [name for name in REQUIRED_VECTOR_FIELDS if name not in handle]
        if absent:
            raise ValueError(f"missing native state/action fields: {absent}")
        n = int(handle[REQUIRED_VECTOR_FIELDS[0]].shape[0])
        if n < 5:
            raise ValueError(f"trajectory too short for phase views: {n}")
        for name in REQUIRED_VECTOR_FIELDS:
            values = np.asarray(handle[name])
            if len(values) != n or not np.isfinite(values).all():
                raise ValueError(f"unaligned or non-finite field: {name}")
        frequency = int(np.asarray(handle["additional_info/frequency"]).item())
        if frequency <= 0:
            raise ValueError(f"invalid frequency: {frequency}")
        camera_receipts = []
        for camera in REQUIRED_CAMERAS:
            name = f"vision/{camera}/colors"
            if name not in handle:
                raise ValueError(f"missing encoded RGB stream: {name}")
            decoded = np.asarray(decoder(handle[name][()]))
            if decoded.ndim != 4 or decoded.shape[0] != n or decoded.shape[-1] != 3 or decoded.dtype != np.uint8:
                raise ValueError(f"invalid decoded RGB stream {name}: {decoded.shape}/{decoded.dtype}")
            camera_receipts.append({
                "camera": camera, "frames": n, "shape": list(decoded.shape[1:]),
                "dtype": str(decoded.dtype), "min": int(decoded.min()), "max": int(decoded.max()),
                "std": float(decoded.std()),
            })
        attrs = {str(k): str(v) for k, v in handle.attrs.items()}
        schema = {
            "transitions": n, "frequency_hz": frequency, "cameras": camera_receipts,
            "attrs": attrs, "state_fields": list(REQUIRED_VECTOR_FIELDS[6:]),
            "action_fields": list(REQUIRED_VECTOR_FIELDS[:6]),
            "instruction": str(np.asarray(handle["instructions"]).item()),
            "data_format_version": str(np.asarray(handle["data_format_version"]).item()),
        }
        return schema


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-archives", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    decoder = load_decoder(args.robotwin_root.resolve())
    archive_rows = list(inventory["files"])
    if args.max_archives is not None:
        archive_rows = archive_rows[: args.max_archives]
    records: list[dict[str, object]] = []
    archive_receipts = []
    total_rgb_frames = 0
    for archive_number, archive_row in enumerate(archive_rows, 1):
        task = str(archive_row["task"])
        archive = args.payload_root / str(archive_row["path"])
        if not archive.is_file():
            raise FileNotFoundError(archive)
        expected_sha = str(archive_row["lfs"]["oid"])
        actual_sha = sha256_file(archive)
        if archive.stat().st_size != int(archive_row["bytes"]) or actual_sha != expected_sha:
            raise ValueError(f"archive integrity mismatch: {archive}")
        semantics = infer_semantics(task)
        episode_count = 0
        archive_transitions = 0
        with zipfile.ZipFile(archive) as bundle:
            prefix = f"{task}/aloha_agilex/"
            seed_member = prefix + "seed.txt"
            scene_member = prefix + "scene_info.json"
            if seed_member not in bundle.namelist() or scene_member not in bundle.namelist():
                raise ValueError(f"archive lacks seed/scene metadata: {archive}")
            seeds = [int(value) for value in bundle.read(seed_member).decode("utf-8").split()]
            scenes = json.loads(bundle.read(scene_member))
            members = sorted(
                (name for name in bundle.namelist() if name.startswith(prefix + "data/") and EPISODE_RE.search(name)),
                key=lambda name: int(EPISODE_RE.search(name).group(1)),  # type: ignore[union-attr]
            )
            if not members or len(seeds) != len(members) or len(scenes) != len(members):
                raise ValueError(f"episode/seed/scene count mismatch for {task}: {len(members)}/{len(seeds)}/{len(scenes)}")
            for member in members:
                episode_id = int(EPISODE_RE.search(member).group(1))  # type: ignore[union-attr]
                payload = bundle.read(member)
                schema = inspect_episode(payload, decoder)
                n = int(schema["transitions"])
                with h5py.File(io.BytesIO(payload), "r") as handle:
                    phases = partition_phases(handle, str(semantics["family"]), n)
                validate_phase_partition(phases, n)
                scene_key = f"episode_{episode_id}"
                if scene_key not in scenes:
                    raise ValueError(f"missing scene metadata {task}:{scene_key}")
                total_rgb_frames += n * len(REQUIRED_CAMERAS)
                archive_transitions += n
                records.append({
                    "schema_version": "official_trajectory_semantic_index_v1",
                    "simulator": "robotwin",
                    "dataset_version": str(inventory["source_revision"]),
                    "parent_episode_id": f"robotwin:{task}:episode_{episode_id:07d}",
                    "native_task_id": task,
                    "scene_or_suite": "robotwin2_demo_clean",
                    "native_transition_count": n,
                    "native_payload": {
                        "storage": "zip_member_hdf5", "archive": archive.resolve().as_posix(),
                        "archive_sha256": actual_sha, "member": member,
                        "member_sha256": sha256_bytes(payload), "member_bytes": len(payload),
                        "state_fields": schema["state_fields"], "action_fields": schema["action_fields"],
                        "camera_fields": [f"vision/{camera}/colors" for camera in REQUIRED_CAMERAS],
                        "control_rate_hz": schema["frequency_hz"],
                    },
                    "native_scene_info": scenes[scene_key],
                    "native_seed": seeds[episode_id],
                    "canonical_task": semantics,
                    "parent_success_evidence": {
                        "kind": "official_native_oracle_and_replay_assert_then_demo_clean_release",
                        "success": True,
                        "scope": "complete_parent_trajectory",
                        "native_oracle_flag_in_payload": False,
                        "evidence_strength": "official_collection_code_requires_plan_success_check_success_joint_legality_and_replay_check_success",
                        "collection_code_evidence": [
                            "scripts/collect_data.py:179-190",
                            "scripts/collect_data.py:278-292"
                        ],
                        "robotwin_code_revision": "ea8b21121ebb3cd201ff5b3fe361944ac94eda3f",
                        "source_repo": inventory["source"],
                        "source_revision": inventory["source_revision"],
                        "source_archive_path": archive_row["path"],
                    },
                    "phase_views": phases,
                    "readback": schema,
                    "provenance": {
                        "class": PROVENANCE, "native_payload_rewritten": False,
                        "source_manifest": args.inventory.resolve().as_posix(),
                        "source_manifest_sha256": sha256_file(args.inventory),
                        "source_archive_sha256": actual_sha,
                    },
                })
                episode_count += 1
        archive_receipts.append({
            "task": task, "archive": archive.as_posix(), "sha256": actual_sha,
            "bytes": archive.stat().st_size, "parent_trajectories": episode_count,
            "native_transitions": archive_transitions, "status": "PASS",
        })
        print(f"[{archive_number}/{len(archive_rows)}] {task}: parents={episode_count} transitions={archive_transitions}", flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    records_path = args.output / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    by_task = Counter(str(row["native_task_id"]) for row in records)
    by_family = Counter(str(row["canonical_task"]["family"]) for row in records)  # type: ignore[index]
    summary = {
        "schema_version": "robotwin_official_index_summary_v1", "provenance_class": PROVENANCE,
        "source_revision": inventory["source_revision"], "archive_count": len(archive_receipts),
        "parent_trajectories": len(records),
        "native_transitions": sum(int(row["native_transition_count"]) for row in records),
        "decoded_rgb_frames": total_rgb_frames,
        "derived_phase_views": sum(len(row["phase_views"]) for row in records),  # type: ignore[arg-type]
        "derived_views_are_new_physical_trajectories": False,
        "by_task": dict(sorted(by_task.items())), "by_family": dict(sorted(by_family.items())),
        "records_jsonl_sha256": sha256_file(records_path), "status": "PASS",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output / "readback_receipt.json").write_text(json.dumps({
        "status": "PASS", "all_archives_hash_verified": True,
        "all_state_action_payloads_finite_and_aligned": True,
        "all_encoded_rgb_streams_decoded": True,
        "per_episode_native_oracle_flag_present": False,
        "official_success_evidence": "official native check_success + planned-joint legality + replay check_success, followed by demo_clean release selection",
        "archives": archive_receipts,
    }, indent=2, sort_keys=True) + "\n")
    with (args.output / "distribution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["dimension", "key", "parent_trajectories"])
        for key, value in sorted(by_family.items()): writer.writerow(["family", key, value])
        for key, value in sorted(by_task.items()): writer.writerow(["task", key, value])
    (args.output / "REPORT.md").write_text(
        "# RoboTwin official trajectory reorganization + label enrichment\n\n"
        f"- Official archives: **{summary['archive_count']}**\n"
        f"- Complete parent trajectories: **{summary['parent_trajectories']}**\n"
        f"- Native transitions: **{summary['native_transitions']}**\n"
        f"- Decoded RGB frames: **{summary['decoded_rgb_frames']}**\n"
        f"- Derived phase views: **{summary['derived_phase_views']}**\n"
        "- New physical trajectories generated: **no**\n"
        "- Native action/state vectors mixed with other simulators: **no**\n"
        "- Success evidence: official collection code requires `plan_success`, native `check_success()`, legal planned joints, and a replay-time `check_success()` assertion before `demo_clean` release. No per-episode Oracle flag is embedded in HDF5.\n"
        "- Phase evidence: gripper-transition or end-effector-motion proxy; not force/contact ground truth.\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
