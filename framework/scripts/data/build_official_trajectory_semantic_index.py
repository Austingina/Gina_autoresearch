#!/usr/bin/env python3
"""Build semantic indexes over official successful robot demonstrations.

Native payloads remain untouched. Derived phase slices are views over a complete
parent demonstration and are never treated as independently Oracle-verified
successes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


CALVIN_BLOCK_SLOT_BY_SCENE = {
    "calvin_scene_A": {"pink": 0, "blue": 1, "red": 2},
    "calvin_scene_B": {"red": 0, "blue": 1, "pink": 2},
    "calvin_scene_C": {"blue": 0, "red": 1, "pink": 2},
}


def calvin_task_progress(scene_obs: np.ndarray, task: str, scene_id: str) -> np.ndarray:
    """Return non-negative native-state progress for supported CALVIN tasks."""

    scene = np.asarray(scene_obs, dtype=np.float64)
    if scene.ndim != 2 or scene.shape[1] < 24:
        raise ValueError("CALVIN scene_obs must have shape [T, >=24]")
    if task.startswith("push_") and task.endswith(("_left", "_right")):
        color = task.split("_")[1]
        mapping = CALVIN_BLOCK_SLOT_BY_SCENE.get(scene_id)
        if mapping is None or color not in mapping:
            raise ValueError(f"Unsupported CALVIN scene/object mapping: {scene_id}/{color}")
        sign = -1.0 if task.endswith("_left") else 1.0
        scalar = scene[:, 6 + mapping[color] * 6]
    elif task in {"open_drawer", "close_drawer"}:
        sign = 1.0 if task == "open_drawer" else -1.0
        scalar = scene[:, 1]
    elif task in {"move_slider_left", "move_slider_right"}:
        sign = 1.0 if task == "move_slider_left" else -1.0
        scalar = scene[:, 0]
    elif task in {"push_into_drawer", "place_in_slider"}:
        starts = scene[0, [6, 7, 8, 12, 13, 14, 18, 19, 20]].reshape(3, 3)
        current = scene[:, [6, 7, 8, 12, 13, 14, 18, 19, 20]].reshape(-1, 3, 3)
        return np.max(np.linalg.norm(current - starts[None, :, :], axis=2), axis=1)
    else:
        raise ValueError(f"Unsupported CALVIN task: {task}")
    return np.maximum(0.0, sign * (scalar - scalar[0]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_calvin_index(dataset_root: Path, manifest_path: Path) -> list[dict[str, object]]:
    """Index complete official CALVIN segments without rewriting native frames."""

    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("CALVIN source manifest must declare complete=true")
    rows = manifest.get("segments_detail")
    if not isinstance(rows, list) or not rows:
        raise ValueError("CALVIN source manifest has no segments_detail")
    manifest_hash = _sha256(manifest_path)
    records: list[dict[str, object]] = []
    for row in rows:
        task = str(row["task"])
        start, end = int(row["start"]), int(row["end"])
        if end < start:
            raise ValueError(f"Invalid CALVIN interval: {start}>{end}")
        frame_paths = [dataset_root / "training" / f"episode_{frame_id:07d}.npz" for frame_id in range(start, end + 1)]
        missing = [path.as_posix() for path in frame_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing CALVIN frames: {missing[:3]}")
        scene_rows: list[np.ndarray] = []
        for path in frame_paths:
            with np.load(path, allow_pickle=False) as payload:
                required = {"scene_obs", "robot_obs", "rel_actions"}
                absent = required - set(payload.files)
                if absent:
                    raise ValueError(f"{path} missing native fields: {sorted(absent)}")
                scene_rows.append(np.asarray(payload["scene_obs"], dtype=np.float64))
        progress = calvin_task_progress(np.stack(scene_rows), task, str(row["scene"]))
        phases = partition_progress_phases(progress)
        phase_views = [
            {
                **phase,
                "native_start": start + int(phase["start"]),
                "native_end": start + int(phase["end"]),
                "independent_success": False,
                "success_scope": "inherited_parent_context_only",
            }
            for phase in phases
        ]
        canonical_task = {
            "family": str(row["intent_family"]),
            "primitive": str(row["primitive"]),
            "entity": str(row["entity"]),
            "direction_or_relation": str(row["direction"]),
        }
        records.append(
            {
                "schema_version": "official_trajectory_semantic_index_v1",
                "simulator": "calvin",
                "dataset_version": str(manifest.get("format", "unknown")),
                "parent_episode_id": f"calvin:{task}:{row['scene']}:{start}-{end}",
                "native_task_id": task,
                "scene_or_suite": str(row["scene"]),
                "native_frame_start": start,
                "native_frame_end": end,
                "native_frame_count": end - start + 1,
                "native_payload": {
                    "storage": "per_frame_npz",
                    "root": dataset_root.as_posix(),
                    "pattern": "training/episode_%07d.npz",
                    "observation_fields": ["robot_obs", "scene_obs", "rgb_static", "rgb_gripper"],
                    "action_field": "rel_actions",
                },
                "canonical_task": canonical_task,
                "parent_success_evidence": {
                    "kind": "official_success_annotation",
                    "success": True,
                    "annotation_index": int(row["annotation_index"]),
                    "scope": "complete_parent_trajectory",
                },
                "phase_views": phase_views,
                "provenance": {
                    "class": "official_trajectory_reorganization_and_label_enrichment",
                    "source_url": manifest.get("source_url"),
                    "source_manifest": manifest_path.as_posix(),
                    "source_manifest_sha256": manifest_hash,
                    "native_payload_rewritten": False,
                },
            }
        )
    return records


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    """Summarize physical parents separately from derived phase views."""

    simulator_counts = Counter(str(row["simulator"]) for row in records)
    task_counts = Counter(f"{row['simulator']}:{row['native_task_id']}" for row in records)
    family_counts = Counter(
        f"{row['simulator']}:{row['canonical_task']['family']}" for row in records  # type: ignore[index]
    )
    domain_counts = Counter(f"{row['simulator']}:{row['scene_or_suite']}" for row in records)
    native_count = sum(
        int(row.get("native_frame_count", row.get("native_transition_count", 0))) for row in records
    )
    phase_count = sum(len(row.get("phase_views", [])) for row in records)  # type: ignore[arg-type]
    return {
        "provenance_class": "official_trajectory_reorganization_and_label_enrichment",
        "parent_trajectories": len(records),
        "native_frames_or_transitions": native_count,
        "derived_phase_views": phase_count,
        "derived_views_are_new_physical_trajectories": False,
        "by_simulator": dict(sorted(simulator_counts.items())),
        "by_task": dict(sorted(task_counts.items())),
        "by_family": dict(sorted(family_counts.items())),
        "by_scene_or_suite": dict(sorted(domain_counts.items())),
    }


def write_outputs(
    records: list[dict[str, object]],
    output_dir: Path,
    *,
    unavailable_simulators: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Write immutable-style semantic index, summary, CSV, and plain report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    distribution_path = output_dir / "distribution.csv"
    report_path = output_dir / "REPORT.md"
    with records_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = summarize_records(records)
    summary["unavailable_simulators"] = dict(sorted((unavailable_simulators or {}).items()))
    summary["records_jsonl_sha256"] = _sha256(records_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total = max(len(records), 1)
    with distribution_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dimension", "key", "parent_trajectories", "percent"])
        writer.writeheader()
        for dimension in ("by_simulator", "by_family", "by_scene_or_suite", "by_task"):
            values = summary[dimension]
            for key, count in values.items():  # type: ignore[union-attr]
                writer.writerow(
                    {
                        "dimension": dimension,
                        "key": key,
                        "parent_trajectories": count,
                        "percent": f"{100.0 * int(count) / total:.6f}",
                    }
                )
    lines = [
        "# Official trajectory reorganization + label enrichment",
        "",
        "> No new physical trajectory was generated. Phase views inherit context from complete official successful parents and are not independent successes.",
        "",
        f"- Parent official trajectories: **{summary['parent_trajectories']}**",
        f"- Native frames/transitions: **{summary['native_frames_or_transitions']}**",
        f"- Derived phase views: **{summary['derived_phase_views']}**",
        f"- Simulators indexed: `{json.dumps(summary['by_simulator'], sort_keys=True)}`",
        f"- Unavailable simulators: `{json.dumps(summary['unavailable_simulators'], sort_keys=True)}`",
        "- Candidate G used for admission/distribution/default input: **no**",
        "- Held-out folds created: **no**",
        "",
        "## Phase evidence",
        "",
        "- CALVIN: task-native state-motion progress proxy.",
        "- LIBERO: gripper-transition proxy when present; otherwise action-motion proxy.",
        "- These are label-enrichment signals, not force/contact ground truth.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "records": records_path,
        "summary": summary_path,
        "distribution": distribution_path,
        "report": report_path,
    }


def infer_libero_task_semantics(task_id: str, language_instruction: str | None = None) -> dict[str, object]:
    """Map a native LIBERO task string into coarse, non-lossy semantic tags."""

    text = (language_instruction or task_id).strip().lower().replace("_", " ")
    if text.startswith("pick up") and "place" in text:
        family, primitive = "pick_place", "grasp_transport_place"
    elif text.startswith("open") and (" and " in text or "put" in text):
        family, primitive = "long_horizon_composite", "open_then_place"
    elif text.startswith("open") or "drawer" in text and "close" in text:
        family, primitive = "articulation", "translate_joint"
    elif text.startswith("push"):
        family, primitive = "planar_push", "push"
    elif text.startswith("turn on"):
        family, primitive = "switch_activation", "rotate_or_press"
    elif " and " in text:
        family, primitive = "long_horizon_composite", "multi_object_place"
    else:
        family, primitive = "placement", "transport_place"
    return {
        "family": family,
        "primitive": primitive,
        "language_instruction": text,
        "semantic_atoms": [token for token in text.split() if token not in {"the", "and", "it", "of"}],
    }


def build_libero_index(
    dataset_root: Path, manifest_path: Path, *, verify_hashes: bool = True
) -> list[dict[str, object]]:
    """Index official LIBERO HDF5 demonstrations and verify terminal success."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to index LIBERO HDF5 files") from exc

    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    suites = manifest.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ValueError("LIBERO inventory manifest has no suites")
    manifest_hash = _sha256(manifest_path)
    records: list[dict[str, object]] = []
    for suite_row in suites:
        suite = str(suite_row["suite"])
        file_rows = suite_row.get("records")
        if not isinstance(file_rows, list):
            raise ValueError(f"LIBERO suite {suite} has no file records")
        for file_row in file_rows:
            path = dataset_root / suite / str(file_row["file"])
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_hash = str(file_row["sha256"])
            if verify_hashes and _sha256(path) != expected_hash:
                raise ValueError(f"LIBERO SHA-256 mismatch: {path}")
            task_id = path.name.removesuffix("_demo.hdf5")
            with h5py.File(path, "r") as handle:
                if "data" not in handle:
                    raise ValueError(f"LIBERO file has no data group: {path}")
                data = handle["data"]
                problem_info = json.loads(str(data.attrs.get("problem_info", "{}")))
                env_args = json.loads(str(data.attrs.get("env_args", "{}")))
                language = str(problem_info.get("language_instruction", task_id.replace("_", " ")))
                control_rate = int(env_args.get("env_kwargs", {}).get("control_freq", 0)) or None
                demo_names = sorted(
                    data.keys(),
                    key=lambda name: int(str(name).split("_")[-1]),
                )
                if len(demo_names) != int(file_row["demos"]):
                    raise ValueError(f"LIBERO demo count mismatch: {path}")
                file_transitions = 0
                for demo_name in demo_names:
                    demo = data[demo_name]
                    required = {"actions", "states", "robot_states", "rewards", "dones", "obs"}
                    absent = required - set(demo.keys())
                    if absent:
                        raise ValueError(f"{path}:{demo_name} missing fields {sorted(absent)}")
                    actions = np.asarray(demo["actions"])
                    rewards = np.asarray(demo["rewards"])
                    dones = np.asarray(demo["dones"])
                    n = len(actions)
                    if not (len(rewards) == len(dones) == len(demo["states"]) == len(demo["robot_states"]) == n):
                        raise ValueError(f"LIBERO modality alignment failure: {path}:{demo_name}")
                    terminal_success = bool(np.any(rewards > 0) and np.any(dones > 0))
                    if not terminal_success:
                        raise ValueError(f"Official LIBERO demo lacks terminal success evidence: {path}:{demo_name}")
                    phases = partition_libero_action_phases(actions, rewards)
                    phase_views = [
                        {
                            **phase,
                            "independent_success": False,
                            "success_scope": "inherited_parent_context_only",
                        }
                        for phase in phases
                    ]
                    records.append(
                        {
                            "schema_version": "official_trajectory_semantic_index_v1",
                            "simulator": "libero",
                            "dataset_version": str(manifest.get("source_revision", "unknown")),
                            "parent_episode_id": f"libero:{suite}:{task_id}:{demo_name}",
                            "native_task_id": task_id,
                            "scene_or_suite": suite,
                            "native_transition_count": n,
                            "native_payload": {
                                "storage": "hdf5_group",
                                "file": path.as_posix(),
                                "group": f"data/{demo_name}",
                                "observation_group": "obs",
                                "action_field": "actions",
                                "control_rate_hz": control_rate,
                            },
                            "canonical_task": infer_libero_task_semantics(task_id, language),
                            "parent_success_evidence": {
                                "kind": "official_terminal_reward_and_done",
                                "success": True,
                                "terminal_reward_count": int(np.count_nonzero(rewards > 0)),
                                "terminal_done_count": int(np.count_nonzero(dones > 0)),
                                "scope": "complete_parent_trajectory",
                            },
                            "phase_views": phase_views,
                            "provenance": {
                                "class": "official_trajectory_reorganization_and_label_enrichment",
                                "source": manifest.get("source"),
                                "source_revision": manifest.get("source_revision"),
                                "source_file_sha256": expected_hash,
                                "source_manifest": manifest_path.as_posix(),
                                "source_manifest_sha256": manifest_hash,
                                "native_payload_rewritten": False,
                            },
                        }
                    )
                    file_transitions += n
                if file_transitions != int(file_row["transitions"]):
                    raise ValueError(f"LIBERO transition count mismatch: {path}")
    return records


def partition_libero_action_phases(
    actions: np.ndarray, rewards: np.ndarray | None = None
) -> list[dict[str, object]]:
    """Partition LIBERO actions with explicit non-contact proxy evidence."""

    action_array = np.asarray(actions, dtype=np.float64)
    if action_array.ndim != 2 or action_array.shape[1] < 7 or len(action_array) < 4:
        raise ValueError("LIBERO actions must have shape [T>=4, >=7]")
    n = len(action_array)
    gripper_changes = np.flatnonzero(np.abs(np.diff(action_array[:, -1])) > 1e-6)
    if len(gripper_changes):
        onset = int(gripper_changes[0] + 1)
        evidence = "gripper_transition_proxy_v1"
    else:
        activity = np.linalg.norm(action_array[:, :6], axis=1)
        cumulative = np.cumsum(activity)
        threshold = cumulative[-1] * 0.10
        onset = int(np.searchsorted(cumulative, threshold)) if cumulative[-1] > 0 else n // 4
        evidence = "action_motion_proxy_v1"

    completion = n - 1
    if rewards is not None:
        reward_array = np.asarray(rewards).reshape(-1)
        if len(reward_array) != n:
            raise ValueError("LIBERO rewards must align with actions")
        successes = np.flatnonzero(reward_array > 0)
        if len(successes):
            completion = int(successes[0])
    onset = min(max(onset, 1), n - 3)
    completion = min(max(completion, onset + 2), n - 1)
    return [
        {"phase": "approach", "start": 0, "end": onset - 1, "boundary_evidence": evidence},
        {"phase": "contact_transition_proxy", "start": onset, "end": onset, "boundary_evidence": evidence},
        {
            "phase": "sustained_interaction",
            "start": onset + 1,
            "end": completion - 1,
            "boundary_evidence": evidence,
        },
        {"phase": "completion", "start": completion, "end": n - 1, "boundary_evidence": evidence},
    ]


def partition_progress_phases(
    progress: Iterable[float], *, motion_floor: float = 0.002
) -> list[dict[str, object]]:
    """Partition a successful trajectory using a state-motion progress proxy."""

    values = [max(0.0, float(value)) for value in progress]
    if len(values) < 4:
        raise ValueError("At least four frames are required for four non-empty phase views")
    if motion_floor <= 0:
        raise ValueError("motion_floor must be positive")

    monotonic: list[float] = []
    running = 0.0
    for value in values:
        running = max(running, value)
        monotonic.append(running)
    total = monotonic[-1]
    if total < motion_floor:
        raise ValueError("Successful trajectory has no measurable state progress")

    onset_threshold = max(motion_floor, total * 0.10)
    completion_threshold = total * 0.90
    onset = next((i for i, value in enumerate(monotonic) if value >= onset_threshold), 1)
    onset = min(max(onset, 1), len(values) - 3)
    completion = next(
        (i for i, value in enumerate(monotonic) if i > onset and value >= completion_threshold),
        len(values) - 1,
    )
    completion = min(max(completion, onset + 2), len(values) - 1)

    evidence = "state_motion_proxy_v1"
    return [
        {"phase": "approach", "start": 0, "end": onset - 1, "boundary_evidence": evidence},
        {
            "phase": "contact_transition_proxy",
            "start": onset,
            "end": onset,
            "boundary_evidence": evidence,
        },
        {
            "phase": "sustained_interaction",
            "start": onset + 1,
            "end": completion - 1,
            "boundary_evidence": evidence,
        },
        {
            "phase": "completion",
            "start": completion,
            "end": len(values) - 1,
            "boundary_evidence": evidence,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin-root", type=Path)
    parser.add_argument("--calvin-manifest", type=Path)
    parser.add_argument("--libero-root", type=Path)
    parser.add_argument("--libero-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-libero-hashes", action="store_true")
    parser.add_argument("--unavailable", action="append", default=[], metavar="SIMULATOR=REASON")
    args = parser.parse_args()
    records: list[dict[str, object]] = []
    if bool(args.calvin_root) != bool(args.calvin_manifest):
        parser.error("--calvin-root and --calvin-manifest must be provided together")
    if bool(args.libero_root) != bool(args.libero_manifest):
        parser.error("--libero-root and --libero-manifest must be provided together")
    if args.calvin_root:
        records.extend(build_calvin_index(args.calvin_root, args.calvin_manifest))
    if args.libero_root:
        records.extend(
            build_libero_index(
                args.libero_root,
                args.libero_manifest,
                verify_hashes=not args.skip_libero_hashes,
            )
        )
    if not records:
        parser.error("At least one simulator dataset must be provided")
    unavailable: dict[str, str] = {}
    for value in args.unavailable:
        if "=" not in value:
            parser.error("--unavailable must use SIMULATOR=REASON")
        simulator, reason = value.split("=", 1)
        unavailable[simulator] = reason
    paths = write_outputs(records, args.output, unavailable_simulators=unavailable)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    print(json.dumps({"output": args.output.resolve().as_posix(), "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
