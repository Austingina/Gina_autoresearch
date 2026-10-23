#!/usr/bin/env python3
"""Parameterized closed-loop CALVIN block-push expert and compositional collector."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

COLORS = ("red", "blue", "pink")
DIRECTIONS = ("left", "right")
ALL_DIRECTIONS = ("left", "right", "forward", "backward")
DIRECTION_FLOWS = {
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "forward": (0.0, 1.0, 0.0),
    "backward": (0.0, -1.0, 0.0),
}
PUSH_TASKS = tuple(f"push_{color}_block_{direction}" for color in COLORS for direction in DIRECTIONS)
ALL_PUSH_TASKS = tuple(f"push_{color}_block_{direction}" for color in COLORS for direction in ALL_DIRECTIONS)


def parse_push_task(task: str) -> dict[str, Any]:
    if task not in ALL_PUSH_TASKS:
        raise KeyError(f"Unsupported push task: {task}")
    parts = task.split("_")
    color, direction = parts[1], parts[-1]
    return {
        "task": task,
        "color": color,
        "direction": direction,
        "object_name": f"block_{color}",
        "flow": np.asarray(DIRECTION_FLOWS[direction], dtype=np.float32),
    }


def wrap_angle_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    return (np.asarray(target) - np.asarray(current) + np.pi) % (2 * np.pi) - np.pi


class ParametricPushExpert:
    """Named-object waypoint state machine with live pose feedback."""

    def __init__(
        self,
        task: str,
        initial_object_pos: np.ndarray,
        *,
        side_offset: float = 0.047,
        contact_height: float = 0.012,
        hover_height: float = 0.080,
        push_travel: float | None = None,
        position_tolerance: float = 0.009,
        orientation_tolerance: float = 0.15,
        stable_steps: int = 3,
    ) -> None:
        self.spec = parse_push_task(task)
        self.task = task
        self.initial_object_pos = np.asarray(initial_object_pos, dtype=np.float32).copy()
        if self.initial_object_pos.shape != (3,):
            raise ValueError("initial_object_pos must have shape (3,)")
        flow = self.spec["flow"]
        self.is_world_y = abs(float(flow[1])) > 0.5
        self.side_offset = float(side_offset)
        self.contact_height = 0.004 if self.is_world_y and contact_height == 0.012 else float(contact_height)
        self.permanent_preemptive_hold = False
        if self.is_world_y and self.spec["direction"] == "forward":
            if self.spec["color"] == "pink":
                self.y_push_standoff = 0.005
                self.y_post_hold_standoff = 0.015
                self.settle_progress = 0.090
            else:
                self.y_push_standoff = 0.018
                self.y_post_hold_standoff = 0.035
                self.settle_progress = 0.095
                self.permanent_preemptive_hold = True
        else:
            self.y_push_standoff = 0.018
            self.y_post_hold_standoff = 0.018
            self.settle_progress = 0.102 if self.is_world_y else float("inf")
        self.y_lateral_gain = 0.8
        self.y_lateral_limit = 0.035
        self.preemptive_hold_steps = 4
        self._preemptive_hold_count = 0
        self._preemptive_hold_done = self.spec["direction"] != "forward"
        self.orientation = np.asarray([3.10, -0.13, 1.50], dtype=np.float32)
        self.gripper_action = -1.0
        if push_travel is None:
            push_travel = 0.125
        contact = self.initial_object_pos - flow * np.float32(side_offset)
        contact[2] += np.float32(self.contact_height)
        hover = contact.copy(); hover[2] += np.float32(hover_height)
        push = contact + flow * np.float32(push_travel)
        self.waypoints = {"hover": hover, "contact": contact, "push": push}
        self.state = "hover"
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.stable_steps = int(stable_steps)
        self._stable_count = 0

    def should_settle(self, progress: float) -> bool:
        if not self.is_world_y:
            return False
        if float(progress) >= 0.102:
            return True
        if self.permanent_preemptive_hold:
            return float(progress) >= self.settle_progress
        if self._preemptive_hold_done:
            return False
        if self._preemptive_hold_count > 0 or float(progress) >= self.settle_progress:
            self._preemptive_hold_count += 1
            if self._preemptive_hold_count >= self.preemptive_hold_steps:
                self._preemptive_hold_done = True
            return True
        return False

    def command(
        self,
        robot_obs: np.ndarray,
        *,
        object_pos: np.ndarray,
        oracle_success: bool,
    ) -> dict[str, Any]:
        robot = np.asarray(robot_obs, dtype=np.float32)
        current_object = np.asarray(object_pos, dtype=np.float32)
        if robot.shape != (15,):
            raise ValueError("robot_obs must have shape (15,)")
        if current_object.shape != (3,):
            raise ValueError("object_pos must have shape (3,)")
        if oracle_success:
            self.state = "done"

        def target_for_state(state: str) -> tuple[np.ndarray, np.ndarray]:
            if state == "done":
                hold = robot[:3].copy()
                return hold, hold
            if not self.is_world_y or state == "hover":
                fixed = self.waypoints[state].copy()
                return fixed, fixed
            flow = self.spec["flow"]
            physical = current_object.copy()
            lane_x = 0.75 * float(self.initial_object_pos[0]) + 0.25 * float(current_object[0])
            physical[0] = np.float32(lane_x)
            if state == "contact":
                standoff = self.side_offset
            elif self._preemptive_hold_done and self.spec["direction"] == "forward":
                standoff = self.y_post_hold_standoff
            else:
                standoff = self.y_push_standoff
            physical -= flow * np.float32(standoff)
            physical[2] = self.initial_object_pos[2] + np.float32(self.contact_height)
            target = physical.copy()
            lateral_error = float(lane_x - robot[0])
            target[0] += np.float32(
                np.clip(self.y_lateral_gain * lateral_error, -self.y_lateral_limit, self.y_lateral_limit)
            )
            return target, physical

        target, convergence_target = target_for_state(self.state)
        if self.state != "done":
            pos_ok = float(np.linalg.norm(robot[:3] - convergence_target)) <= self.position_tolerance
            orn_ok = float(np.linalg.norm(wrap_angle_error(self.orientation, robot[3:6]))) <= self.orientation_tolerance
            if pos_ok and orn_ok:
                self._stable_count += 1
            else:
                self._stable_count = 0
            if self._stable_count >= self.stable_steps:
                self._stable_count = 0
                if self.state == "hover":
                    self.state = "contact"
                elif self.state == "contact":
                    self.state = "push"
                target, convergence_target = target_for_state(self.state)
        absolute_action = np.concatenate([target, self.orientation, [self.gripper_action]]).astype(np.float32)
        return {
            "state": self.state,
            "target_pos": target.copy(),
            "absolute_action": absolute_action,
            "object_pos": np.asarray(object_pos, dtype=np.float32).copy(),
        }


def push_progress(task: str, initial_object_pos: np.ndarray, current_object_pos: np.ndarray) -> float:
    flow = parse_push_task(task)["flow"]
    return float(np.dot(np.asarray(current_object_pos) - np.asarray(initial_object_pos), flow))


def projected_push_progress(current: float, previous: float, *, horizon: int = 3) -> float:
    if horizon < 0:
        raise ValueError("projection horizon must be non-negative")
    positive_velocity = max(float(current) - float(previous), 0.0)
    return float(current) + float(horizon) * positive_velocity


def build_push_guidance(
    task: str,
    robot_obs: np.ndarray,
    initial_object_pos: np.ndarray,
    current_object_pos: np.ndarray,
    *,
    motion_threshold: float,
    side_offset: float = 0.047,
    contact_height: float = 0.012,
) -> dict[str, Any]:
    spec = parse_push_task(task)
    progress = push_progress(task, initial_object_pos, current_object_pos)
    post_contact = bool(progress >= motion_threshold)
    anchor = np.asarray(current_object_pos, dtype=np.float32) - spec["flow"] * np.float32(side_offset)
    anchor[2] += np.float32(contact_height)
    robot = np.asarray(robot_obs, dtype=np.float32)
    attraction = np.zeros(3, dtype=np.float32) if post_contact else anchor - robot[:3]
    return {
        "g": np.concatenate([attraction, spec["flow"]]).astype(np.float32),
        "phase": "contact_transport" if post_contact else "approach",
        "post_contact": post_contact,
        "progress": progress,
        "interaction_anchor": anchor,
    }


def attach_provisional_labels(
    native_step: dict[str, Any],
    guidance: dict[str, Any],
    *,
    emit: bool,
) -> dict[str, Any]:
    """Attach candidate G labels only when explicitly requested."""
    step = dict(native_step)
    if emit:
        for name in ("g", "phase", "post_contact", "interaction_anchor"):
            step[name] = guidance[name]
    return step


def validate_parametric_trajectory(trajectory: dict[str, Any]) -> None:
    """Validate native temporal alignment; candidate semantic labels are optional."""

    required = (
        "robot_obs", "scene_obs", "rgb_static", "rgb_gripper", "actions",
        "rel_actions", "controller_state", "object_pos",
    )
    provisional = ("g", "phase", "post_contact", "interaction_anchor")
    missing = [name for name in required if name not in trajectory]
    if missing:
        raise ValueError(f"Missing trajectory fields: {missing}")
    present_provisional = [name for name in provisional if name in trajectory]
    schema = str(np.asarray(trajectory.get("provisional_label_schema", "legacy_unversioned_g")).item())
    if schema in {"not_emitted", "native_only_v1"} and present_provisional:
        raise ValueError(f"Provisional labels present while provisional_label_schema={schema}")
    if schema == "legacy_unversioned_g" and present_provisional:
        legacy_required = {"g", "phase", "post_contact"}
        if not legacy_required.issubset(present_provisional):
            raise ValueError(f"Legacy provisional label bundle is incomplete: {present_provisional}")
    elif schema not in {"not_emitted", "native_only_v1", "legacy_unversioned_g"}:
        if len(present_provisional) != len(provisional):
            raise ValueError(
                f"Versioned provisional label bundle must be all-or-none; present={present_provisional} expected={list(provisional)}"
            )
    if not present_provisional and schema not in {"not_emitted", "native_only_v1"}:
        raise ValueError(f"Provisional labels absent but schema is {schema!r}")
    provisional_temporal = tuple(name for name in provisional if name in trajectory)
    temporal = required + (("progress",) if "progress" in trajectory else ()) + provisional_temporal
    lengths = {name: len(np.asarray(trajectory[name])) for name in temporal}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Step modalities must be aligned: {lengths}")
    steps = next(iter(lengths.values()))
    if steps < 1:
        raise ValueError("Trajectory must contain at least one step")
    shapes = {
        "robot_obs": (steps, 15), "scene_obs": (steps, 24),
        "actions": (steps, 7), "rel_actions": (steps, 7),
        "object_pos": (steps, 3),
    }
    if "g" in trajectory:
        shapes["g"] = (steps, 6)
    if "interaction_anchor" in trajectory:
        shapes["interaction_anchor"] = (steps, 3)
    for name, expected in shapes.items():
        if np.asarray(trajectory[name]).shape != expected:
            raise ValueError(f"Unexpected {name} shape: {np.asarray(trajectory[name]).shape}; expected {expected}")
    for name in ("rgb_static", "rgb_gripper"):
        value = np.asarray(trajectory[name])
        if value.ndim != 4 or value.shape[0] != steps or value.shape[-1] != 3:
            raise ValueError(f"Unexpected {name} shape: {value.shape}")
    if present_provisional:
        phases = np.asarray(trajectory["phase"]).astype(str)
        post_contact = np.asarray(trajectory["post_contact"], dtype=bool)
        if not set(phases.tolist()) <= {"approach", "contact_transport"}:
            raise ValueError(f"Unexpected phases: {set(phases.tolist())}")
        if not np.array_equal(post_contact, phases == "contact_transport"):
            raise ValueError("phase/post_contact mismatch")
    task = str(np.asarray(trajectory["task"]).item())
    spec = parse_push_task(task)
    object_name = str(np.asarray(trajectory["object_name"]).item())
    direction = str(np.asarray(trajectory["direction"]).item())
    if object_name != spec["object_name"] or direction != spec["direction"]:
        raise ValueError(f"Task/factor mismatch: {task}, {object_name}, {direction}")


def absolute_to_relative_action(
    absolute_action: np.ndarray,
    robot_obs: np.ndarray,
    *,
    max_rel_pos: float = 0.02,
    max_rel_orn: float = 0.05,
) -> np.ndarray:
    """Convert a seven-value absolute pose command to CALVIN-scaled relative action."""

    absolute = np.asarray(absolute_action, dtype=np.float32)
    robot = np.asarray(robot_obs, dtype=np.float32)
    if absolute.shape != (7,) or robot.shape != (15,):
        raise ValueError("Expected absolute_action (7,) and robot_obs (15,)")
    rel_pos = np.clip((absolute[:3] - robot[:3]) / max_rel_pos, -1.0, 1.0)
    rel_orn = np.clip(wrap_angle_error(absolute[3:6], robot[3:6]) / max_rel_orn, -1.0, 1.0)
    return np.concatenate([rel_pos, rel_orn, [absolute[6]]]).astype(np.float32)


def load_reset_bank(
    manifest_path: Path,
    *,
    seed: int,
    tasks: tuple[str, ...] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load exact resets; fixed-scene resets may target all or a declared task subset."""

    selected_tasks = tuple(PUSH_TASKS if tasks is None else tasks)
    unknown = set(selected_tasks) - set(ALL_PUSH_TASKS)
    if unknown:
        raise ValueError(f"Unsupported selected tasks: {sorted(unknown)}")
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    bank: dict[str, list[dict[str, Any]]] = {task: [] for task in selected_tasks}

    if manifest.get("format") == "calvin_fixed_scene_push_resets_v1":
        runtime_scene = str(manifest.get("runtime_scene", ""))
        if runtime_scene != "calvin_scene_D":
            raise ValueError(f"Expected calvin_scene_D fixed reset pool, got {runtime_scene!r}")
        seen_ids: set[str] = set()
        for row in manifest.get("resets", []):
            reset_id = str(row.get("reset_id", ""))
            if not reset_id or reset_id in seen_ids:
                raise ValueError(f"Missing or duplicate fixed-scene reset_id: {reset_id!r}")
            if str(row.get("runtime_scene", row.get("scene", ""))) != runtime_scene:
                raise ValueError(f"Cross-scene reset rejected: {reset_id}")
            robot_obs = np.asarray(row.get("robot_obs"), dtype=np.float32)
            scene_obs = np.asarray(row.get("scene_obs"), dtype=np.float32)
            if robot_obs.shape != (15,) or scene_obs.shape != (24,):
                raise ValueError(f"Invalid embedded reset shapes for {reset_id}")
            eligible = set(row.get("eligible_tasks", selected_tasks))
            invalid_eligible = eligible - set(ALL_PUSH_TASKS)
            if invalid_eligible:
                raise ValueError(f"Fixed reset declares unsupported tasks {sorted(invalid_eligible)}: {reset_id}")
            normalized = dict(row)
            normalized["reset_id"] = reset_id
            normalized["scene"] = runtime_scene
            normalized["runtime_scene"] = runtime_scene
            normalized["robot_obs"] = robot_obs.tolist()
            normalized["scene_obs"] = scene_obs.tolist()
            for task in selected_tasks:
                if task not in eligible:
                    continue
                task_reset = dict(normalized)
                task_reset["task"] = task
                bank[task].append(task_reset)
            seen_ids.add(reset_id)
    else:
        seen: set[tuple[str, int, int]] = set()
        for row in manifest.get("segments_detail", []):
            task = str(row.get("task", ""))
            if task not in bank:
                continue
            normalized = {
                "task": task,
                "scene": str(row.get("scene", "unknown")),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "reset_id": f"episode_{int(row['start']):07d}",
                "source_kind": "source_episode_frame",
            }
            key = (task, normalized["start"], normalized["end"])
            if key not in seen:
                bank[task].append(normalized)
                seen.add(key)

    for task, rows in bank.items():
        rows.sort(
            key=lambda row: hashlib.sha256(
                (
                    f"{seed}|fixed|{row.get('reset_id')}"
                    if row.get("source_kind") == "native_runtime_reset"
                    else f"{seed}|{task}|{row.get('scene')}|{row.get('reset_id')}|{row.get('start', -1)}"
                ).encode("utf-8")
            ).hexdigest()
        )
        if not rows:
            raise ValueError(f"No reset candidates found for {task}")
    return bank


def build_checkerboard_folds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build bridge-connected object×direction folds from oracle-accepted records only."""

    accepted = [dict(row) for row in records if bool(row.get("oracle_success", False))]
    folds: list[dict[str, Any]] = []
    fold_index = 0
    for bridge in COLORS:
        remaining = [color for color in COLORS if color != bridge]
        for polarity in (0, 1):
            first_direction = DIRECTIONS[polarity]
            second_direction = DIRECTIONS[1 - polarity]
            source_cells = [
                f"push_{bridge}_block_left",
                f"push_{bridge}_block_right",
                f"push_{remaining[0]}_block_{first_direction}",
                f"push_{remaining[1]}_block_{second_direction}",
            ]
            heldout_cells = [
                f"push_{remaining[0]}_block_{second_direction}",
                f"push_{remaining[1]}_block_{first_direction}",
            ]
            source_set, heldout_set = set(source_cells), set(heldout_cells)
            if source_set & heldout_set or source_set | heldout_set != set(PUSH_TASKS):
                raise AssertionError("Invalid checkerboard cell partition")
            source_records = [row for row in accepted if row.get("task") in source_set]
            heldout_records = [row for row in accepted if row.get("task") in heldout_set]
            source_objects = {parse_push_task(task)["color"] for task in source_cells}
            source_directions = {parse_push_task(task)["direction"] for task in source_cells}
            for task in heldout_cells:
                spec = parse_push_task(task)
                if spec["color"] not in source_objects or spec["direction"] not in source_directions:
                    raise AssertionError(f"Held-out atoms missing from source: {task}")
            folds.append(
                {
                    "fold": fold_index,
                    "bridge_object": bridge,
                    "polarity": polarity,
                    "source_cells": source_cells,
                    "heldout_cells": heldout_cells,
                    "source_records": source_records,
                    "heldout_records": heldout_records,
                }
            )
            fold_index += 1
    return folds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def named_object_pos(info: dict[str, Any], object_name: str) -> np.ndarray:
    try:
        value = info["scene_info"]["movable_objects"][object_name]["current_pos"]
    except KeyError as exc:
        raise KeyError(f"Live scene info has no movable object {object_name!r}") from exc
    position = np.asarray(value, dtype=np.float32)
    if position.shape != (3,):
        raise ValueError(f"Unexpected {object_name} position shape: {position.shape}")
    return position.copy()


def load_reset_arrays(split_dir: Path, reset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "robot_obs" in reset and "scene_obs" in reset:
        robot_obs = np.asarray(reset["robot_obs"], dtype=np.float32)
        scene_obs = np.asarray(reset["scene_obs"], dtype=np.float32)
        if robot_obs.shape != (15,) or scene_obs.shape != (24,):
            raise ValueError(f"Invalid embedded reset shapes: {robot_obs.shape}, {scene_obs.shape}")
        return robot_obs.copy(), scene_obs.copy()
    path = Path(split_dir) / f"episode_{int(reset['start']):07d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["robot_obs"], dtype=np.float32).copy(),
            np.asarray(payload["scene_obs"], dtype=np.float32).copy(),
        )


def stack_trajectory(steps: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    if not steps:
        raise ValueError("Cannot stack empty rollout")
    trajectory: dict[str, Any] = {}
    native_fields = (
        "robot_obs", "scene_obs", "rgb_static", "rgb_gripper", "actions",
        "rel_actions", "object_pos", "progress",
    )
    for name in native_fields:
        trajectory[name] = np.stack([np.asarray(step[name]) for step in steps])
    for name in ("object_orn", "contact_detected", "task_flow", "live_contact_pos", "surface_normal"):
        if all(name in step for step in steps):
            trajectory[name] = np.stack([np.asarray(step[name]) for step in steps])
    trajectory["controller_state"] = np.asarray([step["controller_state"] for step in steps])
    provisional = ("g", "phase", "post_contact", "interaction_anchor")
    presence = {name: all(name in step for step in steps) for name in provisional}
    if any(presence.values()) and not all(presence.values()):
        raise ValueError(f"Provisional label fields must be emitted together: {presence}")
    if all(presence.values()):
        trajectory["g"] = np.stack([np.asarray(step["g"]) for step in steps])
        trajectory["phase"] = np.asarray([step["phase"] for step in steps])
        trajectory["post_contact"] = np.asarray([step["post_contact"] for step in steps], dtype=bool)
        trajectory["interaction_anchor"] = np.stack([np.asarray(step["interaction_anchor"]) for step in steps])
    for name, value in metadata.items():
        trajectory[name] = np.asarray(value)
    validate_parametric_trajectory(trajectory)
    return trajectory


def lightweight_failure_trajectory(trajectory: dict[str, Any], *, stride: int) -> dict[str, Any]:
    """Keep aligned diagnostic keyframes for failures while preserving the terminal step."""
    if stride < 1:
        raise ValueError("failure stride must be >= 1")
    validate_parametric_trajectory(trajectory)
    steps = len(np.asarray(trajectory["robot_obs"]))
    indices = list(range(0, steps, stride))
    if indices[-1] != steps - 1:
        indices.append(steps - 1)
    temporal_fields = tuple(
        name
        for name in (
            "robot_obs", "scene_obs", "rgb_static", "rgb_gripper", "actions", "rel_actions",
            "g", "phase", "post_contact", "controller_state", "object_pos", "interaction_anchor", "progress",
        )
        if name in trajectory
    )
    thinned = dict(trajectory)
    for name in temporal_fields:
        thinned[name] = np.asarray(trajectory[name])[indices]
    thinned["original_steps"] = np.asarray(steps)
    thinned["failure_storage_mode"] = np.asarray("temporal_stride")
    thinned["failure_stride"] = np.asarray(stride)
    validate_parametric_trajectory(thinned)
    return thinned


def persist_attempt(
    output_root: Path,
    *,
    attempt_id: str,
    trajectory: dict[str, Any],
    receipt: dict[str, Any],
) -> Path:
    validate_parametric_trajectory(trajectory)
    success = bool(np.asarray(trajectory["success"]).item())
    if success != bool(receipt.get("oracle_success", False)):
        raise ValueError("Trajectory success must match oracle receipt")
    bucket = "accepted" if success else "attempts"
    directory = Path(output_root) / bucket
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"{attempt_id}.npz"
    receipt_path = directory / f"{attempt_id}.receipt.json"
    if npz_path.exists() or receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing attempt: {attempt_id}")
    temp_npz = directory / f".{attempt_id}.npz.tmp"
    with open(temp_npz, "wb") as stream:
        np.savez_compressed(stream, **trajectory)
    os.replace(temp_npz, npz_path)
    receipt_payload = dict(receipt)
    receipt_payload["trajectory"] = npz_path.relative_to(output_root).as_posix()
    receipt_payload["trajectory_sha256"] = sha256_file(npz_path)
    temp_receipt = directory / f".{attempt_id}.receipt.json.tmp"
    with open(temp_receipt, "w", encoding="utf-8") as stream:
        json.dump(receipt_payload, stream, indent=2, default=json_default)
        stream.write("\n")
    os.replace(temp_receipt, receipt_path)
    return npz_path


def persist_failure_metadata_only(
    output_root: Path,
    *,
    attempt_id: str,
    receipt: dict[str, Any],
) -> Path:
    if bool(receipt.get("oracle_success", False)):
        raise ValueError("metadata-only failure persistence requires oracle_success=false")
    directory = Path(output_root) / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / f"{attempt_id}.receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing failure receipt: {attempt_id}")
    payload = dict(receipt)
    payload.update({"failure_storage_mode": "metadata_only", "trajectory": None, "trajectory_sha256": None})
    temp = directory / f".{attempt_id}.receipt.json.tmp"
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=json_default)
        stream.write("\n")
    os.replace(temp, receipt_path)
    return receipt_path


def collect_one(
    env: Any,
    oracle: Any,
    split_dir: Path,
    task: str,
    reset: dict[str, Any],
    *,
    max_steps: int,
    motion_threshold: float,
    emit_provisional_g: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = parse_push_task(task)
    robot_reset, scene_reset = load_reset_arrays(split_dir, reset)
    obs = env.reset(robot_obs=robot_reset, scene_obs=scene_reset)
    start_info = env.get_info()
    initial_object_pos = named_object_pos(start_info, spec["object_name"])
    expert = ParametricPushExpert(task, initial_object_pos)
    steps: list[dict[str, Any]] = []
    achieved: list[str] = []
    success_step: int | None = None
    state_history: list[str] = []

    for step_index in range(max_steps):
        current_info = env.get_info()
        current_object_pos = named_object_pos(current_info, spec["object_name"])
        guidance = build_push_guidance(
            task,
            np.asarray(obs["robot_obs"], dtype=np.float32),
            initial_object_pos,
            current_object_pos,
            motion_threshold=motion_threshold,
        )
        command = expert.command(
            np.asarray(obs["robot_obs"], dtype=np.float32),
            object_pos=current_object_pos,
            oracle_success=False,
        )
        absolute_action = command["absolute_action"]
        control_progress = float(guidance["progress"])
        if expert.is_world_y and spec["direction"] == "forward" and steps:
            control_progress = projected_push_progress(
                control_progress,
                float(steps[-1]["progress"]),
                horizon=3,
            )
        if expert.should_settle(control_progress):
            robot_now = np.asarray(obs["robot_obs"], dtype=np.float32)
            absolute_action = np.concatenate([robot_now[:6], [expert.gripper_action]]).astype(np.float32)
            command = dict(command)
            command["state"] = "settle"
            command["absolute_action"] = absolute_action
        rel_action = absolute_to_relative_action(absolute_action, np.asarray(obs["robot_obs"], dtype=np.float32))
        rgb_obs = obs["rgb_obs"]
        steps.append(
            attach_provisional_labels(
                {
                    "robot_obs": np.asarray(obs["robot_obs"], dtype=np.float32).copy(),
                    "scene_obs": np.asarray(obs["scene_obs"], dtype=np.float32).copy(),
                    "rgb_static": np.asarray(rgb_obs["rgb_static"], dtype=np.uint8).copy(),
                    "rgb_gripper": np.asarray(rgb_obs["rgb_gripper"], dtype=np.uint8).copy(),
                    "actions": absolute_action.copy(),
                    "rel_actions": rel_action,
                    "progress": guidance["progress"],
                    "controller_state": command["state"],
                    "object_pos": current_object_pos,
                },
                guidance,
                emit=emit_provisional_g,
            )
        )
        state_history.append(str(command["state"]))
        obs, _, _, final_info = env.step(
            (absolute_action[:3].astype(float), absolute_action[3:6].astype(float), int(absolute_action[6]))
        )
        achieved = sorted(oracle.get_task_info_for_set(start_info, final_info, {task}))
        if task in achieved:
            success_step = step_index
            break

    final_info = env.get_info()
    final_object_pos = named_object_pos(final_info, spec["object_name"])
    final_progress = push_progress(task, initial_object_pos, final_object_pos)
    success = success_step is not None
    trajectory = stack_trajectory(
        steps,
        {
            "task": task,
            "object_name": spec["object_name"],
            "direction": spec["direction"],
            "success": success,
            "provisional_label_schema": "candidate_g_af_v1" if emit_provisional_g else "not_emitted",
            "source_kind": "parameterized_closed_loop_waypoint_expert",
            "action_mode": "absolute_pose_executed_with_relative_action_labels",
            "reset_robot_obs": robot_reset,
            "reset_scene_obs": scene_reset,
            "reset_start": int(reset.get("start", -1)),
            "reset_end": int(reset.get("end", -1)),
            "reset_id": str(reset.get("reset_id", f"episode_{int(reset.get('start', -1)):07d}")),
            "reset_source_kind": str(reset.get("source_kind", "source_episode_frame")),
            "reset_scene": str(reset["scene"]),
            "runtime_scene": str(reset.get("runtime_scene", reset["scene"])),
            "initial_object_pos": initial_object_pos,
            "final_object_pos": final_object_pos,
        },
    )
    receipt = {
        "task": task,
        "family": "block_push",
        "object_name": spec["object_name"],
        "direction": spec["direction"],
        "oracle_success": success,
        "oracle_achieved_tasks": achieved,
        "success_step": success_step,
        "executed_steps": len(steps),
        "final_progress": final_progress,
        "motion_threshold": motion_threshold,
        "reset": reset,
        "initial_object_pos": initial_object_pos,
        "final_object_pos": final_object_pos,
        "controller_states": list(dict.fromkeys(state_history)),
    }
    return trajectory, receipt


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.tmp"
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=json_default)
        stream.write("\n")
    os.replace(temp, path)


def write_fold_manifests(output_root: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    folds = build_checkerboard_folds(records)
    summaries = []
    for fold in folds:
        serializable = dict(fold)
        serializable["format"] = "calvin_object_direction_checkerboard_fold_v1"
        serializable["invariants"] = {
            "pair_overlap": False,
            "all_heldout_objects_seen_in_source": True,
            "all_heldout_directions_seen_in_source": True,
            "accepted_only": True,
        }
        path = (
            Path(output_root)
            / "folds"
            / f"fold_{fold['fold']:02d}_bridge_{fold['bridge_object']}_polarity_{fold['polarity']}.json"
        )
        write_json_atomic(path, serializable)
        summaries.append(
            {
                "fold": fold["fold"],
                "bridge_object": fold["bridge_object"],
                "polarity": fold["polarity"],
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
                "source_cells": fold["source_cells"],
                "heldout_cells": fold["heldout_cells"],
                "source_trajectories": len(fold["source_records"]),
                "heldout_trajectories": len(fold["heldout_records"]),
            }
        )
    return summaries


def validate_collection(output_root: Path) -> dict[str, Any]:
    root = Path(output_root)
    with open(root / "collection_manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format") != "calvin_parametric_push_collection_v1":
        raise ValueError("Unexpected collection format")
    manifest_tasks = tuple(manifest.get("tasks", PUSH_TASKS))
    counts = {task: 0 for task in manifest_tasks}
    verified = 0
    prune_sidecar = root / "failure_payload_prune.json"
    pruned_paths: set[str] = set()
    if prune_sidecar.is_file():
        with open(prune_sidecar, encoding="utf-8") as stream:
            prune_payload = json.load(stream)
        if prune_payload.get("format") != "calvin_failure_payload_prune_v1":
            raise ValueError(f"Unexpected failure prune sidecar: {prune_sidecar}")
        pruned_paths = {str(path) for path in prune_payload.get("pruned_paths", [])}
    pruned_verified = 0
    metadata_only_verified = 0
    for row in manifest.get("results", []):
        success_row = bool(row["oracle_success"])
        row_path = row.get("path")
        if not success_row and row_path is None and row.get("failure_storage_mode") == "metadata_only":
            metadata_only_verified += 1
            continue
        if not success_row and str(row_path) in pruned_paths:
            pruned_verified += 1
            continue
        if not row_path:
            raise FileNotFoundError(f"Manifest row has no trajectory path: {row}")
        path = root / str(row_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Trajectory hash mismatch: {path}")
        with open(path.with_suffix(".receipt.json"), encoding="utf-8") as stream:
            receipt = json.load(stream)
        if receipt.get("trajectory_sha256") != row["sha256"]:
            raise ValueError(f"Receipt hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            trajectory = {name: payload[name] for name in payload.files}
        validate_parametric_trajectory(trajectory)
        success = bool(np.asarray(trajectory["success"]).item())
        if success != bool(row["oracle_success"]):
            raise ValueError(f"Oracle/trajectory mismatch: {path}")
        expected_bucket = "accepted" if success else "attempts"
        if Path(row["path"]).parts[0] != expected_bucket:
            raise ValueError(f"Acceptance bucket mismatch: {path}")
        if success:
            counts[row["task"]] += 1
            verified += 1
    quota = int(manifest["quota_per_cell"])
    complete = all(count >= quota for count in counts.values())
    for fold in manifest.get("folds", []):
        fold_path = root / fold["path"]
        if sha256_file(fold_path) != fold["sha256"]:
            raise ValueError(f"Fold hash mismatch: {fold_path}")
        with open(fold_path, encoding="utf-8") as stream:
            payload = json.load(stream)
        source, heldout = set(payload["source_cells"]), set(payload["heldout_cells"])
        if source & heldout or source | heldout != set(manifest_tasks):
            raise ValueError(f"Fold leakage or missing cells: {fold_path}")
        if not all(row["oracle_success"] for row in payload["source_records"] + payload["heldout_records"]):
            raise ValueError(f"Failed attempt leaked into fold: {fold_path}")
    return {
        "accepted_verified": verified,
        "accepted_per_cell": counts,
        "quota_per_cell": quota,
        "complete": complete,
        "folds_verified": len(manifest.get("folds", [])),
        "failed_payloads_pruned": pruned_verified,
        "failed_metadata_only_verified": metadata_only_verified,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--reset-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quota-per-cell", type=int, default=2)
    parser.add_argument("--max-attempts-per-cell", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--motion-threshold", type=float, default=0.003)
    parser.add_argument(
        "--failure-storage",
        choices=("metadata_only", "stride"),
        default="metadata_only",
        help="Failures default to receipts only; use stride only for targeted diagnostics.",
    )
    parser.add_argument(
        "--failure-stride",
        type=int,
        default=8,
        help="When --failure-storage=stride, persist every Nth failure frame plus terminal frame.",
    )
    parser.add_argument(
        "--emit-provisional-g",
        action="store_true",
        help="Opt in to experimental G/phase labels; disabled by default for native data expansion.",
    )
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--directions",
        default="left,right",
        help="Comma-separated collection directions from left,right,forward,backward.",
    )
    parser.add_argument(
        "--emit-folds",
        action="store_true",
        help="Optionally emit evaluation folds; disabled by default for data-expansion runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.calvin_code = args.calvin_code.resolve()
    args.source_split = args.source_split.resolve()
    args.reset_manifest = args.reset_manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    if (args.out_dir / "collection_manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite collection: {args.out_dir}")
    if args.quota_per_cell < 1 or args.max_attempts_per_cell < args.quota_per_cell:
        raise ValueError("Invalid quota/attempt limit")
    if args.failure_stride < 1:
        raise ValueError("failure stride must be >= 1")
    selected_directions = tuple(part.strip() for part in args.directions.split(",") if part.strip())
    if not selected_directions or len(set(selected_directions)) != len(selected_directions):
        raise ValueError("Directions must be a non-empty unique list")
    invalid_directions = set(selected_directions) - set(ALL_DIRECTIONS)
    if invalid_directions:
        raise ValueError(f"Unsupported directions: {sorted(invalid_directions)}")
    selected_tasks = tuple(
        f"push_{color}_block_{direction}" for color in COLORS for direction in selected_directions
    )
    if args.emit_folds and selected_directions != DIRECTIONS:
        raise ValueError("Evaluation folds are only defined for the legacy left/right task set")

    calvin_models = args.calvin_code / "calvin_models"
    calvin_env = args.calvin_code / "calvin_env"
    sys.path[:0] = [calvin_env.as_posix(), calvin_models.as_posix()]
    import hydra  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415
    from calvin_env.envs.play_table_env import get_env  # noqa: PLC0415

    np.random.seed(args.seed)
    reset_bank = load_reset_bank(args.reset_manifest, seed=args.seed, tasks=selected_tasks)
    task_cfg_path = calvin_models / "conf/callbacks/rollout/tasks/new_playtable_tasks.yaml"
    task_cfg = OmegaConf.load(task_cfg_path)
    for color in COLORS:
        task_cfg.tasks[f"push_{color}_block_forward"] = ["push_object", f"block_{color}", 0.0, 0.1]
        task_cfg.tasks[f"push_{color}_block_backward"] = ["push_object", f"block_{color}", 0.0, -0.1]
    oracle = hydra.utils.instantiate(task_cfg)
    env = get_env(
        args.source_split,
        show_gui=False,
        obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []},
    )
    results: list[dict[str, Any]] = []
    accepted_counts = {task: 0 for task in selected_tasks}
    try:
        for task in selected_tasks:
            candidates = reset_bank[task][: args.max_attempts_per_cell]
            if len(candidates) < args.quota_per_cell:
                raise RuntimeError(f"Insufficient reset candidates for {task}")
            for attempt_index, reset in enumerate(candidates):
                if accepted_counts[task] >= args.quota_per_cell:
                    break
                reset_token = str(reset.get("reset_id", f"episode_{int(reset.get('start', -1)):07d}"))
                attempt_id = f"{task}_{reset_token}_seed_{args.seed}_try_{attempt_index:02d}"
                trajectory, receipt = collect_one(
                    env,
                    oracle,
                    args.source_split,
                    task,
                    reset,
                    max_steps=args.max_steps,
                    motion_threshold=args.motion_threshold,
                    emit_provisional_g=args.emit_provisional_g,
                )
                receipt.update({"attempt_id": attempt_id, "seed": args.seed})
                if not bool(receipt["oracle_success"]) and args.failure_storage == "metadata_only":
                    receipt["stored_steps"] = 0
                    receipt["failure_storage_mode"] = "metadata_only"
                    receipt["failure_stride"] = None
                    receipt_path = persist_failure_metadata_only(
                        args.out_dir, attempt_id=attempt_id, receipt=receipt
                    )
                    row = dict(receipt)
                    row["path"] = None
                    row["receipt_path"] = receipt_path.relative_to(args.out_dir).as_posix()
                    row["sha256"] = None
                else:
                    if not bool(receipt["oracle_success"]) and args.failure_stride > 1:
                        trajectory = lightweight_failure_trajectory(trajectory, stride=args.failure_stride)
                        receipt["failure_storage_mode"] = "temporal_stride"
                        receipt["failure_stride"] = args.failure_stride
                    else:
                        receipt["failure_storage_mode"] = "full_rate"
                        receipt["failure_stride"] = 1
                    receipt["stored_steps"] = len(np.asarray(trajectory["robot_obs"]))
                    path = persist_attempt(args.out_dir, attempt_id=attempt_id, trajectory=trajectory, receipt=receipt)
                    row = dict(receipt)
                    row["path"] = path.relative_to(args.out_dir).as_posix()
                    row["sha256"] = sha256_file(path)
                results.append(row)
                if row["oracle_success"]:
                    accepted_counts[task] += 1
                print(
                    f"[collect] task={task} success={int(row['oracle_success'])} "
                    f"progress={row['final_progress']:.4f} steps={row['executed_steps']} "
                    f"accepted={accepted_counts[task]}/{args.quota_per_cell}",
                    flush=True,
                )
    finally:
        env.close()
        env.ownsPhysicsClient = False

    complete = all(count >= args.quota_per_cell for count in accepted_counts.values())
    folds = write_fold_manifests(args.out_dir, results) if complete and args.emit_folds else []
    successful_rows = [row for row in results if bool(row["oracle_success"])]
    lengths_by_task = {
        task: [int(row["executed_steps"]) for row in successful_rows if row["task"] == task]
        for task in selected_tasks
    }
    distribution = {
        "scope": "data_expansion_only",
        "evaluation_folds_emitted": bool(folds),
        "accepted_per_cell": accepted_counts,
        "attempted_per_cell": {
            task: sum(row["task"] == task for row in results) for task in selected_tasks
        },
        "successful_steps_per_cell": {
            task: sum(lengths_by_task[task]) for task in selected_tasks
        },
        "distinct_reset_ids_per_cell": {
            task: len(
                {
                    str(row["reset"].get("reset_id", row["reset"].get("start", "unknown")))
                    for row in successful_rows
                    if row["task"] == task
                }
            )
            for task in selected_tasks
        },
    }
    manifest = {
        "format": "calvin_parametric_push_collection_v1",
        "provenance": {
            "collector": Path(__file__).resolve(),
            "collector_sha256": sha256_file(Path(__file__).resolve()),
            "calvin_code": args.calvin_code,
            "calvin_task_config": task_cfg_path,
            "calvin_task_config_sha256": sha256_file(task_cfg_path),
            "source_split": args.source_split,
            "reset_manifest": args.reset_manifest,
            "reset_manifest_sha256": sha256_file(args.reset_manifest),
            "controller": "named-object closed-loop absolute-waypoint state machine",
            "object_lookup": "env.get_info scene_info movable_objects by exact object name",
            "acceptance_gate": "CALVIN get_task_info_for_set exact requested task",
            "observations_kind": "fresh simulator observations from this run",
        },
        "seed": args.seed,
        "tasks": list(selected_tasks),
        "factor_axes": {"object": list(COLORS), "direction": list(selected_directions)},
        "quota_per_cell": args.quota_per_cell,
        "max_attempts_per_cell": args.max_attempts_per_cell,
        "max_steps": args.max_steps,
        "motion_threshold": args.motion_threshold,
        "semantic_labels": {
            "candidate_g_emitted": bool(args.emit_provisional_g),
            "schema": "candidate_g_af_v1" if args.emit_provisional_g else "not_emitted",
            "role": "optional_derived_research_label_not_acceptance_or_distribution_gate",
        },
        "failure_storage": {
            "mode": args.failure_storage,
            "stride": args.failure_stride if args.failure_storage == "stride" else None,
            "accepted_trajectories": "always_full_rate",
        },
        "accepted_per_cell": accepted_counts,
        "accepted": sum(accepted_counts.values()),
        "failed": sum(not bool(row["oracle_success"]) for row in results),
        "complete": complete,
        "distribution": distribution,
        "results": results,
        "folds": folds,
    }
    write_json_atomic(args.out_dir / "collection_manifest.json", manifest)
    summary = validate_collection(args.out_dir)
    print(json.dumps(summary, default=json_default), flush=True)
    if not complete or not summary["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
