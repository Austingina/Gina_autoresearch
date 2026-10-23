#!/usr/bin/env python3
"""Geometry-derived contact-pose search for CALVIN block pushing.

The planner derives candidates from live AABBs and task flow. Candidate previews are
simulator diagnostics; only a fresh rerun of the selected candidate may be persisted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, NamedTuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_DIR.as_posix())

import collect_calvin_parametric_push_expert as base


class ContactCandidate(NamedTuple):
    candidate_id: str
    object_name: str
    contact_pos: np.ndarray
    orientation: np.ndarray
    flow: np.ndarray
    half_extents: np.ndarray
    surface_normal: np.ndarray
    height_fraction: float
    tool_clearance: float
    geometry_signature: tuple[float, float]


def _validate_axis_flow(flow: np.ndarray) -> np.ndarray:
    value = np.asarray(flow, dtype=np.float32)
    if value.shape != (3,):
        raise ValueError("flow must have shape (3,)")
    if not np.isclose(float(np.linalg.norm(value)), 1.0, atol=1e-5):
        raise ValueError("flow must be a unit vector")
    if not (np.isclose(abs(float(value[0])), 1.0) or np.isclose(abs(float(value[1])), 1.0)):
        raise ValueError("only planar axis-aligned flows are supported")
    if not np.isclose(float(value[2]), 0.0):
        raise ValueError("push flow must be planar")
    return value


def _rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def generate_oriented_contact_candidates(
    *,
    object_name: str,
    object_pos: np.ndarray,
    object_orn: np.ndarray,
    local_extents: np.ndarray,
    flow: np.ndarray,
    tool_clearance: float = 0.015,
) -> list[ContactCandidate]:
    """Generate contact frames at the center of the OBB face whose normal best opposes flow."""
    position = np.asarray(object_pos, dtype=np.float32)
    extents = np.asarray(local_extents, dtype=np.float32)
    if position.shape != (3,) or extents.shape != (3,) or np.any(extents <= 0):
        raise ValueError("object_pos/local_extents must be positive 3-vectors")
    direction = _validate_axis_flow(flow)
    rotation = _rotation_matrix_from_quaternion(np.asarray(object_orn, dtype=np.float32))
    half_extents = extents * np.float32(0.5)
    ray_world = -direction
    ray_local = rotation.T @ ray_world
    intersections = [
        (float(half_extents[axis] / abs(ray_local[axis])), axis)
        for axis in (0, 1)
        if abs(float(ray_local[axis])) > 1e-6
    ]
    if not intersections:
        raise ValueError("push ray does not intersect a planar OBB face")
    ray_distance, face_axis = min(intersections, key=lambda row: row[0])
    face_sign = 1.0 if float(ray_local[face_axis]) > 0 else -1.0
    surface_normal = rotation[:, face_axis] * np.float32(face_sign)
    face_point = position + ray_world * np.float32(ray_distance)
    contact_base = face_point + ray_world * np.float32(tool_clearance)
    approach_angle = math.atan2(float(ray_world[1]), float(ray_world[0]))
    base_yaw = _wrap_angle(approach_angle + math.pi / 2.0)
    height_fractions = (0.20, 0.35, 0.50, 0.65)
    candidates: list[ContactCandidate] = []
    for height_fraction in height_fractions:
        for wrist_quarter_turn in (0.0, 1.0, 2.0, 3.0):
            yaw = _wrap_angle(base_yaw + wrist_quarter_turn * math.pi / 2.0)
            contact = contact_base.copy()
            contact[2] = position[2] + np.float32(height_fraction - 0.5) * extents[2]
            signature = (float(height_fraction), float(wrist_quarter_turn))
            candidates.append(
                ContactCandidate(
                    candidate_id=f"h{height_fraction:.2f}_q{int(wrist_quarter_turn)}",
                    object_name=object_name,
                    contact_pos=contact,
                    orientation=np.asarray([3.10, -0.13, yaw], dtype=np.float32),
                    flow=direction.copy(),
                    half_extents=half_extents.copy(),
                    surface_normal=surface_normal.copy(),
                    height_fraction=float(height_fraction),
                    tool_clearance=float(tool_clearance),
                    geometry_signature=signature,
                )
            )
    return candidates


def generate_contact_candidates(
    *,
    object_name: str,
    aabb: tuple[np.ndarray, np.ndarray],
    flow: np.ndarray,
    tool_clearance: float = 0.015,
) -> list[ContactCandidate]:
    """Axis-aligned compatibility wrapper around OBB contact generation."""
    lower = np.asarray(aabb[0], dtype=np.float32)
    upper = np.asarray(aabb[1], dtype=np.float32)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
        raise ValueError(f"invalid AABB: {aabb}")
    return generate_oriented_contact_candidates(
        object_name=object_name,
        object_pos=(lower + upper) * np.float32(0.5),
        object_orn=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        local_extents=upper - lower,
        flow=flow,
        tool_clearance=tool_clearance,
    )


class GeometryContactController:
    """Execute one geometry-derived contact frame without object-label-specific parameters."""

    def __init__(
        self,
        candidate: ContactCandidate,
        *,
        hover_height: float = 0.08,
        push_distance: float = 0.02,
        position_tolerance: float = 0.009,
        orientation_tolerance: float = 0.15,
        stable_steps: int = 2,
        preload_step: float = 0.002,
        max_preload: float = 0.02,
    ) -> None:
        self.candidate = candidate
        hover = candidate.contact_pos.copy()
        hover[2] += np.float32(hover_height)
        push = candidate.contact_pos + candidate.flow * np.float32(push_distance)
        self.waypoints = {"hover": hover, "contact": candidate.contact_pos.copy(), "push": push}
        self.state = "hover"
        self.push_distance = float(push_distance)
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.stable_steps = int(stable_steps)
        self.preload_step = float(preload_step)
        self.max_preload = float(max_preload)
        self.preload_depth = 0.0
        self._stable_count = 0

    def update_live_contact_frame(self, candidate: ContactCandidate) -> None:
        if candidate.geometry_signature != self.candidate.geometry_signature:
            raise ValueError("live contact frame must preserve the selected geometry signature")
        if candidate.object_name != self.candidate.object_name:
            raise ValueError("live contact frame object mismatch")
        self.candidate = candidate
        if self.state == "push":
            self.waypoints["push"] = candidate.contact_pos + candidate.flow * np.float32(self.push_distance)

    def command(self, robot_obs: np.ndarray, *, contact_detected: bool = False) -> dict[str, Any]:
        robot = np.asarray(robot_obs, dtype=np.float32)
        if robot.shape != (15,):
            raise ValueError("robot_obs must have shape (15,)")
        if self.state == "contact" and contact_detected:
            self.waypoints["push"] = self.candidate.contact_pos + self.candidate.flow * np.float32(self.push_distance)
            self.state = "push"
            self._stable_count = 0
        target = self.waypoints[self.state]
        orientation_error = (self.candidate.orientation - robot[3:6] + np.pi) % (2 * np.pi) - np.pi
        pos_ok = float(np.linalg.norm(robot[:3] - target)) <= self.position_tolerance
        orn_ok = float(np.linalg.norm(orientation_error)) <= self.orientation_tolerance
        if pos_ok and orn_ok:
            self._stable_count += 1
        else:
            self._stable_count = 0
        if self._stable_count >= self.stable_steps:
            self._stable_count = 0
            if self.state == "hover":
                self.state = "contact"
            elif self.state == "contact" and not contact_detected:
                remaining = max(self.max_preload - self.preload_depth, 0.0)
                increment = min(self.preload_step, remaining)
                self.waypoints["contact"] = self.waypoints["contact"] + self.candidate.flow * np.float32(increment)
                self.preload_depth += increment
            target = self.waypoints[self.state]
        action = np.concatenate([target, self.candidate.orientation, [-1.0]]).astype(np.float32)
        return {
            "state": self.state,
            "target_pos": target.copy(),
            "target_position": target.copy(),
            "absolute_action": action,
        }


def score_preview(
    *,
    progress: float,
    lateral_drift: float,
    z_drop: float,
    tilt: float,
    support_contact: bool,
    robot_object_contacts: int,
    oracle_success: bool,
) -> float:
    """Rank candidate rollouts while rejecting false progress from falling/sliding sideways."""
    score = 4.0 * float(progress)
    score -= 6.0 * abs(float(lateral_drift))
    score -= 10.0 * max(float(z_drop), 0.0)
    score -= abs(float(tilt))
    score += 10.0 if oracle_success else 0.0
    score += 0.02 * min(max(int(robot_object_contacts), 0), 10)
    if not support_contact:
        score -= 5.0
    if int(robot_object_contacts) <= 0:
        score -= 0.2
    return float(score)


def select_candidate_trial(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("cannot select from empty candidate trials")
    successful = [row for row in trials if bool(row.get("oracle_success", False))]
    pool = successful if successful else trials
    return max(pool, key=lambda row: (float(row["score"]), str(row["candidate_id"])))


def select_probe_resets(
    resets: list[dict[str, Any]], reset_ids: tuple[str, ...], *, repeats_per_reset: int
) -> list[tuple[dict[str, Any], int]]:
    """Select exact reset identities and expand deterministic probe repetitions."""
    if repeats_per_reset < 1:
        raise ValueError("repeats_per_reset must be positive")
    if not reset_ids:
        return [(reset, repeat) for reset in resets for repeat in range(repeats_per_reset)]
    by_id: dict[str, dict[str, Any]] = {}
    for reset in resets:
        reset_id = str(reset.get("reset_id", ""))
        if reset_id in by_id:
            raise ValueError(f"duplicate reset id in bank: {reset_id}")
        by_id[reset_id] = reset
    missing = [reset_id for reset_id in reset_ids if reset_id not in by_id]
    if missing:
        raise ValueError(f"missing requested reset ids: {missing}")
    return [(by_id[reset_id], repeat) for reset_id in reset_ids for repeat in range(repeats_per_reset)]


def select_probe_candidate(candidates: list[ContactCandidate], candidate_id: str) -> ContactCandidate:
    """Resolve one exact geometry candidate or fail closed."""
    matches = [candidate for candidate in candidates if candidate.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one candidate {candidate_id!r}, found {len(matches)}")
    return matches[0]


def candidate_payload(candidate: ContactCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "object_name": candidate.object_name,
        "contact_pos": candidate.contact_pos.tolist(),
        "orientation": candidate.orientation.tolist(),
        "flow": candidate.flow.tolist(),
        "half_extents": candidate.half_extents.tolist(),
        "surface_normal": candidate.surface_normal.tolist(),
        "height_fraction": candidate.height_fraction,
        "tool_clearance": candidate.tool_clearance,
        "geometry_signature": list(candidate.geometry_signature),
    }


def quaternion_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(2.0 * math.acos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


def build_geometry_guidance(
    candidate: ContactCandidate,
    robot_obs: np.ndarray,
    initial_object_pos: np.ndarray,
    current_object_pos: np.ndarray,
    *,
    motion_threshold: float,
) -> dict[str, Any]:
    progress = float(np.dot(current_object_pos - initial_object_pos, candidate.flow))
    post_contact = progress >= motion_threshold
    anchor = np.asarray(current_object_pos, dtype=np.float32) + (
        candidate.contact_pos - np.asarray(initial_object_pos, dtype=np.float32)
    )
    attraction = np.zeros(3, dtype=np.float32) if post_contact else anchor - np.asarray(robot_obs[:3], dtype=np.float32)
    return {
        "g": np.concatenate([attraction, candidate.flow]).astype(np.float32),
        "phase": "contact_transport" if post_contact else "approach",
        "post_contact": bool(post_contact),
        "progress": progress,
        "interaction_anchor": anchor,
    }


def deterministic_native_reset(env: Any, robot_reset: np.ndarray, scene_reset: np.ndarray) -> dict[str, Any]:
    """Reset poses and erase movable-body velocity history before a rollout.

    CALVIN's movable-object reset restores pose but leaves PyBullet linear and
    angular velocities untouched. The first pass clears stale velocities; the
    second pass restores exact requested poses after CALVIN's mandatory physics
    step, then quiesces any residual velocity generated by that step.
    """

    def quiesce_movable_bodies() -> None:
        info = env.get_info()
        movable = info["scene_info"]["movable_objects"]
        for object_info in movable.values():
            env.p.resetBaseVelocity(
                int(object_info["uid"]),
                linearVelocity=[0.0, 0.0, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=env.cid,
            )

    env.reset(robot_obs=robot_reset, scene_obs=scene_reset)
    quiesce_movable_bodies()
    env.reset(robot_obs=robot_reset, scene_obs=scene_reset)
    quiesce_movable_bodies()
    return env.get_obs()


def execute_candidate(
    env: Any,
    oracle: Any,
    split_dir: Path,
    task: str,
    reset: dict[str, Any],
    candidate: ContactCandidate,
    *,
    max_steps: int,
    motion_threshold: float,
    collect_trajectory: bool,
    emit_provisional_g: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    spec = base.parse_push_task(task)
    robot_reset, scene_reset = base.load_reset_arrays(split_dir, reset)
    obs = deterministic_native_reset(env, robot_reset, scene_reset)
    start_info = env.get_info()
    start_object_info = start_info["scene_info"]["movable_objects"][spec["object_name"]]
    uid = int(start_object_info["uid"])
    collision_shapes = env.p.getCollisionShapeData(uid, -1, physicsClientId=env.cid)
    if not collision_shapes:
        raise RuntimeError(f"No collision geometry for {spec['object_name']}")
    live_candidates = generate_oriented_contact_candidates(
        object_name=spec["object_name"],
        object_pos=np.asarray(start_object_info["current_pos"], dtype=np.float32),
        object_orn=np.asarray(start_object_info["current_orn"], dtype=np.float32),
        local_extents=np.asarray(collision_shapes[0][3], dtype=np.float32),
        flow=spec["flow"],
    )
    matching_candidates = [row for row in live_candidates if row.geometry_signature == candidate.geometry_signature]
    if len(matching_candidates) != 1:
        raise RuntimeError(f"Could not rederive candidate signature {candidate.geometry_signature}")
    candidate = matching_candidates[0]
    initial_pos = np.asarray(start_object_info["current_pos"], dtype=np.float32)
    initial_orn = np.asarray(start_object_info["current_orn"], dtype=np.float32)
    robot_uid = int(start_info["robot_info"]["uid"])
    start_support_contacts = {
        (int(contact[2]), int(contact[4]))
        for contact in start_object_info["contacts"]
        if int(contact[2]) != robot_uid
    }
    controller = GeometryContactController(candidate, push_distance=0.02)
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    steps: list[dict[str, Any]] = []
    achieved: list[str] = []
    success_step: int | None = None
    robot_object_contacts = 0
    step_index = -1

    for step_index in range(max_steps):
        current_info = env.get_info()
        current_object_info = current_info["scene_info"]["movable_objects"][spec["object_name"]]
        current_pos = np.asarray(current_object_info["current_pos"], dtype=np.float32)
        live_candidates = generate_oriented_contact_candidates(
            object_name=spec["object_name"],
            object_pos=current_pos,
            object_orn=np.asarray(current_object_info["current_orn"], dtype=np.float32),
            local_extents=candidate.half_extents * np.float32(2.0),
            flow=spec["flow"],
        )
        live_candidate = next(
            row for row in live_candidates if row.geometry_signature == candidate.geometry_signature
        )
        controller.update_live_contact_frame(live_candidate)
        contact_detected = any(int(contact[2]) == robot_uid for contact in current_object_info["contacts"])
        guidance = build_geometry_guidance(
            live_candidate,
            np.asarray(obs["robot_obs"], dtype=np.float32),
            initial_pos,
            current_pos,
            motion_threshold=motion_threshold,
        )
        command = controller.command(
            np.asarray(obs["robot_obs"], dtype=np.float32), contact_detected=contact_detected
        )
        absolute_action = command["absolute_action"]
        rel_action = base.absolute_to_relative_action(absolute_action, np.asarray(obs["robot_obs"], dtype=np.float32))
        if collect_trajectory:
            rgb_obs = obs["rgb_obs"]
            steps.append(
                base.attach_provisional_labels(
                    {
                        "robot_obs": np.asarray(obs["robot_obs"], dtype=np.float32).copy(),
                        "scene_obs": np.asarray(obs["scene_obs"], dtype=np.float32).copy(),
                        "rgb_static": np.asarray(rgb_obs["rgb_static"], dtype=np.uint8).copy(),
                        "rgb_gripper": np.asarray(rgb_obs["rgb_gripper"], dtype=np.uint8).copy(),
                        "actions": absolute_action.copy(),
                        "rel_actions": rel_action,
                        "progress": guidance["progress"],
                        "controller_state": command["state"],
                        "object_pos": current_pos,
                        "object_orn": np.asarray(current_object_info["current_orn"], dtype=np.float32),
                        "contact_detected": np.asarray(contact_detected, dtype=bool),
                        "task_flow": np.asarray(spec["flow"], dtype=np.float32),
                        "live_contact_pos": live_candidate.contact_pos.copy(),
                        "surface_normal": live_candidate.surface_normal.copy(),
                    },
                    guidance,
                    emit=emit_provisional_g,
                )
            )
        obs, _, _, final_info = env.step(
            (absolute_action[:3].astype(float), absolute_action[3:6].astype(float), int(absolute_action[6]))
        )
        final_object_info = final_info["scene_info"]["movable_objects"][spec["object_name"]]
        robot_object_contacts += sum(int(contact[2]) == robot_uid for contact in final_object_info["contacts"])
        achieved = sorted(oracle.get_task_info_for_set(start_info, final_info, {task}))
        if task in achieved:
            success_step = step_index
            break
        live_pos = np.asarray(final_object_info["current_pos"], dtype=np.float32)
        delta = live_pos - initial_pos
        progress = float(np.dot(delta, candidate.flow))
        lateral = float(np.linalg.norm(delta - candidate.flow * np.float32(progress)))
        if live_pos[2] < 0.35 or lateral > 0.20:
            break

    final_info = env.get_info()
    final_object_info = final_info["scene_info"]["movable_objects"][spec["object_name"]]
    final_pos = np.asarray(final_object_info["current_pos"], dtype=np.float32)
    final_orn = np.asarray(final_object_info["current_orn"], dtype=np.float32)
    delta = final_pos - initial_pos
    progress = float(np.dot(delta, candidate.flow))
    lateral_vector = delta - candidate.flow * np.float32(progress)
    lateral_drift = float(np.linalg.norm(lateral_vector[:2]))
    z_drop = max(float(initial_pos[2] - final_pos[2]), 0.0)
    tilt = quaternion_angle(initial_orn, final_orn)
    end_support_contacts = {
        (int(contact[2]), int(contact[4]))
        for contact in final_object_info["contacts"]
        if int(contact[2]) != robot_uid
    }
    support_contact = bool(end_support_contacts)
    support_preserved = bool(start_support_contacts) and start_support_contacts.issubset(end_support_contacts)
    success = success_step is not None
    score = score_preview(
        progress=progress,
        lateral_drift=lateral_drift,
        z_drop=z_drop,
        tilt=tilt,
        support_contact=support_preserved,
        robot_object_contacts=robot_object_contacts,
        oracle_success=success,
    )
    receipt = {
        "task": task,
        "family": "block_push",
        "object_name": spec["object_name"],
        "direction": spec["direction"],
        "oracle_success": success,
        "oracle_achieved_tasks": achieved,
        "success_step": success_step,
        "executed_steps": step_index + 1,
        "initial_object_pos": initial_pos.tolist(),
        "final_object_pos": final_pos.tolist(),
        "final_progress": progress,
        "lateral_drift": lateral_drift,
        "z_drop": z_drop,
        "tilt": tilt,
        "support_contact": support_contact,
        "support_preserved": support_preserved,
        "start_support_contacts": sorted(start_support_contacts),
        "end_support_contacts": sorted(end_support_contacts),
        "robot_object_contacts": robot_object_contacts,
        "score": score,
        "candidate": candidate_payload(candidate),
        "reset": reset,
    }
    if not collect_trajectory:
        return None, receipt
    trajectory = base.stack_trajectory(
        steps,
        {
            "task": task,
            "object_name": spec["object_name"],
            "direction": spec["direction"],
            "success": success,
            "provisional_label_schema": "candidate_g_af_v1" if emit_provisional_g else "not_emitted",
            "source_kind": "geometry_contact_pose_search_teacher",
            "action_mode": "absolute_pose_executed_with_relative_action_labels",
            "reset_robot_obs": robot_reset,
            "reset_scene_obs": scene_reset,
            "reset_start": int(reset.get("start", -1)),
            "reset_end": int(reset.get("end", -1)),
            "reset_id": str(reset.get("reset_id", f"episode_{int(reset.get('start', -1)):07d}")),
            "reset_source_kind": str(reset.get("source_kind", "source_episode_frame")),
            "reset_scene": str(reset["scene"]),
            "runtime_scene": str(reset.get("runtime_scene", reset["scene"])),
            "initial_object_pos": initial_pos,
            "final_object_pos": final_pos,
            "planner_candidate_id": candidate.candidate_id,
            "planner_contact_pos": candidate.contact_pos,
            "planner_orientation": candidate.orientation,
            "planner_half_extents": candidate.half_extents,
        },
    )
    return trajectory, receipt


def validate_geometry_collection(output_root: Path) -> dict[str, Any]:
    root = Path(output_root)
    with open(root / "collection_manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("planner_format") != "calvin_geometry_contact_push_collection_v1":
        raise ValueError("Unexpected geometry planner collection format")
    tasks = tuple(manifest["tasks"])
    counts = {task: 0 for task in tasks}
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
    for row in manifest.get("results", []):
        if not row.get("path"):
            if row.get("oracle_success"):
                raise ValueError("Successful planner result cannot omit trajectory path")
            continue
        if not row.get("oracle_success") and str(row["path"]) in pruned_paths:
            pruned_verified += 1
            continue
        path = root / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if base.sha256_file(path) != row["sha256"]:
            raise ValueError(f"Trajectory hash mismatch: {path}")
        with open(path.with_suffix(".receipt.json"), encoding="utf-8") as stream:
            receipt = json.load(stream)
        if receipt.get("trajectory_sha256") != row["sha256"]:
            raise ValueError(f"Receipt hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            trajectory = {name: payload[name] for name in payload.files}
        base.validate_parametric_trajectory(trajectory)
        success = bool(np.asarray(trajectory["success"]).item())
        if success != bool(row["oracle_success"]):
            raise ValueError(f"Oracle/trajectory mismatch: {path}")
        if success:
            counts[row["task"]] += 1
            verified += 1
    complete = all(counts[task] >= int(manifest["quota_per_cell"]) for task in tasks)
    return {
        "accepted_verified": verified,
        "accepted_per_cell": counts,
        "quota_per_cell": int(manifest["quota_per_cell"]),
        "complete": complete,
        "folds_verified": 0,
        "failed_payloads_pruned": pruned_verified,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--reset-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--directions", default="forward,backward")
    parser.add_argument("--colors", default=",".join(base.COLORS))
    parser.add_argument("--quota-per-cell", type=int, default=1)
    parser.add_argument("--max-attempts-per-cell", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--reset-ids", default="")
    parser.add_argument("--fixed-candidate-id", default="")
    parser.add_argument("--repeats-per-reset", type=int, default=1)
    parser.add_argument("--preview-steps", type=int, default=120)
    parser.add_argument("--final-steps", type=int, default=200)
    parser.add_argument("--motion-threshold", type=float, default=0.003)
    parser.add_argument("--failure-storage", choices=("metadata_only", "stride"), default="metadata_only")
    parser.add_argument("--failure-stride", type=int, default=8)
    parser.add_argument("--emit-provisional-g", action="store_true")
    parser.add_argument("--seed", type=int, default=20261021)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.calvin_code = args.calvin_code.resolve()
    args.source_split = args.source_split.resolve()
    args.reset_manifest = args.reset_manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    if (args.out_dir / "collection_manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite collection: {args.out_dir}")
    directions = tuple(part.strip() for part in args.directions.split(",") if part.strip())
    if not directions or set(directions) - set(base.ALL_DIRECTIONS):
        raise ValueError(f"invalid directions: {directions}")
    colors = tuple(part.strip() for part in args.colors.split(",") if part.strip())
    if not colors or set(colors) - set(base.COLORS):
        raise ValueError(f"invalid colors: {colors}")
    tasks = tuple(f"push_{color}_block_{direction}" for color in colors for direction in directions)
    reset_bank = base.load_reset_bank(args.reset_manifest, seed=args.seed, tasks=tasks)
    probe_reset_ids = tuple(part.strip() for part in args.reset_ids.split(",") if part.strip())
    if args.repeats_per_reset < 1:
        raise ValueError("repeats-per-reset must be positive")

    calvin_models = args.calvin_code / "calvin_models"
    calvin_env = args.calvin_code / "calvin_env"
    sys.path[:0] = [calvin_env.as_posix(), calvin_models.as_posix()]
    import hydra  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415
    from calvin_env.envs.play_table_env import get_env  # noqa: PLC0415

    task_cfg_path = calvin_models / "conf/callbacks/rollout/tasks/new_playtable_tasks.yaml"
    task_cfg = OmegaConf.load(task_cfg_path)
    for color in base.COLORS:
        task_cfg.tasks[f"push_{color}_block_forward"] = ["push_object", f"block_{color}", 0.0, 0.1]
        task_cfg.tasks[f"push_{color}_block_backward"] = ["push_object", f"block_{color}", 0.0, -0.1]
    oracle = hydra.utils.instantiate(task_cfg)

    def create_env() -> Any:
        return get_env(
            args.source_split,
            show_gui=False,
            obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []},
        )

    def close_env(env: Any) -> None:
        env.close()
        env.ownsPhysicsClient = False

    def execute_in_fresh_env(
        task: str,
        reset: dict[str, Any],
        candidate: ContactCandidate,
        *,
        max_steps: int,
        collect_trajectory: bool,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        env = create_env()
        try:
            return execute_candidate(
                env,
                oracle,
                args.source_split,
                task,
                reset,
                candidate,
                max_steps=max_steps,
                motion_threshold=args.motion_threshold,
                collect_trajectory=collect_trajectory,
                emit_provisional_g=args.emit_provisional_g if collect_trajectory else False,
            )
        finally:
            close_env(env)

    results: list[dict[str, Any]] = []
    accepted_counts = {task: 0 for task in tasks}
    for task in tasks:
        spec = base.parse_push_task(task)
        if probe_reset_ids:
            attempts = select_probe_resets(
                reset_bank[task], probe_reset_ids, repeats_per_reset=args.repeats_per_reset
            )
        else:
            attempts = select_probe_resets(
                reset_bank[task][: args.max_attempts_per_cell], (), repeats_per_reset=args.repeats_per_reset
            )
        for attempt_index, (reset, repeat_index) in enumerate(attempts):
            if accepted_counts[task] >= args.quota_per_cell:
                break
            geometry_env = create_env()
            try:
                robot_reset, scene_reset = base.load_reset_arrays(args.source_split, reset)
                deterministic_native_reset(geometry_env, robot_reset, scene_reset)
                info = geometry_env.get_info()
                object_info = info["scene_info"]["movable_objects"][spec["object_name"]]
                uid = int(object_info["uid"])
                collision_shapes = geometry_env.p.getCollisionShapeData(
                    uid, -1, physicsClientId=geometry_env.cid
                )
                if not collision_shapes:
                    raise RuntimeError(f"No collision geometry for {spec['object_name']}")
                local_extents = np.asarray(collision_shapes[0][3], dtype=np.float32)
                generated_candidates = generate_oriented_contact_candidates(
                    object_name=spec["object_name"],
                    object_pos=np.asarray(object_info["current_pos"], dtype=np.float32),
                    object_orn=np.asarray(object_info["current_orn"], dtype=np.float32),
                    local_extents=local_extents,
                    flow=spec["flow"],
                )
                if args.fixed_candidate_id:
                    candidates = [select_probe_candidate(generated_candidates, args.fixed_candidate_id)]
                else:
                    candidates = generated_candidates[: args.max_candidates]
            finally:
                close_env(geometry_env)
            trials: list[dict[str, Any]] = []
            for candidate in candidates:
                _, trial = execute_in_fresh_env(
                    task,
                    reset,
                    candidate,
                    max_steps=args.preview_steps,
                    collect_trajectory=False,
                )
                trial["candidate_id"] = candidate.candidate_id
                trials.append(trial)
            selected = select_candidate_trial(trials)
            selected_candidate = next(c for c in candidates if c.candidate_id == selected["candidate_id"])
            reset_token = str(reset.get("reset_id", f"episode_{int(reset.get('start', -1)):07d}"))
            repeat_token = f"_repeat_{repeat_index:02d}" if args.repeats_per_reset > 1 else ""
            attempt_id = (
                f"{task}_{reset_token}_geometry_seed_{args.seed}{repeat_token}_try_{attempt_index:02d}"
            )
            if not bool(selected["oracle_success"]):
                results.append(
                    {
                        "attempt_id": attempt_id,
                        "task": task,
                        "oracle_success": False,
                        "executed_steps": 0,
                        "stored_steps": 0,
                        "final_progress": selected["final_progress"],
                        "score": selected["score"],
                        "selected_candidate": candidate_payload(selected_candidate),
                        "planner_trials": trials,
                        "reset": reset,
                        "path": None,
                    }
                )
                print(
                    f"[planner] task={task} reset={reset_token} preview_success=0 "
                    f"best={selected_candidate.candidate_id} score={selected['score']:.4f}",
                    flush=True,
                )
                continue
            trajectory, receipt = execute_in_fresh_env(
                task,
                reset,
                selected_candidate,
                max_steps=args.final_steps,
                collect_trajectory=True,
            )
            assert trajectory is not None
            receipt.update(
                {
                    "attempt_id": attempt_id,
                    "seed": args.seed,
                    "planner_trials": trials,
                    "selected_candidate": candidate_payload(selected_candidate),
                }
            )
            if not bool(receipt["oracle_success"]) and args.failure_storage == "metadata_only":
                receipt["failure_storage_mode"] = "metadata_only"
                receipt["failure_stride"] = None
                receipt["stored_steps"] = 0
                receipt_path = base.persist_failure_metadata_only(
                    args.out_dir, attempt_id=attempt_id, receipt=receipt
                )
                row = dict(receipt)
                row["path"] = None
                row["receipt_path"] = receipt_path.relative_to(args.out_dir).as_posix()
                row["sha256"] = None
            else:
                if not bool(receipt["oracle_success"]) and args.failure_stride > 1:
                    trajectory = base.lightweight_failure_trajectory(trajectory, stride=args.failure_stride)
                    receipt["failure_storage_mode"] = "temporal_stride"
                    receipt["failure_stride"] = args.failure_stride
                else:
                    receipt["failure_storage_mode"] = "full_rate"
                    receipt["failure_stride"] = 1
                receipt["stored_steps"] = len(np.asarray(trajectory["robot_obs"]))
                path = base.persist_attempt(args.out_dir, attempt_id=attempt_id, trajectory=trajectory, receipt=receipt)
                row = dict(receipt)
                row["path"] = path.relative_to(args.out_dir).as_posix()
                row["sha256"] = base.sha256_file(path)
            results.append(row)
            if row["oracle_success"]:
                accepted_counts[task] += 1
            print(
                f"[planner] task={task} reset={reset_token} preview_success=1 "
                f"final_success={int(row['oracle_success'])} candidate={selected_candidate.candidate_id} "
                f"progress={row['final_progress']:.4f} accepted={accepted_counts[task]}/{args.quota_per_cell}",
                flush=True,
            )

    complete = all(value >= args.quota_per_cell for value in accepted_counts.values())
    manifest = {
        "format": "calvin_parametric_push_collection_v1",
        "planner_format": "calvin_geometry_contact_push_collection_v1",
        "provenance": {
            "collector": Path(__file__).resolve(),
            "collector_sha256": base.sha256_file(Path(__file__).resolve()),
            "base_collector": Path(base.__file__).resolve(),
            "base_collector_sha256": base.sha256_file(Path(base.__file__).resolve()),
            "calvin_code": args.calvin_code,
            "calvin_task_config": task_cfg_path,
            "calvin_task_config_sha256": base.sha256_file(task_cfg_path),
            "source_split": args.source_split,
            "reset_manifest": args.reset_manifest,
            "reset_manifest_sha256": base.sha256_file(args.reset_manifest),
            "planner": "OBB COM-ray contact frame plus short-horizon simulator candidate search",
            "candidate_preview_policy": "diagnostic_only_selected_candidate_rerun_fresh",
            "candidate_execution_isolation": "fresh_calvin_environment_per_preview_and_final_rollout",
            "reset_hygiene": "two_pass_native_pose_reset_with_all_movable_linear_angular_velocity_zeroed",
            "acceptance_gate": "CALVIN get_task_info_for_set exact requested task",
        },
        "seed": args.seed,
        "tasks": list(tasks),
        "factor_axes": {"object": list(colors), "direction": list(directions)},
        "quota_per_cell": args.quota_per_cell,
        "max_attempts_per_cell": args.max_attempts_per_cell,
        "max_candidates": args.max_candidates,
        "probe_mode": {
            "enabled": bool(probe_reset_ids or args.fixed_candidate_id or args.repeats_per_reset != 1),
            "reset_ids": list(probe_reset_ids),
            "fixed_candidate_id": args.fixed_candidate_id or None,
            "repeats_per_reset": args.repeats_per_reset,
            "candidate_search_disabled": bool(args.fixed_candidate_id),
        },
        "preview_steps": args.preview_steps,
        "max_steps": args.final_steps,
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
        "results": results,
        "folds": [],
    }
    base.write_json_atomic(args.out_dir / "collection_manifest.json", manifest)
    summary = validate_geometry_collection(args.out_dir)
    print(json.dumps(summary, default=base.json_default), flush=True)
    if not complete or not summary["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
