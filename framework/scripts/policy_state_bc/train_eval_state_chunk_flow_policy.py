#!/usr/bin/env python3
"""Train/evaluate chunked BC and conditional flow-matching CALVIN state policies."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

for _alias, _target in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
}.items():
    if _alias not in np.__dict__:
        setattr(np, _alias, _target)

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402


def json_default(value: Any) -> Any:
    return base.json_default(value)


SUPPORTED_MODES = {
    "chunk_bc_state",
    "chunk_bc_lang",
    "chunk_bc_query_g",
    "chunk_bc_lang_query_g",
    "chunk_bc_lang_minimal_g_concat",
    "chunk_bc_minimal_g_concat",
    "chunk_bc_minimal_g_film",
    "chunk_bc_minimal_g_tokens",
    "chunk_fm_minimal_g_tokens",
    "chunk_bc_minimal_g_semantic",
    "chunk_bc_minimal_g_frame",
    "chunk_bc_lang_minimal_g_structured",
    "chunk_bc_lang_minimal_g_projected",
    "chunk_bc_minimal_g_projected",
    "chunk_bc_lang_minimal_g_bounded_projected",
    "chunk_bc_minimal_g_bounded_projected",
    "chunk_bc_lang_minimal_g_invariant_bounded_projected",
    "chunk_bc_minimal_g_invariant_bounded_projected",
    "chunk_bc_v7",
    "chunk_bc_v7_scale_aug",
    "chunk_bc_v7_gated",
    "chunk_fm_v7",
}

MASKED_LANG_MINIMAL_G_LEARNED_MODES = {
    "chunk_bc_minimal_g_concat",
    "chunk_bc_minimal_g_film",
    "chunk_bc_minimal_g_tokens",
    "chunk_fm_minimal_g_tokens",
    "chunk_bc_minimal_g_semantic",
    "chunk_bc_minimal_g_frame",
}

LANG_MINIMAL_G_PROJECTED_MODES = {
    "chunk_bc_lang_minimal_g_projected",
    "chunk_bc_lang_minimal_g_bounded_projected",
    "chunk_bc_lang_minimal_g_invariant_bounded_projected",
}
MASKED_LANG_MINIMAL_G_PROJECTED_MODES = {
    "chunk_bc_minimal_g_projected",
    "chunk_bc_minimal_g_bounded_projected",
    "chunk_bc_minimal_g_invariant_bounded_projected",
}
MINIMAL_G_PROJECTED_MODES = LANG_MINIMAL_G_PROJECTED_MODES | MASKED_LANG_MINIMAL_G_PROJECTED_MODES
BOUNDED_MINIMAL_G_PROJECTED_MODES = {
    "chunk_bc_lang_minimal_g_bounded_projected",
    "chunk_bc_minimal_g_bounded_projected",
    "chunk_bc_lang_minimal_g_invariant_bounded_projected",
    "chunk_bc_minimal_g_invariant_bounded_projected",
}
INVARIANT_GAIN_MINIMAL_G_PROJECTED_MODES = {
    "chunk_bc_lang_minimal_g_invariant_bounded_projected",
    "chunk_bc_minimal_g_invariant_bounded_projected",
}


QUERY_G_TOKEN_COUNT = 4
QUERY_G_TOKEN_DIM = 19
QUERY_G_TOKEN_DIM_NO_TYPE = 15
MINIMAL_G_DIM = 6
MINIMAL_G_SCOPES = {"articulation", "translational"}
EXECUTOR_STATE_SCOPES = {"full", "proprio", "visual"}
EXECUTOR_LANGUAGE_MODES = {"none", "onehot", "minilm"}
BLOCK_SCENE_SLICES = {
    "red": slice(6, 9),
    "blue": slice(12, 15),
    "pink": slice(18, 21),
}
BLOCK_PUSH_TASKS = {
    f"push_{color}_block_{side}": (color, np.asarray([sign, 0.0, 0.0], dtype=np.float32))
    for color in BLOCK_SCENE_SLICES
    for side, sign in (("left", -1.0), ("right", 1.0))
}
DINO_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINO_IMAGE_STD = (0.229, 0.224, 0.225)
QUERY_G_FEATURE_GROUPS: dict[str, tuple[int, ...]] = {
    "token_type": tuple(range(0, 4)),
    "rel_eef": tuple(range(4, 7)),
    "rel_part": tuple(range(7, 10)),
    "field_vec": tuple(range(10, 13)),
    "desired_flow": tuple(range(13, 16)),
    "dist_norm": (16,),
    "residual_norm": (17,),
    "contact_conf": (18,),
}


def drop_query_g_token_type(query_g: np.ndarray) -> np.ndarray:
    """Remove the redundant four-way role one-hot from every fixed-slot token."""

    shaped = np.asarray(query_g, dtype=np.float32).reshape(
        query_g.shape[0], QUERY_G_TOKEN_COUNT, QUERY_G_TOKEN_DIM
    )
    return shaped[:, :, 4:].reshape(query_g.shape[0], -1).astype(np.float32)


def zero_query_g_token_type(query_g: np.ndarray) -> np.ndarray:
    """Keep the 76-D layout but remove all fixed-slot role one-hot signal."""

    shaped = np.asarray(query_g, dtype=np.float32).reshape(
        query_g.shape[0], QUERY_G_TOKEN_COUNT, QUERY_G_TOKEN_DIM
    ).copy()
    shaped[:, :, :4] = 0.0
    return shaped.reshape(query_g.shape[0], -1).astype(np.float32)


def query_g_feature_layout(include_token_type: bool) -> tuple[int, dict[str, tuple[int, ...]]]:
    if include_token_type:
        return QUERY_G_TOKEN_DIM, QUERY_G_FEATURE_GROUPS
    return QUERY_G_TOKEN_DIM_NO_TYPE, {
        name: tuple(dim - 4 for dim in dims)
        for name, dims in QUERY_G_FEATURE_GROUPS.items()
        if name != "token_type"
    }


def make_minimal_articulation_g(query_g: np.ndarray) -> np.ndarray:
    """Return coherent attraction and object-flow vectors derived from query-G."""

    shaped = np.asarray(query_g, dtype=np.float32).reshape(
        query_g.shape[0], QUERY_G_TOKEN_COUNT, QUERY_G_TOKEN_DIM
    )
    attraction = shaped[:, 2, 4:7]
    desired_object_flow = shaped[:, 0, 13:16]
    return np.concatenate([attraction, desired_object_flow], axis=1).astype(np.float32)


def make_behavior_block_center_minimal_g(
    robot: np.ndarray,
    scene: np.ndarray,
    task_names: np.ndarray,
    *,
    contact_radius: float = 0.01,
) -> np.ndarray:
    """Build the online counterpart of the demonstration-derived push field."""

    out: list[np.ndarray] = []
    for robot_obs, scene_obs, task_value in zip(robot, scene, task_names):
        task = str(task_value)
        if task not in BLOCK_PUSH_TASKS:
            out.append(np.zeros(6, dtype=np.float32))
            continue
        color, desired_flow = BLOCK_PUSH_TASKS[task]
        eef = np.asarray(robot_obs[:3], dtype=np.float32)
        part = np.asarray(scene_obs[BLOCK_SCENE_SLICES[color]], dtype=np.float32)
        delta = part - eef
        distance = float(np.linalg.norm(delta))
        if distance <= contact_radius:
            attraction = np.zeros(3, dtype=np.float32)
        else:
            attraction = delta * np.float32(1.0 - contact_radius / distance)
        out.append(np.concatenate([attraction, desired_flow]).astype(np.float32))
    return np.stack(out).astype(np.float32)


def apply_minimal_g_control(minimal_g: np.ndarray, control: str) -> np.ndarray:
    """Apply the causal controls used by minimal-G policy evaluations."""

    if control == "normal":
        return minimal_g.astype(np.float32)
    if control == "zero":
        return np.zeros_like(minimal_g, dtype=np.float32)
    controlled = np.asarray(minimal_g, dtype=np.float32).copy()
    if control == "sign_flip":
        controlled[:, 3:6] *= -1.0
    elif control == "zero_attraction":
        controlled[:, :3] = 0.0
    elif control == "zero_flow":
        controlled[:, 3:6] = 0.0
    elif control == "sign_flip_attraction":
        controlled[:, :3] *= -1.0
    else:
        raise ValueError(f"Unsupported minimal-G control: {control}")
    return controlled.astype(np.float32)


def parse_task_segment_limits(items: list[str] | None) -> dict[str, int]:
    limits: dict[str, int] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected TASK=COUNT for --task-segment-limits, got: {item}")
        task, value = item.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError(f"Empty task name in --task-segment-limits item: {item}")
        count = int(value)
        if count < 0:
            raise ValueError(f"Negative segment limit for {task}: {count}")
        limits[task] = count
    return limits


def parse_task_loss_weights(items: list[str] | None) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected TASK=WEIGHT, got: {item}")
        task, value = item.split("=", 1)
        task = task.strip()
        weight = float(value)
        if not task or weight <= 0:
            raise ValueError(f"Invalid task loss weight: {item}")
        weights[task] = weight
    return weights


def parse_reverse_task_pairs(items: list[str] | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected SOURCE=TARGET for --reverse-task-pairs, got: {item}")
        source, target = (part.strip() for part in item.split("=", 1))
        if not source or not target:
            raise ValueError(f"Invalid reverse task pair: {item}")
        pair = (source, target)
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _scene_block_positions(scene: np.ndarray) -> np.ndarray:
    return np.stack([scene[..., field] for field in BLOCK_SCENE_SLICES.values()], axis=-2).astype(np.float32)


def fit_primitive_query_geometry(
    data_root: Path,
    required_tasks: set[str] | None = None,
) -> dict[str, Any]:
    """Fit task-level target regions without exposing future frames at runtime."""

    split_dir = data_root / "training"
    supported_tasks = {"push_into_drawer", "place_in_slider"}
    if required_tasks is not None:
        supported_tasks &= set(required_tasks)
    targets_by_task: dict[str, list[np.ndarray]] = {task: [] for task in supported_tasks}
    deltas_by_task: dict[str, list[np.ndarray]] = {task: [] for task in supported_tasks}
    active_indices_by_task: dict[str, list[int]] = {task: [] for task in supported_tasks}
    for segment in base.load_lang_segments(split_dir):
        task = str(segment["task"])
        if task not in supported_tasks:
            continue
        start_file = base.episode_file(split_dir, int(segment["start"]))
        end_file = base.episode_file(split_dir, int(segment["end"]))
        if not start_file.exists() or not end_file.exists():
            continue
        try:
            with np.load(start_file, allow_pickle=True) as start_data:
                start_scene = np.asarray(start_data["scene_obs"], dtype=np.float32)
            with np.load(end_file, allow_pickle=True) as end_data:
                end_scene = np.asarray(end_data["scene_obs"], dtype=np.float32)
        except Exception:
            continue
        start_blocks = _scene_block_positions(start_scene)
        end_blocks = _scene_block_positions(end_scene)
        block_deltas = end_blocks - start_blocks
        active_idx = int(np.argmax(np.linalg.norm(block_deltas, axis=1)))
        active_delta = block_deltas[active_idx]
        if float(np.linalg.norm(active_delta)) < 0.03:
            continue
        targets_by_task[task].append(end_blocks[active_idx])
        deltas_by_task[task].append(active_delta)
        active_indices_by_task[task].append(active_idx)

    result: dict[str, Any] = {}
    for task in sorted(supported_tasks):
        deltas = deltas_by_task[task]
        if not deltas:
            raise RuntimeError(f"No valid {task} demonstration endpoints were found")
        delta_array = np.stack(deltas).astype(np.float32)
        unit_delta = delta_array / np.maximum(
            np.linalg.norm(delta_array, axis=1, keepdims=True), 1e-6
        )
        flow_dir = base._safe_unit(np.median(unit_delta, axis=0).astype(np.float32))
        target_center = np.median(np.stack(targets_by_task[task]), axis=0).astype(np.float32)
        active_indices = active_indices_by_task[task]
        result[task] = {
            "flow_dir": flow_dir.tolist(),
            "target_center": target_center.tolist(),
            "active_object_selector": "nearest_eef_3d",
            "median_displacement": float(np.median(np.linalg.norm(delta_array, axis=1))),
            "num_segments": int(len(deltas)),
            "active_color_counts": {
                color: int(sum(index == color_idx for index in active_indices))
                for color_idx, color in enumerate(BLOCK_SCENE_SLICES)
            },
        }
    return result


def make_state_lang_parts(
    arrays: dict[str, np.ndarray],
    state_scope: str = "full",
    language_mode: str = "onehot",
) -> tuple[np.ndarray, np.ndarray]:
    if state_scope not in EXECUTOR_STATE_SCOPES:
        raise ValueError(f"Unsupported executor state scope: {state_scope}")
    state_pieces = [arrays["robot_obs"]]
    if state_scope == "full":
        state_pieces.append(arrays["scene_obs"])
    elif state_scope == "visual":
        if "visual_static" not in arrays or "visual_gripper" not in arrays:
            raise RuntimeError("Visual executor state requires static and gripper features")
        state_pieces.extend([arrays["visual_static"], arrays["visual_gripper"]])
    state_x = np.concatenate(state_pieces, axis=1).astype(np.float32)
    if language_mode not in EXECUTOR_LANGUAGE_MODES:
        raise ValueError(f"Unsupported executor language mode: {language_mode}")
    if language_mode == "none":
        lang_x = np.zeros((state_x.shape[0], 0), dtype=np.float32)
    elif language_mode == "minilm":
        if "lang_emb" not in arrays or arrays["lang_emb"].shape[1] == 0:
            raise RuntimeError("MiniLM executor language requires non-empty lang_emb")
        lang_x = arrays["lang_emb"].astype(np.float32)
    else:
        lang_x = arrays["task_oh"].astype(np.float32)
    return state_x, lang_x


def make_articulation_query_g(
    robot: np.ndarray,
    scene: np.ndarray,
    task_names: np.ndarray,
    geometry: dict[str, Any],
    *,
    minimal_g_scope: str = "articulation",
    primitive_geometry: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build compact oracle query-G tokens for articulation and push tasks.

    The task/language is used only to build these oracle tokens. A policy mode
    can then choose whether to also receive the language/task vector directly.
    """

    if minimal_g_scope not in MINIMAL_G_SCOPES:
        raise ValueError(f"Unsupported minimal-G scope: {minimal_g_scope}")

    out: list[np.ndarray] = []

    def token(
        token_type: int,
        q: np.ndarray,
        eef: np.ndarray,
        part: np.ndarray,
        field_vec: np.ndarray,
        desired_flow: np.ndarray,
        residual_norm: float,
        radius: float,
    ) -> np.ndarray:
        type_vec = np.zeros(4, dtype=np.float32)
        type_vec[int(token_type)] = 1.0
        rel_eef = (q - eef).astype(np.float32)
        rel_part = (q - part).astype(np.float32)
        dist = float(np.linalg.norm(q - part))
        dist_norm = np.float32(dist / max(radius, 1e-6))
        contact_conf = np.float32(np.exp(-dist / max(radius, 1e-6)))
        return np.concatenate(
            [
                type_vec,
                rel_eef,
                rel_part,
                np.asarray(field_vec, dtype=np.float32),
                np.asarray(desired_flow, dtype=np.float32),
                np.asarray([dist_norm, residual_norm, contact_conf], dtype=np.float32),
            ]
        ).astype(np.float32)

    for r, s, task_value in zip(robot, scene, task_names):
        task = str(task_value)
        joint_name = base.ARTICULATION_TASK_TO_JOINT.get(task)
        if joint_name is None:
            supported_primitives = set(BLOCK_PUSH_TASKS) | {"push_into_drawer", "place_in_slider"}
            if minimal_g_scope != "translational" or task not in supported_primitives:
                out.append(np.zeros(QUERY_G_TOKEN_COUNT * QUERY_G_TOKEN_DIM, dtype=np.float32))
                continue

            eef = np.asarray(r[:3], dtype=np.float32)
            radius = 0.05
            residual_norm = 1.0
            if task in BLOCK_PUSH_TASKS:
                color, desired_dir = BLOCK_PUSH_TASKS[task]
                part = np.asarray(s[BLOCK_SCENE_SLICES[color]], dtype=np.float32)
                flow_scale = 1.0
            else:
                if not primitive_geometry or task not in primitive_geometry:
                    raise RuntimeError(f"Translational G requires fitted {task} geometry")
                fitted = primitive_geometry[task]
                target_center = np.asarray(fitted["target_center"], dtype=np.float32)
                blocks = _scene_block_positions(np.asarray(s, dtype=np.float32))
                active_idx = int(np.argmin(np.linalg.norm(blocks - eef[None, :], axis=1)))
                part = blocks[active_idx]
                fitted_flow = base._safe_unit(np.asarray(fitted["flow_dir"], dtype=np.float32))
                desired_dir = fitted_flow
                median_displacement = float(max(fitted.get("median_displacement", 0.10), 1e-3))
                remaining = float(np.dot(target_center - part, desired_dir))
                flow_scale = float(np.clip(max(remaining, 0.0) / median_displacement, 0.10, 1.25))
                residual_norm = flow_scale

            desired_flow = desired_dir * np.float32(flow_scale)
            contact_anchor = (
                part if task == "place_in_slider" else part - desired_dir * np.float32(radius)
            )
            side_axis, _ = base._axis_perp_basis(desired_dir)
            side_anchor = part + side_axis * np.float32(radius)
            eef_to_contact = contact_anchor - eef
            eef_field = base._safe_unit(eef_to_contact) * np.float32(
                min(1.25, np.linalg.norm(eef_to_contact) / radius)
            )
            side_field = (part - side_anchor) / np.float32(radius)
            tokens = [
                token(0, eef, eef, part, eef_field, desired_flow, residual_norm, radius),
                token(1, part, eef, part, desired_flow, desired_flow, residual_norm, radius),
                token(2, contact_anchor, eef, part, desired_flow, desired_flow, residual_norm, radius),
                token(3, side_anchor, eef, part, side_field, desired_flow, residual_norm, radius),
            ]
            out.append(np.concatenate(tokens).astype(np.float32))
            continue

        joint_model = geometry["joints"][joint_name]
        joint_idx = int(joint_model["joint_index"])
        joint_range = float(joint_model["joint_range"])
        q_joint = float(s[joint_idx])
        target = base.articulation_target_controls(task, s)[joint_idx]
        residual = float(target - q_joint)
        residual_norm = float(residual / max(joint_range, 1e-6))
        residual_mag = abs(residual_norm)
        desired_sign = 0.0 if residual_mag < 1e-6 else (1.0 if residual >= 0.0 else -1.0)

        eef = np.asarray(r[:3], dtype=np.float32)
        intercept = np.asarray(joint_model["intercept"], dtype=np.float32)
        slope = np.asarray(joint_model["slope"], dtype=np.float32)
        axis = base._safe_unit(np.asarray(joint_model["axis_unit"], dtype=np.float32))
        desired_dir = axis * np.float32(desired_sign)
        part = intercept + slope * np.float32(q_joint)
        b1, _ = base._axis_perp_basis(axis)
        radius = float(max(0.02, min(0.08, float(joint_model.get("contact_dist_threshold", 0.05)))))
        flow_scale = np.float32(min(residual_mag, 1.25))
        desired_flow = desired_dir * flow_scale

        contact_anchor = part - desired_dir * np.float32(radius)
        side_anchor = part + b1 * np.float32(radius)
        eef_to_contact = contact_anchor - eef
        eef_field = base._safe_unit(eef_to_contact) * np.float32(min(1.25, np.linalg.norm(eef_to_contact) / max(radius, 1e-6)))
        contact_field = desired_flow
        part_field = desired_flow
        side_field = (part - side_anchor) / np.float32(max(radius, 1e-6)) * np.float32(min(1.0, max(residual_mag, 0.25)))

        tokens = [
            token(0, eef, eef, part, eef_field, desired_flow, residual_norm, radius),
            token(1, part, eef, part, part_field, desired_flow, residual_norm, radius),
            token(2, contact_anchor, eef, part, contact_field, desired_flow, residual_norm, radius),
            token(3, side_anchor, eef, part, side_field, desired_flow, residual_norm, radius),
        ]
        out.append(np.concatenate(tokens).astype(np.float32))
    return np.stack(out).astype(np.float32)


def apply_query_g_control(query_g: np.ndarray, control: str) -> np.ndarray:
    if control == "normal":
        return query_g.astype(np.float32)
    if control == "zero":
        return np.zeros_like(query_g, dtype=np.float32)
    if control == "sign_flip":
        shaped = query_g.reshape(query_g.shape[0], QUERY_G_TOKEN_COUNT, QUERY_G_TOKEN_DIM).copy()
        shaped[:, :, 10:16] *= -1.0
        shaped[:, :, 17] *= -1.0
        return shaped.reshape(query_g.shape).astype(np.float32)
    if control in {"zero_attraction", "zero_flow", "sign_flip_attraction"}:
        shaped = query_g.reshape(query_g.shape[0], QUERY_G_TOKEN_COUNT, QUERY_G_TOKEN_DIM).copy()
        if control == "zero_attraction":
            shaped[:, 2, 4:7] = 0.0
        elif control == "zero_flow":
            shaped[:, 0, 13:16] = 0.0
        else:
            shaped[:, 2, 4:7] *= -1.0
        return shaped.reshape(query_g.shape).astype(np.float32)
    if control.startswith("mask:"):
        group = control.split(":", 1)[1]
        if group not in QUERY_G_FEATURE_GROUPS:
            raise ValueError(f"Unknown query-G feature group: {group}")
        return query_g.astype(np.float32)
    if control.startswith("task:"):
        return query_g.astype(np.float32)
    raise ValueError(f"Unsupported query-G control: {control}")


def make_condition_parts(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    base_x = base.make_absolute_base(arrays).astype(np.float32)
    g_x = arrays["articulation_g_v7_unified_field"].astype(np.float32)
    return base_x, g_x


def make_minimal_condition_parts(
    arrays: dict[str, np.ndarray],
    state_scope: str = "full",
    language_mode: str = "onehot",
) -> tuple[np.ndarray, np.ndarray]:
    state_x, lang_x = make_state_lang_parts(arrays, state_scope, language_mode)
    base_x = np.concatenate([state_x, lang_x], axis=1).astype(np.float32)
    g_x = arrays["articulation_minimal_g"].astype(np.float32)
    return base_x, g_x


def make_minimal_query_condition_parts(
    arrays: dict[str, np.ndarray],
    state_scope: str = "full",
    mask_language_with_g: bool = True,
    language_mode: str = "onehot",
) -> tuple[np.ndarray, np.ndarray]:
    state_x, lang_x = make_state_lang_parts(arrays, state_scope, language_mode)
    g_x = arrays["articulation_minimal_g"].astype(np.float32)
    if mask_language_with_g:
        g_valid = (np.linalg.norm(g_x, axis=1, keepdims=True) > 1e-6).astype(np.float32)
        executor_lang_x = lang_x * (1.0 - g_valid)
    else:
        executor_lang_x = lang_x
    base_x = np.concatenate([state_x, executor_lang_x], axis=1).astype(np.float32)
    return base_x, g_x


def make_condition(
    arrays: dict[str, np.ndarray],
    mode: str,
    executor_state_scope: str = "full",
    mask_executor_language_with_g: bool = True,
    executor_language_mode: str = "onehot",
) -> tuple[np.ndarray, dict[str, str]]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    state_x, lang_x = make_state_lang_parts(
        arrays, executor_state_scope, executor_language_mode
    )
    query_g = arrays.get("articulation_query_g")
    if mode == "chunk_bc_state":
        return state_x.astype(np.float32), {"condition": "state_only"}
    if mode == "chunk_bc_lang":
        return np.concatenate([state_x, lang_x], axis=1).astype(np.float32), {
            "condition": f"state_plus_language_{executor_language_mode}"
        }
    if mode == "chunk_bc_query_g":
        if query_g is None:
            raise RuntimeError("chunk_bc_query_g requires articulation_query_g")
        return np.concatenate([state_x, query_g], axis=1).astype(np.float32), {"condition": "state_plus_oracle_query_g"}
    if mode == "chunk_bc_lang_query_g":
        if query_g is None:
            raise RuntimeError("chunk_bc_lang_query_g requires articulation_query_g")
        return np.concatenate([state_x, lang_x, query_g], axis=1).astype(np.float32), {
            "condition": f"state_plus_language_{executor_language_mode}_plus_oracle_query_g"
        }
    if mode in {
        "chunk_bc_lang_minimal_g_concat",
        "chunk_bc_lang_minimal_g_structured",
    } | LANG_MINIMAL_G_PROJECTED_MODES:
        minimal_base, minimal_g = make_minimal_condition_parts(
            arrays, executor_state_scope, executor_language_mode
        )
        return np.concatenate([minimal_base, minimal_g], axis=1).astype(np.float32), {
            "condition": "state_plus_language_plus_coherent_attraction_and_object_flow"
        }
    if mode in MASKED_LANG_MINIMAL_G_LEARNED_MODES | MASKED_LANG_MINIMAL_G_PROJECTED_MODES:
        minimal_base, minimal_g = make_minimal_query_condition_parts(
            arrays,
            executor_state_scope,
            mask_language_with_g=mask_executor_language_with_g,
            language_mode=executor_language_mode,
        )
        return np.concatenate([minimal_base, minimal_g], axis=1).astype(np.float32), {
            "condition": "state_plus_minimal_g_with_direct_language_masked_when_g_is_valid"
        }
    base_x, g_x = make_condition_parts(arrays)
    x = np.concatenate(
        [
            base_x,
            g_x,
        ],
        axis=1,
    ).astype(np.float32)
    return x, {"condition": "absolute_base_plus_articulation_g_v7_unified_field"}


PACKED_STATE_ARRAYS_NAME = "state_only_arrays.npz"
PACKED_STATE_FIELDS = ("robot_obs", "scene_obs", "rel_actions")


def load_packed_state_arrays(
    split_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    """Load the optional contiguous cache used to avoid per-frame NPZ I/O."""
    packed_path = split_dir / PACKED_STATE_ARRAYS_NAME
    if not packed_path.exists():
        return None
    with np.load(packed_path, allow_pickle=False) as packed:
        required = ("frame_index", *PACKED_STATE_FIELDS)
        absent = [field for field in required if field not in packed.files]
        if absent:
            raise RuntimeError(f"{packed_path} is missing fields: {absent}")
        arrays = {field: np.asarray(packed[field]) for field in packed.files}

    frame_indices = np.asarray(arrays["frame_index"], dtype=np.int64)
    if frame_indices.ndim != 1 or len(frame_indices) == 0:
        raise RuntimeError(f"Invalid frame_index array in {packed_path}")
    if len(np.unique(frame_indices)) != len(frame_indices):
        raise RuntimeError(f"Duplicate frame indices in {packed_path}")
    for field in PACKED_STATE_FIELDS:
        if len(arrays[field]) != len(frame_indices):
            raise RuntimeError(f"Length mismatch for {field} in {packed_path}")
    if "minimal_g" in arrays and len(arrays["minimal_g"]) != len(frame_indices):
        raise RuntimeError(f"Length mismatch for minimal_g in {packed_path}")

    lookup = np.full(int(frame_indices.max()) + 1, -1, dtype=np.int64)
    lookup[frame_indices] = np.arange(len(frame_indices), dtype=np.int64)
    size_mb = sum(value.nbytes for value in arrays.values()) / (1024**2)
    print(
        f"[data] loaded packed state cache {packed_path} "
        f"frames={len(frame_indices)} arrays={size_mb:.1f} MiB",
        flush=True,
    )
    return arrays, lookup


def collect_chunk_records(
    data_root: Path,
    task_to_id: dict[str, int],
    chunk_size: int,
    max_windows: int,
    max_segments: int,
    stride: int,
    seed: int,
    include_tasks: set[str] | None,
    articulation_geometry: dict[str, Any],
    minimal_g_scope: str = "articulation",
    primitive_geometry: dict[str, Any] | None = None,
    task_segment_limits: dict[str, int] | None = None,
    reverse_open_augmentation: bool = False,
    reverse_open_max_segments: int = 0,
    reverse_task_pairs: list[tuple[str, str]] | None = None,
    reverse_pair_max_segments: int = 0,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rng = random.Random(seed)
    split_dir = data_root / "training"
    packed_state = load_packed_state_arrays(split_dir)

    def read_frame(
        frame_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
        if packed_state is not None:
            packed_arrays, packed_lookup = packed_state
            if frame_index < 0 or frame_index >= len(packed_lookup):
                return None
            position = int(packed_lookup[frame_index])
            if position < 0:
                return None
            frame_g = (
                np.asarray(packed_arrays["minimal_g"][position], dtype=np.float32)
                if "minimal_g" in packed_arrays
                else None
            )
            return (
                np.asarray(packed_arrays["robot_obs"][position], dtype=np.float32),
                np.asarray(packed_arrays["scene_obs"][position], dtype=np.float32),
                np.asarray(packed_arrays["rel_actions"][position], dtype=np.float32),
                frame_g,
            )
        frame_path = base.episode_file(split_dir, frame_index)
        if not frame_path.exists():
            return None
        try:
            with np.load(frame_path, allow_pickle=True) as data:
                return (
                    np.asarray(data["robot_obs"], dtype=np.float32),
                    np.asarray(data["scene_obs"], dtype=np.float32),
                    np.asarray(data["rel_actions"], dtype=np.float32),
                    np.asarray(data["minimal_g"], dtype=np.float32)
                    if "minimal_g" in data.files
                    else None,
                )
        except Exception:
            return None

    segments = base.load_lang_segments(split_dir)
    embedding_rows = [
        np.asarray(segment["emb"], dtype=np.float32).reshape(-1)
        for segment in segments
        if segment.get("emb") is not None
    ]
    language_embedding_dim = int(embedding_rows[0].shape[0]) if embedding_rows else 0
    if any(row.shape[0] != language_embedding_dim for row in embedding_rows):
        raise RuntimeError("Inconsistent language embedding dimensions in CALVIN annotations")
    task_embedding_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    for segment in segments:
        if segment.get("emb") is not None:
            task_embedding_rows[str(segment["task"])].append(
                np.asarray(segment["emb"], dtype=np.float32).reshape(-1)
            )
    task_embeddings = {
        task: np.mean(np.stack(rows), axis=0).astype(np.float32)
        for task, rows in task_embedding_rows.items()
    }
    if max_segments > 0:
        segments = segments[:max_segments]
    rng.shuffle(segments)

    robot_obs: list[np.ndarray] = []
    scene_obs: list[np.ndarray] = []
    task_oh: list[np.ndarray] = []
    lang_emb: list[np.ndarray] = []
    task_names: list[str] = []
    action_chunks: list[np.ndarray] = []
    segment_ids: list[int] = []
    frame_indices: list[int] = []
    reverse_open_flags: list[bool] = []
    reverse_source_flags: list[bool] = []
    precomputed_minimal_g: list[np.ndarray] = []
    precomputed_g_presence: bool | None = None
    used_segments: list[dict[str, Any]] = []
    used_segment_ids: list[int] = []
    used_task_segments: Counter[str] = Counter()

    for seg_id, seg in enumerate(segments):
        task = str(seg["task"])
        if include_tasks is not None and task not in include_tasks:
            continue
        if task_segment_limits is not None and task in task_segment_limits:
            if used_task_segments[task] >= int(task_segment_limits[task]):
                continue
        start, end = int(seg["start"]), int(seg["end"])
        if end - start + 1 < chunk_size:
            continue

        frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = {}
        segment_ok = True
        for idx in range(start, end + 1):
            frame = read_frame(idx)
            if frame is None:
                segment_ok = False
                break
            r, s, a, frame_g = frame
            has_precomputed_g = frame_g is not None
            if precomputed_g_presence is None:
                precomputed_g_presence = has_precomputed_g
            elif precomputed_g_presence != has_precomputed_g:
                raise RuntimeError(
                    "Mixed NPZ records with and without minimal_g are not supported"
                )
            frames[idx] = (r, s, a, frame_g)
        if not segment_ok:
            continue

        any_window = False
        for idx in range(start, end - chunk_size + 2, max(1, stride)):
            if max_windows > 0 and len(action_chunks) >= max_windows:
                break
            r, s, _a0, frame_g = frames[idx]
            chunk = np.stack([frames[j][2] for j in range(idx, idx + chunk_size)]).astype(np.float32)
            robot_obs.append(r)
            scene_obs.append(s)
            task_oh.append(base.one_hot(task, task_to_id))
            segment_embedding = (
                np.asarray(seg["emb"], dtype=np.float32).reshape(-1)
                if seg.get("emb") is not None
                else task_embeddings.get(task, np.zeros(language_embedding_dim, dtype=np.float32))
            )
            lang_emb.append(segment_embedding)
            task_names.append(task)
            action_chunks.append(chunk)
            segment_ids.append(seg_id)
            frame_indices.append(idx)
            reverse_open_flags.append(False)
            reverse_source_flags.append(False)
            if frame_g is not None:
                precomputed_minimal_g.append(frame_g)
            any_window = True
        if any_window:
            used_segments.append(seg)
            used_segment_ids.append(seg_id)
            used_task_segments[task] += 1
        if max_windows > 0 and len(action_chunks) >= max_windows:
            break

    active_reverse_pairs = list(reverse_task_pairs or [])
    if reverse_open_augmentation and ("open_drawer", "close_drawer") not in active_reverse_pairs:
        active_reverse_pairs.append(("open_drawer", "close_drawer"))
    for source_task, target_task in active_reverse_pairs:
        if max_windows > 0 and len(action_chunks) >= max_windows:
            break
        augmented_segments = 0
        pair_limit = (
            reverse_open_max_segments
            if source_task == "open_drawer"
            and target_task == "close_drawer"
            and reverse_open_augmentation
            and reverse_open_max_segments > 0
            else reverse_pair_max_segments
        )
        for seg, source_segment_id in zip(used_segments, used_segment_ids):
            if str(seg["task"]) != source_task:
                continue
            if pair_limit > 0 and augmented_segments >= pair_limit:
                break
            start, end = int(seg["start"]), int(seg["end"])
            if read_frame(end + 1) is None:
                continue

            frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            segment_ok = True
            for idx in range(start, end + 2):
                frame = read_frame(idx)
                if frame is None:
                    segment_ok = False
                    break
                frames[idx] = frame[:3]
            if not segment_ok:
                continue

            reverse_states: list[tuple[np.ndarray, np.ndarray, int]] = []
            reverse_actions: list[np.ndarray] = []
            for idx in range(end, start - 1, -1):
                current_robot, current_scene, _ = frames[idx + 1]
                target_robot, _, source_action = frames[idx]
                reverse_action = source_action.copy()
                reverse_action[:6] *= -1.0
                reverse_action[-1] = 1.0 if float(target_robot[-1]) >= 0.0 else -1.0
                reverse_states.append((current_robot, current_scene, idx + 1))
                reverse_actions.append(reverse_action.astype(np.float32))

            any_reverse_window = False
            for reverse_idx in range(0, len(reverse_actions) - chunk_size + 1, max(1, stride)):
                if max_windows > 0 and len(action_chunks) >= max_windows:
                    break
                r, s, frame_idx = reverse_states[reverse_idx]
                chunk = np.stack(reverse_actions[reverse_idx : reverse_idx + chunk_size]).astype(np.float32)
                robot_obs.append(r)
                scene_obs.append(s)
                task_oh.append(base.one_hot(target_task, task_to_id))
                lang_emb.append(
                    task_embeddings.get(
                        target_task,
                        np.zeros(language_embedding_dim, dtype=np.float32),
                    )
                )
                task_names.append(target_task)
                action_chunks.append(chunk)
                # Keep source and reverse windows in the same train/validation group.
                segment_ids.append(source_segment_id)
                frame_indices.append(frame_idx)
                reverse_open_flags.append(
                    source_task == "open_drawer" and target_task == "close_drawer"
                )
                reverse_source_flags.append(True)
                any_reverse_window = True
            if any_reverse_window:
                augmented_segments += 1
            if max_windows > 0 and len(action_chunks) >= max_windows:
                break

    if not action_chunks:
        raise RuntimeError("No chunk train records were collected")

    arrays = {
        "robot_obs": np.stack(robot_obs).astype(np.float32),
        "scene_obs": np.stack(scene_obs).astype(np.float32),
        "task_oh": np.stack(task_oh).astype(np.float32),
        "lang_emb": np.stack(lang_emb).astype(np.float32),
        "task_name": np.asarray(task_names, dtype=object),
        "action_chunks": np.stack(action_chunks).astype(np.float32),
        "segment_id": np.asarray(segment_ids, dtype=np.int64),
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
        "is_reverse_open": np.asarray(reverse_open_flags, dtype=bool),
        "is_reverse_source": np.asarray(reverse_source_flags, dtype=bool),
    }
    arrays["articulation_g_v7_unified_field"] = base.make_articulation_unified_field_g_v7(
        arrays["robot_obs"],
        arrays["scene_obs"],
        arrays["task_name"],
        articulation_geometry,
    )
    arrays["articulation_query_g"] = make_articulation_query_g(
        arrays["robot_obs"],
        arrays["scene_obs"],
        arrays["task_name"],
        articulation_geometry,
        minimal_g_scope=minimal_g_scope,
        primitive_geometry=primitive_geometry,
    )
    arrays["articulation_minimal_g"] = make_minimal_articulation_g(arrays["articulation_query_g"])
    if precomputed_minimal_g:
        if len(precomputed_minimal_g) != len(action_chunks):
            raise RuntimeError(
                "Precomputed minimal_g cannot currently be combined with reverse augmentation"
            )
        precomputed_array = np.stack(precomputed_minimal_g).astype(np.float32)
        if precomputed_array.shape != (len(action_chunks), 6):
            raise RuntimeError(
                f"Expected precomputed minimal_g shape {(len(action_chunks), 6)}, "
                f"got {precomputed_array.shape}"
            )
        arrays["articulation_minimal_g"] = precomputed_array
        arrays["uses_precomputed_minimal_g"] = np.ones(len(action_chunks), dtype=bool)
    return arrays, used_segments


def attach_visual_features(
    arrays: dict[str, np.ndarray], feature_dir: Path
) -> dict[str, Any]:
    manifest_path = feature_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing visual feature manifest: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    frame_parts: list[np.ndarray] = []
    static_parts: list[np.ndarray] = []
    gripper_parts: list[np.ndarray] = []
    for shard_path in sorted(feature_dir.glob("shard_*.npz")):
        with np.load(shard_path, allow_pickle=False) as data:
            frame_parts.append(np.asarray(data["frame_index"], dtype=np.int64))
            static_parts.append(np.asarray(data["static"], dtype=np.float32))
            gripper_parts.append(np.asarray(data["gripper"], dtype=np.float32))
    if not frame_parts:
        raise FileNotFoundError(f"No visual feature shards found under {feature_dir}")

    feature_frames = np.concatenate(frame_parts)
    static_features = np.concatenate(static_parts)
    gripper_features = np.concatenate(gripper_parts)
    order = np.argsort(feature_frames)
    feature_frames = feature_frames[order]
    static_features = static_features[order]
    gripper_features = gripper_features[order]
    requested_frames = np.asarray(arrays["frame_index"], dtype=np.int64)
    positions = np.searchsorted(feature_frames, requested_frames)
    valid = positions < len(feature_frames)
    if np.any(valid):
        valid[valid] &= feature_frames[positions[valid]] == requested_frames[valid]
    if not np.all(valid):
        missing = np.unique(requested_frames[~valid])[:10].tolist()
        raise RuntimeError(
            f"Visual feature cache is missing {int(np.sum(~valid))} policy records; "
            f"first frame indices: {missing}"
        )
    arrays["visual_static"] = static_features[positions].astype(np.float32)
    arrays["visual_gripper"] = gripper_features[positions].astype(np.float32)
    feature_dim = int(arrays["visual_static"].shape[1])
    if arrays["visual_gripper"].shape[1] != feature_dim:
        raise RuntimeError("Static and gripper visual feature dimensions differ")
    manifest["feature_dir"] = str(feature_dir.resolve())
    manifest["feature_dim"] = feature_dim
    return manifest


class ChunkMLP(nn.Module):
    def __init__(self, cond_dim: int, out_dim: int, hidden: int, layers: int):
        super().__init__()
        modules: list[nn.Module] = [nn.Linear(cond_dim, hidden), nn.ReLU()]
        for _ in range(max(0, layers - 1)):
            modules.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        modules.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


class GatedChunkMLP(nn.Module):
    """Predict a no-G action plus a learned-gated G residual."""

    def __init__(self, base_dim: int, g_dim: int, out_dim: int, hidden: int, layers: int, gate_max: float):
        super().__init__()
        self.base_dim = int(base_dim)
        self.g_dim = int(g_dim)
        self.gate_max = float(gate_max)

        base_modules: list[nn.Module] = [nn.Linear(base_dim, hidden), nn.ReLU()]
        residual_modules: list[nn.Module] = [nn.Linear(base_dim + g_dim, hidden), nn.ReLU()]
        gate_modules: list[nn.Module] = [nn.Linear(base_dim + g_dim, hidden), nn.ReLU()]
        for _ in range(max(0, layers - 1)):
            base_modules.extend([nn.Linear(hidden, hidden), nn.ReLU()])
            residual_modules.extend([nn.Linear(hidden, hidden), nn.ReLU()])
            gate_modules.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        base_modules.append(nn.Linear(hidden, out_dim))
        residual_modules.append(nn.Linear(hidden, out_dim))
        gate_modules.append(nn.Linear(hidden, 1))
        self.base_net = nn.Sequential(*base_modules)
        self.residual_net = nn.Sequential(*residual_modules)
        self.gate_net = nn.Sequential(*gate_modules)

    def forward(self, cond: torch.Tensor, return_gate: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        base_x = cond[:, : self.base_dim]
        g_x = cond[:, self.base_dim : self.base_dim + self.g_dim]
        pair = torch.cat([base_x, g_x], dim=-1)
        gate = torch.sigmoid(self.gate_net(pair)) * self.gate_max
        pred = self.base_net(base_x) + gate * self.residual_net(pair)
        if return_gate:
            return pred, gate
        return pred


class StructuredMinimalGChunkMLP(nn.Module):
    """FiLM executor with a sign-preserving low-frequency field residual."""

    def __init__(
        self,
        base_dim: int,
        g_dim: int,
        out_dim: int,
        chunk_size: int,
        hidden: int,
        layers: int,
        action_std: np.ndarray | torch.Tensor,
        approach_gain_init: float = 0.20,
        flow_gain_init: float = 0.20,
        readiness_distance: float = 0.10,
        readiness_temperature: float = 0.025,
    ):
        super().__init__()
        if g_dim != MINIMAL_G_DIM:
            raise ValueError(f"Structured minimal-G model expects {MINIMAL_G_DIM} G dims, got {g_dim}")
        self.base_dim = int(base_dim)
        self.g_dim = int(g_dim)
        self.chunk_size = int(chunk_size)
        self.readiness_distance = float(readiness_distance)
        self.readiness_temperature = float(readiness_temperature)

        self.base_in = nn.Linear(base_dim, hidden)
        self.g_mod = nn.Sequential(nn.Linear(g_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(max(0, layers - 1)))
        self.out = nn.Linear(hidden, out_dim)
        self.approach_gain_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(approach_gain_init, 1e-6))), dtype=torch.float32)
        )
        self.flow_gain_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(flow_gain_init, 1e-6))), dtype=torch.float32)
        )
        action_std_t = torch.as_tensor(action_std, dtype=torch.float32).reshape(-1)
        if action_std_t.numel() != out_dim:
            raise ValueError(f"Expected {out_dim} action std values, got {action_std_t.numel()}")
        self.register_buffer("action_std", action_std_t.clamp_min(1e-6))

    def field_prior(self, g_raw: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attraction = g_raw[:, :3]
        object_flow = g_raw[:, 3:6]
        distance = torch.linalg.vector_norm(attraction, dim=-1, keepdim=True)
        attraction_dir = attraction / distance.clamp_min(1e-6)
        attraction_strength = torch.clamp(distance / max(self.readiness_distance, 1e-6), max=1.0)
        readiness = torch.sigmoid(
            (self.readiness_distance - distance) / max(self.readiness_temperature, 1e-6)
        )
        approach_gain = torch.nn.functional.softplus(self.approach_gain_raw)
        flow_gain = torch.nn.functional.softplus(self.flow_gain_raw)
        xyz = (
            (1.0 - readiness) * approach_gain * attraction_strength * attraction_dir
            + readiness * flow_gain * object_flow
        )
        valid = ((distance > 1e-6) | (torch.linalg.vector_norm(object_flow, dim=-1, keepdim=True) > 1e-6)).float()
        xyz = xyz * valid
        prior_raw = torch.zeros(
            (g_raw.shape[0], self.chunk_size, 7), dtype=g_raw.dtype, device=g_raw.device
        )
        prior_raw[:, :, :3] = xyz[:, None, :]
        prior_n = prior_raw.reshape(g_raw.shape[0], -1) / self.action_std[None, :]
        return prior_n, {
            "readiness": readiness,
            "field_direction": (1.0 - readiness) * attraction_dir + readiness * object_flow,
            "approach_gain": approach_gain,
            "flow_gain": flow_gain,
            "valid": valid,
        }

    def forward(
        self,
        base_n: torch.Tensor,
        g_n: torch.Tensor,
        g_raw: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = torch.relu(self.base_in(base_n))
        gamma, beta = self.g_mod(g_n).chunk(2, dim=-1)
        h = h * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        for block in self.blocks:
            h = torch.relu(block(h))
        learned = self.out(h)
        prior_n, aux = self.field_prior(g_raw)
        pred = learned + prior_n
        if return_aux:
            aux["prior_n"] = prior_n
            return pred, aux
        return pred


class FiLMMinimalGChunkMLP(nn.Module):
    """Learn an unconstrained action chunk while using G only for FiLM conditioning."""

    def __init__(self, base_dim: int, g_dim: int, out_dim: int, hidden: int, layers: int):
        super().__init__()
        if g_dim != MINIMAL_G_DIM:
            raise ValueError(f"Minimal-G FiLM model expects {MINIMAL_G_DIM} G dims, got {g_dim}")
        self.base_dim = int(base_dim)
        self.g_dim = int(g_dim)
        self.base_in = nn.Linear(base_dim, hidden)
        self.g_mod = nn.Sequential(nn.Linear(g_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(max(0, layers - 1)))
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, base_n: torch.Tensor, g_n: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.base_in(base_n))
        gamma, beta = self.g_mod(g_n).chunk(2, dim=-1)
        h = h * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        for block in self.blocks:
            h = torch.relu(block(h))
        return self.out(h)


def transformer_head_count(hidden: int) -> int:
    for heads in (8, 4, 2):
        if hidden % heads == 0:
            return heads
    return 1


class MinimalGTokenChunkTransformer(nn.Module):
    """Decode an action chunk by cross-attending raw attraction and flow tokens."""

    def __init__(self, base_dim: int, g_dim: int, chunk_size: int, hidden: int, layers: int):
        super().__init__()
        if g_dim != MINIMAL_G_DIM:
            raise ValueError(f"Minimal-G token model expects {MINIMAL_G_DIM} G dims, got {g_dim}")
        self.base_dim = int(base_dim)
        self.g_dim = int(g_dim)
        self.chunk_size = int(chunk_size)
        self.base_encoder = nn.Sequential(nn.Linear(base_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.anchor_encoder = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.flow_encoder = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, hidden) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=transformer_head_count(hidden),
            dim_feedforward=4 * hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=max(1, layers), norm=nn.LayerNorm(hidden))
        self.action_out = nn.Linear(hidden, 7)

    def memory(self, base_n: torch.Tensor, g_raw: torch.Tensor) -> torch.Tensor:
        state = self.base_encoder(base_n)
        anchor = self.anchor_encoder(g_raw[:, :3])
        flow = self.flow_encoder(g_raw[:, 3:6])
        return torch.stack([state, anchor, flow], dim=1)

    def forward(self, base_n: torch.Tensor, g_raw: torch.Tensor) -> torch.Tensor:
        query = self.action_queries.expand(base_n.shape[0], -1, -1)
        decoded = self.decoder(query, self.memory(base_n, g_raw))
        return self.action_out(decoded).reshape(base_n.shape[0], -1)


class ResidualAdapterMinimalGTokenPolicy(nn.Module):
    """Preserve a frozen token policy while learning a compact G-conditioned correction."""

    def __init__(
        self,
        base_policy: MinimalGTokenChunkTransformer,
        base_dim: int,
        g_dim: int,
        out_dim: int,
        hidden: int,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.base_dim = int(base_dim)
        self.g_dim = int(g_dim)
        self.chunk_size = int(base_policy.chunk_size)
        self.output_scale = float(output_scale)
        self.base_policy = base_policy
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        self.adapter = nn.Sequential(
            nn.Linear(base_dim + g_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.residual_out = nn.Linear(hidden, out_dim)
        self.gate_out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)

    def forward(
        self,
        base_n: torch.Tensor,
        g_raw: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            base_prediction = self.base_policy(base_n, g_raw)
        features = self.adapter(torch.cat([base_n, g_raw], dim=-1))
        gate = torch.sigmoid(self.gate_out(features))
        residual = self.output_scale * gate * self.residual_out(features)
        prediction = base_prediction + residual
        if return_aux:
            return prediction, {"gate": gate, "residual": residual}
        return prediction


class RoutedResidualAdapterMinimalGTokenPolicy(nn.Module):
    """Use a learned state-and-G router over mechanism-specialized residual experts."""

    def __init__(
        self,
        base_policy: MinimalGTokenChunkTransformer,
        base_dim: int,
        g_dim: int,
        out_dim: int,
        hidden: int,
        num_experts: int,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("Routed residual adapter requires at least two experts")
        self.base_dim = int(base_dim)
        self.g_dim = int(g_dim)
        self.chunk_size = int(base_policy.chunk_size)
        self.num_experts = int(num_experts)
        self.base_policy = base_policy
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        input_dim = base_dim + g_dim
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            for _ in range(num_experts)
        )
        self.residual_outs = nn.ModuleList(nn.Linear(hidden, out_dim) for _ in range(num_experts))
        self.gate_outs = nn.ModuleList(nn.Linear(hidden, 1) for _ in range(num_experts))
        router_hidden = max(32, hidden)
        self.router = nn.Sequential(
            nn.Linear(input_dim, router_hidden),
            nn.SiLU(),
            nn.Linear(router_hidden, num_experts),
        )
        for residual_out, gate_out in zip(self.residual_outs, self.gate_outs):
            nn.init.zeros_(residual_out.weight)
            nn.init.zeros_(residual_out.bias)
            nn.init.zeros_(gate_out.weight)
            nn.init.zeros_(gate_out.bias)

    def forward(
        self,
        base_n: torch.Tensor,
        g_raw: torch.Tensor,
        expert_index: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            base_prediction = self.base_policy(base_n, g_raw)
        inputs = torch.cat([base_n, g_raw], dim=-1)
        residuals = []
        for expert, residual_out, gate_out in zip(self.experts, self.residual_outs, self.gate_outs):
            features = expert(inputs)
            residuals.append(torch.sigmoid(gate_out(features)) * residual_out(features))
        residual_stack = torch.stack(residuals, dim=1)
        expert_predictions = base_prediction[:, None, :] + residual_stack
        router_logits = self.router(inputs)
        router_probs = torch.softmax(router_logits, dim=-1)
        if expert_index is None:
            prediction = torch.sum(expert_predictions * router_probs[:, :, None], dim=1)
        else:
            prediction = expert_predictions[
                torch.arange(expert_predictions.shape[0], device=expert_predictions.device),
                expert_index,
            ]
        if return_aux:
            return prediction, {
                "expert_predictions": expert_predictions,
                "router_logits": router_logits,
                "router_probs": router_probs,
            }
        return prediction


class MinimalGTokenFlowTransformer(MinimalGTokenChunkTransformer):
    """Conditional flow velocity model with action tokens attending state and G."""

    def __init__(self, base_dim: int, g_dim: int, chunk_size: int, hidden: int, layers: int, time_dim: int = 32):
        super().__init__(base_dim, g_dim, chunk_size, hidden, layers)
        self.time_dim = int(time_dim)
        self.noisy_action_encoder = nn.Linear(7, hidden)
        self.time_encoder = nn.Sequential(nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        base_n: torch.Tensor,
        g_raw: torch.Tensor,
    ) -> torch.Tensor:
        noisy_actions = x_t.reshape(x_t.shape[0], self.chunk_size, 7)
        time_token = self.time_encoder(time_embedding(t, self.time_dim))[:, None, :]
        query = self.noisy_action_encoder(noisy_actions) + self.action_queries + time_token
        decoded = self.decoder(query, self.memory(base_n, g_raw))
        return self.action_out(decoded).reshape(x_t.shape[0], -1)


class SemanticMinimalGChunkMLP(nn.Module):
    """Use A/F signs only through learned, positive vector-basis gains.

    The residual action remains fully unconstrained. Raw signed G cannot enter
    an arbitrary FiLM/code path; invariant G magnitudes may modulate the state
    trunk, while A and F directions enter translation through learned gains.
    """

    def __init__(
        self,
        base_dim: int,
        g_dim: int,
        out_dim: int,
        chunk_size: int,
        hidden: int,
        layers: int,
        action_std: np.ndarray | torch.Tensor,
        gain_init: float = 0.05,
    ):
        super().__init__()
        if g_dim != MINIMAL_G_DIM:
            raise ValueError(f"Semantic minimal-G model expects {MINIMAL_G_DIM} G dims, got {g_dim}")
        self.base_dim = int(base_dim)
        self.g_dim = int(g_dim)
        self.chunk_size = int(chunk_size)
        self.base_in = nn.Linear(base_dim, hidden)
        self.invariant_mod = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(max(0, layers - 1)))
        self.residual_out = nn.Linear(hidden, out_dim)
        self.gain_out = nn.Linear(hidden, 2 * chunk_size)
        nn.init.zeros_(self.gain_out.weight)
        nn.init.constant_(self.gain_out.bias, math.log(math.expm1(max(gain_init, 1e-6))))

        action_std_t = torch.as_tensor(action_std, dtype=torch.float32).reshape(-1)
        if action_std_t.numel() != out_dim:
            raise ValueError(f"Expected {out_dim} action std values, got {action_std_t.numel()}")
        self.register_buffer("action_std", action_std_t.clamp_min(1e-6))

    def forward(
        self,
        base_n: torch.Tensor,
        g_raw: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attraction = g_raw[:, :3]
        flow = g_raw[:, 3:6]
        attraction_norm = torch.linalg.vector_norm(attraction, dim=-1, keepdim=True)
        flow_norm = torch.linalg.vector_norm(flow, dim=-1, keepdim=True)
        attraction_dir = attraction / attraction_norm.clamp_min(1e-6)
        flow_dir = flow / flow_norm.clamp_min(1e-6)
        alignment_sq = torch.sum(attraction_dir * flow_dir, dim=-1, keepdim=True).square()
        invariant_g = torch.cat([attraction_norm, flow_norm, alignment_sq], dim=-1)

        h = torch.relu(self.base_in(base_n) + 0.1 * self.invariant_mod(invariant_g))
        for block in self.blocks:
            h = torch.relu(block(h))
        residual_n = self.residual_out(h)
        gains = torch.nn.functional.softplus(self.gain_out(h)).reshape(-1, self.chunk_size, 2)
        valid_a = (attraction_norm > 1e-6).float()
        valid_f = (flow_norm > 1e-6).float()
        gains_a = gains[:, :, 0] * valid_a
        gains_f = gains[:, :, 1] * valid_f
        basis_xyz = (
            gains_a[:, :, None] * attraction_dir[:, None, :]
            + gains_f[:, :, None] * flow_dir[:, None, :]
        )
        basis_raw = torch.zeros(
            (g_raw.shape[0], self.chunk_size, 7), dtype=g_raw.dtype, device=g_raw.device
        )
        basis_raw[:, :, :3] = basis_xyz
        basis_n = basis_raw.reshape(g_raw.shape[0], -1) / self.action_std[None, :]
        pred_n = residual_n + basis_n
        if return_aux:
            return pred_n, {
                "basis_n": basis_n,
                "basis_raw": basis_raw,
                "residual_n": residual_n,
                "attraction_gain": gains_a,
                "flow_gain": gains_f,
                "invariant_g": invariant_g,
            }
        return pred_n


class FrameMinimalGChunkMLP(nn.Module):
    """Predict a full action in a G-aligned, full-rank translation frame."""

    def __init__(
        self,
        base_dim: int,
        g_dim: int,
        out_dim: int,
        chunk_size: int,
        hidden: int,
        layers: int,
        action_mean: np.ndarray | torch.Tensor,
        action_std: np.ndarray | torch.Tensor,
    ):
        super().__init__()
        if g_dim != MINIMAL_G_DIM:
            raise ValueError(f"Frame minimal-G model expects {MINIMAL_G_DIM} G dims, got {g_dim}")
        self.base_dim = int(base_dim)
        self.g_dim = int(g_dim)
        self.chunk_size = int(chunk_size)
        self.base_in = nn.Linear(base_dim, hidden)
        self.frame_mod = nn.Sequential(
            nn.Linear(5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(max(0, layers - 1)))
        self.out = nn.Linear(hidden, out_dim)

        action_mean_t = torch.as_tensor(action_mean, dtype=torch.float32).reshape(chunk_size, 7)
        action_std_t = torch.as_tensor(action_std, dtype=torch.float32).reshape(chunk_size, 7)
        self.register_buffer("action_mean", action_mean_t)
        self.register_buffer("action_std", action_std_t.clamp_min(1e-6))

    @staticmethod
    def interaction_frame(g_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flow = g_raw[:, 3:6]
        flow_norm = torch.linalg.vector_norm(flow, dim=-1, keepdim=True)
        valid = flow_norm > 1e-6
        default_flow = torch.tensor([1.0, 0.0, 0.0], dtype=g_raw.dtype, device=g_raw.device)[None, :]
        e_flow = torch.where(valid, flow / flow_norm.clamp_min(1e-6), default_flow)

        world_up = torch.tensor([0.0, 0.0, 1.0], dtype=g_raw.dtype, device=g_raw.device)[None, :]
        e_up_raw = world_up - torch.sum(world_up * e_flow, dim=-1, keepdim=True) * e_flow
        up_norm = torch.linalg.vector_norm(e_up_raw, dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=g_raw.dtype, device=g_raw.device)[None, :]
        fallback = fallback - torch.sum(fallback * e_flow, dim=-1, keepdim=True) * e_flow
        e_up_raw = torch.where(up_norm > 1e-4, e_up_raw, fallback)
        e_up = e_up_raw / torch.linalg.vector_norm(e_up_raw, dim=-1, keepdim=True).clamp_min(1e-6)
        e_side = torch.linalg.cross(e_up, e_flow, dim=-1)
        frame = torch.stack([e_flow, e_side, e_up], dim=-1)
        return frame, valid.float()

    def forward(
        self,
        base_n: torch.Tensor,
        g_raw: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        frame, valid = self.interaction_frame(g_raw)
        attraction = g_raw[:, :3]
        attraction_local = torch.einsum("bij,bi->bj", frame, attraction)
        attraction_norm = torch.linalg.vector_norm(attraction, dim=-1, keepdim=True)
        flow_norm = torch.linalg.vector_norm(g_raw[:, 3:6], dim=-1, keepdim=True)
        frame_features = torch.cat([attraction_local, attraction_norm, flow_norm], dim=-1)

        h = torch.relu(self.base_in(base_n) + 0.1 * self.frame_mod(frame_features))
        for block in self.blocks:
            h = torch.relu(block(h))
        local_action = self.out(h).reshape(-1, self.chunk_size, 7)
        global_xyz = torch.einsum("bij,btj->bti", frame, local_action[:, :, :3])
        xyz_n = (global_xyz - self.action_mean[None, :, :3]) / self.action_std[None, :, :3]
        pred_n = torch.cat([xyz_n, local_action[:, :, 3:]], dim=-1).reshape(g_raw.shape[0], -1)
        if return_aux:
            return pred_n, {
                "frame": frame,
                "frame_valid": valid,
                "frame_features": frame_features,
                "local_action": local_action,
                "global_xyz": global_xyz,
            }
        return pred_n


class ProjectedMinimalGChunkMLP(nn.Module):
    """Predict field-parallel gain and an orthogonal learned correction."""

    def __init__(
        self,
        base_dim: int,
        g_dim: int,
        out_dim: int,
        chunk_size: int,
        hidden: int,
        layers: int,
        action_mean: np.ndarray | torch.Tensor,
        action_std: np.ndarray | torch.Tensor,
        readiness_distance: float = 0.10,
        readiness_temperature: float = 0.025,
        field_gain_max: float = 1.0,
        field_gain_init: float = 0.20,
        orthogonal_ratio_max: float | None = None,
        invariant_gain: bool = False,
    ):
        super().__init__()
        if g_dim != MINIMAL_G_DIM:
            raise ValueError(f"Projected minimal-G model expects {MINIMAL_G_DIM} G dims, got {g_dim}")
        self.base_dim = int(base_dim)
        self.g_dim = int(g_dim)
        self.chunk_size = int(chunk_size)
        self.readiness_distance = float(readiness_distance)
        self.readiness_temperature = float(readiness_temperature)
        self.field_gain_max = float(field_gain_max)
        self.invariant_gain = bool(invariant_gain)
        self.orthogonal_ratio_max = (
            None if orthogonal_ratio_max is None else float(orthogonal_ratio_max)
        )
        if self.orthogonal_ratio_max is not None and self.orthogonal_ratio_max <= 0:
            raise ValueError("orthogonal_ratio_max must be positive")

        self.base_in = nn.Linear(base_dim, hidden)
        self.g_mod = nn.Sequential(nn.Linear(g_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(max(0, layers - 1)))
        self.residual_out = nn.Linear(hidden, out_dim)
        init_ratio = float(np.clip(field_gain_init / max(field_gain_max, 1e-6), 1e-4, 1.0 - 1e-4))
        init_logit = math.log(init_ratio / (1.0 - init_ratio))
        if self.invariant_gain:
            gain_hidden = max(32, hidden // 8)
            self.field_gain_out = None
            self.invariant_gain_net = nn.Sequential(
                nn.Linear(3, gain_hidden),
                nn.SiLU(),
                nn.Linear(gain_hidden, chunk_size),
            )
            nn.init.zeros_(self.invariant_gain_net[-1].weight)
            nn.init.constant_(self.invariant_gain_net[-1].bias, init_logit)
        else:
            self.field_gain_out = nn.Linear(hidden, chunk_size)
            self.invariant_gain_net = None
            nn.init.zeros_(self.field_gain_out.weight)
            nn.init.constant_(self.field_gain_out.bias, init_logit)

        action_mean_t = torch.as_tensor(action_mean, dtype=torch.float32).reshape(-1)
        action_std_t = torch.as_tensor(action_std, dtype=torch.float32).reshape(-1)
        if action_mean_t.numel() != out_dim or action_std_t.numel() != out_dim:
            raise ValueError(f"Expected {out_dim} action statistics values")
        self.register_buffer("action_mean", action_mean_t)
        self.register_buffer("action_std", action_std_t.clamp_min(1e-6))

    def field_frame(self, g_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        attraction = g_raw[:, :3]
        object_flow = g_raw[:, 3:6]
        distance = torch.linalg.vector_norm(attraction, dim=-1, keepdim=True)
        flow_norm = torch.linalg.vector_norm(object_flow, dim=-1, keepdim=True)
        attraction_dir = attraction / distance.clamp_min(1e-6)
        flow_dir = object_flow / flow_norm.clamp_min(1e-6)
        readiness = torch.sigmoid(
            (self.readiness_distance - distance) / max(self.readiness_temperature, 1e-6)
        )
        blended = (1.0 - readiness) * attraction_dir + readiness * flow_dir
        field_direction = blended / torch.linalg.vector_norm(blended, dim=-1, keepdim=True).clamp_min(1e-6)
        valid = ((distance > 1e-6) | (flow_norm > 1e-6)).float()
        return {
            "distance": distance,
            "flow_norm": flow_norm,
            "readiness": readiness,
            "field_direction": field_direction * valid,
            "valid": valid,
        }

    def forward(
        self,
        base_n: torch.Tensor,
        g_n: torch.Tensor,
        g_raw: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = torch.relu(self.base_in(base_n))
        gamma, beta = self.g_mod(g_n).chunk(2, dim=-1)
        h = h * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        for block in self.blocks:
            h = torch.relu(block(h))

        unconstrained_n = self.residual_out(h)
        unconstrained_raw = unconstrained_n * self.action_std[None, :] + self.action_mean[None, :]
        raw = unconstrained_raw.reshape(g_raw.shape[0], self.chunk_size, 7)
        frame = self.field_frame(g_raw)
        direction = frame["field_direction"][:, None, :]
        valid = frame["valid"][:, None, :]

        xyz = raw[:, :, :3]
        parallel = torch.sum(xyz * direction, dim=-1, keepdim=True)
        orthogonal = xyz - parallel * direction
        if self.invariant_gain:
            gain_features = torch.cat(
                [
                    frame["distance"] / max(self.readiness_distance, 1e-6),
                    frame["flow_norm"],
                    frame["readiness"],
                ],
                dim=-1,
            )
            if self.invariant_gain_net is None:
                raise RuntimeError("Invariant gain network is not initialized")
            field_gain_logits = self.invariant_gain_net(gain_features)
        else:
            if self.field_gain_out is None:
                raise RuntimeError("Field gain head is not initialized")
            field_gain_logits = self.field_gain_out(h)
        field_gain = self.field_gain_max * torch.sigmoid(field_gain_logits)
        orthogonal_scale = torch.ones_like(field_gain[:, :, None])
        if self.orthogonal_ratio_max is not None:
            orthogonal_norm = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)
            max_orthogonal_norm = self.orthogonal_ratio_max * field_gain[:, :, None]
            orthogonal_scale = torch.clamp(
                max_orthogonal_norm / orthogonal_norm.clamp_min(1e-6),
                max=1.0,
            )
            orthogonal = orthogonal * orthogonal_scale
        field_component = field_gain[:, :, None] * direction
        projected_xyz = orthogonal + field_component
        final_xyz = valid * projected_xyz + (1.0 - valid) * xyz
        final_raw = torch.cat([final_xyz, raw[:, :, 3:]], dim=-1)
        pred_n = (final_raw.reshape(g_raw.shape[0], -1) - self.action_mean[None, :]) / self.action_std[None, :]

        if return_aux:
            frame.update(
                {
                    "field_gain": field_gain,
                    "invariant_gain_features": (
                        gain_features if self.invariant_gain else None
                    ),
                    "field_component_raw": field_component,
                    "orthogonal_component_raw": orthogonal,
                    "orthogonal_scale": orthogonal_scale,
                    "unconstrained_raw": unconstrained_raw.reshape(g_raw.shape[0], self.chunk_size, 7),
                }
            )
            return pred_n, frame
        return pred_n


def time_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, t[:, None]], dim=-1)
    return emb


class FlowMLP(nn.Module):
    def __init__(self, cond_dim: int, action_dim: int, hidden: int, layers: int, time_dim: int = 32):
        super().__init__()
        self.time_dim = time_dim
        in_dim = cond_dim + action_dim + time_dim
        modules: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(max(0, layers - 1)):
            modules.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        modules.append(nn.Linear(hidden, action_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, time_embedding(t, self.time_dim), cond], dim=-1))


def segment_train_val_indices(segment_ids: np.ndarray, seed: int, val_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.unique(segment_ids)
    rng.shuffle(unique)
    val_n = max(1, int(round(len(unique) * val_fraction)))
    val_segments = set(int(x) for x in unique[:val_n])
    val_mask = np.asarray([int(s) in val_segments for s in segment_ids], dtype=bool)
    val_idx = np.nonzero(val_mask)[0]
    train_idx = np.nonzero(~val_mask)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        order = rng.permutation(len(segment_ids))
        val_n_rows = max(1, int(0.1 * len(order)))
        return order[val_n_rows:], order[:val_n_rows]
    return train_idx, val_idx


INVERSE_GOAL_PAIR_SPECS: dict[str, tuple[str, str, int | slice]] = {
    "slider": ("move_slider_left", "move_slider_right", 0),
    "red_block": ("push_red_block_left", "push_red_block_right", slice(6, 9)),
    "blue_block": ("push_blue_block_left", "push_blue_block_right", slice(12, 15)),
    "pink_block": ("push_pink_block_left", "push_pink_block_right", slice(18, 21)),
}


def nearest_indices_batched(query: np.ndarray, reference: np.ndarray, batch_size: int = 512) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(query), batch_size):
        batch = query[start : start + batch_size]
        distance_sq = np.sum((batch[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        output.append(np.argmin(distance_sq, axis=1))
    return np.concatenate(output)


def build_inverse_goal_train_pairs(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    *,
    max_eef_distance_m: float,
    max_mechanism_fraction: float,
) -> dict[str, np.ndarray]:
    """Mine near-state opposite-goal pairs, indexed relative to the train split."""
    names = arrays["task_name"][train_idx].astype(str)
    robot = arrays["robot_obs"][train_idx].astype(np.float64)
    scene = arrays["scene_obs"][train_idx].astype(np.float64)
    output: dict[str, np.ndarray] = {}

    for family, (task0, task1, mechanism_index) in INVERSE_GOAL_PAIR_SPECS.items():
        position0 = np.flatnonzero(names == task0)
        position1 = np.flatnonzero(names == task1)
        if not len(position0) or not len(position1):
            continue

        mechanism0 = np.asarray(scene[position0, mechanism_index], dtype=np.float64)
        mechanism1 = np.asarray(scene[position1, mechanism_index], dtype=np.float64)
        if mechanism0.ndim == 1:
            mechanism0 = mechanism0[:, None]
            mechanism1 = mechanism1[:, None]
        compact0 = np.concatenate([robot[position0, :3], mechanism0], axis=1)
        compact1 = np.concatenate([robot[position1, :3], mechanism1], axis=1)
        compact = np.concatenate([compact0, compact1], axis=0)
        mean = np.mean(compact, axis=0)
        std = np.std(compact, axis=0)
        std[std < 1e-6] = 1.0
        compact0_n = (compact0 - mean) / std
        compact1_n = (compact1 - mean) / std
        nearest01 = nearest_indices_batched(compact0_n, compact1_n)
        nearest10 = nearest_indices_batched(compact1_n, compact0_n)
        mechanism_span = max(float(np.linalg.norm(np.ptp(compact[:, 3:], axis=0))), 1e-6)

        candidates: list[np.ndarray] = []
        for source, target, nearest, reverse in (
            (position0, position1, nearest01, False),
            (position1, position0, nearest10, True),
        ):
            matched_target = target[nearest]
            eef_distance = np.linalg.norm(robot[source, :3] - robot[matched_target, :3], axis=1)
            source_mechanism = np.asarray(scene[source, mechanism_index], dtype=np.float64)
            target_mechanism = np.asarray(scene[matched_target, mechanism_index], dtype=np.float64)
            if source_mechanism.ndim == 1:
                mechanism_distance = np.abs(source_mechanism - target_mechanism)
            else:
                mechanism_distance = np.linalg.norm(source_mechanism - target_mechanism, axis=1)
            keep = (eef_distance < max_eef_distance_m) & (
                mechanism_distance < max_mechanism_fraction * mechanism_span
            )
            if reverse:
                candidates.append(np.stack([matched_target[keep], source[keep]], axis=1))
            else:
                candidates.append(np.stack([source[keep], matched_target[keep]], axis=1))

        pairs = np.concatenate(candidates, axis=0).astype(np.int64)
        if len(pairs):
            output[family] = np.unique(pairs, axis=0)
    return output


def save_checkpoint(
    path: Path,
    *,
    mode: str,
    checkpoint_epoch: int | None,
    model: nn.Module,
    chunk_size: int,
    cond_mean: np.ndarray,
    cond_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    hidden: int,
    layers: int,
    articulation_geometry: dict[str, Any],
    minimal_g_scope: str,
    primitive_geometry: dict[str, Any] | None,
    zero_task_onehot: bool,
    fm_steps: int,
    sample_seed: int,
    base_dim: int | None = None,
    g_dim: int | None = None,
    gate_max: float = 1.0,
    g_scale_aug_values: list[float] | None = None,
    g_dropout_prob: float = 0.0,
    query_g_include_token_type: bool = True,
    query_g_token_type_mode: str = "normal",
    field_alignment_weight: float = 0.0,
    structured_field_config: dict[str, float] | None = None,
    inverse_pair_loss_weight: float = 0.0,
    inverse_pair_config: dict[str, Any] | None = None,
    source_distill_weight: float = 0.0,
    residual_adapter_config: dict[str, Any] | None = None,
    executor_state_scope: str = "full",
    executor_language_mode: str = "onehot",
    visual_encoder_name: str | None = None,
    visual_feature_dim: int = 0,
    mask_executor_language_with_g: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "mode": mode,
        "checkpoint_epoch": checkpoint_epoch,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "chunk_size": chunk_size,
        "cond_dim": int(cond_mean.shape[0]),
        "action_dim": int(chunk_size * 7),
        "cond_mean": cond_mean.astype(np.float32),
        "cond_std": cond_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "hidden": hidden,
        "layers": layers,
        "articulation_geometry": articulation_geometry,
        "minimal_g_scope": str(minimal_g_scope),
        "primitive_geometry": {} if primitive_geometry is None else primitive_geometry,
        "zero_task_onehot": zero_task_onehot,
        "fm_steps": fm_steps,
        "sample_seed": sample_seed,
        "base_dim": None if base_dim is None else int(base_dim),
        "g_dim": None if g_dim is None else int(g_dim),
        "gate_max": float(gate_max),
        "g_scale_aug_values": [] if g_scale_aug_values is None else [float(v) for v in g_scale_aug_values],
        "g_dropout_prob": float(g_dropout_prob),
        "query_g_include_token_type": bool(query_g_include_token_type),
        "query_g_token_type_mode": str(query_g_token_type_mode),
        "field_alignment_weight": float(field_alignment_weight),
        "structured_field_config": {} if structured_field_config is None else dict(structured_field_config),
        "inverse_pair_loss_weight": float(inverse_pair_loss_weight),
        "inverse_pair_config": {} if inverse_pair_config is None else dict(inverse_pair_config),
        "source_distill_weight": float(source_distill_weight),
        "residual_adapter_config": (
            {} if residual_adapter_config is None else dict(residual_adapter_config)
        ),
        "executor_state_scope": str(executor_state_scope),
        "executor_language_mode": str(executor_language_mode),
        "visual_encoder_name": visual_encoder_name,
        "visual_feature_dim": int(visual_feature_dim),
        "mask_executor_language_with_g": bool(mask_executor_language_with_g),
    }
    torch.save(ckpt, path)


def sample_flow(model: FlowMLP, cond: torch.Tensor, steps: int, generator: torch.Generator | None = None) -> torch.Tensor:
    x = torch.randn((cond.shape[0], model.net[-1].out_features), device=cond.device, generator=generator)
    dt = 1.0 / max(1, steps)
    for i in range(max(1, steps)):
        t = torch.full((cond.shape[0],), i / max(1, steps), device=cond.device)
        x = x + model(x, t, cond) * dt
    return x


def sample_minimal_g_token_flow(
    model: MinimalGTokenFlowTransformer,
    base_n: torch.Tensor,
    g_raw: torch.Tensor,
    steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    x = torch.randn(
        (base_n.shape[0], model.chunk_size * 7),
        device=base_n.device,
        generator=generator,
    )
    dt = 1.0 / max(1, steps)
    for i in range(max(1, steps)):
        t = torch.full((base_n.shape[0],), i / max(1, steps), device=base_n.device)
        x = x + model(x, t, base_n, g_raw) * dt
    return x


def train_mode(
    arrays: dict[str, np.ndarray],
    mode: str,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    hidden: int,
    layers: int,
    lr: float,
    device: str,
    seed: int,
    loader_seed: int,
    split_seed: int,
    checkpoint_epochs: set[int],
    articulation_geometry: dict[str, Any],
    minimal_g_scope: str,
    primitive_geometry: dict[str, Any] | None,
    zero_task_onehot: bool,
    fm_steps: int,
    g_scale_aug_values: list[float],
    g_dropout_prob: float,
    gate_max: float,
    query_g_include_token_type: bool,
    query_g_token_type_mode: str,
    field_alignment_weight: float,
    orthogonal_ratio_max: float,
    first_action_loss_weight: float,
    inverse_pair_loss_weight: float,
    inverse_pairs_per_batch: int,
    inverse_pair_max_eef_distance_m: float,
    inverse_pair_max_mechanism_fraction: float,
    init_checkpoint: Path | None = None,
    source_distill_weight: float = 0.0,
    residual_adapter: bool = False,
    residual_adapter_hidden: int = 256,
    reverse_source_loss_weight: float = 1.0,
    reverse_task_loss_weights: dict[str, float] | None = None,
    residual_adapter_expert_tasks: list[str] | None = None,
    residual_router_loss_weight: float = 1.0,
    residual_adapter_source_expert: bool = False,
    executor_state_scope: str = "full",
    executor_language_mode: str = "onehot",
    visual_encoder_name: str | None = None,
    visual_feature_dim: int = 0,
    mask_executor_language_with_g: bool = True,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if mode in {
        "chunk_bc_lang_minimal_g_concat",
        "chunk_bc_lang_minimal_g_structured",
    } | LANG_MINIMAL_G_PROJECTED_MODES:
        base_x, g_x = make_minimal_condition_parts(
            arrays, executor_state_scope, executor_language_mode
        )
    elif mode in MASKED_LANG_MINIMAL_G_LEARNED_MODES | MASKED_LANG_MINIMAL_G_PROJECTED_MODES:
        base_x, g_x = make_minimal_query_condition_parts(
            arrays,
            executor_state_scope,
            mask_language_with_g=mask_executor_language_with_g,
            language_mode=executor_language_mode,
        )
    else:
        base_x, g_x = make_condition_parts(arrays)
    cond, cond_meta = make_condition(
        arrays,
        mode,
        executor_state_scope=executor_state_scope,
        mask_executor_language_with_g=mask_executor_language_with_g,
        executor_language_mode=executor_language_mode,
    )
    y = arrays["action_chunks"].reshape(len(arrays["action_chunks"]), -1).astype(np.float32)
    cond_mean, cond_std = cond.mean(axis=0), cond.std(axis=0) + 1e-6
    if mode in {
        "chunk_bc_lang",
        "chunk_bc_lang_query_g",
        "chunk_bc_lang_minimal_g_concat",
        "chunk_bc_lang_minimal_g_structured",
    } | MASKED_LANG_MINIMAL_G_LEARNED_MODES | MINIMAL_G_PROJECTED_MODES:
        state_dim = int(
            make_state_lang_parts(arrays, executor_state_scope, "none")[0].shape[1]
        )
        if executor_language_mode == "onehot":
            task_dim = int(arrays["task_oh"].shape[1])
            task_slice = slice(state_dim, state_dim + task_dim)
            cond_mean[task_slice] = 0.0
            cond_std[task_slice] = 1.0
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-6
    init_ckpt: dict[str, Any] | None = None
    if init_checkpoint is not None:
        init_ckpt = torch.load(init_checkpoint, map_location="cpu")
        if str(init_ckpt.get("mode")) != mode:
            raise ValueError(
                f"Initialization checkpoint mode {init_ckpt.get('mode')!r} does not match {mode!r}"
            )
        init_cond_mean = np.asarray(init_ckpt["cond_mean"], dtype=np.float32)
        init_cond_std = np.asarray(init_ckpt["cond_std"], dtype=np.float32)
        init_y_mean = np.asarray(init_ckpt["y_mean"], dtype=np.float32)
        init_y_std = np.asarray(init_ckpt["y_std"], dtype=np.float32)
        if init_cond_mean.shape != cond_mean.shape or init_y_mean.shape != y_mean.shape:
            raise ValueError(
                "Initialization checkpoint normalization dimensions do not match the current dataset"
            )
        cond_mean, cond_std = init_cond_mean, init_cond_std
        y_mean, y_std = init_y_mean, init_y_std
    y_n = (y - y_mean) / y_std
    base_dim = int(base_x.shape[1])
    g_dim = int(g_x.shape[1])

    train_idx, val_idx = segment_train_val_indices(arrays["segment_id"], split_seed)
    inverse_pair_families: dict[str, np.ndarray] = {}
    inverse_pair_pool = np.empty((0, 2), dtype=np.int64)
    if inverse_pair_loss_weight > 0:
        if mode not in {"chunk_bc_minimal_g_tokens", "chunk_fm_minimal_g_tokens"}:
            raise ValueError("inverse-pair loss is currently defined only for minimal-G token BC/FM")
        if inverse_pairs_per_batch <= 0:
            raise ValueError("inverse_pairs_per_batch must be positive when inverse-pair loss is enabled")
        inverse_pair_families = build_inverse_goal_train_pairs(
            arrays,
            train_idx,
            max_eef_distance_m=inverse_pair_max_eef_distance_m,
            max_mechanism_fraction=inverse_pair_max_mechanism_fraction,
        )
        if not inverse_pair_families:
            raise RuntimeError("inverse-pair loss was enabled but no qualifying train pairs were found")
        pair_rng = np.random.default_rng(loader_seed + 42_424)
        family_target = max(len(pairs) for pairs in inverse_pair_families.values())
        balanced_families = []
        for pairs in inverse_pair_families.values():
            selected = pair_rng.choice(len(pairs), size=family_target, replace=len(pairs) < family_target)
            balanced_families.append(pairs[selected])
        inverse_pair_pool = np.concatenate(balanced_families, axis=0).astype(np.int64)
        pair_rng.shuffle(inverse_pair_pool)
    train_cond = torch.from_numpy(cond[train_idx])
    train_base = torch.from_numpy(base_x[train_idx])
    train_g = torch.from_numpy(g_x[train_idx])
    train_y = torch.from_numpy(y_n[train_idx])
    train_reverse = torch.from_numpy(arrays["is_reverse_source"][train_idx].astype(bool))
    reverse_task_loss_weights = reverse_task_loss_weights or {}
    residual_adapter_expert_tasks = residual_adapter_expert_tasks or []
    expert_task_to_id = {task: index for index, task in enumerate(residual_adapter_expert_tasks)}
    source_expert_index = (
        len(residual_adapter_expert_tasks) if residual_adapter_source_expert else -1
    )
    train_task_weight = torch.from_numpy(
        np.asarray(
            [reverse_task_loss_weights.get(str(task), 1.0) for task in arrays["task_name"][train_idx]],
            dtype=np.float32,
        )
    )
    train_expert_index = torch.from_numpy(
        np.asarray(
            [
                expert_task_to_id.get(str(task), source_expert_index)
                for task in arrays["task_name"][train_idx]
            ],
            dtype=np.int64,
        )
    )
    val_cond = torch.from_numpy(((cond[val_idx] - cond_mean) / cond_std).astype(np.float32)).to(device)
    val_y_raw = torch.from_numpy(y[val_idx]).to(device)

    out_dim = y_n.shape[1]
    chunk_size = int(arrays["action_chunks"].shape[1])
    if first_action_loss_weight <= 0:
        raise ValueError("first_action_loss_weight must be positive")

    def action_chunk_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error_sq = (prediction - target).reshape(-1, chunk_size, 7).square()
        if first_action_loss_weight == 1.0:
            return torch.mean(error_sq)
        weights = torch.ones((1, chunk_size, 1), dtype=error_sq.dtype, device=error_sq.device)
        weights[:, 0, :] = float(first_action_loss_weight)
        denom = error_sq.shape[0] * 7.0 * torch.sum(weights)
        return torch.sum(error_sq * weights) / denom

    def action_chunk_mse_per_sample(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        error_sq = (prediction - target).reshape(-1, chunk_size, 7).square()
        weights = torch.ones((1, chunk_size, 1), dtype=error_sq.dtype, device=error_sq.device)
        weights[:, 0, :] = float(first_action_loss_weight)
        return torch.sum(error_sq * weights, dim=(1, 2)) / (7.0 * torch.sum(weights))

    if mode in {
        "chunk_bc_state",
        "chunk_bc_lang",
        "chunk_bc_query_g",
        "chunk_bc_lang_query_g",
        "chunk_bc_lang_minimal_g_concat",
        "chunk_bc_minimal_g_concat",
        "chunk_bc_v7",
        "chunk_bc_v7_scale_aug",
    }:
        model: nn.Module = ChunkMLP(cond.shape[1], out_dim, hidden=hidden, layers=layers).to(device)
    elif mode == "chunk_bc_minimal_g_film":
        model = FiLMMinimalGChunkMLP(base_dim, g_dim, out_dim, hidden=hidden, layers=layers).to(device)
    elif mode == "chunk_bc_minimal_g_tokens":
        model = MinimalGTokenChunkTransformer(
            base_dim,
            g_dim,
            chunk_size=chunk_size,
            hidden=hidden,
            layers=layers,
        ).to(device)
    elif mode == "chunk_fm_minimal_g_tokens":
        model = MinimalGTokenFlowTransformer(
            base_dim,
            g_dim,
            chunk_size=chunk_size,
            hidden=hidden,
            layers=layers,
        ).to(device)
    elif mode == "chunk_bc_minimal_g_semantic":
        model = SemanticMinimalGChunkMLP(
            base_dim,
            g_dim,
            out_dim,
            chunk_size=chunk_size,
            hidden=hidden,
            layers=layers,
            action_std=y_std,
        ).to(device)
    elif mode == "chunk_bc_minimal_g_frame":
        model = FrameMinimalGChunkMLP(
            base_dim,
            g_dim,
            out_dim,
            chunk_size=chunk_size,
            hidden=hidden,
            layers=layers,
            action_mean=y_mean,
            action_std=y_std,
        ).to(device)
    elif mode == "chunk_bc_lang_minimal_g_structured":
        model = StructuredMinimalGChunkMLP(
            base_dim,
            g_dim,
            out_dim,
            chunk_size=int(arrays["action_chunks"].shape[1]),
            hidden=hidden,
            layers=layers,
            action_std=y_std,
        ).to(device)
    elif mode in MINIMAL_G_PROJECTED_MODES:
        model = ProjectedMinimalGChunkMLP(
            base_dim,
            g_dim,
            out_dim,
            chunk_size=int(arrays["action_chunks"].shape[1]),
            hidden=hidden,
            layers=layers,
            action_mean=y_mean,
            action_std=y_std,
            orthogonal_ratio_max=(
                orthogonal_ratio_max if mode in BOUNDED_MINIMAL_G_PROJECTED_MODES else None
            ),
            invariant_gain=mode in INVARIANT_GAIN_MINIMAL_G_PROJECTED_MODES,
        ).to(device)
    elif mode == "chunk_bc_v7_gated":
        model = GatedChunkMLP(base_dim, g_dim, out_dim, hidden=hidden, layers=layers, gate_max=gate_max).to(device)
    elif mode == "chunk_fm_v7":
        model = FlowMLP(cond.shape[1], out_dim, hidden=hidden, layers=layers).to(device)
    else:
        raise ValueError(mode)

    if init_ckpt is not None:
        model.load_state_dict(init_ckpt["model_state"], strict=True)

    teacher: nn.Module | None = None
    residual_adapter_config: dict[str, Any] = {}
    if residual_adapter:
        if init_ckpt is None:
            raise ValueError("--residual-adapter requires --init-checkpoint")
        if mode != "chunk_bc_minimal_g_tokens":
            raise ValueError("Residual adaptation is currently implemented for minimal-G token BC")
        if residual_adapter_hidden <= 0:
            raise ValueError("--residual-adapter-hidden must be positive")
        if reverse_source_loss_weight <= 0:
            raise ValueError("--reverse-source-loss-weight must be positive")
        if residual_adapter_source_expert and not residual_adapter_expert_tasks:
            raise ValueError(
                "--residual-adapter-source-expert requires --residual-adapter-expert-tasks"
            )
        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        if residual_adapter_expert_tasks:
            if residual_router_loss_weight <= 0:
                raise ValueError("--residual-router-loss-weight must be positive")
            model = RoutedResidualAdapterMinimalGTokenPolicy(
                model,
                base_dim,
                g_dim,
                out_dim,
                hidden=residual_adapter_hidden,
                num_experts=(
                    len(residual_adapter_expert_tasks)
                    + int(residual_adapter_source_expert)
                ),
            ).to(device)
        else:
            model = ResidualAdapterMinimalGTokenPolicy(
                model,
                base_dim,
                g_dim,
                out_dim,
                hidden=residual_adapter_hidden,
            ).to(device)
        residual_adapter_config = {
            "hidden": int(residual_adapter_hidden),
            "reverse_source_loss_weight": float(reverse_source_loss_weight),
            "reverse_task_loss_weights": dict(reverse_task_loss_weights),
            "expert_tasks": list(residual_adapter_expert_tasks),
            "source_expert": bool(residual_adapter_source_expert),
            "router_loss_weight": float(residual_router_loss_weight),
            "output_scale": 1.0,
        }
    elif source_distill_weight > 0:
        if init_ckpt is None:
            raise ValueError("--source-distill-weight requires --init-checkpoint")
        if mode != "chunk_bc_minimal_g_tokens":
            raise ValueError("Source distillation is currently implemented for minimal-G token BC")
        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    structured_field_config: dict[str, float] = {}
    if isinstance(model, StructuredMinimalGChunkMLP):
        structured_field_config = {
            "readiness_distance": float(model.readiness_distance),
            "readiness_temperature": float(model.readiness_temperature),
        }
    elif isinstance(model, ProjectedMinimalGChunkMLP):
        structured_field_config = {
            "readiness_distance": float(model.readiness_distance),
            "readiness_temperature": float(model.readiness_temperature),
            "field_gain_max": float(model.field_gain_max),
        }
        if model.orthogonal_ratio_max is not None:
            structured_field_config["orthogonal_ratio_max"] = float(model.orthogonal_ratio_max)
        structured_field_config["invariant_gain"] = bool(model.invariant_gain)

    opt = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )
    loader_gen = torch.Generator()
    loader_gen.manual_seed(loader_seed)
    loader = DataLoader(
        TensorDataset(
            train_cond,
            train_base,
            train_g,
            train_y,
            train_reverse,
            train_task_weight,
            train_expert_index,
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_gen,
    )
    inverse_pair_loader: DataLoader | None = None
    if len(inverse_pair_pool):
        pair_loader_gen = torch.Generator()
        pair_loader_gen.manual_seed(loader_seed + 42_424)
        inverse_pair_loader = DataLoader(
            TensorDataset(torch.from_numpy(inverse_pair_pool)),
            batch_size=inverse_pairs_per_batch,
            shuffle=True,
            drop_last=False,
            generator=pair_loader_gen,
        )

    history: list[dict[str, Any]] = []
    best_val_first_action_mse = float("inf")
    best_val_epoch = 0
    best_val_model_state: dict[str, torch.Tensor] | None = None
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_mean_t = torch.from_numpy(cond_mean.astype(np.float32)).to(device)
    cond_std_t = torch.from_numpy(cond_std.astype(np.float32)).to(device)
    y_mean_t = torch.from_numpy(y_mean.astype(np.float32)).to(device)
    y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(device)
    val_g_raw = torch.from_numpy(g_x[val_idx].astype(np.float32)).to(device)
    aug_values = [float(v) for v in g_scale_aug_values] if g_scale_aug_values else [1.0]
    aug_values_t = torch.tensor(aug_values, dtype=torch.float32, device=device)
    use_train_g_aug = mode in {
        "chunk_bc_v7_scale_aug",
        "chunk_bc_v7_gated",
        "chunk_bc_minimal_g_tokens",
        "chunk_fm_minimal_g_tokens",
    }
    if use_train_g_aug and g_dropout_prob >= 1.0:
        val_g_raw = torch.zeros_like(val_g_raw)
    inverse_pair_config = {
        "families": {name: int(len(pairs)) for name, pairs in inverse_pair_families.items()},
        "balanced_pool_pairs": int(len(inverse_pair_pool)),
        "pairs_per_batch": int(inverse_pairs_per_batch),
        "max_eef_distance_m": float(inverse_pair_max_eef_distance_m),
        "max_mechanism_fraction": float(inverse_pair_max_mechanism_fraction),
    }

    def make_cond_n_t(
        base_b: torch.Tensor,
        g_b: torch.Tensor,
        *,
        train_aug: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_b = base_b.to(device)
        g_b = g_b.to(device)
        if train_aug and use_train_g_aug:
            idx = torch.randint(0, len(aug_values), (g_b.shape[0],), device=device)
            scale = aug_values_t[idx][:, None]
            if g_dropout_prob > 0:
                keep = (torch.rand((g_b.shape[0], 1), device=device) >= g_dropout_prob).float()
                scale = scale * keep
            g_b = g_b * scale
        cond_b = torch.cat([base_b, g_b], dim=-1)
        return (cond_b - cond_mean_t) / cond_std_t, g_b

    def inverse_pair_difference_loss(pair_positions: torch.Tensor) -> torch.Tensor:
        pair_count = int(pair_positions.shape[0])
        flat_positions = pair_positions.reshape(-1)
        pair_base = train_base[flat_positions]
        pair_g = train_g[flat_positions]
        pair_y = train_y[flat_positions].to(device).reshape(pair_count, 2, out_dim)
        pair_cond_n, pair_g_raw = make_cond_n_t(pair_base, pair_g, train_aug=False)
        if mode == "chunk_bc_minimal_g_tokens":
            pair_prediction = model(pair_cond_n[:, :base_dim], pair_g_raw).reshape(pair_count, 2, out_dim)
            pair_target = pair_y
        elif mode == "chunk_fm_minimal_g_tokens":
            shared_x0 = torch.randn((pair_count, out_dim), device=device)
            shared_t = torch.rand((pair_count,), device=device)
            x_t = (
                (1.0 - shared_t[:, None, None]) * shared_x0[:, None, :]
                + shared_t[:, None, None] * pair_y
            )
            pair_target = pair_y - shared_x0[:, None, :]
            pair_prediction = model(
                x_t.reshape(-1, out_dim),
                shared_t.repeat_interleave(2),
                pair_cond_n[:, :base_dim],
                pair_g_raw,
            ).reshape(pair_count, 2, out_dim)
        else:
            raise RuntimeError(f"Unsupported inverse-pair mode: {mode}")
        return action_chunk_mse(
            pair_prediction[:, 0] - pair_prediction[:, 1],
            pair_target[:, 0] - pair_target[:, 1],
        )

    def eval_metrics() -> tuple[float, float, float, float | None, float | None]:
        model.eval()
        with torch.no_grad():
            gate_mean: float | None = None
            field_gain_mean: float | None = None
            if mode in {
                "chunk_bc_state",
                "chunk_bc_lang",
                "chunk_bc_query_g",
                "chunk_bc_lang_query_g",
                "chunk_bc_lang_minimal_g_concat",
                "chunk_bc_minimal_g_concat",
                "chunk_bc_v7",
                "chunk_bc_v7_scale_aug",
            }:
                pred_n = model(val_cond)
            elif mode == "chunk_bc_minimal_g_film":
                pred_n = model(
                    val_cond[:, :base_dim],
                    val_cond[:, base_dim : base_dim + g_dim],
                )
            elif mode == "chunk_bc_minimal_g_tokens":
                pred_n = model(val_cond[:, :base_dim], val_g_raw)
            elif mode == "chunk_fm_minimal_g_tokens":
                gen = torch.Generator(device=device)
                gen.manual_seed(seed + 100_003)
                pred_n = sample_minimal_g_token_flow(
                    model,
                    val_cond[:, :base_dim],
                    val_g_raw,
                    steps=fm_steps,
                    generator=gen,
                )
            elif mode == "chunk_bc_minimal_g_semantic":
                pred_n, aux = model(
                    val_cond[:, :base_dim], val_g_raw, return_aux=True
                )
                valid = torch.linalg.vector_norm(val_g_raw, dim=-1) > 1e-6
                if torch.any(valid):
                    gate_mean = float(
                        (aux["attraction_gain"][valid].mean() + aux["flow_gain"][valid].mean()).cpu()
                        / 2.0
                    )
            elif mode == "chunk_bc_minimal_g_frame":
                pred_n = model(val_cond[:, :base_dim], val_g_raw)
            elif mode == "chunk_bc_lang_minimal_g_structured":
                pred_n_raw, aux = model(
                    val_cond[:, :base_dim], val_cond[:, base_dim : base_dim + g_dim], val_g_raw, return_aux=True
                )
                pred_n = pred_n_raw
                gate_mean = float(aux["readiness"].detach().mean().cpu())
            elif mode in MINIMAL_G_PROJECTED_MODES:
                pred_n_raw, aux = model(
                    val_cond[:, :base_dim], val_cond[:, base_dim : base_dim + g_dim], val_g_raw, return_aux=True
                )
                pred_n = pred_n_raw
                gate_mean = float(aux["readiness"].detach().mean().cpu())
                valid = aux["valid"].detach().squeeze(-1) > 0
                if torch.any(valid):
                    field_gain_mean = float(aux["field_gain"].detach()[valid].mean().cpu())
            elif mode == "chunk_bc_v7_gated":
                pred_n_raw, gate = model(val_cond, return_gate=True)
                pred_n = pred_n_raw
                gate_mean = float(gate.detach().mean().cpu())
            else:
                gen = torch.Generator(device=device)
                gen.manual_seed(seed + 100_003)
                pred_n = sample_flow(model, val_cond, steps=fm_steps, generator=gen)
            pred = pred_n.cpu().numpy() * y_std + y_mean
        val_raw = val_y_raw.cpu().numpy()
        mse = float(np.mean((pred - val_raw) ** 2))
        first_action_mse = float(
            np.mean(
                (
                    pred.reshape(-1, chunk_size, 7)[:, 0]
                    - val_raw.reshape(-1, chunk_size, 7)[:, 0]
                )
                ** 2
            )
        )
        first_grip = float(np.mean((pred[:, 6] >= 0) == (val_raw[:, 6] >= 0)))
        return mse, first_action_mse, first_grip, gate_mean, field_gain_mean

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        field_losses = []
        inverse_pair_losses = []
        source_distill_losses = []
        router_losses = []
        inverse_pair_iterator = iter(inverse_pair_loader) if inverse_pair_loader is not None else None
        for cond_raw_b, base_b, g_b, y_b, reverse_b, task_weight_b, expert_index_b in loader:
            if use_train_g_aug:
                cond_b, g_raw_b = make_cond_n_t(base_b, g_b, train_aug=True)
            else:
                cond_raw_b = cond_raw_b.to(device)
                cond_b = (cond_raw_b - cond_mean_t) / cond_std_t
                g_raw_b = g_b.to(device)
            y_b = y_b.to(device)
            if mode in {
                "chunk_bc_state",
                "chunk_bc_lang",
                "chunk_bc_query_g",
                "chunk_bc_lang_query_g",
                "chunk_bc_lang_minimal_g_concat",
                "chunk_bc_minimal_g_concat",
                "chunk_bc_v7",
                "chunk_bc_v7_scale_aug",
                "chunk_bc_v7_gated",
            }:
                pred = model(cond_b)
                loss = action_chunk_mse(pred, y_b)
            elif mode == "chunk_bc_minimal_g_film":
                pred = model(
                    cond_b[:, :base_dim],
                    cond_b[:, base_dim : base_dim + g_dim],
                )
                loss = action_chunk_mse(pred, y_b)
            elif mode == "chunk_bc_minimal_g_tokens":
                routed_aux: dict[str, torch.Tensor] | None = None
                if isinstance(model, RoutedResidualAdapterMinimalGTokenPolicy):
                    pred, routed_aux = model(
                        cond_b[:, :base_dim], g_raw_b, return_aux=True
                    )
                else:
                    pred = model(cond_b[:, :base_dim], g_raw_b)
                if residual_adapter:
                    if teacher is None:
                        raise RuntimeError("Residual adapter teacher was not initialized")
                    reverse_mask = reverse_b.to(device)
                    source_mask = ~reverse_mask
                    loss_terms = []
                    if torch.any(reverse_mask):
                        reverse_prediction = pred[reverse_mask]
                        if isinstance(model, RoutedResidualAdapterMinimalGTokenPolicy):
                            reverse_expert_index = expert_index_b.to(device)[reverse_mask]
                            if torch.any(reverse_expert_index < 0):
                                raise RuntimeError(
                                    "Every reversed record must map to --residual-adapter-expert-tasks"
                                )
                            reverse_prediction, reverse_aux = model(
                                cond_b[reverse_mask, :base_dim],
                                g_raw_b[reverse_mask],
                                expert_index=reverse_expert_index,
                                return_aux=True,
                            )
                            router_loss = torch.nn.functional.cross_entropy(
                                reverse_aux["router_logits"], reverse_expert_index
                            )
                            loss_terms.append(float(residual_router_loss_weight) * router_loss)
                            router_losses.append(float(router_loss.detach().cpu()))
                        per_sample_reverse_loss = action_chunk_mse_per_sample(
                            reverse_prediction, y_b[reverse_mask]
                        )
                        reverse_weights = task_weight_b.to(device)[reverse_mask]
                        reverse_loss = torch.sum(per_sample_reverse_loss * reverse_weights) / torch.sum(
                            reverse_weights
                        ).clamp_min(1e-6)
                        loss_terms.append(float(reverse_source_loss_weight) * reverse_loss)
                    if torch.any(source_mask):
                        with torch.no_grad():
                            source_target = teacher(
                                cond_b[source_mask, :base_dim], g_raw_b[source_mask]
                            )
                        if (
                            routed_aux is not None
                            and residual_adapter_source_expert
                            and isinstance(model, RoutedResidualAdapterMinimalGTokenPolicy)
                        ):
                            source_expert_indices = expert_index_b.to(device)[source_mask]
                            source_prediction, source_aux = model(
                                cond_b[source_mask, :base_dim],
                                g_raw_b[source_mask],
                                expert_index=source_expert_indices,
                                return_aux=True,
                            )
                            preserve_loss = action_chunk_mse(source_prediction, source_target)
                            source_router_loss = torch.nn.functional.cross_entropy(
                                source_aux["router_logits"], source_expert_indices
                            )
                            loss_terms.append(
                                float(residual_router_loss_weight) * source_router_loss
                            )
                            router_losses.append(float(source_router_loss.detach().cpu()))
                        elif routed_aux is not None:
                            source_expert_predictions = routed_aux["expert_predictions"][source_mask]
                            expanded_source_target = source_target[:, None, :].expand_as(
                                source_expert_predictions
                            )
                            preserve_loss = action_chunk_mse(
                                source_expert_predictions.reshape(-1, out_dim),
                                expanded_source_target.reshape(-1, out_dim),
                            )
                        else:
                            preserve_loss = action_chunk_mse(pred[source_mask], source_target)
                        loss_terms.append(float(source_distill_weight) * preserve_loss)
                        source_distill_losses.append(float(preserve_loss.detach().cpu()))
                    if not loss_terms:
                        raise RuntimeError("Residual adapter batch contained no samples")
                    loss = torch.stack(loss_terms).sum()
                else:
                    loss = action_chunk_mse(pred, y_b)
            elif mode == "chunk_fm_minimal_g_tokens":
                x0 = torch.randn_like(y_b)
                t = torch.rand((y_b.shape[0],), device=device)
                x_t = (1.0 - t[:, None]) * x0 + t[:, None] * y_b
                target_v = y_b - x0
                pred_v = model(x_t, t, cond_b[:, :base_dim], g_raw_b)
                loss = action_chunk_mse(pred_v, target_v)
            elif mode == "chunk_bc_minimal_g_semantic":
                pred, aux = model(cond_b[:, :base_dim], g_b.to(device), return_aux=True)
                bc_loss = action_chunk_mse(pred, y_b)
                residual_raw = (
                    aux["residual_n"] * y_std_t[None, :]
                ).reshape(-1, chunk_size, 7)[:, :, :3]
                attraction = g_b.to(device)[:, :3]
                flow = g_b.to(device)[:, 3:6]
                attraction_norm = torch.linalg.vector_norm(attraction, dim=-1, keepdim=True)
                attraction_dir = attraction / attraction_norm.clamp_min(1e-6)
                flow_orth = flow - torch.sum(flow * attraction_dir, dim=-1, keepdim=True) * attraction_dir
                flow_orth_norm = torch.linalg.vector_norm(flow_orth, dim=-1, keepdim=True)
                flow_orth_dir = flow_orth / flow_orth_norm.clamp_min(1e-6)
                projection_a = torch.sum(residual_raw * attraction_dir[:, None, :], dim=-1).square()
                projection_f = torch.sum(residual_raw * flow_orth_dir[:, None, :], dim=-1).square()
                projection_a *= (attraction_norm > 1e-6).float()
                projection_f *= (flow_orth_norm > 1e-6).float()
                residual_energy = torch.sum(residual_raw.square(), dim=-1).clamp_min(1e-4)
                valid = (
                    (attraction_norm > 1e-6)
                    | (torch.linalg.vector_norm(flow, dim=-1, keepdim=True) > 1e-6)
                ).float()
                residual_leakage = (projection_a + projection_f) / residual_energy
                field_loss = torch.sum(residual_leakage * valid) / (
                    valid.sum().clamp_min(1.0) * chunk_size
                )
                loss = bc_loss + float(field_alignment_weight) * field_loss
                field_losses.append(float(field_loss.detach().cpu()))
            elif mode == "chunk_bc_minimal_g_frame":
                pred = model(cond_b[:, :base_dim], g_b.to(device))
                loss = action_chunk_mse(pred, y_b)
            elif mode == "chunk_bc_lang_minimal_g_structured":
                g_raw_b = g_b.to(device)
                pred, aux = model(
                    cond_b[:, :base_dim], cond_b[:, base_dim : base_dim + g_dim], g_raw_b, return_aux=True
                )
                bc_loss = action_chunk_mse(pred, y_b)
                pred_raw = pred * y_std_t[None, :] + y_mean_t[None, :]
                pred_xyz = pred_raw.reshape(pred_raw.shape[0], -1, 7)[:, :, :3].mean(dim=1)
                field_direction = aux["field_direction"]
                cosine = torch.nn.functional.cosine_similarity(pred_xyz, field_direction, dim=-1, eps=1e-6)
                valid = aux["valid"].squeeze(-1)
                field_loss = torch.sum((1.0 - cosine) * valid) / valid.sum().clamp_min(1.0)
                loss = bc_loss + float(field_alignment_weight) * field_loss
                field_losses.append(float(field_loss.detach().cpu()))
            elif mode in MINIMAL_G_PROJECTED_MODES:
                g_raw_b = g_b.to(device)
                pred = model(cond_b[:, :base_dim], cond_b[:, base_dim : base_dim + g_dim], g_raw_b)
                loss = action_chunk_mse(pred, y_b)
            else:
                x0 = torch.randn_like(y_b)
                t = torch.rand((y_b.shape[0],), device=device)
                x_t = (1.0 - t[:, None]) * x0 + t[:, None] * y_b
                target_v = y_b - x0
                pred_v = model(x_t, t, cond_b)
                loss = action_chunk_mse(pred_v, target_v)
            if teacher is not None and not residual_adapter:
                source_mask = ~reverse_b.to(device)
                if torch.any(source_mask):
                    with torch.no_grad():
                        teacher_prediction = teacher(
                            cond_b[source_mask, :base_dim], g_raw_b[source_mask]
                        )
                    distill_loss = action_chunk_mse(
                        pred[source_mask], teacher_prediction
                    )
                    loss = loss + float(source_distill_weight) * distill_loss
                    source_distill_losses.append(float(distill_loss.detach().cpu()))
            if inverse_pair_iterator is not None:
                try:
                    pair_positions_b = next(inverse_pair_iterator)[0]
                except StopIteration:
                    inverse_pair_iterator = iter(inverse_pair_loader)
                    pair_positions_b = next(inverse_pair_iterator)[0]
                pair_loss = inverse_pair_difference_loss(pair_positions_b)
                loss = loss + float(inverse_pair_loss_weight) * pair_loss
                inverse_pair_losses.append(float(pair_loss.detach().cpu()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_mse, val_first_action_mse, val_grip, val_gate_mean, val_field_gain_mean = eval_metrics()
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_chunk_mse": val_mse,
            "val_first_action_mse": val_first_action_mse,
            "val_first_gripper_acc": val_grip,
        }
        if field_losses:
            rec["train_field_alignment_loss"] = float(np.mean(field_losses))
        if inverse_pair_losses:
            rec["train_inverse_pair_difference_loss"] = float(np.mean(inverse_pair_losses))
        if source_distill_losses:
            rec["train_source_distill_loss"] = float(np.mean(source_distill_losses))
        if router_losses:
            rec["train_router_loss"] = float(np.mean(router_losses))
        if isinstance(model, StructuredMinimalGChunkMLP):
            rec["approach_gain"] = float(torch.nn.functional.softplus(model.approach_gain_raw).detach().cpu())
            rec["flow_gain"] = float(torch.nn.functional.softplus(model.flow_gain_raw).detach().cpu())
        if val_gate_mean is not None:
            rec["val_gate_mean"] = val_gate_mean
        if val_field_gain_mean is not None:
            rec["val_field_gain_mean"] = val_field_gain_mean
        history.append(rec)
        if val_first_action_mse < best_val_first_action_mse:
            best_val_first_action_mse = val_first_action_mse
            best_val_epoch = epoch
            best_val_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        gate_msg = "" if val_gate_mean is None else f" gate={val_gate_mean:.3f}"
        if isinstance(model, StructuredMinimalGChunkMLP):
            gate_msg += f" gains={rec['approach_gain']:.3f}/{rec['flow_gain']:.3f}"
        elif val_field_gain_mean is not None:
            gate_msg += f" field_gain={val_field_gain_mean:.3f}"
        print(
            f"[{mode}] epoch {epoch}/{epochs} loss={rec['train_loss']:.5f} "
            f"val_chunk_mse={val_mse:.5f} val_first_mse={val_first_action_mse:.5f} "
            f"first_grip={val_grip:.3f}{gate_msg}",
            flush=True,
        )
        if epoch in checkpoint_epochs:
            save_checkpoint(
                out_dir / "checkpoints" / f"epoch_{epoch:04d}" / "policy.pt",
                mode=mode,
                checkpoint_epoch=epoch,
                model=model,
                chunk_size=int(arrays["action_chunks"].shape[1]),
                cond_mean=cond_mean,
                cond_std=cond_std,
                y_mean=y_mean,
                y_std=y_std,
                hidden=hidden,
                layers=layers,
                articulation_geometry=articulation_geometry,
                minimal_g_scope=minimal_g_scope,
                primitive_geometry=primitive_geometry,
                zero_task_onehot=zero_task_onehot,
                fm_steps=fm_steps,
                sample_seed=seed + 17,
                base_dim=base_dim,
                g_dim=g_dim,
                gate_max=gate_max,
                g_scale_aug_values=aug_values,
                g_dropout_prob=g_dropout_prob,
                query_g_include_token_type=query_g_include_token_type,
                query_g_token_type_mode=query_g_token_type_mode,
                field_alignment_weight=field_alignment_weight,
                structured_field_config=structured_field_config,
                inverse_pair_loss_weight=inverse_pair_loss_weight,
                inverse_pair_config=inverse_pair_config,
                source_distill_weight=source_distill_weight,
                residual_adapter_config=residual_adapter_config,
                executor_state_scope=executor_state_scope,
                executor_language_mode=executor_language_mode,
                visual_encoder_name=visual_encoder_name,
                visual_feature_dim=visual_feature_dim,
                mask_executor_language_with_g=mask_executor_language_with_g,
            )

    save_checkpoint(
        out_dir / "policy.pt",
        mode=mode,
        checkpoint_epoch=None,
        model=model,
        chunk_size=int(arrays["action_chunks"].shape[1]),
        cond_mean=cond_mean,
        cond_std=cond_std,
        y_mean=y_mean,
        y_std=y_std,
        hidden=hidden,
        layers=layers,
        articulation_geometry=articulation_geometry,
        minimal_g_scope=minimal_g_scope,
        primitive_geometry=primitive_geometry,
        zero_task_onehot=zero_task_onehot,
        fm_steps=fm_steps,
        sample_seed=seed + 17,
        base_dim=base_dim,
        g_dim=g_dim,
        gate_max=gate_max,
        g_scale_aug_values=aug_values,
        g_dropout_prob=g_dropout_prob,
        query_g_include_token_type=query_g_include_token_type,
        query_g_token_type_mode=query_g_token_type_mode,
        field_alignment_weight=field_alignment_weight,
        structured_field_config=structured_field_config,
        inverse_pair_loss_weight=inverse_pair_loss_weight,
        inverse_pair_config=inverse_pair_config,
        source_distill_weight=source_distill_weight,
        residual_adapter_config=residual_adapter_config,
        executor_state_scope=executor_state_scope,
        executor_language_mode=executor_language_mode,
        visual_encoder_name=visual_encoder_name,
        visual_feature_dim=visual_feature_dim,
        mask_executor_language_with_g=mask_executor_language_with_g,
    )
    if best_val_model_state is None or best_val_epoch <= 0:
        raise RuntimeError("Training did not produce a best validation checkpoint")
    model.load_state_dict(best_val_model_state)
    save_checkpoint(
        out_dir / "policy_best_val.pt",
        mode=mode,
        checkpoint_epoch=best_val_epoch,
        model=model,
        chunk_size=int(arrays["action_chunks"].shape[1]),
        cond_mean=cond_mean,
        cond_std=cond_std,
        y_mean=y_mean,
        y_std=y_std,
        hidden=hidden,
        layers=layers,
        articulation_geometry=articulation_geometry,
        minimal_g_scope=minimal_g_scope,
        primitive_geometry=primitive_geometry,
        zero_task_onehot=zero_task_onehot,
        fm_steps=fm_steps,
        sample_seed=seed + 17,
        base_dim=base_dim,
        g_dim=g_dim,
        gate_max=gate_max,
        g_scale_aug_values=aug_values,
        g_dropout_prob=g_dropout_prob,
        query_g_include_token_type=query_g_include_token_type,
        query_g_token_type_mode=query_g_token_type_mode,
        field_alignment_weight=field_alignment_weight,
        structured_field_config=structured_field_config,
        inverse_pair_loss_weight=inverse_pair_loss_weight,
        inverse_pair_config=inverse_pair_config,
        source_distill_weight=source_distill_weight,
        residual_adapter_config=residual_adapter_config,
        executor_state_scope=executor_state_scope,
        executor_language_mode=executor_language_mode,
        visual_encoder_name=visual_encoder_name,
        visual_feature_dim=visual_feature_dim,
        mask_executor_language_with_g=mask_executor_language_with_g,
    )
    metrics = {
        "history": history,
        "final": history[-1],
        "best_val_first_action": history[best_val_epoch - 1],
        "num_records": int(len(y)),
        "num_train_records": int(len(train_idx)),
        "num_val_records": int(len(val_idx)),
        "condition_meta": cond_meta,
        "base_dim": base_dim,
        "g_dim": g_dim,
        "g_scale_aug_values": aug_values if use_train_g_aug else [1.0],
        "g_dropout_prob": float(g_dropout_prob if use_train_g_aug else 0.0),
        "gate_max": float(gate_max),
        "query_g_include_token_type": bool(query_g_include_token_type),
        "query_g_token_type_mode": str(query_g_token_type_mode),
        "field_alignment_weight": float(field_alignment_weight),
        "structured_field_config": structured_field_config,
        "first_action_loss_weight": float(first_action_loss_weight),
        "inverse_pair_loss_weight": float(inverse_pair_loss_weight),
        "inverse_pair_config": inverse_pair_config,
        "source_distill_weight": float(source_distill_weight),
        "residual_adapter_config": residual_adapter_config,
        "executor_state_scope": str(executor_state_scope),
        "executor_language_mode": str(executor_language_mode),
        "visual_encoder_name": visual_encoder_name,
        "visual_feature_dim": int(visual_feature_dim),
        "mask_executor_language_with_g": bool(mask_executor_language_with_g),
    }
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=json_default)
    return metrics


def load_task_language_embeddings(data_root: Path) -> dict[str, np.ndarray]:
    embedding_path = data_root / "validation" / "lang_annotations" / "embeddings.npy"
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"MiniLM evaluation requires CALVIN validation embeddings: {embedding_path}"
        )
    loaded = np.load(embedding_path, allow_pickle=True)
    raw = loaded.item() if isinstance(loaded, np.ndarray) and loaded.shape == () else loaded
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected CALVIN language embedding format: {embedding_path}")

    embeddings: dict[str, np.ndarray] = {}
    embedding_dim: int | None = None
    for task, payload in raw.items():
        value = payload.get("emb") if isinstance(payload, dict) else payload
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.size == 0:
            continue
        vector = array.reshape(-1, array.shape[-1]).mean(axis=0).astype(np.float32)
        if embedding_dim is None:
            embedding_dim = int(vector.shape[0])
        elif vector.shape[0] != embedding_dim:
            raise RuntimeError(
                f"Inconsistent CALVIN validation embedding dimensions in {embedding_path}"
            )
        embeddings[str(task)] = vector
    if not embeddings:
        raise RuntimeError(f"No task embeddings found in {embedding_path}")
    return embeddings


class ChunkFlowPolicy:
    def __init__(
        self,
        ckpt_path: Path,
        task_to_id: dict[str, int],
        text_to_task: dict[str, str],
        device: str,
        task_language_embeddings: dict[str, np.ndarray] | None = None,
        g_scale: float = 1.0,
        query_g_control: str = "normal",
        language_control: str = "normal",
        action_execution_horizon: int = 1,
        temporal_ensemble: bool = False,
        temporal_ensemble_decay: float = 0.25,
        online_minimal_g_mode: str = "generated_query",
        sample_seed_override: int | None = None,
        reseed_each_episode: bool = True,
    ):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.mode = ckpt["mode"]
        self.task_to_id = task_to_id
        self.text_to_task = text_to_task
        self.task_language_embeddings = task_language_embeddings or {}
        self.device = torch.device(device)
        self.g_scale = float(g_scale)
        self.query_g_control = str(query_g_control)
        self.language_control = str(language_control)
        self.chunk_size = int(ckpt["chunk_size"])
        self.action_execution_horizon = int(action_execution_horizon)
        if not 1 <= self.action_execution_horizon <= self.chunk_size:
            raise ValueError(
                "action_execution_horizon must be between 1 and the checkpoint chunk size "
                f"({self.chunk_size}), got {self.action_execution_horizon}"
            )
        self.temporal_ensemble = bool(temporal_ensemble)
        self.temporal_ensemble_decay = float(temporal_ensemble_decay)
        self.online_minimal_g_mode = str(online_minimal_g_mode)
        if self.online_minimal_g_mode not in {"generated_query", "behavior_block_center"}:
            raise ValueError(f"Unsupported online minimal-G mode: {self.online_minimal_g_mode}")
        if self.temporal_ensemble and self.action_execution_horizon != 1:
            raise ValueError("temporal ensembling requires action_execution_horizon=1")
        if self.temporal_ensemble_decay < 0:
            raise ValueError("temporal_ensemble_decay must be non-negative")
        self.cond_mean = ckpt["cond_mean"].astype(np.float32)
        self.cond_std = ckpt["cond_std"].astype(np.float32)
        self.y_mean = ckpt["y_mean"].astype(np.float32)
        self.y_std = ckpt["y_std"].astype(np.float32)
        self.articulation_geometry = ckpt["articulation_geometry"]
        self.minimal_g_scope = str(ckpt.get("minimal_g_scope", "articulation"))
        self.primitive_geometry = ckpt.get("primitive_geometry") or {}
        self.zero_task_onehot = bool(ckpt.get("zero_task_onehot", False))
        self.fm_steps = int(ckpt.get("fm_steps", 10))
        self.sample_seed = int(ckpt.get("sample_seed", 0) if sample_seed_override is None else sample_seed_override)
        self.reseed_each_episode = bool(reseed_each_episode)
        self.query_g_include_token_type = bool(ckpt.get("query_g_include_token_type", True))
        self.query_g_token_type_mode = str(ckpt.get("query_g_token_type_mode", "normal"))
        self.executor_state_scope = str(ckpt.get("executor_state_scope", "full"))
        self.executor_language_mode = str(ckpt.get("executor_language_mode", "onehot"))
        if self.executor_language_mode not in EXECUTOR_LANGUAGE_MODES:
            raise RuntimeError(
                f"Unsupported checkpoint executor language mode: {self.executor_language_mode}"
            )
        self.mask_executor_language_with_g = bool(
            ckpt.get("mask_executor_language_with_g", True)
        )
        self.visual_encoder_name = ckpt.get("visual_encoder_name")
        self.visual_feature_dim = int(ckpt.get("visual_feature_dim", 0))
        self.visual_encoder: nn.Module | None = None
        if self.executor_state_scope == "visual":
            if not self.visual_encoder_name or self.visual_feature_dim <= 0:
                raise RuntimeError(
                    f"Visual checkpoint is missing encoder metadata: {ckpt_path}"
                )
            from transformers import Dinov2Model

            self.visual_encoder = Dinov2Model.from_pretrained(
                self.visual_encoder_name
            ).to(self.device).eval()
            for parameter in self.visual_encoder.parameters():
                parameter.requires_grad_(False)
            if int(self.visual_encoder.config.hidden_size) != self.visual_feature_dim:
                raise RuntimeError(
                    "Visual checkpoint feature dimension does not match the frozen encoder"
                )
        if self.mode in {
            "chunk_bc_state",
            "chunk_bc_lang",
            "chunk_bc_query_g",
            "chunk_bc_lang_query_g",
            "chunk_bc_lang_minimal_g_concat",
            "chunk_bc_minimal_g_concat",
            "chunk_bc_v7",
            "chunk_bc_v7_scale_aug",
        }:
            self.model: nn.Module = ChunkMLP(ckpt["cond_dim"], ckpt["action_dim"], ckpt["hidden"], ckpt["layers"])
        elif self.mode == "chunk_bc_minimal_g_film":
            base_dim = int(ckpt.get("base_dim") or 0)
            g_dim = int(ckpt.get("g_dim") or 0)
            self.model = FiLMMinimalGChunkMLP(
                base_dim,
                g_dim,
                ckpt["action_dim"],
                ckpt["hidden"],
                ckpt["layers"],
            )
        elif self.mode == "chunk_bc_minimal_g_tokens":
            base_dim = int(ckpt.get("base_dim") or 0)
            g_dim = int(ckpt.get("g_dim") or 0)
            base_policy = MinimalGTokenChunkTransformer(
                base_dim,
                g_dim,
                chunk_size=self.chunk_size,
                hidden=ckpt["hidden"],
                layers=ckpt["layers"],
            )
            residual_config = ckpt.get("residual_adapter_config") or {}
            if residual_config:
                expert_tasks = list(residual_config.get("expert_tasks") or [])
                if expert_tasks:
                    self.model = RoutedResidualAdapterMinimalGTokenPolicy(
                        base_policy,
                        base_dim,
                        g_dim,
                        ckpt["action_dim"],
                        hidden=int(residual_config["hidden"]),
                        num_experts=(
                            len(expert_tasks)
                            + int(bool(residual_config.get("source_expert", False)))
                        ),
                    )
                else:
                    self.model = ResidualAdapterMinimalGTokenPolicy(
                        base_policy,
                        base_dim,
                        g_dim,
                        ckpt["action_dim"],
                        hidden=int(residual_config["hidden"]),
                        output_scale=float(residual_config.get("output_scale", 1.0)),
                    )
            else:
                self.model = base_policy
        elif self.mode == "chunk_fm_minimal_g_tokens":
            base_dim = int(ckpt.get("base_dim") or 0)
            g_dim = int(ckpt.get("g_dim") or 0)
            self.model = MinimalGTokenFlowTransformer(
                base_dim,
                g_dim,
                chunk_size=self.chunk_size,
                hidden=ckpt["hidden"],
                layers=ckpt["layers"],
            )
        elif self.mode == "chunk_bc_minimal_g_semantic":
            base_dim = int(ckpt.get("base_dim") or 0)
            g_dim = int(ckpt.get("g_dim") or 0)
            self.model = SemanticMinimalGChunkMLP(
                base_dim,
                g_dim,
                ckpt["action_dim"],
                chunk_size=self.chunk_size,
                hidden=ckpt["hidden"],
                layers=ckpt["layers"],
                action_std=ckpt["y_std"],
            )
        elif self.mode == "chunk_bc_minimal_g_frame":
            base_dim = int(ckpt.get("base_dim") or 0)
            g_dim = int(ckpt.get("g_dim") or 0)
            self.model = FrameMinimalGChunkMLP(
                base_dim,
                g_dim,
                ckpt["action_dim"],
                chunk_size=self.chunk_size,
                hidden=ckpt["hidden"],
                layers=ckpt["layers"],
                action_mean=ckpt["y_mean"],
                action_std=ckpt["y_std"],
            )
        elif self.mode == "chunk_bc_lang_minimal_g_structured":
            base_dim = int(ckpt.get("base_dim") or 0)
            g_dim = int(ckpt.get("g_dim") or 0)
            config = ckpt.get("structured_field_config") or {}
            self.model = StructuredMinimalGChunkMLP(
                base_dim,
                g_dim,
                ckpt["action_dim"],
                chunk_size=self.chunk_size,
                hidden=ckpt["hidden"],
                layers=ckpt["layers"],
                action_std=ckpt["y_std"],
                readiness_distance=float(config.get("readiness_distance", 0.10)),
                readiness_temperature=float(config.get("readiness_temperature", 0.025)),
            )
        elif self.mode in MINIMAL_G_PROJECTED_MODES:
            base_dim = int(ckpt.get("base_dim") or 0)
            g_dim = int(ckpt.get("g_dim") or 0)
            config = ckpt.get("structured_field_config") or {}
            self.model = ProjectedMinimalGChunkMLP(
                base_dim,
                g_dim,
                ckpt["action_dim"],
                chunk_size=self.chunk_size,
                hidden=ckpt["hidden"],
                layers=ckpt["layers"],
                action_mean=ckpt["y_mean"],
                action_std=ckpt["y_std"],
                readiness_distance=float(config.get("readiness_distance", 0.10)),
                readiness_temperature=float(config.get("readiness_temperature", 0.025)),
                field_gain_max=float(config.get("field_gain_max", 1.0)),
                orthogonal_ratio_max=(
                    None
                    if config.get("orthogonal_ratio_max") is None
                    else float(config["orthogonal_ratio_max"])
                ),
                invariant_gain=bool(config.get("invariant_gain", False)),
            )
        elif self.mode == "chunk_bc_v7_gated":
            base_dim = int(ckpt.get("base_dim") or 0)
            g_dim = int(ckpt.get("g_dim") or 0)
            if base_dim <= 0 or g_dim <= 0:
                raise RuntimeError(f"Gated checkpoint is missing base_dim/g_dim: {ckpt_path}")
            self.model = GatedChunkMLP(
                base_dim,
                g_dim,
                ckpt["action_dim"],
                ckpt["hidden"],
                ckpt["layers"],
                gate_max=float(ckpt.get("gate_max", 1.0)),
            )
        elif self.mode == "chunk_fm_v7":
            self.model = FlowMLP(ckpt["cond_dim"], ckpt["action_dim"], ckpt["hidden"], ckpt["layers"])
        else:
            raise ValueError(self.mode)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        self.generator: torch.Generator | None = None
        self.last_debug: dict[str, Any] = {}
        self.pending_actions: list[np.ndarray] = []
        self.action_predictions: list[tuple[int, np.ndarray]] = []
        self.step_index = 0

    def _encode_visual_observation(
        self, obs: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.visual_encoder is None:
            raise RuntimeError("Visual observation requested without a visual encoder")
        rgb_obs = obs.get("rgb_obs")
        if not isinstance(rgb_obs, dict):
            raise KeyError("CALVIN visual policy requires obs['rgb_obs']")
        images = []
        for camera in ("rgb_static", "rgb_gripper"):
            image = np.asarray(rgb_obs[camera], dtype=np.uint8)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Unexpected {camera} image shape: {image.shape}")
            tensor = torch.from_numpy(image).to(self.device, dtype=torch.float32)
            tensor = tensor.permute(2, 0, 1)[None, :] / 255.0
            tensor = torch.nn.functional.interpolate(
                tensor, size=(224, 224), mode="bilinear", align_corners=False
            )
            images.append(tensor)
        pixels = torch.cat(images, dim=0)
        mean = torch.tensor(DINO_IMAGE_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(DINO_IMAGE_STD, device=self.device).view(1, 3, 1, 1)
        pixels = (pixels - mean) / std
        amp_enabled = self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if amp_enabled else torch.float32,
            enabled=amp_enabled,
        ):
            features = self.visual_encoder(pixel_values=pixels).last_hidden_state[:, 0]
        features_np = features.float().cpu().numpy().astype(np.float32)
        return features_np[0], features_np[1]

    def reset(self) -> None:
        if self.generator is None or self.reseed_each_episode:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(self.sample_seed)
        self.last_debug = {}
        self.pending_actions = []
        self.action_predictions = []
        self.step_index = 0

    def _task_from_goal(self, goal: str) -> str:
        if goal in self.text_to_task:
            return self.text_to_task[goal]
        lower = goal.lower()
        for text, task in self.text_to_task.items():
            if text.lower() == lower:
                return task
        return next(iter(self.task_to_id))

    def step(self, obs: dict[str, Any], goal: str) -> np.ndarray:
        if self.pending_actions:
            self.last_debug = {}
            return self.pending_actions.pop(0)

        task = self._task_from_goal(goal)
        language_task = task
        if self.language_control.startswith("task:"):
            language_task = self.language_control.split(":", 1)[1]
            if language_task not in self.task_to_id:
                raise ValueError(f"Unknown language-control task: {language_task}")
        elif self.language_control not in {"normal", "zero"}:
            raise ValueError(f"Unsupported language control: {self.language_control}")
        robot = np.asarray(obs["robot_obs"], dtype=np.float32)
        scene = np.asarray(obs["scene_obs"], dtype=np.float32)
        arrays = {
            "robot_obs": robot[None, :],
            "scene_obs": scene[None, :],
            "task_name": np.asarray([task], dtype=object),
            "task_oh": base.one_hot(language_task, self.task_to_id)[None, :],
        }
        if self.executor_state_scope == "visual":
            visual_static, visual_gripper = self._encode_visual_observation(obs)
            arrays["visual_static"] = visual_static[None, :]
            arrays["visual_gripper"] = visual_gripper[None, :]
        if self.executor_language_mode == "minilm":
            if language_task not in self.task_language_embeddings:
                raise KeyError(
                    f"Missing CALVIN MiniLM embedding for evaluation task: {language_task}"
                )
            arrays["lang_emb"] = self.task_language_embeddings[language_task][None, :]
        if self.zero_task_onehot or self.language_control == "zero":
            arrays["task_oh"] = np.zeros_like(arrays["task_oh"], dtype=np.float32)
            if "lang_emb" in arrays:
                arrays["lang_emb"] = np.zeros_like(arrays["lang_emb"], dtype=np.float32)
        arrays["articulation_g_v7_unified_field"] = base.make_articulation_unified_field_g_v7(
            arrays["robot_obs"],
            arrays["scene_obs"],
            arrays["task_name"],
            self.articulation_geometry,
        )
        query_task = task
        if self.query_g_control.startswith("task:"):
            query_task = self.query_g_control.split(":", 1)[1]
        arrays["articulation_query_g"] = make_articulation_query_g(
            arrays["robot_obs"],
            arrays["scene_obs"],
            np.asarray([query_task], dtype=object),
            self.articulation_geometry,
            minimal_g_scope=self.minimal_g_scope,
            primitive_geometry=self.primitive_geometry,
        )
        arrays["articulation_query_g"] = apply_query_g_control(arrays["articulation_query_g"], self.query_g_control)
        if self.online_minimal_g_mode == "behavior_block_center":
            if self.query_g_control.startswith(("task:", "mask:")):
                raise ValueError(
                    f"Control {self.query_g_control!r} is unsupported for behavior block-center G"
                )
            arrays["articulation_minimal_g"] = apply_minimal_g_control(
                make_behavior_block_center_minimal_g(
                    arrays["robot_obs"], arrays["scene_obs"], arrays["task_name"]
                ),
                self.query_g_control,
            )
        else:
            arrays["articulation_minimal_g"] = make_minimal_articulation_g(
                arrays["articulation_query_g"]
            )
        arrays["articulation_minimal_g"] = (
            arrays["articulation_minimal_g"] * np.float32(self.g_scale)
        ).astype(np.float32)
        if self.query_g_token_type_mode == "zero":
            arrays["articulation_query_g"] = zero_query_g_token_type(arrays["articulation_query_g"])
        if not self.query_g_include_token_type:
            arrays["articulation_query_g"] = drop_query_g_token_type(arrays["articulation_query_g"])
        arrays["articulation_g_v7_unified_field"] = (
            arrays["articulation_g_v7_unified_field"] * np.float32(self.g_scale)
        ).astype(np.float32)
        cond, _ = make_condition(
            arrays,
            self.mode,
            executor_state_scope=self.executor_state_scope,
            mask_executor_language_with_g=self.mask_executor_language_with_g,
            executor_language_mode=self.executor_language_mode,
        )
        cond_n = (cond.astype(np.float32) - self.cond_mean) / self.cond_std
        self.last_debug = {}
        if self.query_g_control.startswith("mask:"):
            if self.mode not in {"chunk_bc_query_g", "chunk_bc_lang_query_g"}:
                raise ValueError(f"Query-G feature masking is unsupported for mode: {self.mode}")
            group = self.query_g_control.split(":", 1)[1]
            token_dim, feature_groups = query_g_feature_layout(self.query_g_include_token_type)
            if group not in feature_groups:
                raise ValueError(
                    f"Query-G feature group {group!r} is unavailable for "
                    f"include_token_type={self.query_g_include_token_type}"
                )
            local_dims = feature_groups[group]
            g_start = cond_n.shape[1] - QUERY_G_TOKEN_COUNT * token_dim
            global_dims = [
                g_start + token_idx * token_dim + local_dim
                for token_idx in range(QUERY_G_TOKEN_COUNT)
                for local_dim in local_dims
            ]
            cond_n[:, global_dims] = 0.0
        cond_t = torch.from_numpy(cond_n).to(self.device)
        with torch.no_grad():
            if self.mode in {
                "chunk_bc_state",
                "chunk_bc_lang",
                "chunk_bc_query_g",
                "chunk_bc_lang_query_g",
                "chunk_bc_lang_minimal_g_concat",
                "chunk_bc_minimal_g_concat",
                "chunk_bc_v7",
                "chunk_bc_v7_scale_aug",
                "chunk_bc_v7_gated",
            }:
                pred_n = self.model(cond_t)
            elif self.mode == "chunk_bc_minimal_g_film":
                base_dim = int(getattr(self.model, "base_dim"))
                g_dim = int(getattr(self.model, "g_dim"))
                pred_n = self.model(
                    cond_t[:, :base_dim],
                    cond_t[:, base_dim : base_dim + g_dim],
                )
            elif self.mode == "chunk_bc_minimal_g_tokens":
                base_dim = int(getattr(self.model, "base_dim"))
                g_raw_t = torch.from_numpy(arrays["articulation_minimal_g"]).to(self.device)
                pred_n = self.model(cond_t[:, :base_dim], g_raw_t)
            elif self.mode == "chunk_fm_minimal_g_tokens":
                base_dim = int(getattr(self.model, "base_dim"))
                g_raw_t = torch.from_numpy(arrays["articulation_minimal_g"]).to(self.device)
                if self.generator is None:
                    self.reset()
                pred_n = sample_minimal_g_token_flow(
                    self.model,
                    cond_t[:, :base_dim],
                    g_raw_t,
                    steps=self.fm_steps,
                    generator=self.generator,
                )
            elif self.mode == "chunk_bc_minimal_g_semantic":
                base_dim = int(getattr(self.model, "base_dim"))
                g_raw_t = torch.from_numpy(arrays["articulation_minimal_g"]).to(self.device)
                pred_n, aux = self.model(
                    cond_t[:, :base_dim], g_raw_t, return_aux=True
                )
                basis_action = aux["basis_raw"][0, 0].cpu().numpy()
                residual_action = (
                    aux["residual_n"].cpu().numpy() * self.y_std + self.y_mean
                ).reshape(1, self.chunk_size, 7)[0, 0]
                self.last_debug = {
                    "attraction": arrays["articulation_minimal_g"][0, :3].copy(),
                    "object_flow": arrays["articulation_minimal_g"][0, 3:6].copy(),
                    "attraction_gain": float(aux["attraction_gain"][0, 0].cpu()),
                    "flow_gain": float(aux["flow_gain"][0, 0].cpu()),
                    "prior_action": basis_action.copy(),
                    "learned_action": residual_action.copy(),
                }
            elif self.mode == "chunk_bc_minimal_g_frame":
                base_dim = int(getattr(self.model, "base_dim"))
                g_raw_t = torch.from_numpy(arrays["articulation_minimal_g"]).to(self.device)
                pred_n, aux = self.model(cond_t[:, :base_dim], g_raw_t, return_aux=True)
                local_action = aux["local_action"][0, 0].cpu().numpy()
                global_action = np.zeros(7, dtype=np.float32)
                global_action[:3] = aux["global_xyz"][0, 0].cpu().numpy()
                global_action[3:] = (
                    pred_n.cpu().numpy() * self.y_std + self.y_mean
                ).reshape(1, self.chunk_size, 7)[0, 0, 3:]
                self.last_debug = {
                    "attraction": arrays["articulation_minimal_g"][0, :3].copy(),
                    "object_flow": arrays["articulation_minimal_g"][0, 3:6].copy(),
                    "frame": aux["frame"][0].cpu().numpy().copy(),
                    "local_action": local_action.copy(),
                    "prior_action": global_action.copy(),
                    "learned_action": np.zeros(7, dtype=np.float32),
                }
            elif self.mode == "chunk_bc_lang_minimal_g_structured":
                base_dim = int(getattr(self.model, "base_dim"))
                g_dim = int(getattr(self.model, "g_dim"))
                g_raw_t = torch.from_numpy(arrays["articulation_minimal_g"]).to(self.device)
                pred_n, aux = self.model(
                    cond_t[:, :base_dim],
                    cond_t[:, base_dim : base_dim + g_dim],
                    g_raw_t,
                    return_aux=True,
                )
                prior_n = aux["prior_n"]
                learned_n = pred_n - prior_n
                prior_action = (prior_n.cpu().numpy() * self.y_std).reshape(1, self.chunk_size, 7)[0, 0]
                learned_action = (
                    learned_n.cpu().numpy() * self.y_std + self.y_mean
                ).reshape(1, self.chunk_size, 7)[0, 0]
                self.last_debug = {
                    "attraction": arrays["articulation_minimal_g"][0, :3].copy(),
                    "object_flow": arrays["articulation_minimal_g"][0, 3:6].copy(),
                    "readiness": float(aux["readiness"][0, 0].cpu()),
                    "approach_gain": float(aux["approach_gain"].cpu()),
                    "flow_gain": float(aux["flow_gain"].cpu()),
                    "prior_action": prior_action.copy(),
                    "learned_action": learned_action.copy(),
                }
            elif self.mode in MINIMAL_G_PROJECTED_MODES:
                base_dim = int(getattr(self.model, "base_dim"))
                g_dim = int(getattr(self.model, "g_dim"))
                g_raw_t = torch.from_numpy(arrays["articulation_minimal_g"]).to(self.device)
                pred_n, aux = self.model(
                    cond_t[:, :base_dim],
                    cond_t[:, base_dim : base_dim + g_dim],
                    g_raw_t,
                    return_aux=True,
                )
                field_action = np.zeros(7, dtype=np.float32)
                field_action[:3] = aux["field_component_raw"][0, 0].cpu().numpy()
                orthogonal_action = aux["unconstrained_raw"][0, 0].cpu().numpy().copy()
                orthogonal_action[:3] = aux["orthogonal_component_raw"][0, 0].cpu().numpy()
                self.last_debug = {
                    "attraction": arrays["articulation_minimal_g"][0, :3].copy(),
                    "object_flow": arrays["articulation_minimal_g"][0, 3:6].copy(),
                    "readiness": float(aux["readiness"][0, 0].cpu()),
                    "field_gain": float(aux["field_gain"][0, 0].cpu()),
                    "orthogonal_scale": float(aux["orthogonal_scale"][0, 0, 0].cpu()),
                    "prior_action": field_action,
                    "learned_action": orthogonal_action,
                }
            else:
                pred_n = sample_flow(self.model, cond_t, self.fm_steps, self.generator)
        chunk = pred_n.cpu().numpy() * self.y_std + self.y_mean
        actions = chunk.reshape(self.chunk_size, 7)
        if self.last_debug:
            self.last_debug["unclipped_action"] = actions[0].copy()
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
        actions[:, -1] = np.where(actions[:, -1] >= 0, 1.0, -1.0)
        if self.temporal_ensemble:
            self.action_predictions.append((self.step_index, actions.copy()))
            candidates: list[np.ndarray] = []
            ages: list[int] = []
            retained: list[tuple[int, np.ndarray]] = []
            for start_step, prediction in self.action_predictions:
                offset = self.step_index - start_step
                if 0 <= offset < len(prediction):
                    candidates.append(prediction[offset])
                    ages.append(offset)
                    retained.append((start_step, prediction))
            weights = np.exp(-self.temporal_ensemble_decay * np.asarray(ages, dtype=np.float32))
            weights /= np.sum(weights)
            action = np.sum(np.stack(candidates) * weights[:, None], axis=0).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
            action[-1] = 1.0 if action[-1] >= 0 else -1.0
            self.action_predictions = retained
            self.step_index += 1
        else:
            self.pending_actions = [
                action.copy() for action in actions[: self.action_execution_horizon]
            ]
            action = self.pending_actions.pop(0)
        if self.last_debug:
            self.last_debug["final_action"] = action.copy()
        return action


def evaluate_policy_ckpt(
    calvin_code: Path,
    data_root: Path,
    eval_dir: Path,
    ckpt_path: Path,
    task_to_id: dict[str, int],
    text_to_task: dict[str, str],
    num_sequences: int,
    ep_len: int,
    device: str,
    eval_mode: str,
    eval_task_filter: set[str] | None,
    eval_scan_sequences: int,
    eval_sequence_workers: int = 8,
    eval_sequence_offset: int = 0,
    g_scale: float = 1.0,
    query_g_control: str = "normal",
    online_minimal_g_mode: str = "generated_query",
    language_control: str = "normal",
    action_execution_horizon: int = 1,
    temporal_ensemble: bool = False,
    temporal_ensemble_decay: float = 0.25,
    state_only_observation: bool = False,
    policy_sample_seed: int | None = None,
    fm_reseed_each_episode: bool = True,
    profile_eval_timing: bool = False,
) -> dict[str, Any]:
    calvin_models = calvin_code / "calvin_models"
    calvin_env = calvin_code / "calvin_env"
    calvin_agent = calvin_models / "calvin_agent"
    sys.path.insert(0, calvin_models.as_posix())
    sys.path.insert(0, calvin_env.as_posix())
    os.chdir(calvin_agent)

    from calvin_agent.evaluation import evaluate_policy as eval_mod
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_agent.evaluation.utils import count_success
    import hydra
    from omegaconf import OmegaConf

    eval_mod.NUM_SEQUENCES = num_sequences
    eval_mod.EP_LEN = ep_len
    checkpoint_meta = torch.load(ckpt_path, map_location="cpu")
    executor_language_mode = str(checkpoint_meta.get("executor_language_mode", "onehot"))
    task_language_embeddings = (
        load_task_language_embeddings(data_root)
        if executor_language_mode == "minilm"
        else None
    )
    model = ChunkFlowPolicy(
        ckpt_path,
        task_to_id,
        text_to_task,
        device=device,
        task_language_embeddings=task_language_embeddings,
        g_scale=g_scale,
        query_g_control=query_g_control,
        online_minimal_g_mode=online_minimal_g_mode,
        language_control=language_control,
        action_execution_horizon=action_execution_horizon,
        temporal_ensemble=temporal_ensemble,
        temporal_ensemble_decay=temporal_ensemble_decay,
        sample_seed_override=policy_sample_seed,
        reseed_each_episode=fm_reseed_each_episode,
    )
    if model.executor_state_scope == "visual":
        from calvin_env.envs.play_table_env import get_env

        env = get_env(
            data_root / "validation",
            show_gui=False,
            obs_space={
                "rgb_obs": ["rgb_static", "rgb_gripper"],
                "depth_obs": [],
            },
        )
    else:
        env = eval_mod.make_env(data_root)
    if state_only_observation:
        env.get_obs = env.get_state_obs
    eval_timing = {
        "policy_seconds": 0.0,
        "policy_calls": 0,
        "env_seconds": 0.0,
        "env_calls": 0,
    }
    if profile_eval_timing:
        model_step = model.step
        env_step = env.step

        def timed_model_step(*args: Any, **kwargs: Any) -> np.ndarray:
            start = time.perf_counter()
            result = model_step(*args, **kwargs)
            eval_timing["policy_seconds"] += time.perf_counter() - start
            eval_timing["policy_calls"] += 1
            return result

        def timed_env_step(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = env_step(*args, **kwargs)
            eval_timing["env_seconds"] += time.perf_counter() - start
            eval_timing["env_calls"] += 1
            return result

        model.step = timed_model_step  # type: ignore[method-assign]
        env.step = timed_env_step  # type: ignore[method-assign]
    eval_dir.mkdir(parents=True, exist_ok=True)
    if eval_mode == "long_horizon":
        results = eval_mod.evaluate_policy(model, env, epoch=ckpt_path.parent.name, eval_log_dir=eval_dir, debug=False)
        summary = {
            "eval_mode": eval_mode,
            "action_execution_horizon": int(action_execution_horizon),
            "temporal_ensemble": bool(temporal_ensemble),
            "temporal_ensemble_decay": float(temporal_ensemble_decay),
            "state_only_observation": bool(state_only_observation),
            "policy_sample_seed": policy_sample_seed,
            "fm_reseed_each_episode": bool(fm_reseed_each_episode),
            "avg_seq_len": float(np.mean(results)) if results else 0.0,
            "chain_sr": {str(i + 1): float(sr) for i, sr in enumerate(count_success(results))},
            "raw_results": results,
        }
    else:
        conf_dir = calvin_models / "conf"
        task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        task_oracle = hydra.utils.instantiate(task_cfg)
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
        selected: list[tuple[Any, str, int]] = []
        if eval_mode == "first_task":
            matched = 0
            for sequence_index, (initial_state, seq) in enumerate(
                get_sequences(
                    max(num_sequences + eval_sequence_offset, eval_scan_sequences),
                    num_workers=eval_sequence_workers,
                )
            ):
                first_task = seq[0]
                if eval_task_filter is None or first_task in eval_task_filter:
                    if matched >= eval_sequence_offset:
                        selected.append((initial_state, first_task, sequence_index))
                    matched += 1
                if len(selected) >= num_sequences:
                    break
        else:
            if eval_task_filter is None:
                raise RuntimeError("balanced_first_task requires --eval-task-filter")
            buckets: dict[str, list[tuple[Any, str, int]]] = {task: [] for task in sorted(eval_task_filter)}
            matched_by_task: Counter[str] = Counter()
            required_per_task = num_sequences + eval_sequence_offset
            for sequence_index, (initial_state, seq) in enumerate(
                get_sequences(
                    max(required_per_task * max(1, len(buckets)) * 10, eval_scan_sequences),
                    num_workers=eval_sequence_workers,
                )
            ):
                first_task = seq[0]
                if first_task in buckets:
                    if matched_by_task[first_task] >= eval_sequence_offset and len(buckets[first_task]) < num_sequences:
                        buckets[first_task].append((initial_state, first_task, sequence_index))
                    matched_by_task[first_task] += 1
                if all(len(items) >= num_sequences for items in buckets.values()):
                    break
            selected = [item for task in sorted(buckets) for item in buckets[task]]
        if not selected:
            raise RuntimeError(f"No first-task eval sequences matched filter: {sorted(eval_task_filter or [])}")
        selected_tasks = [task for _, task, _ in selected]
        selected_sequence_indices = [int(index) for _, _, index in selected]
        progress_path = eval_dir / "eval_progress.json"
        progress_signature = {
            "checkpoint": ckpt_path.resolve().as_posix(),
            "num_sequences": int(num_sequences),
            "ep_len": int(ep_len),
            "eval_mode": eval_mode,
            "eval_scan_sequences": int(eval_scan_sequences),
            "eval_sequence_offset": int(eval_sequence_offset),
            "g_scale": float(g_scale),
            "query_g_control": str(query_g_control),
            "online_minimal_g_mode": str(online_minimal_g_mode),
            "language_control": str(language_control),
            "action_execution_horizon": int(action_execution_horizon),
            "temporal_ensemble": bool(temporal_ensemble),
            "temporal_ensemble_decay": float(temporal_ensemble_decay),
            "state_only_observation": bool(state_only_observation),
            "policy_sample_seed": policy_sample_seed,
            "fm_reseed_each_episode": bool(fm_reseed_each_episode),
            "selected_tasks": selected_tasks,
            "selected_sequence_indices": selected_sequence_indices,
        }
        results: list[int] = []
        if progress_path.exists():
            with open(progress_path, encoding="utf-8") as f:
                progress = json.load(f)
            for key, expected in progress_signature.items():
                if progress.get(key) != expected:
                    raise RuntimeError(
                        f"Evaluation progress protocol mismatch for {key}: "
                        f"{progress.get(key)!r} != {expected!r}"
                    )
            results = [int(value) for value in progress.get("raw_results", [])]
            if len(results) > len(selected):
                raise RuntimeError(
                    f"Evaluation progress has {len(results)} results for only {len(selected)} selected sequences"
                )
            print(
                f"[resume:{ckpt_path.parent.name}] {len(results)}/{len(selected)} completed rollouts",
                flush=True,
            )

        def save_progress() -> None:
            payload = {**progress_signature, "raw_results": results}
            tmp_path = progress_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=json_default)
            os.replace(tmp_path, progress_path)

        task_counts: Counter[str] = Counter()
        task_success: Counter[str] = Counter()
        for subtask, success in zip(selected_tasks, results):
            task_counts[subtask] += 1
            task_success[subtask] += int(success)
        plans: dict[str, list[Any]] = defaultdict(list)
        for initial_state, subtask, _sequence_index in selected[len(results) :]:
            timing_before = eval_timing.copy()
            robot_obs, scene_obs = eval_mod.get_env_state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            model.reset()
            success = bool(eval_mod.rollout(env, model, task_oracle, subtask, val_annotations, plans, debug=False))
            task_counts[subtask] += 1
            task_success[subtask] += int(success)
            results.append(int(success))
            save_progress()
            done = len(results)
            if profile_eval_timing:
                policy_seconds = eval_timing["policy_seconds"] - timing_before["policy_seconds"]
                policy_calls = eval_timing["policy_calls"] - timing_before["policy_calls"]
                env_seconds = eval_timing["env_seconds"] - timing_before["env_seconds"]
                env_calls = eval_timing["env_calls"] - timing_before["env_calls"]
                print(
                    f"[timing:{ckpt_path.parent.name}] rollout={done} "
                    f"policy={policy_seconds:.3f}s/{policy_calls} "
                    f"({policy_seconds / max(1, policy_calls):.3f}s/step) "
                    f"env={env_seconds:.3f}s/{env_calls} "
                    f"({env_seconds / max(1, env_calls):.3f}s/step)",
                    flush=True,
                )
            interval = max(1, len(selected) // 10)
            if done == 1 or done == len(selected) or done % interval == 0:
                print(
                    f"[eval:{ckpt_path.parent.name}] {done}/{len(selected)} "
                    f"last={subtask}:{int(success)} sr={float(np.mean(results)):.3f}",
                    flush=True,
                )
        task_info = {
            task: {"success": int(task_success[task]), "total": int(total), "sr": float(task_success[task] / total)}
            for task, total in sorted(task_counts.items())
        }
        summary = {
            "eval_mode": eval_mode,
            "g_scale": float(g_scale),
            "query_g_control": str(query_g_control),
            "online_minimal_g_mode": str(online_minimal_g_mode),
            "language_control": str(language_control),
            "action_execution_horizon": int(action_execution_horizon),
            "temporal_ensemble": bool(temporal_ensemble),
            "temporal_ensemble_decay": float(temporal_ensemble_decay),
            "state_only_observation": bool(state_only_observation),
            "policy_sample_seed": policy_sample_seed,
            "fm_reseed_each_episode": bool(fm_reseed_each_episode),
            "avg_seq_len": float(np.mean(results)) if results else 0.0,
            "chain_sr": {"1": float(np.mean(results)) if results else 0.0},
            "raw_results": results,
            "task_info": task_info,
            "eval_sequence_offset": int(eval_sequence_offset),
            "selected_tasks": selected_tasks,
            "selected_sequence_indices": selected_sequence_indices,
        }
    with open(eval_dir / "state_bc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=sorted(SUPPORTED_MODES), required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize policy weights and normalization statistics from a compatible checkpoint.",
    )
    parser.add_argument(
        "--source-distill-weight",
        type=float,
        default=0.0,
        help="Preserve an initialization policy on non-reversed source windows during fine-tuning.",
    )
    parser.add_argument(
        "--residual-adapter",
        action="store_true",
        help="Freeze the initialization policy and train only a compact G-conditioned residual adapter.",
    )
    parser.add_argument("--residual-adapter-hidden", type=int, default=256)
    parser.add_argument(
        "--reverse-source-loss-weight",
        type=float,
        default=1.0,
        help="Relative inverse-augmentation loss weight for residual-adapter training.",
    )
    parser.add_argument(
        "--reverse-task-loss-weights",
        nargs="*",
        default=None,
        help="Optional TASK=WEIGHT balancing within reverse-augmentation records.",
    )
    parser.add_argument(
        "--residual-adapter-expert-tasks",
        nargs="*",
        default=None,
        help="Reverse target tasks assigned to learned residual experts; inference uses the learned router.",
    )
    parser.add_argument("--residual-router-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--residual-adapter-source-expert",
        action="store_true",
        help="Add a routed no-op residual expert supervised by all non-reversed source records.",
    )
    parser.add_argument("--include-tasks", nargs="+", default=None)
    parser.add_argument(
        "--task-segment-limits",
        nargs="*",
        default=None,
        help="Optional per-task training demo limits, formatted as TASK=COUNT, e.g. close_drawer=5.",
    )
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument(
        "--reverse-open-augmentation",
        action="store_true",
        help="Add close-drawer chunks made by time-reversing used open-drawer demonstrations.",
    )
    parser.add_argument(
        "--reverse-open-max-segments",
        type=int,
        default=0,
        help="Maximum used open-drawer segments to reverse; 0 uses every available segment.",
    )
    parser.add_argument(
        "--reverse-task-pairs",
        nargs="*",
        default=None,
        help="Physically validated SOURCE=TARGET pairs to augment by reversing source trajectories.",
    )
    parser.add_argument(
        "--reverse-pair-max-segments",
        type=int,
        default=0,
        help="Maximum source segments to reverse per --reverse-task-pairs entry; 0 uses all.",
    )
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Model/optimizer initialization seed. Defaults to --seed for backward compatibility.",
    )
    parser.add_argument(
        "--loader-seed",
        type=int,
        default=None,
        help="DataLoader shuffle seed. Defaults to --seed for backward compatibility.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Seed used to order/collect demonstrations. Defaults to --seed for backward compatibility.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed used for the segment train/validation split. Defaults to --seed for backward compatibility.",
    )
    parser.add_argument("--num-sequences", type=int, default=25)
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-mode", choices=["long_horizon", "first_task", "balanced_first_task"], default="balanced_first_task")
    parser.add_argument("--eval-task-filter", nargs="+", default=None)
    parser.add_argument("--eval-scan-sequences", type=int, default=5000)
    parser.add_argument(
        "--eval-sequence-workers",
        type=int,
        default=8,
        help="Workers used by CALVIN's deterministic standard-sequence generator.",
    )
    parser.add_argument(
        "--eval-sequence-offset",
        type=int,
        default=0,
        help="Skip this many matching CALVIN sequences per eval task before selecting rollouts.",
    )
    parser.add_argument("--checkpoint-epochs", nargs="*", type=int, default=[])
    parser.add_argument("--eval-checkpoints", action="store_true")
    parser.add_argument(
        "--skip-eval",
        type=int,
        choices=[0, 1],
        default=0,
        help="Train/save requested modes without creating CALVIN environments or rollouts.",
    )
    parser.add_argument("--zero-task-onehot", action="store_true")
    parser.add_argument(
        "--executor-state-scope",
        choices=sorted(EXECUTOR_STATE_SCOPES),
        default="full",
        help="Use all privileged state or robot proprioception only in the policy state token.",
    )
    parser.add_argument(
        "--executor-language-mode",
        choices=sorted(EXECUTOR_LANGUAGE_MODES),
        default="onehot",
        help="Use no direct executor language, legacy one-hot, or official CALVIN MiniLM embeddings.",
    )
    parser.add_argument(
        "--visual-feature-dir",
        type=Path,
        default=None,
        help="Frozen visual feature shards required by --executor-state-scope visual.",
    )
    parser.add_argument(
        "--visual-encoder-name",
        default="facebook/dinov2-small",
        help="Frozen Hugging Face DINOv2 encoder used for cached and online features.",
    )
    parser.add_argument(
        "--mask-executor-language-with-g",
        type=int,
        choices=[0, 1],
        default=1,
        help="Mask direct task language when compact G is nonzero (legacy=1); keep it for fair language+G tests with 0.",
    )
    parser.add_argument("--articulation-geometry-json", type=Path, required=True)
    parser.add_argument(
        "--minimal-g-scope",
        choices=sorted(MINIMAL_G_SCOPES),
        default="articulation",
        help="Tasks receiving nonzero compact [attraction, object-flow] G.",
    )
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--g-scale-aug-values", nargs="+", type=float, default=[1.0])
    parser.add_argument("--g-dropout-prob", type=float, default=0.0)
    parser.add_argument("--gate-max", type=float, default=2.0)
    parser.add_argument(
        "--field-alignment-weight",
        type=float,
        default=0.05,
        help="Auxiliary low-frequency field-alignment weight for structured minimal-G BC.",
    )
    parser.add_argument(
        "--orthogonal-ratio-max",
        type=float,
        default=2.0,
        help="Maximum orthogonal/field action ratio for bounded projected minimal-G modes.",
    )
    parser.add_argument(
        "--first-action-loss-weight",
        type=float,
        default=1.0,
        help="Relative loss weight of the first action in each training chunk.",
    )
    parser.add_argument(
        "--inverse-pair-loss-weight",
        type=float,
        default=0.0,
        help="Weight on matched inverse-goal predicted-action difference loss for token BC/FM.",
    )
    parser.add_argument(
        "--inverse-pairs-per-batch",
        type=int,
        default=128,
        help="Number of mined inverse-goal pairs used by each training update.",
    )
    parser.add_argument("--inverse-pair-max-eef-distance-m", type=float, default=0.05)
    parser.add_argument("--inverse-pair-max-mechanism-fraction", type=float, default=0.10)
    parser.add_argument(
        "--action-execution-horizon",
        type=int,
        default=1,
        help="Number of predicted chunk actions to execute open-loop before replanning.",
    )
    parser.add_argument(
        "--eval-dir-suffix",
        default="",
        help="Suffix for evaluation and comparison outputs, e.g. _execute4.",
    )
    parser.add_argument(
        "--temporal-ensemble",
        type=int,
        choices=[0, 1],
        default=0,
        help="Ensemble overlapping chunk predictions for the current timestep.",
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.25,
        help="Exponential age decay for temporal ensembling; newer predictions receive more weight.",
    )
    parser.add_argument(
        "--drop-query-g-token-type",
        action="store_true",
        help="Remove the redundant four-way role one-hot from each fixed-slot query-G token.",
    )
    parser.add_argument(
        "--zero-query-g-token-type",
        action="store_true",
        help="Keep the 76-D layout but set every query-G token role one-hot to zero during train and eval.",
    )
    args = parser.parse_args()

    args.calvin_code = args.calvin_code.resolve()
    args.data_root = args.data_root.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.visual_feature_dir is not None:
        args.visual_feature_dir = args.visual_feature_dir.resolve()
    if args.init_checkpoint is not None:
        args.init_checkpoint = args.init_checkpoint.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.temporal_ensemble = bool(args.temporal_ensemble)
    if args.action_execution_horizon < 1 or args.action_execution_horizon > args.chunk_size:
        parser.error("--action-execution-horizon must be in [1, --chunk-size]")
    if args.temporal_ensemble and args.action_execution_horizon != 1:
        parser.error("--temporal-ensemble requires --action-execution-horizon 1")
    if args.temporal_ensemble_decay < 0:
        parser.error("--temporal-ensemble-decay must be non-negative")
    if args.inverse_pair_loss_weight < 0:
        parser.error("--inverse-pair-loss-weight must be non-negative")
    if args.source_distill_weight < 0:
        parser.error("--source-distill-weight must be non-negative")
    if args.residual_adapter_hidden <= 0:
        parser.error("--residual-adapter-hidden must be positive")
    if args.reverse_source_loss_weight <= 0:
        parser.error("--reverse-source-loss-weight must be positive")
    if args.residual_router_loss_weight <= 0:
        parser.error("--residual-router-loss-weight must be positive")
    if args.reverse_open_max_segments < 0:
        parser.error("--reverse-open-max-segments must be non-negative")
    if args.reverse_pair_max_segments < 0:
        parser.error("--reverse-pair-max-segments must be non-negative")
    if args.inverse_pairs_per_batch <= 0:
        parser.error("--inverse-pairs-per-batch must be positive")
    if args.inverse_pair_max_eef_distance_m <= 0:
        parser.error("--inverse-pair-max-eef-distance-m must be positive")
    if not 0 < args.inverse_pair_max_mechanism_fraction <= 1:
        parser.error("--inverse-pair-max-mechanism-fraction must be in (0, 1]")
    valid_suffix_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if args.eval_dir_suffix and any(char not in valid_suffix_chars for char in args.eval_dir_suffix):
        parser.error("--eval-dir-suffix may contain only letters, digits, '_', '-', and '.'")
    articulation_geometry = base.load_articulation_geometry(args.articulation_geometry_json)
    if articulation_geometry is None:
        raise RuntimeError("--articulation-geometry-json is required")
    supported_primitive_tasks = {"push_into_drawer", "place_in_slider"}
    required_primitive_tasks = (
        supported_primitive_tasks
        if args.include_tasks is None
        else set(args.include_tasks) & supported_primitive_tasks
    )
    primitive_geometry = (
        fit_primitive_query_geometry(args.data_root, required_primitive_tasks)
        if args.minimal_g_scope == "translational" and required_primitive_tasks
        else {}
    )
    task_segment_limits = parse_task_segment_limits(args.task_segment_limits)
    reverse_task_loss_weights = parse_task_loss_weights(args.reverse_task_loss_weights)
    reverse_task_pairs = parse_reverse_task_pairs(args.reverse_task_pairs)
    init_seed = args.seed if args.init_seed is None else int(args.init_seed)
    loader_seed = args.seed if args.loader_seed is None else int(args.loader_seed)
    data_seed = args.seed if args.data_seed is None else int(args.data_seed)
    split_seed = args.seed if args.split_seed is None else int(args.split_seed)

    checkpoint_epochs = sorted({epoch for epoch in args.checkpoint_epochs if epoch > 0})
    if args.eval_checkpoints and not checkpoint_epochs:
        checkpoint_epochs = [args.epochs]

    text_to_task, _task_to_text = base.load_eval_annotation_maps(args.calvin_code)
    segments = base.load_lang_segments(args.data_root / "training")
    task_to_id = base.build_task_index(segments, text_to_task)
    arrays, used_segments = collect_chunk_records(
        args.data_root,
        task_to_id,
        chunk_size=args.chunk_size,
        max_windows=args.max_windows,
        max_segments=args.max_segments,
        stride=args.stride,
        seed=data_seed,
        include_tasks=set(args.include_tasks) if args.include_tasks else None,
        articulation_geometry=articulation_geometry,
        minimal_g_scope=args.minimal_g_scope,
        primitive_geometry=primitive_geometry,
        task_segment_limits=task_segment_limits,
        reverse_open_augmentation=args.reverse_open_augmentation,
        reverse_open_max_segments=args.reverse_open_max_segments,
        reverse_task_pairs=reverse_task_pairs,
        reverse_pair_max_segments=args.reverse_pair_max_segments,
    )
    visual_feature_meta: dict[str, Any] = {}
    if args.executor_state_scope == "visual":
        if args.visual_feature_dir is None:
            parser.error("--executor-state-scope visual requires --visual-feature-dir")
        visual_feature_meta = attach_visual_features(arrays, args.visual_feature_dir)
        cached_encoder_name = str(visual_feature_meta.get("model_name", ""))
        if cached_encoder_name != args.visual_encoder_name:
            parser.error(
                "--visual-encoder-name does not match the cached feature manifest: "
                f"{args.visual_encoder_name!r} != {cached_encoder_name!r}"
            )
    if args.zero_task_onehot:
        arrays["task_oh"] = np.zeros_like(arrays["task_oh"], dtype=np.float32)
    if args.drop_query_g_token_type and args.zero_query_g_token_type:
        parser.error("--drop-query-g-token-type and --zero-query-g-token-type are mutually exclusive")
    query_g_include_token_type = not args.drop_query_g_token_type
    query_g_token_type_mode = "zero" if args.zero_query_g_token_type else "normal"
    if query_g_token_type_mode == "zero":
        arrays["articulation_query_g"] = zero_query_g_token_type(arrays["articulation_query_g"])
    if not query_g_include_token_type:
        arrays["articulation_query_g"] = drop_query_g_token_type(arrays["articulation_query_g"])

    run_meta = {
        "data_root": args.data_root,
        "modes": args.modes,
        "init_checkpoint": args.init_checkpoint,
        "source_distill_weight": args.source_distill_weight,
        "residual_adapter": args.residual_adapter,
        "residual_adapter_hidden": args.residual_adapter_hidden,
        "reverse_source_loss_weight": args.reverse_source_loss_weight,
        "reverse_task_loss_weights": reverse_task_loss_weights,
        "residual_adapter_expert_tasks": args.residual_adapter_expert_tasks,
        "residual_router_loss_weight": args.residual_router_loss_weight,
        "residual_adapter_source_expert": args.residual_adapter_source_expert,
        "include_tasks": args.include_tasks,
        "task_segment_limits": task_segment_limits,
        "max_windows": args.max_windows,
        "actual_windows": int(len(arrays["action_chunks"])),
        "max_segments": args.max_segments,
        "actual_segments": len(used_segments),
        "actual_segments_by_task": dict(Counter(str(seg["task"]) for seg in used_segments)),
        "actual_windows_by_task": dict(Counter(str(task) for task in arrays["task_name"])),
        "reverse_open_augmentation": args.reverse_open_augmentation,
        "reverse_open_max_segments": args.reverse_open_max_segments,
        "reverse_open_windows": int(np.sum(arrays["is_reverse_open"])),
        "reverse_open_source_segments": int(len(np.unique(arrays["segment_id"][arrays["is_reverse_open"]]))),
        "reverse_task_pairs": [f"{source}={target}" for source, target in reverse_task_pairs],
        "reverse_pair_max_segments": args.reverse_pair_max_segments,
        "reverse_source_windows": int(np.sum(arrays["is_reverse_source"])),
        "reverse_source_segments": int(
            len(np.unique(arrays["segment_id"][arrays["is_reverse_source"]]))
        ),
        "stride": args.stride,
        "chunk_size": args.chunk_size,
        "epochs": args.epochs,
        "train_seed": args.seed,
        "init_seed": init_seed,
        "loader_seed": loader_seed,
        "data_seed": data_seed,
        "split_seed": split_seed,
        "num_sequences": args.num_sequences,
        "ep_len": args.ep_len,
        "eval_mode": args.eval_mode,
        "eval_task_filter": args.eval_task_filter,
        "eval_scan_sequences": args.eval_scan_sequences,
        "eval_sequence_workers": args.eval_sequence_workers,
        "eval_sequence_offset": args.eval_sequence_offset,
        "checkpoint_epochs": checkpoint_epochs,
        "eval_checkpoints": args.eval_checkpoints,
        "skip_eval": bool(args.skip_eval),
        "zero_task_onehot": args.zero_task_onehot,
        "executor_state_scope": args.executor_state_scope,
        "executor_language_mode": args.executor_language_mode,
        "visual_feature_dir": args.visual_feature_dir,
        "visual_feature_meta": visual_feature_meta,
        "visual_encoder_name": (
            args.visual_encoder_name if args.executor_state_scope == "visual" else None
        ),
        "mask_executor_language_with_g": bool(args.mask_executor_language_with_g),
        "articulation_geometry_json": args.articulation_geometry_json,
        "minimal_g_scope": args.minimal_g_scope,
        "primitive_geometry": primitive_geometry,
        "fm_steps": args.fm_steps,
        "g_scale_aug_values": args.g_scale_aug_values,
        "g_dropout_prob": args.g_dropout_prob,
        "gate_max": args.gate_max,
        "field_alignment_weight": args.field_alignment_weight,
        "orthogonal_ratio_max": args.orthogonal_ratio_max,
        "first_action_loss_weight": args.first_action_loss_weight,
        "inverse_pair_loss_weight": args.inverse_pair_loss_weight,
        "inverse_pairs_per_batch": args.inverse_pairs_per_batch,
        "inverse_pair_max_eef_distance_m": args.inverse_pair_max_eef_distance_m,
        "inverse_pair_max_mechanism_fraction": args.inverse_pair_max_mechanism_fraction,
        "action_execution_horizon": args.action_execution_horizon,
        "eval_dir_suffix": args.eval_dir_suffix,
        "temporal_ensemble": args.temporal_ensemble,
        "temporal_ensemble_decay": args.temporal_ensemble_decay,
        "query_g_include_token_type": query_g_include_token_type,
        "query_g_token_dim": QUERY_G_TOKEN_DIM if query_g_include_token_type else QUERY_G_TOKEN_DIM_NO_TYPE,
        "query_g_token_type_mode": query_g_token_type_mode,
        "tasks": task_to_id,
    }
    with open(args.out_dir / f"run_meta{args.eval_dir_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, default=json_default)

    comparison_rows: list[dict[str, Any]] = []
    for mode in args.modes:
        variant_dir = args.out_dir / mode
        if not (variant_dir / "policy.pt").exists():
            train_metrics = train_mode(
                arrays,
                mode,
                variant_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden=args.hidden,
                layers=args.layers,
                lr=args.lr,
                device=args.device,
                seed=init_seed,
                loader_seed=loader_seed,
                split_seed=split_seed,
                checkpoint_epochs=set(checkpoint_epochs),
                articulation_geometry=articulation_geometry,
                minimal_g_scope=args.minimal_g_scope,
                primitive_geometry=primitive_geometry,
                zero_task_onehot=args.zero_task_onehot,
                fm_steps=args.fm_steps,
                g_scale_aug_values=args.g_scale_aug_values,
                g_dropout_prob=args.g_dropout_prob,
                gate_max=args.gate_max,
                query_g_include_token_type=query_g_include_token_type,
                query_g_token_type_mode=query_g_token_type_mode,
                field_alignment_weight=args.field_alignment_weight,
                orthogonal_ratio_max=args.orthogonal_ratio_max,
                first_action_loss_weight=args.first_action_loss_weight,
                inverse_pair_loss_weight=args.inverse_pair_loss_weight,
                inverse_pairs_per_batch=args.inverse_pairs_per_batch,
                inverse_pair_max_eef_distance_m=args.inverse_pair_max_eef_distance_m,
                inverse_pair_max_mechanism_fraction=args.inverse_pair_max_mechanism_fraction,
                init_checkpoint=args.init_checkpoint,
                source_distill_weight=args.source_distill_weight,
                residual_adapter=args.residual_adapter,
                residual_adapter_hidden=args.residual_adapter_hidden,
                reverse_source_loss_weight=args.reverse_source_loss_weight,
                reverse_task_loss_weights=reverse_task_loss_weights,
                residual_adapter_expert_tasks=args.residual_adapter_expert_tasks,
                residual_router_loss_weight=args.residual_router_loss_weight,
                residual_adapter_source_expert=args.residual_adapter_source_expert,
                executor_state_scope=args.executor_state_scope,
                executor_language_mode=args.executor_language_mode,
                visual_encoder_name=(
                    args.visual_encoder_name
                    if args.executor_state_scope == "visual"
                    else None
                ),
                visual_feature_dim=int(visual_feature_meta.get("feature_dim", 0)),
                mask_executor_language_with_g=bool(args.mask_executor_language_with_g),
            )
        else:
            with open(variant_dir / "train_metrics.json", encoding="utf-8") as f:
                train_metrics = json.load(f)

        if args.skip_eval:
            continue

        if args.eval_checkpoints:
            eval_jobs = [
                (
                    epoch,
                    variant_dir / "checkpoints" / f"epoch_{epoch:04d}" / "policy.pt",
                    variant_dir / f"task_eval_epoch_{epoch:04d}{args.eval_dir_suffix}",
                )
                for epoch in checkpoint_epochs
            ]
        else:
            eval_jobs = [
                (
                    None,
                    variant_dir / "policy.pt",
                    variant_dir / f"task_eval{args.eval_dir_suffix}",
                )
            ]

        for checkpoint_epoch, ckpt_path, eval_dir in eval_jobs:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
            eval_summary = evaluate_policy_ckpt(
                args.calvin_code,
                args.data_root,
                eval_dir,
                ckpt_path,
                task_to_id,
                text_to_task,
                num_sequences=args.num_sequences,
                ep_len=args.ep_len,
                device=args.device,
                eval_mode=args.eval_mode,
                eval_task_filter=set(args.eval_task_filter) if args.eval_task_filter else None,
                eval_scan_sequences=args.eval_scan_sequences,
                eval_sequence_workers=args.eval_sequence_workers,
                eval_sequence_offset=args.eval_sequence_offset,
                action_execution_horizon=args.action_execution_horizon,
                temporal_ensemble=args.temporal_ensemble,
                temporal_ensemble_decay=args.temporal_ensemble_decay,
                state_only_observation=False,
            )
            metric = base.train_metric_for_epoch(train_metrics, checkpoint_epoch)
            row = {
                "mode": mode,
                "checkpoint_epoch": checkpoint_epoch,
                "train_val_chunk_mse": metric.get("val_chunk_mse"),
                "train_val_first_gripper_acc": metric.get("val_first_gripper_acc"),
                "action_execution_horizon": args.action_execution_horizon,
                "temporal_ensemble": args.temporal_ensemble,
                "temporal_ensemble_decay": args.temporal_ensemble_decay,
                "avg_seq_len": eval_summary["avg_seq_len"],
                **{f"sr_{k}": v for k, v in eval_summary["chain_sr"].items()},
            }
            for task, info in eval_summary.get("task_info", {}).items():
                row[f"task_{task}_success"] = info["success"]
                row[f"task_{task}_total"] = info["total"]
                row[f"task_{task}_sr"] = info["sr"]
            comparison_rows.append(row)
            print(f"[result] {json.dumps(row, default=json_default)}", flush=True)

    with open(args.out_dir / f"comparison{args.eval_dir_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, indent=2, default=json_default)
    with open(args.out_dir / f"comparison{args.eval_dir_suffix}.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = list(dict.fromkeys(key for row in comparison_rows for key in row.keys()))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)
    print(json.dumps(comparison_rows, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
