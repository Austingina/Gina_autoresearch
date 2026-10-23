#!/usr/bin/env python3
"""Train and evaluate lightweight CALVIN state-BC policies.

This is a task-metric bridge for early no-G vs oracle-G comparisons. It is not
intended to replace the final vision/flow policy stack.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import random
import sys
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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def train_metric_for_epoch(train_metrics: dict[str, Any], epoch: int | None) -> dict[str, Any]:
    if epoch is None:
        return train_metrics.get("final", {})
    for item in train_metrics.get("history", []):
        if int(item.get("epoch", -1)) == int(epoch):
            return item
    return {}


def episode_file(split_dir: Path, index: int) -> Path:
    return split_dir / f"episode_{index:07d}.npz"


def install_numpy_pickle_compat_aliases() -> None:
    """Allow NumPy 1.x to read object arrays pickled by NumPy 2.x."""

    if not hasattr(np, "core"):
        return
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def load_lang_segments(split_dir: Path) -> list[dict[str, Any]]:
    lang_file = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    install_numpy_pickle_compat_aliases()
    lang = np.load(lang_file, allow_pickle=True).item()
    tasks = lang["language"]["task"]
    anns = lang["language"]["ann"]
    embs = lang["language"].get("emb", None)
    indices = lang["info"]["indx"]
    segments = []
    for i, (start, end) in enumerate(indices):
        segments.append(
            {
                "task": tasks[i],
                "ann": anns[i],
                "emb": None if embs is None else np.asarray(embs[i]).reshape(-1).astype(np.float32),
                "start": int(start),
                "end": int(end),
            }
        )
    return segments


def load_eval_annotation_maps(calvin_code: Path) -> tuple[dict[str, str], dict[str, str]]:
    from omegaconf import OmegaConf

    val_yaml = calvin_code / "calvin_models" / "conf" / "annotations" / "new_playtable_validation.yaml"
    val_ann = OmegaConf.to_container(OmegaConf.load(val_yaml), resolve=True)
    text_to_task: dict[str, str] = {}
    task_to_text: dict[str, str] = {}
    for task, texts in val_ann.items():
        if not texts:
            continue
        task_to_text[task] = texts[0]
        for text in texts:
            text_to_task[text] = task
    return text_to_task, task_to_text


def build_task_index(segments: list[dict[str, Any]], text_to_task: dict[str, str]) -> dict[str, int]:
    tasks = sorted({s["task"] for s in segments} | set(text_to_task.values()))
    return {task: i for i, task in enumerate(tasks)}


def one_hot(task: str, task_to_id: dict[str, int]) -> np.ndarray:
    out = np.zeros(len(task_to_id), dtype=np.float32)
    if task in task_to_id:
        out[task_to_id[task]] = 1.0
    return out


def articulation_target_controls(task: str, scene: np.ndarray, fallback_target_scene: np.ndarray | None = None) -> np.ndarray:
    target = np.asarray(scene[:2] if fallback_target_scene is None else fallback_target_scene[:2], dtype=np.float32).copy()
    if task == "move_slider_left":
        target[0] = 0.28
    elif task == "move_slider_right":
        target[0] = 0.0
    elif task == "open_drawer":
        target[1] = 0.22
    elif task == "close_drawer":
        target[1] = 0.0
    return target.astype(np.float32)


ARTICULATION_TASK_TO_JOINT = {
    "move_slider_left": "slider",
    "move_slider_right": "slider",
    "open_drawer": "drawer",
    "close_drawer": "drawer",
}

ARTICULATION_JOINT_META = {
    "slider": {"index": 0, "range": 0.28},
    "drawer": {"index": 1, "range": 0.22},
}

ARTICULATION_FIELD_JOINT_ORDER = ("slider", "drawer")
ARTICULATION_V5_TOKEN_COUNT = 6
ARTICULATION_V5_TOKEN_DIM = 11


def load_articulation_geometry(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _articulation_geometry_terms(
    robot: np.ndarray,
    scene: np.ndarray,
    task_names: np.ndarray,
    geometry: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray, np.float32, np.ndarray]]:
    if geometry is None:
        raise RuntimeError("articulation G-v2 modes require --articulation-geometry-json")

    terms: list[tuple[np.ndarray, np.ndarray, np.float32, np.ndarray]] = []
    for r, s, task_value in zip(robot, scene, task_names):
        task = str(task_value)
        joint_name = ARTICULATION_TASK_TO_JOINT.get(task)
        if joint_name is None:
            terms.append(
                (
                    np.zeros(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.float32(0.0),
                    np.zeros(3, dtype=np.float32),
                )
            )
            continue

        joint_model = geometry["joints"][joint_name]
        joint_idx = int(joint_model["joint_index"])
        joint_range = float(joint_model["joint_range"])
        q = float(s[joint_idx])
        target = articulation_target_controls(task, s)[joint_idx]
        residual = float(target - q)
        residual_norm = residual / max(joint_range, 1e-6)

        intercept = np.asarray(joint_model["intercept"], dtype=np.float32)
        slope = np.asarray(joint_model["slope"], dtype=np.float32)
        axis = np.asarray(joint_model["axis_unit"], dtype=np.float32)
        part_point = intercept + slope * q
        target_affordance = part_point - r[:3]

        done = abs(residual_norm) < 0.08
        contact_threshold = float(joint_model.get("contact_dist_threshold", 0.06))
        far = float(np.linalg.norm(target_affordance)) > contact_threshold
        if done:
            stage = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        elif far:
            stage = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            stage = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        terms.append((target_affordance.astype(np.float32), axis.astype(np.float32), np.float32(residual_norm), stage))
    return terms


def _safe_unit(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        return (vec / norm).astype(np.float32)
    if fallback is None:
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray(fallback, dtype=np.float32)


def _axis_perp_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = _safe_unit(axis, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(axis, ref))) > 0.9:
        ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    b1 = _safe_unit(np.cross(axis, ref), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    b2 = _safe_unit(np.cross(axis, b1), np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    return b1, b2


def _articulation_token(
    robot: np.ndarray,
    anchor: np.ndarray,
    attractive_flow: np.ndarray,
    repulsive_flow: np.ndarray,
    attract_weight: float,
    repel_weight: float,
) -> np.ndarray:
    return np.concatenate(
        [
            (anchor - robot[:3]).astype(np.float32),
            np.asarray(attractive_flow, dtype=np.float32),
            np.asarray(repulsive_flow, dtype=np.float32),
            np.asarray([attract_weight, repel_weight], dtype=np.float32),
        ]
    ).astype(np.float32)


def make_articulation_compact_g(
    robot: np.ndarray,
    scene: np.ndarray,
    task_names: np.ndarray,
    geometry: dict[str, Any],
) -> np.ndarray:
    out: list[np.ndarray] = []
    for target_affordance, axis, residual_norm, stage in _articulation_geometry_terms(robot, scene, task_names, geometry):
        residual_motion = axis * residual_norm
        out.append(np.concatenate([target_affordance, residual_motion, stage]).astype(np.float32))
    return np.stack(out).astype(np.float32)


def make_articulation_split_g(
    robot: np.ndarray,
    scene: np.ndarray,
    task_names: np.ndarray,
    geometry: dict[str, Any],
) -> np.ndarray:
    out: list[np.ndarray] = []
    for target_affordance, axis, residual_norm, stage in _articulation_geometry_terms(robot, scene, task_names, geometry):
        residual_scalar = np.asarray([residual_norm], dtype=np.float32)
        out.append(np.concatenate([target_affordance, axis, residual_scalar, stage]).astype(np.float32))
    return np.stack(out).astype(np.float32)


def make_articulation_mechanism_g_v4(
    robot: np.ndarray,
    scene: np.ndarray,
    task_names: np.ndarray,
    geometry: dict[str, Any],
) -> np.ndarray:
    out: list[np.ndarray] = []
    for _, axis, residual_norm, _ in _articulation_geometry_terms(robot, scene, task_names, geometry):
        signed_linear_twist = axis * residual_norm
        signed_angular_twist = np.zeros(3, dtype=np.float32)
        residual_magnitude = np.asarray([abs(float(residual_norm))], dtype=np.float32)
        out.append(np.concatenate([signed_linear_twist, signed_angular_twist, residual_magnitude]).astype(np.float32))
    return np.stack(out).astype(np.float32)


def make_articulation_potential_g_v5(
    robot: np.ndarray,
    scene: np.ndarray,
    task_names: np.ndarray,
    geometry: dict[str, Any],
) -> np.ndarray:
    out: list[np.ndarray] = []
    zero_token = np.zeros(ARTICULATION_V5_TOKEN_DIM, dtype=np.float32)

    for r, s, task_value in zip(robot, scene, task_names):
        task = str(task_value)
        joint_name = ARTICULATION_TASK_TO_JOINT.get(task)
        if joint_name is None:
            out.append(np.zeros(ARTICULATION_V5_TOKEN_COUNT * ARTICULATION_V5_TOKEN_DIM, dtype=np.float32))
            continue

        joint_model = geometry["joints"][joint_name]
        joint_idx = int(joint_model["joint_index"])
        joint_range = float(joint_model["joint_range"])
        q = float(s[joint_idx])
        target = articulation_target_controls(task, s)[joint_idx]
        residual = float(target - q)
        residual_norm = residual / max(joint_range, 1e-6)
        residual_mag = abs(residual_norm)
        desired_sign = 0.0 if residual_mag < 1e-6 else (1.0 if residual >= 0.0 else -1.0)

        intercept = np.asarray(joint_model["intercept"], dtype=np.float32)
        slope = np.asarray(joint_model["slope"], dtype=np.float32)
        axis = np.asarray(joint_model["axis_unit"], dtype=np.float32)
        part_point = intercept + slope * q
        desired_dir = axis * np.float32(desired_sign)
        b1, _ = _axis_perp_basis(axis)
        radius = np.float32(max(0.02, min(0.08, float(joint_model.get("contact_dist_threshold", 0.05)))))

        anchor_offsets = [
            np.zeros(3, dtype=np.float32),
            -desired_dir * radius,
            desired_dir * radius,
            b1 * radius,
            -b1 * radius,
        ]
        anchor_weights = [1.0, 0.75, 0.75, 0.5, 0.5]
        tokens: list[np.ndarray] = []
        for offset, weight in zip(anchor_offsets, anchor_weights):
            anchor = part_point + offset
            attract_weight = np.float32(weight * min(residual_mag, 1.25))
            attractive_flow = desired_dir * attract_weight
            repulsive_flow = np.zeros(3, dtype=np.float32)
            tokens.append(_articulation_token(r, anchor, attractive_flow, repulsive_flow, float(attract_weight), 0.0))

        other_joint = next((name for name in ARTICULATION_FIELD_JOINT_ORDER if name != joint_name and name in geometry["joints"]), None)
        if other_joint is not None:
            other_model = geometry["joints"][other_joint]
            other_idx = int(other_model["joint_index"])
            other_q = float(s[other_idx])
            other_point = np.asarray(other_model["intercept"], dtype=np.float32) + np.asarray(other_model["slope"], dtype=np.float32) * other_q
            away_from_other = _safe_unit(r[:3] - other_point, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
            repel_weight = np.float32(0.5)
            tokens.append(_articulation_token(r, other_point, np.zeros(3, dtype=np.float32), away_from_other * repel_weight, 0.0, float(repel_weight)))

        while len(tokens) < ARTICULATION_V5_TOKEN_COUNT:
            tokens.append(zero_token.copy())
        out.append(np.concatenate(tokens[:ARTICULATION_V5_TOKEN_COUNT]).astype(np.float32))
    return np.stack(out).astype(np.float32)


def make_articulation_potential_g_v5_attraction_only(
    robot: np.ndarray,
    scene: np.ndarray,
    task_names: np.ndarray,
    geometry: dict[str, Any],
) -> np.ndarray:
    out: list[np.ndarray] = []
    zero_token = np.zeros(ARTICULATION_V5_TOKEN_DIM, dtype=np.float32)

    for r, s, task_value in zip(robot, scene, task_names):
        task = str(task_value)
        joint_name = ARTICULATION_TASK_TO_JOINT.get(task)
        if joint_name is None:
            out.append(np.zeros(ARTICULATION_V5_TOKEN_COUNT * ARTICULATION_V5_TOKEN_DIM, dtype=np.float32))
            continue

        joint_model = geometry["joints"][joint_name]
        joint_idx = int(joint_model["joint_index"])
        joint_range = float(joint_model["joint_range"])
        q = float(s[joint_idx])
        target = articulation_target_controls(task, s)[joint_idx]
        residual = float(target - q)
        residual_norm = residual / max(joint_range, 1e-6)
        residual_mag = abs(residual_norm)
        desired_sign = 0.0 if residual_mag < 1e-6 else (1.0 if residual >= 0.0 else -1.0)

        intercept = np.asarray(joint_model["intercept"], dtype=np.float32)
        slope = np.asarray(joint_model["slope"], dtype=np.float32)
        axis = np.asarray(joint_model["axis_unit"], dtype=np.float32)
        part_point = intercept + slope * q
        desired_dir = axis * np.float32(desired_sign)
        b1, _ = _axis_perp_basis(axis)
        radius = np.float32(max(0.02, min(0.08, float(joint_model.get("contact_dist_threshold", 0.05)))))

        anchor_offsets = [
            np.zeros(3, dtype=np.float32),
            -desired_dir * radius,
            desired_dir * radius,
            b1 * radius,
            -b1 * radius,
        ]
        anchor_weights = [1.0, 0.75, 0.75, 0.5, 0.5]
        tokens: list[np.ndarray] = []
        for offset, weight in zip(anchor_offsets, anchor_weights):
            anchor = part_point + offset
            attract_weight = np.float32(weight * min(residual_mag, 1.25))
            attractive_flow = desired_dir * attract_weight
            tokens.append(_articulation_token(r, anchor, attractive_flow, np.zeros(3, dtype=np.float32), float(attract_weight), 0.0))

        while len(tokens) < ARTICULATION_V5_TOKEN_COUNT:
            tokens.append(zero_token.copy())
        out.append(np.concatenate(tokens[:ARTICULATION_V5_TOKEN_COUNT]).astype(np.float32))
    return np.stack(out).astype(np.float32)


def make_articulation_interaction_field_g_v6(
    robot: np.ndarray,
    scene: np.ndarray,
    task_names: np.ndarray,
    geometry: dict[str, Any],
) -> np.ndarray:
    out: list[np.ndarray] = []
    zero_token = np.zeros(ARTICULATION_V5_TOKEN_DIM, dtype=np.float32)

    for r, s, task_value in zip(robot, scene, task_names):
        task = str(task_value)
        joint_name = ARTICULATION_TASK_TO_JOINT.get(task)
        if joint_name is None:
            out.append(np.zeros(ARTICULATION_V5_TOKEN_COUNT * ARTICULATION_V5_TOKEN_DIM, dtype=np.float32))
            continue

        joint_model = geometry["joints"][joint_name]
        joint_idx = int(joint_model["joint_index"])
        joint_range = float(joint_model["joint_range"])
        q = float(s[joint_idx])
        target = articulation_target_controls(task, s)[joint_idx]
        residual = float(target - q)
        residual_norm = residual / max(joint_range, 1e-6)
        residual_mag = abs(residual_norm)
        desired_sign = 0.0 if residual_mag < 1e-6 else (1.0 if residual >= 0.0 else -1.0)

        intercept = np.asarray(joint_model["intercept"], dtype=np.float32)
        slope = np.asarray(joint_model["slope"], dtype=np.float32)
        axis = np.asarray(joint_model["axis_unit"], dtype=np.float32)
        part_point = intercept + slope * q
        desired_dir = axis * np.float32(desired_sign)
        b1, _ = _axis_perp_basis(axis)
        radius = np.float32(max(0.02, min(0.08, float(joint_model.get("contact_dist_threshold", 0.05)))))
        flow_weight = np.float32(min(residual_mag, 1.25))
        side_repel_weight = np.float32(0.45 * min(1.0, max(residual_mag, 0.25)))
        axial_repel_weight = np.float32(0.35 * min(1.0, max(residual_mag, 0.25)))

        tokens = [
            _articulation_token(r, part_point, desired_dir * flow_weight, np.zeros(3, dtype=np.float32), float(flow_weight), 0.0),
            _articulation_token(r, part_point - desired_dir * radius, desired_dir * flow_weight * np.float32(0.75), np.zeros(3, dtype=np.float32), float(flow_weight * np.float32(0.75)), 0.0),
            _articulation_token(r, part_point + desired_dir * radius, desired_dir * flow_weight * np.float32(0.75), np.zeros(3, dtype=np.float32), float(flow_weight * np.float32(0.75)), 0.0),
            _articulation_token(r, part_point + b1 * radius, np.zeros(3, dtype=np.float32), -b1 * side_repel_weight, 0.0, float(side_repel_weight)),
            _articulation_token(r, part_point - b1 * radius, np.zeros(3, dtype=np.float32), b1 * side_repel_weight, 0.0, float(side_repel_weight)),
            _articulation_token(
                r,
                part_point - desired_dir * radius * np.float32(1.5),
                np.zeros(3, dtype=np.float32),
                desired_dir * axial_repel_weight,
                0.0,
                float(axial_repel_weight),
            ),
        ]

        while len(tokens) < ARTICULATION_V5_TOKEN_COUNT:
            tokens.append(zero_token.copy())
        out.append(np.concatenate(tokens[:ARTICULATION_V5_TOKEN_COUNT]).astype(np.float32))
    return np.stack(out).astype(np.float32)


def make_articulation_unified_field_g_v7(
    robot: np.ndarray,
    scene: np.ndarray,
    task_names: np.ndarray,
    geometry: dict[str, Any],
) -> np.ndarray:
    out: list[np.ndarray] = []
    zero_token = np.zeros(ARTICULATION_V5_TOKEN_DIM, dtype=np.float32)

    for r, s, task_value in zip(robot, scene, task_names):
        task = str(task_value)
        joint_name = ARTICULATION_TASK_TO_JOINT.get(task)
        if joint_name is None:
            out.append(np.zeros(ARTICULATION_V5_TOKEN_COUNT * ARTICULATION_V5_TOKEN_DIM, dtype=np.float32))
            continue

        joint_model = geometry["joints"][joint_name]
        joint_idx = int(joint_model["joint_index"])
        joint_range = float(joint_model["joint_range"])
        q = float(s[joint_idx])
        target = articulation_target_controls(task, s)[joint_idx]
        residual = float(target - q)
        residual_norm = residual / max(joint_range, 1e-6)
        residual_mag = abs(residual_norm)
        desired_sign = 0.0 if residual_mag < 1e-6 else (1.0 if residual >= 0.0 else -1.0)

        intercept = np.asarray(joint_model["intercept"], dtype=np.float32)
        slope = np.asarray(joint_model["slope"], dtype=np.float32)
        axis = np.asarray(joint_model["axis_unit"], dtype=np.float32)
        part_point = intercept + slope * q
        desired_dir = axis * np.float32(desired_sign)
        b1, _ = _axis_perp_basis(axis)
        radius = np.float32(max(0.02, min(0.08, float(joint_model.get("contact_dist_threshold", 0.05)))))
        flow_weight = np.float32(min(residual_mag, 1.25))
        corridor_weight = np.float32(0.35 * min(1.0, max(residual_mag, 0.25)))
        backstop_weight = np.float32(0.25 * min(1.0, max(residual_mag, 0.25)))

        offsets = [
            np.zeros(3, dtype=np.float32),
            -desired_dir * radius,
            desired_dir * radius,
            b1 * radius,
            -b1 * radius,
            -desired_dir * radius * np.float32(1.5),
        ]
        flow_scales = [1.0, 0.9, 0.65, 0.75, 0.75, 0.8]
        tokens: list[np.ndarray] = []
        for offset, flow_scale in zip(offsets, flow_scales):
            axis_coord = np.float32(np.dot(offset, desired_dir))
            lateral = offset - desired_dir * axis_coord
            corridor_correction = np.zeros(3, dtype=np.float32)
            if float(np.linalg.norm(lateral)) > 1e-6:
                corridor_correction = -lateral / max(float(radius), 1e-6) * corridor_weight
            backstop_correction = np.zeros(3, dtype=np.float32)
            if axis_coord < 0.0:
                backstop_correction = desired_dir * backstop_weight * np.float32(min(1.5, abs(float(axis_coord)) / max(float(radius), 1e-6)))
            field = desired_dir * flow_weight * np.float32(flow_scale) + corridor_correction + backstop_correction
            field_weight = np.float32(np.linalg.norm(field))
            tokens.append(_articulation_token(r, part_point + offset, field, np.zeros(3, dtype=np.float32), float(field_weight), 0.0))

        while len(tokens) < ARTICULATION_V5_TOKEN_COUNT:
            tokens.append(zero_token.copy())
        out.append(np.concatenate(tokens[:ARTICULATION_V5_TOKEN_COUNT]).astype(np.float32))
    return np.stack(out).astype(np.float32)


def make_articulation_contact_g_v3(
    robot: np.ndarray,
    scene: np.ndarray,
    task_names: np.ndarray,
    geometry: dict[str, Any],
) -> np.ndarray:
    out: list[np.ndarray] = []
    task_contact_models = geometry.get("task_contact_models", {})
    for r, s, task_value in zip(robot, scene, task_names):
        task = str(task_value)
        joint_name = ARTICULATION_TASK_TO_JOINT.get(task)
        if joint_name is None:
            out.append(np.zeros(10, dtype=np.float32))
            continue

        joint_model = geometry["joints"][joint_name]
        joint_idx = int(joint_model["joint_index"])
        joint_range = float(joint_model["joint_range"])
        q = float(s[joint_idx])
        target = articulation_target_controls(task, s)[joint_idx]
        residual = float(target - q)
        residual_mag = abs(residual) / max(joint_range, 1e-6)
        desired_sign = 0.0 if residual_mag < 1e-6 else (1.0 if residual >= 0.0 else -1.0)

        intercept = np.asarray(joint_model["intercept"], dtype=np.float32)
        slope = np.asarray(joint_model["slope"], dtype=np.float32)
        axis = np.asarray(joint_model["axis_unit"], dtype=np.float32)
        part_point = intercept + slope * q
        desired_motion = axis * np.float32(desired_sign)

        contact_model = task_contact_models.get(task, {})
        if "eef_offset_from_part_point" in contact_model:
            eef_offset = np.asarray(contact_model["eef_offset_from_part_point"], dtype=np.float32)
            effective_contact_point = part_point + eef_offset
        else:
            contact_offset = float(contact_model.get("contact_offset_along_desired_motion", 0.03))
            effective_contact_point = part_point - contact_offset * desired_motion
        effective_contact_affordance = effective_contact_point - r[:3]

        done = residual_mag < 0.08
        contact_threshold = float(joint_model.get("contact_dist_threshold", 0.06))
        far = float(np.linalg.norm(effective_contact_affordance)) > contact_threshold
        if done:
            stage = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        elif far:
            stage = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            stage = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        out.append(
            np.concatenate(
                [
                    effective_contact_affordance.astype(np.float32),
                    desired_motion.astype(np.float32),
                    np.asarray([residual_mag], dtype=np.float32),
                    stage,
                ]
            ).astype(np.float32)
        )
    return np.stack(out).astype(np.float32)


def collect_records(
    data_root: Path,
    task_to_id: dict[str, int],
    max_samples: int,
    max_segments: int,
    stride: int,
    seed: int,
    include_tasks: set[str] | None = None,
    articulation_geometry: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    split_dir = data_root / "training"
    segments = load_lang_segments(split_dir)
    if max_segments > 0:
        segments = segments[:max_segments]
    rng.shuffle(segments)

    robot_obs: list[np.ndarray] = []
    scene_obs: list[np.ndarray] = []
    task_oh: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    task_names: list[str] = []
    g_oracle: list[np.ndarray] = []
    target_scenes: list[np.ndarray] = []
    articulation_targets: list[np.ndarray] = []
    demo_starts: dict[str, list[np.ndarray]] = defaultdict(list)
    demo_targets: dict[str, list[np.ndarray]] = defaultdict(list)
    used_segments: list[dict[str, Any]] = []

    for seg in segments:
        task = seg["task"]
        if include_tasks is not None and task not in include_tasks:
            continue
        start, end = seg["start"], seg["end"]
        start_file = episode_file(split_dir, start)
        end_file = episode_file(split_dir, end)
        if not start_file.exists() or not end_file.exists():
            continue
        try:
            start_npz = np.load(start_file, allow_pickle=True)
            end_npz = np.load(end_file, allow_pickle=True)
            start_scene = np.asarray(start_npz["scene_obs"], dtype=np.float32)
            target_scene = np.asarray(end_npz["scene_obs"], dtype=np.float32)
        except Exception:
            continue
        demo_starts[task].append(start_scene)
        demo_targets[task].append(target_scene)
        used_segments.append(seg)

        frame_indices = list(range(start, end + 1, max(1, stride)))
        for idx in frame_indices:
            if max_samples > 0 and len(actions) >= max_samples:
                break
            f = episode_file(split_dir, idx)
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
                r = np.asarray(d["robot_obs"], dtype=np.float32)
                s = np.asarray(d["scene_obs"], dtype=np.float32)
                a = np.asarray(d["rel_actions"], dtype=np.float32)
            except Exception:
                continue
            robot_obs.append(r)
            scene_obs.append(s)
            task_oh.append(one_hot(task, task_to_id))
            actions.append(a)
            task_names.append(task)
            g_oracle.append(target_scene - s)
            target_scenes.append(target_scene)
            articulation_targets.append(articulation_target_controls(task, s, target_scene))
        if max_samples > 0 and len(actions) >= max_samples:
            break

    if not actions:
        raise RuntimeError("No train records were collected")

    demos = {
        task: {
            "starts": np.stack(demo_starts[task]).astype(np.float32),
            "targets": np.stack(demo_targets[task]).astype(np.float32),
        }
        for task in demo_starts
        if demo_starts[task]
    }
    arrays = {
        "robot_obs": np.stack(robot_obs).astype(np.float32),
        "scene_obs": np.stack(scene_obs).astype(np.float32),
        "task_oh": np.stack(task_oh).astype(np.float32),
        "task_name": np.asarray(task_names, dtype=object),
        "g_oracle": np.stack(g_oracle).astype(np.float32),
        "target_scene": np.stack(target_scenes).astype(np.float32),
        "articulation_target": np.stack(articulation_targets).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
    }
    if articulation_geometry is not None:
        arrays["articulation_g_v2"] = make_articulation_compact_g(
            arrays["robot_obs"],
            arrays["scene_obs"],
            arrays["task_name"],
            articulation_geometry,
        )
        arrays["articulation_g_v2_split"] = make_articulation_split_g(
            arrays["robot_obs"],
            arrays["scene_obs"],
            arrays["task_name"],
            articulation_geometry,
        )
        arrays["articulation_g_v4_mechanism"] = make_articulation_mechanism_g_v4(
            arrays["robot_obs"],
            arrays["scene_obs"],
            arrays["task_name"],
            articulation_geometry,
        )
        arrays["articulation_g_v5_potential"] = make_articulation_potential_g_v5(
            arrays["robot_obs"],
            arrays["scene_obs"],
            arrays["task_name"],
            articulation_geometry,
        )
        arrays["articulation_g_v5_attraction_only"] = make_articulation_potential_g_v5_attraction_only(
            arrays["robot_obs"],
            arrays["scene_obs"],
            arrays["task_name"],
            articulation_geometry,
        )
        arrays["articulation_g_v6_interaction_field"] = make_articulation_interaction_field_g_v6(
            arrays["robot_obs"],
            arrays["scene_obs"],
            arrays["task_name"],
            articulation_geometry,
        )
        arrays["articulation_g_v7_unified_field"] = make_articulation_unified_field_g_v7(
            arrays["robot_obs"],
            arrays["scene_obs"],
            arrays["task_name"],
            articulation_geometry,
        )
        arrays["articulation_g_v3_contact"] = make_articulation_contact_g_v3(
            arrays["robot_obs"],
            arrays["scene_obs"],
            arrays["task_name"],
            articulation_geometry,
        )
    return arrays, demos, used_segments


def _block_positions(scene: np.ndarray) -> np.ndarray:
    return np.stack([scene[:, 6:9], scene[:, 12:15], scene[:, 18:21]], axis=1)


def _block_eulers(scene: np.ndarray) -> np.ndarray:
    return np.concatenate([scene[:, 9:12], scene[:, 15:18], scene[:, 21:24]], axis=1)


def _interaction_scale(robot: np.ndarray, scene: np.ndarray) -> np.ndarray:
    eef = robot[:, None, :3]
    blocks = _block_positions(scene)
    dists = np.linalg.norm(blocks - eef, axis=2)
    return np.maximum(np.sqrt(np.mean(dists**2, axis=1, keepdims=True)), 0.05).astype(np.float32)


def make_absolute_base(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([arrays["robot_obs"], arrays["scene_obs"], arrays["task_oh"]], axis=1).astype(np.float32)


def make_canonical_base(arrays: dict[str, np.ndarray]) -> np.ndarray:
    robot = arrays["robot_obs"]
    scene = arrays["scene_obs"]
    scale = _interaction_scale(robot, scene)
    rel_blocks = ((_block_positions(scene) - robot[:, None, :3]) / scale[:, None, :]).reshape(len(robot), -1)
    pieces = [
        robot[:, 3:],
        scene[:, :6],
        rel_blocks,
        _block_eulers(scene),
        arrays["task_oh"],
    ]
    return np.concatenate(pieces, axis=1).astype(np.float32)


def make_structured_oracle_g(arrays: dict[str, np.ndarray]) -> np.ndarray:
    robot = arrays["robot_obs"]
    scene = arrays["scene_obs"]
    target_scene = arrays["target_scene"]
    scale = _interaction_scale(robot, scene)
    eef = robot[:, None, :3]
    blocks = _block_positions(scene)
    target_blocks = _block_positions(target_scene)
    rel_blocks = (blocks - eef) / scale[:, None, :]
    block_delta = (target_blocks - blocks) / scale[:, None, :]
    target_rel = (target_blocks - eef) / scale[:, None, :]
    control_delta = target_scene[:, :6] - scene[:, :6]
    rel_norms = np.linalg.norm(rel_blocks, axis=2)
    delta_norms = np.linalg.norm(block_delta, axis=2)
    pieces = [
        block_delta.reshape(len(robot), -1),
        target_rel.reshape(len(robot), -1),
        control_delta,
        target_scene[:, :6],
        rel_norms,
        delta_norms,
        arrays["task_oh"],
    ]
    return np.concatenate(pieces, axis=1).astype(np.float32)


def make_articulation_oracle_g(arrays: dict[str, np.ndarray]) -> np.ndarray:
    controls = arrays["scene_obs"][:, :2]
    target = arrays["articulation_target"]
    delta = target - controls
    pieces = [
        controls,
        target,
        delta,
        np.abs(delta),
        np.sign(delta),
        arrays["task_oh"],
    ]
    return np.concatenate(pieces, axis=1).astype(np.float32)


def make_policy_inputs(arrays: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray | str]:
    if mode == "no_g":
        return {"kind": "plain", "x": make_absolute_base(arrays)}
    if mode == "oracle_g":
        return {"kind": "plain", "x": np.concatenate([make_absolute_base(arrays), arrays["g_oracle"]], axis=1).astype(np.float32)}
    if mode == "canonical_no_g":
        return {"kind": "plain", "x": make_canonical_base(arrays)}
    if mode == "canonical_g_concat":
        return {
            "kind": "plain",
            "x": np.concatenate([make_canonical_base(arrays), make_structured_oracle_g(arrays)], axis=1).astype(np.float32),
        }
    if mode == "canonical_g_film":
        return {"kind": "film", "base": make_canonical_base(arrays), "g": make_structured_oracle_g(arrays)}
    if mode == "canonical_articulation_g_concat":
        return {
            "kind": "plain",
            "x": np.concatenate([make_canonical_base(arrays), make_articulation_oracle_g(arrays)], axis=1).astype(np.float32),
        }
    if mode == "canonical_articulation_g_film":
        return {"kind": "film", "base": make_canonical_base(arrays), "g": make_articulation_oracle_g(arrays)}
    if mode == "articulation_g_concat":
        return {
            "kind": "plain",
            "x": np.concatenate([make_absolute_base(arrays), make_articulation_oracle_g(arrays)], axis=1).astype(np.float32),
        }
    if mode == "articulation_g_film":
        return {"kind": "film", "base": make_absolute_base(arrays), "g": make_articulation_oracle_g(arrays)}
    if mode == "articulation_g_v2_concat":
        return {
            "kind": "plain",
            "x": np.concatenate([make_absolute_base(arrays), arrays["articulation_g_v2"]], axis=1).astype(np.float32),
        }
    if mode == "articulation_g_v2_film":
        return {"kind": "film", "base": make_absolute_base(arrays), "g": arrays["articulation_g_v2"].astype(np.float32)}
    if mode == "articulation_g_v2_split_concat":
        return {
            "kind": "plain",
            "x": np.concatenate([make_absolute_base(arrays), arrays["articulation_g_v2_split"]], axis=1).astype(np.float32),
        }
    if mode == "articulation_g_v2_split_film":
        return {"kind": "film", "base": make_absolute_base(arrays), "g": arrays["articulation_g_v2_split"].astype(np.float32)}
    if mode == "articulation_g_v3_contact_concat":
        return {
            "kind": "plain",
            "x": np.concatenate([make_absolute_base(arrays), arrays["articulation_g_v3_contact"]], axis=1).astype(np.float32),
        }
    if mode == "articulation_g_v3_contact_film":
        return {"kind": "film", "base": make_absolute_base(arrays), "g": arrays["articulation_g_v3_contact"].astype(np.float32)}
    if mode == "articulation_g_v4_mechanism_concat":
        return {
            "kind": "plain",
            "x": np.concatenate([make_absolute_base(arrays), arrays["articulation_g_v4_mechanism"]], axis=1).astype(np.float32),
        }
    if mode == "articulation_g_v4_mechanism_film":
        return {"kind": "film", "base": make_absolute_base(arrays), "g": arrays["articulation_g_v4_mechanism"].astype(np.float32)}
    if mode == "articulation_g_v5_potential_concat":
        return {
            "kind": "plain",
            "x": np.concatenate([make_absolute_base(arrays), arrays["articulation_g_v5_potential"]], axis=1).astype(np.float32),
        }
    if mode == "articulation_g_v5_potential_film":
        return {"kind": "film", "base": make_absolute_base(arrays), "g": arrays["articulation_g_v5_potential"].astype(np.float32)}
    if mode == "articulation_g_v5_attraction_only_concat":
        return {
            "kind": "plain",
            "x": np.concatenate([make_absolute_base(arrays), arrays["articulation_g_v5_attraction_only"]], axis=1).astype(np.float32),
        }
    if mode == "articulation_g_v5_attraction_only_film":
        return {"kind": "film", "base": make_absolute_base(arrays), "g": arrays["articulation_g_v5_attraction_only"].astype(np.float32)}
    if mode == "articulation_g_v6_interaction_field_concat":
        return {
            "kind": "plain",
            "x": np.concatenate([make_absolute_base(arrays), arrays["articulation_g_v6_interaction_field"]], axis=1).astype(np.float32),
        }
    if mode == "articulation_g_v6_interaction_field_film":
        return {"kind": "film", "base": make_absolute_base(arrays), "g": arrays["articulation_g_v6_interaction_field"].astype(np.float32)}
    if mode == "articulation_g_v7_unified_field_concat":
        return {
            "kind": "plain",
            "x": np.concatenate([make_absolute_base(arrays), arrays["articulation_g_v7_unified_field"]], axis=1).astype(np.float32),
        }
    if mode == "articulation_g_v7_unified_field_film":
        return {"kind": "film", "base": make_absolute_base(arrays), "g": arrays["articulation_g_v7_unified_field"].astype(np.float32)}
    raise ValueError(f"Unknown policy input mode: {mode}")


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int):
        super().__init__()
        blocks: list[nn.Module] = []
        dim = in_dim
        for _ in range(layers):
            blocks.extend([nn.Linear(dim, hidden), nn.ReLU()])
            dim = hidden
        blocks.append(nn.Linear(dim, 7))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMMLP(nn.Module):
    def __init__(self, base_dim: int, g_dim: int, hidden: int, layers: int):
        super().__init__()
        self.in_layer = nn.Linear(base_dim, hidden)
        self.g_net = nn.Sequential(nn.Linear(g_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden * 2))
        self.hidden_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(max(0, layers - 1)))
        self.out = nn.Linear(hidden, 7)

    def forward(self, base: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.in_layer(base))
        gamma, beta = self.g_net(g).chunk(2, dim=-1)
        h = h * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))
        return self.out(h)


def train_variant(
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
    articulation_geometry: dict[str, Any] | None,
    checkpoint_epochs: set[int] | None = None,
    zero_task_onehot: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    y = arrays["actions"].astype(np.float32)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-6
    y_n = (y - y_mean) / y_std
    inputs = make_policy_inputs(arrays, mode)

    if inputs["kind"] == "plain":
        x = inputs["x"]
        assert isinstance(x, np.ndarray)
        x_mean, x_std = x.mean(axis=0), x.std(axis=0) + 1e-6
        x_n = (x - x_mean) / x_std
        dataset_tensors = [torch.from_numpy(x_n), torch.from_numpy(y_n)]
        model: nn.Module = MLP(x_n.shape[1], hidden=hidden, layers=layers).to(device)
        ckpt_dims = {"arch": "mlp", "in_dim": x_n.shape[1], "x_mean": x_mean, "x_std": x_std}
    else:
        base = inputs["base"]
        g = inputs["g"]
        assert isinstance(base, np.ndarray) and isinstance(g, np.ndarray)
        base_mean, base_std = base.mean(axis=0), base.std(axis=0) + 1e-6
        g_mean, g_std = g.mean(axis=0), g.std(axis=0) + 1e-6
        base_n = (base - base_mean) / base_std
        g_n = (g - g_mean) / g_std
        dataset_tensors = [torch.from_numpy(base_n), torch.from_numpy(g_n), torch.from_numpy(y_n)]
        model = FiLMMLP(base_n.shape[1], g_n.shape[1], hidden=hidden, layers=layers).to(device)
        ckpt_dims = {
            "arch": "film",
            "base_dim": base_n.shape[1],
            "g_dim": g_n.shape[1],
            "base_mean": base_mean,
            "base_std": base_std,
            "g_mean": g_mean,
            "g_std": g_std,
        }

    n = len(y_n)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    val_n = max(1, int(0.1 * n))
    val_idx, train_idx = order[:val_n], order[val_n:]
    train_ds = TensorDataset(*[t[train_idx] for t in dataset_tensors])
    val_y = torch.from_numpy(y[val_idx]).to(device)
    if inputs["kind"] == "plain":
        assert "x_n" in locals()
        val_inputs = (torch.from_numpy(x_n[val_idx]).to(device),)
    else:
        assert "base_n" in locals() and "g_n" in locals()
        val_inputs = (torch.from_numpy(base_n[val_idx]).to(device), torch.from_numpy(g_n[val_idx]).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader_gen = torch.Generator()
    loader_gen.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=loader_gen)

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_epochs = checkpoint_epochs or set()

    def save_checkpoint(path: Path, checkpoint_epoch: int | None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "mode": mode,
            "checkpoint_epoch": checkpoint_epoch,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "hidden": hidden,
            "layers": layers,
            "y_mean": y_mean,
            "y_std": y_std,
            "zero_task_onehot": zero_task_onehot,
            **ckpt_dims,
        }
        if articulation_geometry is not None:
            ckpt["articulation_geometry"] = articulation_geometry
        torch.save(ckpt, path)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            *xb, yb = batch
            xb = [item.to(device) for item in xb]
            yb = yb.to(device)
            pred = model(*xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred = model(*val_inputs).cpu().numpy() * y_std + y_mean
        val_mse = float(np.mean((pred - val_y.cpu().numpy()) ** 2))
        val_grip = float(np.mean((pred[:, -1] >= 0) == (val_y.cpu().numpy()[:, -1] >= 0)))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mse": val_mse, "val_gripper_acc": val_grip})
        print(f"[{mode}] epoch {epoch}/{epochs} train_loss={history[-1]['train_loss']:.5f} val_mse={val_mse:.5f} grip={val_grip:.3f}", flush=True)
        if epoch in checkpoint_epochs:
            save_checkpoint(out_dir / "checkpoints" / f"epoch_{epoch:04d}" / "policy.pt", checkpoint_epoch=epoch)

    save_checkpoint(out_dir / "policy.pt", checkpoint_epoch=None)
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "final": history[-1], "num_records": int(n)}, f, indent=2, default=json_default)
    return {"history": history, "final": history[-1], "num_records": int(n)}


class StateBCPolicy:
    def __init__(
        self,
        ckpt_path: Path,
        task_to_id: dict[str, int],
        text_to_task: dict[str, str],
        demos: dict[str, dict[str, np.ndarray]],
        device: str,
    ):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.mode = ckpt["mode"]
        self.task_to_id = task_to_id
        self.text_to_task = text_to_task
        self.demos = demos
        self.device = torch.device(device)
        self.arch = ckpt.get("arch", "mlp")
        if self.arch == "film":
            self.model = FiLMMLP(ckpt["base_dim"], ckpt["g_dim"], ckpt["hidden"], ckpt["layers"])
            self.base_mean = ckpt["base_mean"].astype(np.float32)
            self.base_std = ckpt["base_std"].astype(np.float32)
            self.g_mean = ckpt["g_mean"].astype(np.float32)
            self.g_std = ckpt["g_std"].astype(np.float32)
        else:
            self.model = MLP(ckpt["in_dim"], ckpt["hidden"], ckpt["layers"])
            self.x_mean = ckpt["x_mean"].astype(np.float32)
            self.x_std = ckpt["x_std"].astype(np.float32)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        self.y_mean = ckpt["y_mean"].astype(np.float32)
        self.y_std = ckpt["y_std"].astype(np.float32)
        self.articulation_geometry = ckpt.get("articulation_geometry", None)
        self.zero_task_onehot = bool(ckpt.get("zero_task_onehot", False))
        self.current_lang: str | None = None

    def reset(self) -> None:
        self.current_lang = None

    def _task_from_goal(self, goal: str) -> str:
        if goal in self.text_to_task:
            return self.text_to_task[goal]
        lower = goal.lower()
        for text, task in self.text_to_task.items():
            if text.lower() == lower:
                return task
        return next(iter(self.task_to_id))

    def _oracle_target_scene(self, task: str, scene: np.ndarray) -> np.ndarray:
        if task not in self.demos:
            return scene.astype(np.float32)
        starts = self.demos[task]["starts"]
        targets = self.demos[task]["targets"]
        distances = np.mean((starts - scene[None, :]) ** 2, axis=1)
        return targets[int(np.argmin(distances))].astype(np.float32)

    def step(self, obs: dict[str, Any], goal: str) -> np.ndarray:
        task = self._task_from_goal(goal)
        robot = np.asarray(obs["robot_obs"], dtype=np.float32)
        scene = np.asarray(obs["scene_obs"], dtype=np.float32)
        target_scene = self._oracle_target_scene(task, scene)
        arrays = {
            "robot_obs": robot[None, :],
            "scene_obs": scene[None, :],
            "target_scene": target_scene[None, :],
            "g_oracle": (target_scene - scene)[None, :],
            "articulation_target": articulation_target_controls(task, scene, target_scene)[None, :],
            "task_name": np.asarray([task], dtype=object),
            "task_oh": one_hot(task, self.task_to_id)[None, :],
        }
        if self.zero_task_onehot:
            arrays["task_oh"] = np.zeros_like(arrays["task_oh"], dtype=np.float32)
        if self.articulation_geometry is not None:
            arrays["articulation_g_v2"] = make_articulation_compact_g(
                arrays["robot_obs"],
                arrays["scene_obs"],
                arrays["task_name"],
                self.articulation_geometry,
            )
            arrays["articulation_g_v2_split"] = make_articulation_split_g(
                arrays["robot_obs"],
                arrays["scene_obs"],
                arrays["task_name"],
                self.articulation_geometry,
            )
            arrays["articulation_g_v4_mechanism"] = make_articulation_mechanism_g_v4(
                arrays["robot_obs"],
                arrays["scene_obs"],
                arrays["task_name"],
                self.articulation_geometry,
            )
            arrays["articulation_g_v5_potential"] = make_articulation_potential_g_v5(
                arrays["robot_obs"],
                arrays["scene_obs"],
                arrays["task_name"],
                self.articulation_geometry,
            )
            arrays["articulation_g_v5_attraction_only"] = make_articulation_potential_g_v5_attraction_only(
                arrays["robot_obs"],
                arrays["scene_obs"],
                arrays["task_name"],
                self.articulation_geometry,
            )
            arrays["articulation_g_v6_interaction_field"] = make_articulation_interaction_field_g_v6(
                arrays["robot_obs"],
                arrays["scene_obs"],
                arrays["task_name"],
                self.articulation_geometry,
            )
            arrays["articulation_g_v7_unified_field"] = make_articulation_unified_field_g_v7(
                arrays["robot_obs"],
                arrays["scene_obs"],
                arrays["task_name"],
                self.articulation_geometry,
            )
            arrays["articulation_g_v3_contact"] = make_articulation_contact_g_v3(
                arrays["robot_obs"],
                arrays["scene_obs"],
                arrays["task_name"],
                self.articulation_geometry,
            )
        inputs = make_policy_inputs(arrays, self.mode)
        with torch.no_grad():
            if self.arch == "film":
                base = inputs["base"]
                g = inputs["g"]
                assert isinstance(base, np.ndarray) and isinstance(g, np.ndarray)
                base = (base.astype(np.float32) - self.base_mean) / self.base_std
                g = (g.astype(np.float32) - self.g_mean) / self.g_std
                pred = self.model(
                    torch.from_numpy(base).to(self.device),
                    torch.from_numpy(g).to(self.device),
                ).cpu().numpy()[0]
            else:
                x = inputs["x"]
                assert isinstance(x, np.ndarray)
                x = (x.astype(np.float32) - self.x_mean) / self.x_std
                pred = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()[0]
        action = pred * self.y_std + self.y_mean
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        action[-1] = 1.0 if action[-1] >= 0 else -1.0
        return action


def evaluate_variant(
    calvin_code: Path,
    data_root: Path,
    eval_dir: Path,
    ckpt_path: Path,
    task_to_id: dict[str, int],
    text_to_task: dict[str, str],
    demos: dict[str, dict[str, np.ndarray]],
    num_sequences: int,
    ep_len: int,
    device: str,
    eval_mode: str,
    eval_task_filter: set[str] | None,
    eval_scan_sequences: int,
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
    from collections import Counter, defaultdict
    import hydra
    from omegaconf import OmegaConf

    eval_mod.NUM_SEQUENCES = num_sequences
    eval_mod.EP_LEN = ep_len
    model = StateBCPolicy(ckpt_path, task_to_id, text_to_task, demos, device=device)
    env = eval_mod.make_env(data_root)
    eval_dir.mkdir(parents=True, exist_ok=True)
    if eval_mode == "long_horizon":
        results = eval_mod.evaluate_policy(model, env, epoch=ckpt_path.parent.name, eval_log_dir=eval_dir, debug=False)
        summary = {
            "eval_mode": eval_mode,
            "avg_seq_len": float(np.mean(results)) if results else 0.0,
            "chain_sr": {str(i + 1): float(sr) for i, sr in enumerate(count_success(results))},
            "raw_results": results,
        }
    else:
        conf_dir = calvin_models / "conf"
        task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        task_oracle = hydra.utils.instantiate(task_cfg)
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
        eval_mod.EP_LEN = ep_len
        selected = []
        if eval_mode == "first_task":
            for initial_state, seq in get_sequences(max(num_sequences, eval_scan_sequences)):
                first_task = seq[0]
                if eval_task_filter is None or first_task in eval_task_filter:
                    selected.append((initial_state, first_task))
                if len(selected) >= num_sequences:
                    break
        else:
            if eval_task_filter is None:
                raise RuntimeError("balanced_first_task requires --eval-task-filter")
            buckets: dict[str, list[tuple[Any, str]]] = {task: [] for task in sorted(eval_task_filter)}
            for initial_state, seq in get_sequences(max(num_sequences * max(1, len(buckets)) * 10, eval_scan_sequences)):
                first_task = seq[0]
                if first_task in buckets and len(buckets[first_task]) < num_sequences:
                    buckets[first_task].append((initial_state, first_task))
                if all(len(items) >= num_sequences for items in buckets.values()):
                    break
            selected = [item for task in sorted(buckets) for item in buckets[task]]
        if not selected:
            raise RuntimeError(f"No first-task eval sequences matched filter: {sorted(eval_task_filter or [])}")
        task_counts: Counter[str] = Counter()
        task_success: Counter[str] = Counter()
        results = []
        plans: dict[str, list[Any]] = defaultdict(list)
        for initial_state, subtask in selected:
            robot_obs, scene_obs = eval_mod.get_env_state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            success = bool(eval_mod.rollout(env, model, task_oracle, subtask, val_annotations, plans, debug=False))
            task_counts[subtask] += 1
            task_success[subtask] += int(success)
            results.append(int(success))
            done = len(results)
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
            "avg_seq_len": float(np.mean(results)) if results else 0.0,
            "chain_sr": {"1": float(np.mean(results)) if results else 0.0},
            "raw_results": results,
            "task_info": task_info,
            "selected_tasks": [task for _, task in selected],
        }
    with open(eval_dir / "state_bc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["no_g", "oracle_g"],
        choices=[
            "no_g",
            "oracle_g",
            "canonical_no_g",
            "canonical_g_concat",
            "canonical_g_film",
            "canonical_articulation_g_concat",
            "canonical_articulation_g_film",
            "articulation_g_concat",
            "articulation_g_film",
            "articulation_g_v2_concat",
            "articulation_g_v2_film",
            "articulation_g_v2_split_concat",
            "articulation_g_v2_split_film",
            "articulation_g_v3_contact_concat",
            "articulation_g_v3_contact_film",
            "articulation_g_v4_mechanism_concat",
            "articulation_g_v4_mechanism_film",
            "articulation_g_v5_potential_concat",
            "articulation_g_v5_potential_film",
            "articulation_g_v5_attraction_only_concat",
            "articulation_g_v5_attraction_only_film",
            "articulation_g_v6_interaction_field_concat",
            "articulation_g_v6_interaction_field_film",
            "articulation_g_v7_unified_field_concat",
            "articulation_g_v7_unified_field_film",
        ],
    )
    parser.add_argument("--include-tasks", nargs="+", default=None)
    parser.add_argument("--articulation-geometry-json", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--ep-len", type=int, default=120)
    parser.add_argument("--eval-mode", choices=["long_horizon", "first_task", "balanced_first_task"], default="long_horizon")
    parser.add_argument("--eval-task-filter", nargs="+", default=None)
    parser.add_argument("--eval-scan-sequences", type=int, default=1000)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--checkpoint-epochs", nargs="*", type=int, default=[])
    parser.add_argument("--eval-checkpoints", action="store_true")
    parser.add_argument("--zero-task-onehot", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.calvin_code = args.calvin_code.resolve()
    args.data_root = args.data_root.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.articulation_geometry_json is not None:
        args.articulation_geometry_json = args.articulation_geometry_json.resolve()
    if args.checkpoint_root is not None:
        args.checkpoint_root = args.checkpoint_root.resolve()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_epochs = sorted({epoch for epoch in args.checkpoint_epochs if epoch > 0})
    if args.eval_checkpoints and not checkpoint_epochs:
        checkpoint_epochs = [args.epochs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    articulation_geometry = load_articulation_geometry(args.articulation_geometry_json)

    text_to_task, task_to_text = load_eval_annotation_maps(args.calvin_code)
    segments = load_lang_segments(args.data_root / "training")
    task_to_id = build_task_index(segments, text_to_task)
    arrays, demos, used_segments = collect_records(
        args.data_root,
        task_to_id=task_to_id,
        max_samples=args.max_samples,
        max_segments=args.max_segments,
        stride=args.stride,
        seed=args.seed,
        include_tasks=set(args.include_tasks) if args.include_tasks else None,
        articulation_geometry=articulation_geometry,
    )
    if args.zero_task_onehot:
        arrays["task_oh"] = np.zeros_like(arrays["task_oh"], dtype=np.float32)

    with open(args.out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_root": args.data_root,
                "modes": args.modes,
                "include_tasks": args.include_tasks,
                "max_samples": args.max_samples,
                "actual_samples": int(len(arrays["actions"])),
                "max_segments": args.max_segments,
                "actual_segments": len(used_segments),
                "stride": args.stride,
                "epochs": args.epochs,
                "num_sequences": args.num_sequences,
                "ep_len": args.ep_len,
                "eval_mode": args.eval_mode,
                "eval_task_filter": args.eval_task_filter,
                "eval_scan_sequences": args.eval_scan_sequences,
                "checkpoint_epochs": checkpoint_epochs,
                "eval_checkpoints": args.eval_checkpoints,
                "zero_task_onehot": args.zero_task_onehot,
                "articulation_geometry_json": args.articulation_geometry_json,
                "tasks": task_to_id,
            },
            f,
            indent=2,
            default=json_default,
        )
    np.savez_compressed(
        args.out_dir / "demo_oracle_targets.npz",
        **{f"{task}__starts": val["starts"] for task, val in demos.items()},
        **{f"{task}__targets": val["targets"] for task, val in demos.items()},
    )

    rows = []
    for mode in args.modes:
        print(f"[variant] {mode}", flush=True)
        variant_dir = args.out_dir / mode
        if args.eval_only:
            ckpt_root = args.checkpoint_root if args.checkpoint_root is not None else args.out_dir
            metrics_path = ckpt_root / mode / "train_metrics.json"
            if metrics_path.exists():
                train_metrics = json.load(open(metrics_path, encoding="utf-8"))
            else:
                train_metrics = {"final": {"val_mse": None, "val_gripper_acc": None}}
        else:
            train_metrics = train_variant(
                arrays,
                mode=mode,
                out_dir=variant_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden=args.hidden,
                layers=args.layers,
                lr=args.lr,
                device=args.device,
                seed=args.seed,
                articulation_geometry=articulation_geometry,
                checkpoint_epochs=set(checkpoint_epochs),
                zero_task_onehot=args.zero_task_onehot,
            )

        if args.eval_checkpoints:
            ckpt_base = (args.checkpoint_root if args.eval_only and args.checkpoint_root is not None else args.out_dir) / mode
            eval_jobs = [
                (
                    epoch,
                    ckpt_base / "checkpoints" / f"epoch_{epoch:04d}" / "policy.pt",
                    variant_dir / f"task_eval_epoch_{epoch:04d}",
                )
                for epoch in checkpoint_epochs
            ]
        else:
            ckpt_base = (args.checkpoint_root if args.eval_only and args.checkpoint_root is not None else args.out_dir) / mode
            eval_jobs = [(None, ckpt_base / "policy.pt", variant_dir / "task_eval")]

        for checkpoint_epoch, ckpt_path, eval_dir in eval_jobs:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint for mode {mode}: {ckpt_path}")
            metric = train_metric_for_epoch(train_metrics, checkpoint_epoch)
            eval_summary = evaluate_variant(
                calvin_code=args.calvin_code,
                data_root=args.data_root,
                eval_dir=eval_dir,
                ckpt_path=ckpt_path,
                task_to_id=task_to_id,
                text_to_task=text_to_task,
                demos=demos,
                num_sequences=args.num_sequences,
                ep_len=args.ep_len,
                device=args.device,
                eval_mode=args.eval_mode,
                eval_task_filter=set(args.eval_task_filter) if args.eval_task_filter else None,
                eval_scan_sequences=args.eval_scan_sequences,
            )
            row = {
                "mode": mode,
                "checkpoint_epoch": checkpoint_epoch,
                "train_val_mse": metric.get("val_mse"),
                "train_val_gripper_acc": metric.get("val_gripper_acc"),
                "avg_seq_len": eval_summary["avg_seq_len"],
                **{f"sr_{k}": v for k, v in eval_summary["chain_sr"].items()},
            }
            for task, info in eval_summary.get("task_info", {}).items():
                row[f"{task}_success"] = info["success"]
                row[f"{task}_total"] = info["total"]
                row[f"{task}_sr"] = info["sr"]
            rows.append(row)

    with open(args.out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=json_default)
    with open(args.out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
