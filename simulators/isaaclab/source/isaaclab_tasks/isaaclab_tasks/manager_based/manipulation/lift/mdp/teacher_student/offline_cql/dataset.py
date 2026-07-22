# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load lenient-teacher mask+depth HDF5 episodes for offline CQL."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

import cv2

from .rewards import (
    STUDENT_STATE_CUBE_SLICE,
    compute_offline_reach_rewards,
)


@dataclass
class ActionStats:
    """Per-dimension action min/max for normalization to [-1, 1]."""

    low: np.ndarray
    high: np.ndarray

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        denom = np.maximum(self.high - self.low, 1.0e-6)
        scaled = (actions - self.low) / denom
        return (2.0 * scaled - 1.0).astype(np.float32)

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(actions, dtype=np.float32) + 1.0) * 0.5
        return (scaled * (self.high - self.low) + self.low).astype(np.float32)


def _demo_sort_key(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def _read_dataset(group: h5py.Group, path: str) -> np.ndarray:
    node = group
    for part in path.split("/"):
        if part not in node:
            raise KeyError(f"Missing dataset path '{path}' in demo group.")
        node = node[part]
    return np.asarray(node)


def _optional_dataset(group: h5py.Group, path: str) -> np.ndarray | None:
    try:
        return _read_dataset(group, path)
    except KeyError:
        return None


def _stack_visual_obs(
    ee_mask: np.ndarray,
    ee_depth: np.ndarray,
    tp_mask: np.ndarray,
    tp_depth: np.ndarray,
) -> np.ndarray:
    """Return visual observation (T, C, H, W) with C=8."""
    if ee_depth.ndim == 4 and ee_depth.shape[-1] == 1:
        ee_depth = ee_depth[..., 0]
    if tp_depth.ndim == 4 and tp_depth.shape[-1] == 1:
        tp_depth = tp_depth[..., 0]

    ee_mask = ee_mask.astype(np.float32)
    tp_mask = tp_mask.astype(np.float32)
    ee_depth = ee_depth.astype(np.float32)[..., np.newaxis]
    tp_depth = tp_depth.astype(np.float32)[..., np.newaxis]

    visual = np.concatenate([ee_mask, ee_depth, tp_mask, tp_depth], axis=-1)  # (T, H, W, 8)
    return np.transpose(visual, (0, 3, 1, 2))


def _resize_visual(visual: np.ndarray, image_size: int) -> np.ndarray:
    """Resize (T, C, H, W) visual tensor; OpenCV supports at most 4 channels per call."""
    t, c, h, w = visual.shape
    if h == image_size and w == image_size:
        return visual
    resized = np.empty((t, c, image_size, image_size), dtype=np.float32)
    for ti in range(t):
        for ci in range(c):
            resized[ti, ci] = cv2.resize(visual[ti, ci], (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized


def resolve_maskdepth_dataset_paths(dataset_path: str) -> list[str]:
    """Resolve a single HDF5 file or a directory of ``*.hdf5`` files for loading."""
    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if path.is_file():
        if path.suffix.lower() not in (".hdf5", ".h5"):
            raise ValueError(f"Expected an HDF5 file, got: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Dataset file is empty: {path}")
        return [str(path)]

    if path.is_dir():
        files = sorted(
            candidate
            for candidate in path.glob("*.hdf5")
            if candidate.is_file() and candidate.stat().st_size > 0
        )
        if not files:
            raise FileNotFoundError(f"No non-empty *.hdf5 files found in directory: {path}")
        return [str(file_path) for file_path in files]

    raise ValueError(f"Dataset path must be a file or directory: {dataset_path}")


def _load_episodes_from_hdf5_file(
    dataset_path: str,
    *,
    obs_mode: str,
    image_size: int | None,
    source_label: str,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    episodes: list[dict[str, np.ndarray]] = []
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None

    with h5py.File(dataset_path, "r") as hdf5_file:
        if "data" not in hdf5_file:
            raise KeyError(f"Missing 'data' group in dataset file: {dataset_path}")
        demo_names = sorted(hdf5_file["data"].keys(), key=_demo_sort_key)

        for demo_name in demo_names:
            group = hdf5_file["data"][demo_name]
            actions = _read_dataset(group, "env_actions").astype(np.float32)
            if actions.ndim == 1:
                actions = actions[:, np.newaxis]
            progress_delta = _optional_dataset(group, "rewards/progress_delta")
            if progress_delta is not None:
                progress_delta = progress_delta.astype(np.float32).reshape(-1)

            ee_pose = _read_dataset(group, "ee_pose").astype(np.float32)
            student_state = _read_dataset(group, "obs/student_state").astype(np.float32)
            success_score = _optional_dataset(group, "rewards/success_score")
            if success_score is not None:
                success_score = success_score.astype(np.float32).reshape(-1)

            num_steps = min(len(actions), len(ee_pose), len(student_state))
            actions = actions[:num_steps]
            ee_pose = ee_pose[:num_steps]
            student_state = student_state[:num_steps]
            if success_score is not None:
                success_score = success_score[:num_steps]
            if progress_delta is not None:
                progress_delta = progress_delta[:num_steps]

            ee_pos = ee_pose[:, :3]
            cube_pos = student_state[:, STUDENT_STATE_CUBE_SLICE]
            reward_parts = compute_offline_reach_rewards(ee_pos, cube_pos, success_score=success_score)
            rewards = reward_parts["rewards"]

            if action_min is None:
                action_min = actions.min(axis=0)
                action_max = actions.max(axis=0)
            else:
                action_min = np.minimum(action_min, actions.min(axis=0))
                action_max = np.maximum(action_max, actions.max(axis=0))

            episode: dict[str, np.ndarray] = {
                "actions": actions,
                "rewards": rewards,
                "reach_reward": reward_parts["reach_reward"],
                "near_reward": reward_parts["near_reward"],
                "success_bonus": reward_parts["success_bonus"],
                "ee_cube_distance": reward_parts["ee_cube_distance"],
                "demo_name": np.array(f"{source_label}/{demo_name}"),
                "success": np.array(bool(group.attrs.get("success", True))),
                "source_file": np.array(source_label),
            }
            if progress_delta is not None:
                episode["progress_delta"] = progress_delta

            if obs_mode in ("vision", "vision_proprio"):
                ee_mask = _read_dataset(group, "obs/ee_mask")
                ee_depth = _read_dataset(group, "obs/ee_depth")
                tp_mask = _read_dataset(group, "obs/third_person_mask")
                tp_depth = _read_dataset(group, "obs/third_person_depth")
                visual = _stack_visual_obs(ee_mask, ee_depth, tp_mask, tp_depth)[:num_steps]
                if image_size is not None and visual.shape[-1] != image_size:
                    visual = _resize_visual(visual, image_size)
                episode["visual"] = visual

            if obs_mode in ("proprio", "vision_proprio"):
                proprio = _read_dataset(group, "obs/student_state").astype(np.float32)[:num_steps]
                episode["proprio"] = proprio

            episodes.append(episode)

    if action_min is None or action_max is None:
        raise ValueError(f"No demos found in dataset file: {dataset_path}")
    return episodes, action_min, action_max


def load_maskdepth_hdf5_episodes(
    dataset_path: str,
    *,
    obs_mode: str = "vision",
    image_size: int | None = 64,
    max_demos: int | None = None,
) -> tuple[list[dict[str, np.ndarray]], ActionStats]:
    """Load episodes from one HDF5 file or all ``*.hdf5`` files in a directory.

    ``dataset_path`` may be:
    - a single ``.hdf5`` file
    - a directory containing one or more non-empty ``*.hdf5`` files

    Each episode dict contains:
    - ``visual`` (optional): float32 (T, C, H, W)
    - ``proprio`` (optional): float32 (T, D)
    - ``actions``: float32 (T, A) from ``env_actions``
    - ``rewards``: float32 (T,) reach-distance scheme (recomputed at load; not ``progress_reward``)
    - ``reach_reward``, ``near_reward``, ``success_bonus``: float32 (T,) reward components
    - ``progress_delta`` (optional): logged diagnostic from HDF5 only
    """
    if obs_mode not in ("vision", "proprio", "vision_proprio"):
        raise ValueError(f"Unsupported obs_mode={obs_mode!r}")

    dataset_files = resolve_maskdepth_dataset_paths(dataset_path)
    episodes: list[dict[str, np.ndarray]] = []
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None

    for file_path in dataset_files:
        source_label = Path(file_path).stem
        file_episodes, file_min, file_max = _load_episodes_from_hdf5_file(
            file_path,
            obs_mode=obs_mode,
            image_size=image_size,
            source_label=source_label,
        )
        episodes.extend(file_episodes)
        if action_min is None:
            action_min = file_min
            action_max = file_max
        else:
            action_min = np.minimum(action_min, file_min)
            action_max = np.maximum(action_max, file_max)

    if max_demos is not None:
        episodes = episodes[:max_demos]

    assert action_min is not None and action_max is not None
    stats = ActionStats(low=action_min, high=action_max)
    return episodes, stats
