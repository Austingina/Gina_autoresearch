# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load oracle HDF5 demonstrations into the teacher-student episode buffer."""

from __future__ import annotations

import os
import re

import h5py
import numpy as np
import torch

from .kl_trainer import TwistEpisodeReplayBuffer
from .teacher import GRIPPER_BINARY_CLOSE_THRESHOLD_M, GRIPPER_CLOSE_SNAP_M, GRIPPER_OPEN_MAX_M


def _demo_sort_key(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def _read_dataset(group: h5py.Group, path: str) -> np.ndarray:
    node = group
    for part in path.split("/"):
        node = node[part]
    return np.asarray(node)


def _try_read(group: h5py.Group, path: str) -> np.ndarray | None:
    try:
        return _read_dataset(group, path)
    except KeyError:
        return None


def _reconstruct_student_obs(
    ee_pose: np.ndarray,
    actions: np.ndarray,
    *,
    object_position: np.ndarray | None,
    target_object_position: np.ndarray | None,
    goal_position: np.ndarray | None,
    control_dt: float,
    open_max: float = 0.04,
) -> np.ndarray:
    """Build 10D student proprio obs: ee_pos(3), gripper(1), cube_pos(3), lift_goal(3)."""
    num_steps = ee_pose.shape[0]
    ee_pos = ee_pose[:, :3].astype(np.float32)

    if object_position is not None:
        cube_pos = object_position[:, :3].astype(np.float32)
    elif goal_position is not None:
        cube_pos = np.repeat(goal_position[0:1, :3], num_steps, axis=0).astype(np.float32)
    else:
        cube_pos = np.full((num_steps, 3), [0.5, 0.0, 0.055], dtype=np.float32)

    if target_object_position is not None:
        lift_goal = target_object_position[:, :3].astype(np.float32)
    elif goal_position is not None:
        lift_goal = np.repeat(goal_position[0:1, :3], num_steps, axis=0).astype(np.float32)
    else:
        lift_goal = cube_pos.copy()

    opening = np.full((num_steps, 1), open_max, dtype=np.float32)
    if actions.shape[-1] >= 7:
        current = open_max
        for t in range(num_steps):
            opening[t, 0] = current
            current = float(np.clip(current + actions[t, 6] * control_dt, 0.0, open_max))

    return np.concatenate([ee_pos, opening, cube_pos, lift_goal], axis=-1)


def _integrate_gripper_opening(
    actions: np.ndarray,
    *,
    control_dt: float,
    open_max: float = GRIPPER_OPEN_MAX_M,
) -> np.ndarray:
    """Integrate recorded gripper opening-rate actions into finger opening (m)."""
    num_steps = actions.shape[0]
    opening = np.full((num_steps,), open_max, dtype=np.float32)
    if actions.shape[-1] < 7:
        return opening

    current = open_max
    for t in range(num_steps):
        opening[t] = current
        current = float(np.clip(current + actions[t, 6] * control_dt, 0.0, open_max))
    return opening


def _opening_to_binary_gripper_label_np(
    opening: np.ndarray,
    *,
    open_max: float = GRIPPER_OPEN_MAX_M,
    close_snap_m: float = GRIPPER_CLOSE_SNAP_M,
    close_threshold_m: float = GRIPPER_BINARY_CLOSE_THRESHOLD_M,
) -> np.ndarray:
    """Map finger opening (m) to binary teacher labels (+1 open, -1 close)."""
    opening = np.clip(opening, 0.0, open_max)
    close = opening <= close_snap_m
    fully_open = opening >= open_max - 1e-3
    return np.where(
        close,
        -1.0,
        np.where(fully_open, 1.0, np.where(opening < close_threshold_m, -1.0, 1.0)),
    ).astype(np.float32)


def load_oracle_demos_into_buffer(
    hdf5_path: str,
    buffer: TwistEpisodeReplayBuffer,
    *,
    device: str | torch.device,
    max_demos: int | None = None,
    control_dt: float = 0.02,
    binary_gripper: bool = False,
) -> int:
    """Load episodes from an oracle goal-conditioned HDF5 dataset into the replay buffer."""
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"Oracle demo file not found: {hdf5_path}")

    loaded = 0
    with h5py.File(hdf5_path, "r") as hdf5_file:
        if "data" not in hdf5_file:
            raise KeyError(f"Missing 'data' group in {hdf5_path}")
        demo_names = sorted(hdf5_file["data"].keys(), key=_demo_sort_key)
        if max_demos is not None:
            demo_names = demo_names[:max_demos]

        for demo_name in demo_names:
            demo_group = hdf5_file["data"][demo_name]
            actions = _read_dataset(demo_group, "actions").astype(np.float32)
            ee_pose = _read_dataset(demo_group, "ee_pose").astype(np.float32)
            if actions.shape[0] != ee_pose.shape[0]:
                continue

            obs = _reconstruct_student_obs(
                ee_pose,
                actions,
                object_position=_try_read(demo_group, "obs/object_position"),
                target_object_position=_try_read(demo_group, "obs/target_object_position"),
                goal_position=_try_read(demo_group, "goals/position"),
                control_dt=control_dt,
            )
            teacher = actions.astype(np.float32)
            if teacher.shape[-1] < 7:
                pad = np.zeros((teacher.shape[0], 7 - teacher.shape[-1]), dtype=np.float32)
                teacher = np.concatenate([teacher, pad], axis=-1)
            if binary_gripper:
                integrated_opening = _integrate_gripper_opening(teacher, control_dt=control_dt)
                teacher[:, 6] = _opening_to_binary_gripper_label_np(integrated_opening)

            obs_t = torch.as_tensor(obs, device=device)
            teacher_t = torch.as_tensor(teacher[:, :7], device=device)
            add_kwargs: dict = {}
            if hasattr(buffer, "success_reserve_ratio"):
                add_kwargs["teacher_success"] = True
            buffer.add_episode(obs_t, teacher_t, **add_kwargs)
            loaded += 1

    return loaded
