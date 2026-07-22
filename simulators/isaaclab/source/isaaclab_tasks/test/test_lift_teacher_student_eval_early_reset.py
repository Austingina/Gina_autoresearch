# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for eval early-reset env_id conversion (no Isaac Sim required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_EVAL_UTILS_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_tasks/manager_based/manipulation/lift/mdp/teacher_student/eval_utils.py"
)
_spec = importlib.util.spec_from_file_location("lift_eval_utils", _EVAL_UTILS_PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
early_reset_env_ids = _mod.early_reset_env_ids


def _simulate_camera_pose_reset(all_poses: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    """Mimic Camera._update_poses indexing: env_ids must be integer indices."""
    if env_ids.dtype == torch.bool:
        selected_count = int(env_ids.sum().item())
        # Isaac assigns full-batch poses into bool-selected rows -> shape mismatch.
        return all_poses[env_ids]
    return all_poses[env_ids]


def test_early_reset_env_ids_empty():
    ids = early_reset_env_ids(None, torch.zeros(4, dtype=torch.bool))
    assert ids.numel() == 0
    assert ids.dtype == torch.long


def test_early_reset_env_ids_combines_masks():
    num_envs = 32
    early_mask = torch.zeros(num_envs, dtype=torch.bool)
    success_early = torch.zeros(num_envs, dtype=torch.bool)
    early_mask[[2, 5, 17]] = True
    success_early[[5, 9, 11, 19, 21, 24, 27, 28, 30]] = True

    ids = early_reset_env_ids(early_mask, success_early, max_env_id=16)
    expected = torch.tensor([2, 5, 9, 11], dtype=torch.long)
    assert torch.equal(ids, expected)


def test_early_reset_single_env_squeezes_scalar():
    mask = torch.zeros(4, dtype=torch.bool)
    mask[1] = True
    ids = early_reset_env_ids(mask)
    assert ids.shape == (1,)
    assert ids.item() == 1


def test_bool_mask_reproduces_camera_shape_mismatch():
    """Old bug: bool mask with partial True vs full pose batch."""
    num_envs = 32
    success_early = torch.zeros(num_envs, dtype=torch.bool)
    success_early[:10] = True
    all_poses = torch.randn(num_envs, 3)

    bool_mask = success_early
    selected = all_poses[bool_mask]
    assert selected.shape == (10, 3)

    # Assignment pattern that fails inside Isaac camera reset when RHS is unsliced.
    target = torch.zeros(num_envs, 3)
    try:
        target[bool_mask] = all_poses  # [10,3] <- [32,3]
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "bool-index assignment with full pose batch should mismatch"


def test_integer_env_ids_compatible_with_camera_pose_reset():
    num_envs = 32
    eval_envs = 16
    success_early = torch.zeros(num_envs, dtype=torch.bool)
    success_early[[1, 3, 5, 7, 9, 11, 13, 15, 20, 25]] = True

    env_ids = early_reset_env_ids(success_early, max_env_id=eval_envs)
    assert env_ids.tolist() == [1, 3, 5, 7, 9, 11, 13, 15]

    all_poses = torch.randn(num_envs, 3)
    updated = _simulate_camera_pose_reset(all_poses, env_ids)
    assert updated.shape == (len(env_ids), 3)

    target = torch.zeros(num_envs, 3)
    target[env_ids] = all_poses[env_ids]
    assert target[env_ids].shape == (len(env_ids), 3)


def test_eval_loop_dummy_state_machine():
    """Minimal fake eval step: done handling uses integer reset ids only."""
    num_envs = 32
    eval_envs = 16
    episodes_done = 0
    num_episodes = 50

    terminated = torch.zeros(num_envs, dtype=torch.bool)
    truncated = torch.zeros(num_envs, dtype=torch.bool)
    success_early = torch.zeros(num_envs, dtype=torch.bool)
    success_early[4] = True
    success_early[18] = True  # outside eval_envs

    early_mask = torch.zeros(num_envs, dtype=torch.bool)
    early_mask[7] = True

    dones = terminated | truncated | success_early | early_mask
    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
    manual_reset_ids = early_reset_env_ids(early_mask, success_early, max_env_id=eval_envs)

    for env_id in done_ids.tolist():
        if env_id >= eval_envs:
            continue
        episodes_done += 1

    assert episodes_done == 2  # env 4 and 7 only
    assert manual_reset_ids.tolist() == [4, 7]
    assert all(int(i) < eval_envs for i in manual_reset_ids.tolist())
    assert episodes_done < num_episodes
