# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for offline CQL reach-distance rewards."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_OFFLINE_CQL_ROOT = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_tasks/manager_based/manipulation/lift/mdp/teacher_student"
)
if str(_OFFLINE_CQL_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFFLINE_CQL_ROOT))

from offline_cql.rewards import (  # noqa: E402
    NEAR_REWARD_SCALE,
    REACH_REWARD_SCALE,
    SUCCESS_BONUS,
    compute_offline_reach_rewards,
)


def test_reach_reward_scales_distance_delta() -> None:
    ee_pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    cube_pos = np.array([[0.05, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32)
    parts = compute_offline_reach_rewards(ee_pos, cube_pos)

    assert parts["reach_reward"][0] == pytest.approx(0.03 * REACH_REWARD_SCALE)
    assert parts["reach_delta"][0] == pytest.approx(0.03)


def test_near_reward_sensitive_close_to_cube() -> None:
    close = compute_offline_reach_rewards(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.05, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32),
    )
    far = compute_offline_reach_rewards(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.30, 0.0, 0.0], [0.27, 0.0, 0.0]], dtype=np.float32),
    )

    assert close["reach_reward"][0] == pytest.approx(far["reach_reward"][0])
    assert close["near_reward"][0] > far["near_reward"][0] * 100.0


def test_success_bonus_only_on_success_transition() -> None:
    ee_pos = np.zeros((3, 3), dtype=np.float32)
    cube_pos = np.full((3, 3), 0.1, dtype=np.float32)
    success_score = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    parts = compute_offline_reach_rewards(ee_pos, cube_pos, success_score=success_score)

    assert parts["success_bonus"][0] == 0.0
    assert parts["success_bonus"][1] == pytest.approx(SUCCESS_BONUS)
    assert parts["success_bonus"][2] == pytest.approx(SUCCESS_BONUS)


def test_reward_total_is_sum_of_components() -> None:
    ee_pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    cube_pos = np.array([[0.05, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32)
    parts = compute_offline_reach_rewards(ee_pos, cube_pos, success_score=np.zeros(2, dtype=np.float32))

    expected_near = np.exp(-30.0 * 0.02) * NEAR_REWARD_SCALE
    assert parts["near_reward"][0] == pytest.approx(expected_near)
    assert parts["rewards"][0] == pytest.approx(parts["reach_reward"][0] + parts["near_reward"][0])
