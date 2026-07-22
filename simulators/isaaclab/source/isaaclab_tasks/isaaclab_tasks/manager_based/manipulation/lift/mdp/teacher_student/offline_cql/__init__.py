# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline CQL-SAC with LSTM for lift-cube mask+depth datasets."""

from .agent import CQLSACAgentLSTM
from .buffer import EpisodeSequenceBuffer
from .checkpoint import build_agent_from_checkpoint, load_checkpoint, pack_checkpoint
from .dataset import ActionStats, load_maskdepth_hdf5_episodes, resolve_maskdepth_dataset_paths
from .eval_offline import OfflineEvalMetrics, evaluate_on_holdout, split_train_holdout_episodes
from .policy import OfflineCQLPolicy
from .rewards import (
    NEAR_DISTANCE_DECAY,
    NEAR_REWARD_SCALE,
    REACH_REWARD_SCALE,
    SUCCESS_BONUS,
    compute_offline_reach_rewards,
    format_reward_summary,
    summarize_dataset_rewards,
)

__all__ = [
    "ActionStats",
    "CQLSACAgentLSTM",
    "EpisodeSequenceBuffer",
    "OfflineCQLPolicy",
    "OfflineEvalMetrics",
    "build_agent_from_checkpoint",
    "evaluate_on_holdout",
    "load_checkpoint",
    "pack_checkpoint",
    "load_maskdepth_hdf5_episodes",
    "resolve_maskdepth_dataset_paths",
    "split_train_holdout_episodes",
    "NEAR_DISTANCE_DECAY",
    "NEAR_REWARD_SCALE",
    "REACH_REWARD_SCALE",
    "SUCCESS_BONUS",
    "compute_offline_reach_rewards",
    "format_reward_summary",
    "summarize_dataset_rewards",
]
