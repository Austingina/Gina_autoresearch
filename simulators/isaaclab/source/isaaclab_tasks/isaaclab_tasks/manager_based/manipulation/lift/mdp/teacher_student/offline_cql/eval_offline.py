# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hold-out dataset evaluation (no Isaac Sim required)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from .agent import CQLSACAgentLSTM
from .dataset import ActionStats
from .policy import OfflineCQLPolicy


@dataclass
class OfflineEvalMetrics:
    action_mse: float
    action_mae: float
    mean_q1: float
    num_sequences: int


def split_train_holdout_episodes(
    episodes: list[dict[str, np.ndarray]],
    holdout_fraction: float,
    *,
    seed: int = 42,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
    if holdout_fraction <= 0.0 or len(episodes) < 2:
        return episodes, []
    holdout_count = max(1, int(round(len(episodes) * holdout_fraction)))
    holdout_count = min(holdout_count, len(episodes) - 1)
    indices = list(range(len(episodes)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    holdout_set = set(indices[:holdout_count])
    train = [episodes[i] for i in range(len(episodes)) if i not in holdout_set]
    holdout = [episodes[i] for i in holdout_set]
    return train, holdout


@torch.no_grad()
def evaluate_on_holdout(
    agent: CQLSACAgentLSTM,
    action_stats: ActionStats,
    holdout_episodes: list[dict[str, np.ndarray]],
    *,
    seq_len: int,
    obs_mode: str,
    device: torch.device,
    num_sequences: int = 64,
) -> OfflineEvalMetrics:
    """Evaluate deterministic policy against recorded ``env_actions`` on hold-out demos."""
    if not holdout_episodes:
        return OfflineEvalMetrics(action_mse=0.0, action_mae=0.0, mean_q1=0.0, num_sequences=0)

    policy = OfflineCQLPolicy(agent, action_stats, seq_len=seq_len, device=device)
    agent.encoder.eval()
    agent.actor.eval()
    agent.critic1.eval()
    agent.critic2.eval()

    starts: list[tuple[int, int]] = []
    for ep_idx, episode in enumerate(holdout_episodes):
        ep_len = len(episode["actions"])
        if ep_len >= seq_len:
            for start in range(0, ep_len - seq_len + 1):
                starts.append((ep_idx, start))

    if not starts:
        return OfflineEvalMetrics(action_mse=0.0, action_mae=0.0, mean_q1=0.0, num_sequences=0)

    sample_count = min(num_sequences, len(starts))
    sampled = random.sample(starts, sample_count)

    mse_sum = 0.0
    mae_sum = 0.0
    q_sum = 0.0

    for ep_idx, start in sampled:
        episode = holdout_episodes[ep_idx]
        end = start + seq_len
        policy.reset(1)

        target_norm = action_stats.normalize(episode["actions"][start:end])
        pred_norm_list = []

        for t in range(start, end):
            visual = episode["visual"][t : t + 1] if "visual" in episode else None
            proprio = episode["proprio"][t : t + 1] if "proprio" in episode else None
            pred = policy.act(visual=visual, proprio=proprio)
            pred_norm_list.append(action_stats.normalize(pred)[0])

        pred_norm = np.stack(pred_norm_list, axis=0)
        diff = pred_norm - target_norm
        mse_sum += float(np.mean(diff**2))
        mae_sum += float(np.mean(np.abs(diff)))

        visual_t = proprio_t = None
        if obs_mode in ("vision", "vision_proprio"):
            visual_t = torch.as_tensor(episode["visual"][start:end][None], dtype=torch.float32, device=device)
        if obs_mode in ("proprio", "vision_proprio"):
            proprio_t = torch.as_tensor(episode["proprio"][start:end][None], dtype=torch.float32, device=device)
        encoded = agent.encode(visual_seq=visual_t, proprio_seq=proprio_t)
        features = encoded.reshape(-1, encoded.shape[-1])
        actions_t = torch.as_tensor(pred_norm, dtype=torch.float32, device=device)
        q1 = agent.critic1(features, actions_t)
        q_sum += float(q1.mean().item())

    return OfflineEvalMetrics(
        action_mse=mse_sum / sample_count,
        action_mae=mae_sum / sample_count,
        mean_q1=q_sum / sample_count,
        num_sequences=sample_count,
    )
