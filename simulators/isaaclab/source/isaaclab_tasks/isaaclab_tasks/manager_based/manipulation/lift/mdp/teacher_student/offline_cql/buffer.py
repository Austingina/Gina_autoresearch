# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode-aware replay buffer with fixed-length LSTM sequence sampling."""

from __future__ import annotations

import random

import numpy as np
import torch

from .dataset import ActionStats


class EpisodeSequenceBuffer:
    """Sample consecutive in-episode sequences for LSTM CQL training."""

    def __init__(
        self,
        episodes: list[dict[str, np.ndarray]],
        *,
        action_stats: ActionStats,
        seq_len: int = 20,
        obs_mode: str = "vision",
    ):
        self.episodes = episodes
        self.action_stats = action_stats
        self.seq_len = seq_len
        self.obs_mode = obs_mode
        self.starts: list[tuple[int, int]] = []
        for ep_idx, episode in enumerate(episodes):
            ep_len = len(episode["actions"])
            if ep_len >= seq_len:
                for start in range(0, ep_len - seq_len + 1):
                    self.starts.append((ep_idx, start))

        if not self.starts:
            raise ValueError(f"No valid sequence starts found with seq_len={seq_len}.")

    def __len__(self) -> int:
        return len(self.starts)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        """Return (states, actions, rewards, next_states, dones) for LSTM training."""
        visual_batch = []
        proprio_batch = []
        actions_batch = []
        rewards_batch = []
        next_visual_batch = []
        next_proprio_batch = []
        dones_batch = []

        for _ in range(batch_size):
            ep_idx, start = random.choice(self.starts)
            episode = self.episodes[ep_idx]
            end = start + self.seq_len
            ep_len = len(episode["actions"])

            actions = self.action_stats.normalize(episode["actions"][start:end])
            rewards = episode["rewards"][start:end]
            dones = np.zeros(self.seq_len, dtype=np.float32)
            if end >= ep_len:
                dones[-1] = 1.0

            actions_batch.append(actions)
            rewards_batch.append(rewards)
            dones_batch.append(dones)

            if "visual" in episode:
                visual_batch.append(episode["visual"][start:end])
                next_frames = episode["visual"][start + 1 : end]
                if len(next_frames) < self.seq_len:
                    pad = np.repeat(next_frames[-1][None], self.seq_len - len(next_frames), axis=0)
                    next_frames = np.concatenate([next_frames, pad], axis=0)
                next_visual_batch.append(next_frames)

            if "proprio" in episode:
                proprio_batch.append(episode["proprio"][start:end])
                next_proprio = episode["proprio"][start + 1 : end]
                if len(next_proprio) < self.seq_len:
                    pad = np.repeat(next_proprio[-1][None], self.seq_len - len(next_proprio), axis=0)
                    next_proprio = np.concatenate([next_proprio, pad], axis=0)
                next_proprio_batch.append(next_proprio)

        actions_t = torch.as_tensor(np.stack(actions_batch), dtype=torch.float32, device=device)
        rewards_t = torch.as_tensor(np.stack(rewards_batch), dtype=torch.float32, device=device)
        dones_t = torch.as_tensor(np.stack(dones_batch), dtype=torch.float32, device=device)

        outputs: list[torch.Tensor] = []
        if visual_batch:
            outputs.extend(
                [
                    torch.as_tensor(np.stack(visual_batch), dtype=torch.float32, device=device),
                    torch.as_tensor(np.stack(next_visual_batch), dtype=torch.float32, device=device),
                ]
            )
        if proprio_batch:
            outputs.extend(
                [
                    torch.as_tensor(np.stack(proprio_batch), dtype=torch.float32, device=device),
                    torch.as_tensor(np.stack(next_proprio_batch), dtype=torch.float32, device=device),
                ]
            )

        return tuple(outputs + [actions_t, rewards_t, dones_t])
