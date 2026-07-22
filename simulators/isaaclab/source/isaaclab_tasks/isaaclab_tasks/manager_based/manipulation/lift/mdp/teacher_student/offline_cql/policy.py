# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout policy wrapper with LSTM history for offline CQL."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .agent import CQLSACAgentLSTM
from .dataset import ActionStats


class OfflineCQLPolicy:
    """Deterministic policy with fixed-length LSTM observation history."""

    def __init__(
        self,
        agent: CQLSACAgentLSTM,
        action_stats: ActionStats,
        *,
        seq_len: int,
        device: torch.device,
    ):
        self.agent = agent
        self.action_stats = action_stats
        self.seq_len = seq_len
        self.device = device
        self.obs_mode = agent.obs_mode
        self._visual_hist: list[deque[np.ndarray]] = []
        self._proprio_hist: list[deque[np.ndarray]] = []

    def reset(self, num_envs: int) -> None:
        self._visual_hist = [deque(maxlen=self.seq_len) for _ in range(num_envs)]
        self._proprio_hist = [deque(maxlen=self.seq_len) for _ in range(num_envs)]

    def reset_envs(self, env_ids: torch.Tensor | list[int]) -> None:
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        for env_id in env_ids:
            self._visual_hist[env_id].clear()
            self._proprio_hist[env_id].clear()

    def _pad_sequence(self, items: deque[np.ndarray]) -> np.ndarray:
        if len(items) == 0:
            raise RuntimeError("Cannot act before any observation is appended.")
        seq = list(items)
        while len(seq) < self.seq_len:
            seq.insert(0, seq[0])
        return np.stack(seq[-self.seq_len :], axis=0)

    @torch.no_grad()
    def act(
        self,
        *,
        visual: np.ndarray | None = None,
        proprio: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute env actions for a batch of environments.

        Args:
            visual: (N, C, H, W) optional visual observation for all envs.
            proprio: (N, D) optional proprio observation for all envs.

        Returns:
            Denormalized env actions with shape (N, action_dim).
        """
        num_envs = visual.shape[0] if visual is not None else proprio.shape[0]
        if not self._visual_hist:
            self.reset(num_envs)

        visual_seq_batch = []
        proprio_seq_batch = []
        for env_id in range(num_envs):
            if visual is not None:
                self._visual_hist[env_id].append(visual[env_id])
                visual_seq_batch.append(self._pad_sequence(self._visual_hist[env_id]))
            if proprio is not None:
                self._proprio_hist[env_id].append(proprio[env_id])
                proprio_seq_batch.append(self._pad_sequence(self._proprio_hist[env_id]))

        visual_t = None
        proprio_t = None
        if visual_seq_batch:
            visual_t = torch.as_tensor(np.stack(visual_seq_batch), dtype=torch.float32, device=self.device)
        if proprio_seq_batch:
            proprio_t = torch.as_tensor(np.stack(proprio_seq_batch), dtype=torch.float32, device=self.device)

        encoded = self.agent.encode(visual_seq=visual_t, proprio_seq=proprio_t)
        features = encoded[:, -1, :]
        action_norm = self.agent.actor.deterministic(features)
        action_norm_np = action_norm.detach().cpu().numpy()
        return self.action_stats.denormalize(action_norm_np)
