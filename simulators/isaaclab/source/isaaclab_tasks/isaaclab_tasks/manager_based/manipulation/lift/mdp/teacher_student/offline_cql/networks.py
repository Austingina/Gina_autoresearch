# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LSTM + CNN encoders, actor, and twin critics for offline CQL."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, *, layers: int = 2) -> nn.Sequential:
    modules: list[nn.Module] = []
    dim = in_dim
    for _ in range(layers - 1):
        modules.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
        dim = hidden_dim
    modules.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*modules)


class VisualCNNEncoder(nn.Module):
    """Encode each frame independently: (B, T, C, H, W) -> (B, T, feat_dim)."""

    def __init__(self, in_channels: int = 8, feat_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, feat_dim)

    def forward(self, visual_seq: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = visual_seq.shape
        x = visual_seq.reshape(b * t, c, h, w)
        x = self.net(x).flatten(1)
        x = self.proj(x)
        return x.reshape(b, t, -1)


class LSTMFeatureEncoder(nn.Module):
    """Fuse optional visual CNN features and proprio with an LSTM."""

    def __init__(
        self,
        *,
        visual_channels: int = 8,
        proprio_dim: int = 0,
        cnn_feat_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        use_visual: bool = True,
    ):
        super().__init__()
        self.use_visual = use_visual
        self.use_proprio = proprio_dim > 0
        self.lstm_hidden_dim = lstm_hidden_dim

        if use_visual:
            self.visual_cnn = VisualCNNEncoder(in_channels=visual_channels, feat_dim=cnn_feat_dim)
        else:
            self.visual_cnn = None

        input_dim = (cnn_feat_dim if use_visual else 0) + (proprio_dim if self.use_proprio else 0)
        if input_dim <= 0:
            raise ValueError("At least one of visual or proprio inputs must be enabled.")

        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)

    def forward(
        self,
        *,
        visual_seq: torch.Tensor | None = None,
        proprio_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = []
        if self.use_visual:
            if visual_seq is None:
                raise ValueError("visual_seq is required when use_visual=True.")
            feats.append(self.visual_cnn(visual_seq))
        if self.use_proprio:
            if proprio_seq is None:
                raise ValueError("proprio_seq is required when use_proprio=True.")
            feats.append(proprio_seq)
        x = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]
        encoded, _ = self.lstm(x)
        return encoded


class GaussianActor(nn.Module):
    """Gaussian policy with tanh squashing (actions in [-1, 1])."""

    def __init__(self, feat_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.backbone = _make_mlp(feat_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(features)
        mean = self.mean(h)
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mean)
        return mean, log_std

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(features)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1.0e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, features: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(features)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    """Twin-Q style critic head Q(s, a)."""

    def __init__(self, feat_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = _make_mlp(feat_dim + action_dim, hidden_dim, 1, layers=3)

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([features, actions], dim=-1)
        return self.net(x)


def flatten_time_features(encoded_seq: torch.Tensor) -> torch.Tensor:
    """(B, T, D) -> (B*T, D)."""
    b, t, d = encoded_seq.shape
    return encoded_seq.reshape(b * t, d)


def random_action_log_prob(action_dim: int, device: torch.device) -> float:
    return float(math.log(0.5**action_dim))
