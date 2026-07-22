# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student policy networks."""

from __future__ import annotations

from typing import TypeAlias

import torch
import torch.nn as nn
from torch.distributions import Normal

RNNState: TypeAlias = tuple[torch.Tensor, torch.Tensor]


def init_rnn_state(batch_size: int, hidden_dim: int, device: str | torch.device) -> RNNState:
    """Create a zero LSTM hidden state ``(h, c)`` with shape ``(1, batch, hidden_dim)``."""
    h = torch.zeros(1, batch_size, hidden_dim, device=device)
    c = torch.zeros(1, batch_size, hidden_dim, device=device)
    return (h, c)


def reset_rnn_state(state: RNNState, env_ids: torch.Tensor | slice) -> None:
    """Zero LSTM state for selected batch indices."""
    state[0][:, env_ids, :] = 0.0
    state[1][:, env_ids, :] = 0.0


def detach_rnn_state(state: RNNState) -> RNNState:
    """Detach LSTM state after truncated BPTT backward."""
    return (state[0].detach(), state[1].detach())


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class StudentTwistPolicy(nn.Module):
    """Proprio student policy for 7D EE command imitation (6D twist velocity + gripper opening rate)."""

    def __init__(
        self,
        obs_dim: int = 10,
        action_dim: int = 7,
        hidden_dim: int = 128,
        rnn_hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.encoder = nn.LSTM(obs_dim, rnn_hidden_dim, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(rnn_hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def init_hidden(self, batch_size: int, device: str | torch.device) -> RNNState:
        return init_rnn_state(batch_size, self.rnn_hidden_dim, device)

    def reset_hidden(self, hidden: RNNState, env_ids: torch.Tensor | slice) -> None:
        reset_rnn_state(hidden, env_ids)

    def forward_rnn(
        self,
        obs_seq: torch.Tensor,
        hidden: RNNState | None = None,
    ) -> tuple[torch.Tensor, RNNState]:
        """Encode observation sequences. obs_seq shape: (batch, time, obs_dim)."""
        encoded, hidden_out = self.encoder(obs_seq, hidden)
        features = encoded[:, -1, :]
        return features, hidden_out

    def policy_dist(
        self,
        obs_seq: torch.Tensor,
        hidden: RNNState | None = None,
    ) -> tuple[Normal, RNNState]:
        """Return the student action distribution and updated RNN hidden state."""
        features, hidden_out = self.forward_rnn(obs_seq, hidden)
        mean = self.actor(features)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), hidden_out

    def act(
        self,
        obs_seq: torch.Tensor,
        hidden: RNNState | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, RNNState, torch.Tensor]:
        """Sample an action for environment rollout."""
        dist, hidden_out = self.policy_dist(obs_seq, hidden)
        action = dist.mean if deterministic else dist.rsample()
        return action, hidden_out, dist.mean


class StudentTwistPolicyBinaryGripper(nn.Module):
    """Dual-head student: shared LSTM representation, separate arm and gripper heads.

    Training uses decoupled losses:
    - Arm KL backpropagates through the encoder + arm head.
    - Gripper BCE backpropagates through the gripper head only (encoder frozen).
    """

    def __init__(
        self,
        obs_dim: int = 10,
        hidden_dim: int = 128,
        rnn_hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.arm_action_dim = 6
        self.action_dim = 7
        self.rnn_hidden_dim = rnn_hidden_dim

        self.encoder = nn.LSTM(obs_dim, rnn_hidden_dim, batch_first=True)
        self.arm_head = _make_mlp(rnn_hidden_dim, hidden_dim, self.arm_action_dim)
        self.gripper_head = _make_mlp(rnn_hidden_dim, hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(self.arm_action_dim))

    def init_hidden(self, batch_size: int, device: str | torch.device) -> RNNState:
        return init_rnn_state(batch_size, self.rnn_hidden_dim, device)

    def reset_hidden(self, hidden: RNNState, env_ids: torch.Tensor | slice) -> None:
        reset_rnn_state(hidden, env_ids)

    def forward_rnn(
        self,
        obs_seq: torch.Tensor,
        hidden: RNNState | None = None,
    ) -> tuple[torch.Tensor, RNNState]:
        encoded, hidden_out = self.encoder(obs_seq, hidden)
        features = encoded[:, -1, :]
        return features, hidden_out

    def encode_step(
        self,
        obs_seq: torch.Tensor,
        hidden: RNNState | None = None,
    ) -> tuple[torch.Tensor, RNNState]:
        """Run one RNN step and return the latest spatiotemporal features."""
        return self.forward_rnn(obs_seq, hidden)

    def arm_dist_from_features(self, features: torch.Tensor) -> Normal:
        arm_mean = self.arm_head(features)
        arm_std = self.log_std.exp().expand_as(arm_mean)
        return Normal(arm_mean, arm_std)

    def gripper_logit_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.gripper_head(features)

    def policy_dist(
        self,
        obs_seq: torch.Tensor,
        hidden: RNNState | None = None,
    ) -> tuple[Normal, torch.Tensor, RNNState]:
        """Return arm Gaussian, gripper logit (B, 1), and updated RNN hidden state."""
        features, hidden_out = self.encode_step(obs_seq, hidden)
        arm_dist = self.arm_dist_from_features(features)
        gripper_logit = self.gripper_logit_from_features(features)
        return arm_dist, gripper_logit, hidden_out

    def arm_parameters(self) -> list[torch.nn.Parameter]:
        """Encoder + arm head (+ arm log_std). Updated by arm KL."""
        return list(self.encoder.parameters()) + list(self.arm_head.parameters()) + [self.log_std]

    def gripper_parameters(self) -> list[torch.nn.Parameter]:
        """Gripper head only. Updated by gripper BCE."""
        return list(self.gripper_head.parameters())

    @staticmethod
    def _logit_to_binary(gripper_logit: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
        gripper_prob = torch.sigmoid(gripper_logit)
        if deterministic:
            open_mask = gripper_prob >= 0.5
        else:
            open_mask = torch.bernoulli(gripper_prob) > 0.5
        return torch.where(open_mask, torch.ones_like(gripper_prob), -torch.ones_like(gripper_prob))

    def act(
        self,
        obs_seq: torch.Tensor,
        hidden: RNNState | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, RNNState, torch.Tensor]:
        """Sample a 7D action: 6D arm twist + binary gripper (+1 open, -1 close)."""
        arm_dist, gripper_logit, hidden_out = self.policy_dist(obs_seq, hidden)
        arm_action = arm_dist.mean if deterministic else arm_dist.rsample()
        gripper_binary = self._logit_to_binary(gripper_logit, deterministic=deterministic)
        action = torch.cat([arm_action, gripper_binary], dim=-1)
        return action, hidden_out, action

    @classmethod
    def load_encoder_and_arm_from_continuous(
        cls,
        policy: StudentTwistPolicyBinaryGripper,
        continuous_state: dict[str, torch.Tensor],
    ) -> None:
        """Initialize arm head from a ``StudentTwistPolicy`` checkpoint (encoder skipped if GRU)."""
        encoder_state = {
            k.removeprefix("encoder."): v for k, v in continuous_state.items() if k.startswith("encoder.")
        }
        try:
            policy.encoder.load_state_dict(encoder_state)
        except RuntimeError as exc:
            print(f"[WARN] Could not load encoder weights (architecture may differ): {exc}")

        for layer_idx in (0, 2):
            policy.arm_head[layer_idx].weight.data.copy_(continuous_state[f"actor.{layer_idx}.weight"])
            policy.arm_head[layer_idx].bias.data.copy_(continuous_state[f"actor.{layer_idx}.bias"])

        policy.arm_head[4].weight.data.copy_(continuous_state["actor.4.weight"][: policy.arm_action_dim])
        policy.arm_head[4].bias.data.copy_(continuous_state["actor.4.bias"][: policy.arm_action_dim])
        policy.log_std.data.copy_(continuous_state["log_std"][: policy.arm_action_dim])

    @classmethod
    def load_from_checkpoint(
        cls,
        policy: StudentTwistPolicyBinaryGripper,
        checkpoint: dict,
        *,
        obs_dim: int,
    ) -> None:
        """Load weights from a training checkpoint dict."""
        ckpt_obs_dim = int(checkpoint.get("obs_dim", obs_dim))
        if ckpt_obs_dim != obs_dim:
            raise ValueError(f"Checkpoint obs_dim={ckpt_obs_dim} does not match env obs_dim={obs_dim}")

        policy_type = checkpoint.get("policy_type", "binary_gripper_dual_head")
        state = checkpoint["model"]

        if policy_type in ("continuous", "StudentTwistPolicy"):
            cls.load_encoder_and_arm_from_continuous(policy, state)
            return

        if policy_type in ("binary_gripper_dual_head", "binary_gripper_dual_head_dagger"):
            try:
                policy.load_state_dict(state)
            except RuntimeError as exc:
                print(
                    "[WARN] Full checkpoint load failed (GRU->LSTM migration?): "
                    f"{exc}. Loading arm/gripper heads only."
                )
                cls._load_heads_only(policy, state)
            return

        if policy_type in ("binary_gripper",):
            cls._load_legacy_binary_gripper(policy, state)
            return

        raise ValueError(f"Unsupported checkpoint policy_type={policy_type!r}")

    @classmethod
    def _load_heads_only(cls, policy: StudentTwistPolicyBinaryGripper, state: dict[str, torch.Tensor]) -> None:
        """Load arm/gripper heads when encoder architecture changed."""
        head_state = {k: v for k, v in state.items() if not k.startswith("encoder.")}
        policy.load_state_dict(head_state, strict=False)

    @classmethod
    def _load_legacy_binary_gripper(
        cls,
        policy: StudentTwistPolicyBinaryGripper,
        state: dict[str, torch.Tensor],
    ) -> None:
        """Legacy single-actor checkpoint: split arm layers and init gripper head from arm trunk."""
        try:
            encoder_state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
            policy.encoder.load_state_dict(encoder_state)
        except RuntimeError as exc:
            print(f"[WARN] Could not load legacy encoder: {exc}")
        for layer_idx in (0, 2):
            policy.arm_head[layer_idx].weight.data.copy_(state[f"actor.{layer_idx}.weight"])
            policy.arm_head[layer_idx].bias.data.copy_(state[f"actor.{layer_idx}.bias"])
            policy.gripper_head[layer_idx].weight.data.copy_(state[f"actor.{layer_idx}.weight"])
            policy.gripper_head[layer_idx].bias.data.copy_(state[f"actor.{layer_idx}.bias"])
        policy.arm_head[4].weight.data.copy_(state["actor.4.weight"][: policy.arm_action_dim])
        policy.arm_head[4].bias.data.copy_(state["actor.4.bias"][: policy.arm_action_dim])
        policy.gripper_head[4].weight.data.copy_(state["actor.4.weight"][policy.arm_action_dim :])
        policy.gripper_head[4].bias.data.copy_(state["actor.4.bias"][policy.arm_action_dim :])
        policy.log_std.data.copy_(state["log_std"])
