# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision dual-head student policy for DAgger (mask + depth, optional proprio)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .networks import RNNState, _make_mlp, init_rnn_state, reset_rnn_state
from .offline_cql.networks import VisualCNNEncoder


class StudentTwistPolicyBinaryGripperVision(nn.Module):
    """Dual-head student with CNN + LSTM encoder (matches offline CQL visual stack)."""

    def __init__(
        self,
        *,
        visual_channels: int = 8,
        proprio_dim: int = 0,
        cnn_feat_dim: int = 128,
        hidden_dim: int = 128,
        rnn_hidden_dim: int = 128,
        enable_aux_delta_head: bool = True,
    ):
        super().__init__()
        self.visual_channels = visual_channels
        self.proprio_dim = proprio_dim
        self.cnn_feat_dim = cnn_feat_dim
        self.arm_action_dim = 6
        self.action_dim = 7
        self.rnn_hidden_dim = rnn_hidden_dim
        self.enable_aux_delta_head = bool(enable_aux_delta_head)

        self.visual_cnn = VisualCNNEncoder(in_channels=visual_channels, feat_dim=cnn_feat_dim)
        lstm_in = cnn_feat_dim + proprio_dim
        self.encoder = nn.LSTM(lstm_in, rnn_hidden_dim, batch_first=True)
        self.arm_head = _make_mlp(rnn_hidden_dim, hidden_dim, self.arm_action_dim)
        self.gripper_head = _make_mlp(rnn_hidden_dim, hidden_dim, 1)
        self.aux_delta_head = (
            _make_mlp(cnn_feat_dim, hidden_dim, 3) if self.enable_aux_delta_head else None
        )
        self.log_std = nn.Parameter(torch.zeros(self.arm_action_dim))

    def init_hidden(self, batch_size: int, device: str | torch.device) -> RNNState:
        return init_rnn_state(batch_size, self.rnn_hidden_dim, device)

    def reset_hidden(self, hidden: RNNState, env_ids: torch.Tensor | slice) -> None:
        reset_rnn_state(hidden, env_ids)

    def visual_features(self, visual_seq: torch.Tensor) -> torch.Tensor:
        """Encode visual frames independently: (B,T,C,H,W) -> (B,T,cnn_feat_dim)."""
        return self.visual_cnn(visual_seq)

    def encode_from_visual_features(
        self,
        cnn_feat: torch.Tensor,
        hidden: RNNState | None,
        proprio_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RNNState]:
        if proprio_seq is not None:
            if self.proprio_dim <= 0:
                raise ValueError("proprio_seq provided but proprio_dim=0.")
            x = torch.cat([cnn_feat, proprio_seq], dim=-1)
        else:
            if self.proprio_dim > 0:
                raise ValueError("proprio_dim>0 but proprio_seq is None.")
            x = cnn_feat
        encoded, hidden_out = self.encoder(x, hidden)
        features = encoded[:, -1, :]
        return features, hidden_out

    def _encode_sequence(
        self,
        visual_seq: torch.Tensor,
        hidden: RNNState | None,
        proprio_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RNNState]:
        cnn_feat = self.visual_features(visual_seq)
        return self.encode_from_visual_features(cnn_feat, hidden, proprio_seq)

    def encode_step(
        self,
        visual_seq: torch.Tensor,
        hidden: RNNState | None = None,
        proprio_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RNNState]:
        return self._encode_sequence(visual_seq, hidden, proprio_seq)

    def arm_dist_from_features(self, features: torch.Tensor) -> Normal:
        arm_mean = self.arm_head(features)
        arm_std = self.log_std.exp().expand_as(arm_mean)
        return Normal(arm_mean, arm_std)

    def gripper_logit_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.gripper_head(features)

    def aux_delta_from_cnn_features(self, cnn_feat: torch.Tensor) -> torch.Tensor:
        """Predict cube position relative to EE from CNN features, in robot-root frame."""
        if self.aux_delta_head is None:
            raise RuntimeError("Auxiliary cube-EE head is disabled for this policy.")
        return self.aux_delta_head(cnn_feat)

    def aux_delta_from_visual(self, visual_seq: torch.Tensor) -> torch.Tensor:
        return self.aux_delta_from_cnn_features(self.visual_features(visual_seq))

    def policy_dist(
        self,
        visual_seq: torch.Tensor,
        hidden: RNNState | None = None,
        proprio_seq: torch.Tensor | None = None,
    ) -> tuple[Normal, torch.Tensor, RNNState]:
        features, hidden_out = self.encode_step(visual_seq, hidden, proprio_seq)
        return self.arm_dist_from_features(features), self.gripper_logit_from_features(features), hidden_out

    def arm_parameters(self) -> list[torch.nn.Parameter]:
        params = list(self.visual_cnn.parameters()) + list(self.encoder.parameters())
        params += list(self.arm_head.parameters()) + [self.log_std]
        return params

    def gripper_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.gripper_head.parameters())

    def aux_parameters(self) -> list[torch.nn.Parameter]:
        if self.aux_delta_head is None:
            return []
        return list(self.visual_cnn.parameters()) + list(self.aux_delta_head.parameters())

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
        visual_seq: torch.Tensor,
        hidden: RNNState | None = None,
        proprio_seq: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, RNNState, torch.Tensor]:
        arm_dist, gripper_logit, hidden_out = self.policy_dist(visual_seq, hidden, proprio_seq)
        arm_action = arm_dist.mean if deterministic else arm_dist.rsample()
        gripper_binary = self._logit_to_binary(gripper_logit, deterministic=deterministic)
        action = torch.cat([arm_action, gripper_binary], dim=-1)
        return action, hidden_out, action

    @staticmethod
    def resolve_visual_channels_from_checkpoint(checkpoint: dict, *, default: int = 8) -> int:
        """Read visual_channels from checkpoint meta, falling back to first conv weights."""
        declared = int(checkpoint.get("visual_channels", default))
        state = checkpoint.get("model")
        if isinstance(state, dict):
            weight = state.get("visual_cnn.net.0.weight")
            if weight is not None:
                inferred = int(weight.shape[1])
                if inferred != declared:
                    print(
                        f"[WARN] checkpoint visual_channels={declared} "
                        f"but weights expect {inferred}; using {inferred}"
                    )
                return inferred
        return declared

    @staticmethod
    def resolve_aux_delta_head_from_checkpoint(checkpoint: dict, *, default: bool = True) -> bool:
        """Infer whether a checkpoint contains the optional cube-EE auxiliary head."""
        if "aux_delta_head_enabled" in checkpoint:
            return bool(checkpoint["aux_delta_head_enabled"])
        state = checkpoint.get("model")
        if isinstance(state, dict):
            return any(key.startswith("aux_delta_head.") for key in state)
        return bool(default)

    @classmethod
    def load_from_checkpoint(
        cls,
        policy: StudentTwistPolicyBinaryGripperVision,
        checkpoint: dict,
        *,
        visual_channels: int | None = None,
        proprio_dim: int | None = None,
    ) -> None:
        ckpt_visual = cls.resolve_visual_channels_from_checkpoint(
            checkpoint, default=int(visual_channels if visual_channels is not None else 8)
        )
        ckpt_proprio = int(checkpoint.get("proprio_dim", proprio_dim if proprio_dim is not None else 0))
        if visual_channels is not None and ckpt_visual != visual_channels:
            raise ValueError(
                f"Checkpoint visual dim {ckpt_visual} != requested policy visual_channels={visual_channels}"
            )
        if proprio_dim is not None and ckpt_proprio != proprio_dim:
            raise ValueError(
                f"Checkpoint proprio dim {ckpt_proprio} != requested proprio_dim={proprio_dim}"
            )
        if ckpt_visual != policy.visual_channels or ckpt_proprio != policy.proprio_dim:
            raise ValueError(
                f"Checkpoint dims ({ckpt_visual}, {ckpt_proprio}) "
                f"!= policy ({policy.visual_channels}, {policy.proprio_dim})"
            )
        policy_type = checkpoint.get("policy_type", "binary_gripper_dual_head_vision_dagger")
        if policy_type not in ("binary_gripper_dual_head_vision_dagger",):
            raise ValueError(f"Unsupported vision checkpoint policy_type={policy_type!r}")
        missing, unexpected = policy.load_state_dict(checkpoint["model"], strict=False)
        allowed_missing = {
            "aux_delta_head.0.weight",
            "aux_delta_head.0.bias",
            "aux_delta_head.2.weight",
            "aux_delta_head.2.bias",
        }
        unexpected = [key for key in unexpected if not key.startswith("aux_delta_head.")]
        missing = [key for key in missing if key not in allowed_missing]
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint state mismatch while loading vision policy: missing={missing}, unexpected={unexpected}"
            )
