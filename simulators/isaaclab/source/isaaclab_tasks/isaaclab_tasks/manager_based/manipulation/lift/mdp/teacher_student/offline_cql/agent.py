# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CQL-SAC agent with shared LSTM encoder (Offline_RL_Active_Tracking style)."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from .networks import GaussianActor, LSTMFeatureEncoder, QNetwork, flatten_time_features, random_action_log_prob


@dataclass
class CQLLearnStats:
    critic1_loss: float
    critic2_loss: float
    actor_loss: float
    alpha_loss: float
    cql1_loss: float
    cql2_loss: float
    q1_mean: float
    q2_mean: float
    alpha: float


class CQLSACAgentLSTM:
    """Offline CQL-SAC with CNN+LSTM encoder."""

    def __init__(
        self,
        *,
        action_dim: int,
        device: torch.device,
        obs_mode: str = "vision",
        visual_channels: int = 8,
        proprio_dim: int = 10,
        hidden_dim: int = 256,
        cnn_feat_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        learning_rate: float = 3e-5,
        gamma: float = 0.99,
        tau: float = 5e-3,
        alpha_lr: float = 3e-5,
        cql_weight: float = 1.0,
        cql_temp: float = 1.0,
        target_action_gap: float = 10.0,
        with_lagrange: bool = False,
        grad_clip: float = 1.0,
        num_random_actions: int = 10,
    ):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.cql_weight = cql_weight
        self.cql_temp = cql_temp
        self.target_action_gap = target_action_gap
        self.with_lagrange = with_lagrange
        self.grad_clip = grad_clip
        self.num_random_actions = num_random_actions
        self.obs_mode = obs_mode

        use_visual = obs_mode in ("vision", "vision_proprio")
        use_proprio = obs_mode in ("proprio", "vision_proprio")

        self.encoder = LSTMFeatureEncoder(
            visual_channels=visual_channels,
            proprio_dim=proprio_dim if use_proprio else 0,
            cnn_feat_dim=cnn_feat_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            use_visual=use_visual,
        ).to(device)

        feat_dim = lstm_hidden_dim
        self.actor = GaussianActor(feat_dim, action_dim, hidden_dim).to(device)
        self.critic1 = QNetwork(feat_dim, action_dim, hidden_dim).to(device)
        self.critic2 = QNetwork(feat_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.log_alpha = torch.tensor([0.0], device=device, requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()

        self.cql_log_alpha = torch.zeros(1, device=device, requires_grad=True)

        encoder_params = list(self.encoder.parameters())
        self.encoder_optimizer = optim.Adam(encoder_params, lr=learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.cql_alpha_optimizer = optim.Adam([self.cql_log_alpha], lr=learning_rate)

        self._random_log_prob = random_action_log_prob(action_dim, device)

    def encode(
        self,
        *,
        visual_seq: torch.Tensor | None = None,
        proprio_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(visual_seq=visual_seq, proprio_seq=proprio_seq)

    def _parse_batch(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.obs_mode == "vision":
            visual, next_visual, actions, rewards, dones = batch
            return visual, None, actions, rewards, dones, next_visual, None
        if self.obs_mode == "proprio":
            proprio, next_proprio, actions, rewards, dones = batch
            return None, proprio, actions, rewards, dones, None, next_proprio
        visual, next_visual, proprio, next_proprio, actions, rewards, dones = batch
        return visual, proprio, actions, rewards, dones, next_visual, next_proprio

    def learn(self, batch: tuple[torch.Tensor, ...]) -> CQLLearnStats:
        visual, proprio, actions, rewards, dones, next_visual, next_proprio = self._parse_batch(batch)

        encoded = self.encode(visual_seq=visual, proprio_seq=proprio)
        with torch.no_grad():
            next_encoded = self.encode(visual_seq=next_visual, proprio_seq=next_proprio)

        batch_size, seq_len, _ = encoded.shape
        flat_states = flatten_time_features(encoded)
        flat_next_states = flatten_time_features(next_encoded)
        flat_actions = actions.reshape(batch_size * seq_len, -1)
        flat_rewards = rewards.reshape(batch_size * seq_len, 1)
        flat_dones = dones.reshape(batch_size * seq_len, 1)

        # ---------------- actor + alpha ---------------- #
        current_alpha = self.log_alpha.exp().detach()
        new_actions, log_pi = self.actor.sample(flat_states)
        q1_pi = self.critic1(flat_states, new_actions)
        q2_pi = self.critic2(flat_states, new_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (current_alpha * log_pi - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (log_pi.detach() + self.action_dim)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------- critic targets ---------------- #
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(flat_next_states)
            q1_next = self.critic1_target(flat_next_states, next_actions)
            q2_next = self.critic2_target(flat_next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_targets = flat_rewards + self.gamma * (1.0 - flat_dones) * q_next

        q1 = self.critic1(flat_states, flat_actions)
        q2 = self.critic2(flat_states, flat_actions)
        critic1_loss = F.mse_loss(q1, q_targets)
        critic2_loss = F.mse_loss(q2, q_targets)

        # ---------------- CQL penalty ---------------- #
        num_repeat = self.num_random_actions
        repeat_states = flat_states.unsqueeze(1).repeat(1, num_repeat, 1).view(-1, flat_states.shape[-1])
        repeat_next = flat_next_states.unsqueeze(1).repeat(1, num_repeat, 1).view(-1, flat_next_states.shape[-1])
        random_actions = torch.empty(repeat_states.shape[0], self.action_dim, device=self.device).uniform_(-1.0, 1.0)

        policy_actions, policy_log_pi = self.actor.sample(repeat_states)
        next_policy_actions, next_policy_log_pi = self.actor.sample(repeat_next)

        q1_rand = self.critic1(repeat_states, random_actions)
        q2_rand = self.critic2(repeat_states, random_actions)
        q1_pi_all = self.critic1(repeat_states, policy_actions) - policy_log_pi.detach()
        q2_pi_all = self.critic2(repeat_states, policy_actions) - policy_log_pi.detach()
        q1_next_pi = self.critic1(repeat_next, next_policy_actions) - next_policy_log_pi.detach()
        q2_next_pi = self.critic2(repeat_next, next_policy_actions) - next_policy_log_pi.detach()

        n_flat = batch_size * seq_len

        def _reshape(values: torch.Tensor) -> torch.Tensor:
            return values.view(n_flat, num_repeat, 1)

        cat_q1 = torch.cat([_reshape(q1_rand), _reshape(q1_pi_all), _reshape(q1_next_pi)], dim=1)
        cat_q2 = torch.cat([_reshape(q2_rand), _reshape(q2_pi_all), _reshape(q2_next_pi)], dim=1)

        cql1 = torch.logsumexp(cat_q1 / self.cql_temp, dim=1).mean() * self.cql_weight * self.cql_temp - q1.mean()
        cql2 = torch.logsumexp(cat_q2 / self.cql_temp, dim=1).mean() * self.cql_weight * self.cql_temp - q2.mean()
        cql1_scaled = cql1 * self.cql_weight
        cql2_scaled = cql2 * self.cql_weight

        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1.0e6)
            cql1_scaled = cql_alpha * (cql1_scaled - self.target_action_gap)
            cql2_scaled = cql_alpha * (cql2_scaled - self.target_action_gap)
            cql_alpha_loss = -(cql1_scaled + cql2_scaled).mean() * 0.5
            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()

        total_c1 = critic1_loss + cql1_scaled
        total_c2 = critic2_loss + cql2_scaled

        self.encoder_optimizer.zero_grad()
        self.critic1_optimizer.zero_grad()
        total_c1.backward(retain_graph=True)
        clip_grad_norm_(list(self.encoder.parameters()) + list(self.critic1.parameters()), self.grad_clip)
        self.critic1_optimizer.step()

        self.encoder_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        total_c2.backward()
        clip_grad_norm_(list(self.encoder.parameters()) + list(self.critic2.parameters()), self.grad_clip)
        self.critic2_optimizer.step()
        self.encoder_optimizer.step()

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return CQLLearnStats(
            critic1_loss=float(critic1_loss.item()),
            critic2_loss=float(critic2_loss.item()),
            actor_loss=float(actor_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            cql1_loss=float(cql1_scaled.item()),
            cql2_loss=float(cql2_scaled.item()),
            q1_mean=float(q1.mean().item()),
            q2_mean=float(q2.mean().item()),
            alpha=float(self.alpha.item()),
        )

    def _soft_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def to(self, device: str | torch.device) -> CQLSACAgentLSTM:
        """Move model parameters (and optimizer state) to ``device``."""
        device = torch.device(device)
        self.device = device
        self.encoder.to(device)
        self.actor.to(device)
        self.critic1.to(device)
        self.critic2.to(device)
        self.critic1_target.to(device)
        self.critic2_target.to(device)
        self.log_alpha.data = self.log_alpha.data.to(device)
        self.cql_log_alpha.data = self.cql_log_alpha.data.to(device)
        self.alpha = self.log_alpha.exp().detach()
        for optimizer in (
            self.encoder_optimizer,
            self.actor_optimizer,
            self.critic1_optimizer,
            self.critic2_optimizer,
            self.alpha_optimizer,
            self.cql_alpha_optimizer,
        ):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        return self

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "action_dim": self.action_dim,
            "obs_mode": self.obs_mode,
        }

    def load_state_dict(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.critic1_target.load_state_dict(state["critic1_target"])
        self.critic2_target.load_state_dict(state["critic2_target"])
        self.log_alpha.data.copy_(state["log_alpha"])
