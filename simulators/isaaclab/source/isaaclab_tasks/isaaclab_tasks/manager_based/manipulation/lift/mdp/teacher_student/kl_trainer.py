# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode replay buffer and BPTT KL optimization for teacher-student training."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from .networks import detach_rnn_state, init_rnn_state


@dataclass
class EpisodeData:
    """One completed rollout episode."""

    obs: torch.Tensor
    teacher: torch.Tensor
    teacher_phase: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return int(self.obs.shape[0])


class TwistEpisodeReplayBuffer:
    """Episode ring buffer aligned with active_tracking_rl train_mode=6."""

    def __init__(self, max_episodes: int, device: str | torch.device):
        self.max_episodes = max_episodes
        self.device = device
        self.episodes: list[EpisodeData] = []
        self.total_steps = 0

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def num_steps(self) -> int:
        return self.total_steps

    @property
    def max_episode_length(self) -> int:
        if not self.episodes:
            return 0
        return max(ep.length for ep in self.episodes)

    def add_episode(
        self,
        obs: torch.Tensor,
        teacher: torch.Tensor,
        teacher_phase: torch.Tensor | None = None,
    ) -> None:
        """Append one episode. obs: (T, obs_dim), teacher: (T, action_dim)."""
        if obs.shape[0] == 0:
            return
        phase = None
        if teacher_phase is not None:
            phase = teacher_phase.detach().clone().to(dtype=torch.long)
        episode = EpisodeData(
            obs=obs.detach().clone(),
            teacher=teacher.detach().clone(),
            teacher_phase=phase,
        )
        self.episodes.append(episode)
        self.total_steps += episode.length
        if len(self.episodes) > self.max_episodes:
            removed = self.episodes.pop(0)
            self.total_steps -= removed.length

    def add_in_progress_episodes(
        self,
        ep_obs: torch.Tensor,
        ep_teacher: torch.Tensor,
        ep_step: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        ep_teacher_phase: torch.Tensor | None = None,
    ) -> int:
        """Flush partially collected episodes into the buffer. Returns number of episodes added."""
        if env_ids is None:
            env_ids = torch.arange(ep_step.shape[0], device=ep_step.device)

        added = 0
        for env_id in env_ids.tolist():
            length = int(ep_step[env_id].item())
            if length <= 0:
                continue
            phase = ep_teacher_phase[env_id, :length] if ep_teacher_phase is not None else None
            self.add_episode(ep_obs[env_id, :length], ep_teacher[env_id, :length], teacher_phase=phase)
            ep_step[env_id] = 0
            added += 1
        return added

    def _segment_pools(
        self,
        valid_episodes: list[EpisodeData],
        seq_len: int,
        min_grasp_phase: int,
    ) -> tuple[list[tuple[EpisodeData, int]], list[tuple[EpisodeData, int]]]:
        """Build uniform and grasp-phase segment pools for prioritized sampling."""
        all_segments: list[tuple[EpisodeData, int]] = []
        grasp_segments: list[tuple[EpisodeData, int]] = []
        for episode in valid_episodes:
            max_start = episode.length - seq_len
            has_phase = episode.teacher_phase is not None
            for start in range(max_start + 1):
                all_segments.append((episode, start))
                if has_phase and int(episode.teacher_phase[start : start + seq_len].max().item()) >= min_grasp_phase:
                    grasp_segments.append((episode, start))
        return all_segments, grasp_segments

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        *,
        grasp_sample_ratio: float = 0.0,
        min_grasp_phase: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Sample trajectory segments for BPTT.

        When ``grasp_sample_ratio > 0``, each batch item is drawn from segments that contain at
        least one step with ``teacher_phase >= min_grasp_phase`` with that probability; otherwise
        sampling is uniform over all valid segments.

        Returns (obs_batch, teacher_batch, grasp_sample_hit_rate).
        """
        if len(self.episodes) == 0:
            raise RuntimeError("Episode replay buffer is empty.")

        valid_episodes = [ep for ep in self.episodes if ep.length >= seq_len]
        if len(valid_episodes) == 0:
            max_len = max(ep.length for ep in self.episodes)
            seq_len = max(2, min(seq_len, max_len))
            valid_episodes = [ep for ep in self.episodes if ep.length >= seq_len]
        if len(valid_episodes) == 0:
            raise RuntimeError("No episode is long enough for sequence sampling.")

        all_segments, grasp_segments = self._segment_pools(valid_episodes, seq_len, min_grasp_phase)

        obs_dim = valid_episodes[0].obs.shape[-1]
        action_dim = valid_episodes[0].teacher.shape[-1]
        obs_batch = torch.zeros(batch_size, seq_len, obs_dim, device=self.device)
        teacher_batch = torch.zeros(batch_size, seq_len, action_dim, device=self.device)

        grasp_hits = 0
        use_grasp_bias = grasp_sample_ratio > 0.0 and len(grasp_segments) > 0
        for _ in range(batch_size):
            if use_grasp_bias and random.random() < grasp_sample_ratio:
                episode, start = grasp_segments[random.randrange(len(grasp_segments))]
                grasp_hits += 1
            else:
                episode, start = all_segments[random.randrange(len(all_segments))]
            obs_batch[_] = episode.obs[start : start + seq_len]
            teacher_batch[_] = episode.teacher[start : start + seq_len]

        grasp_hit_rate = grasp_hits / batch_size if use_grasp_bias else 0.0
        return obs_batch, teacher_batch, grasp_hit_rate


# Backward-compatible alias.
TwistReplayBuffer = TwistEpisodeReplayBuffer


def gaussian_kl_to_teacher(
    student_dist: Normal,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    dim_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute KL(student || teacher) for diagonal Gaussians. Returns shape (batch,)."""
    teacher_dist = Normal(teacher_mean, teacher_std.expand_as(teacher_mean))
    kl = kl_divergence(student_dist, teacher_dist)
    if dim_weights is not None:
        kl = kl * dim_weights.to(kl.device)
    return kl.sum(-1)


def gaussian_kl_arm_gripper_unweighted(
    student_dist: Normal,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step unweighted KL for arm dims (0:6) and gripper dim (6). Returns (batch,), (batch,)."""
    teacher_dist = Normal(teacher_mean, teacher_std.expand_as(teacher_mean))
    kl = kl_divergence(student_dist, teacher_dist)
    arm_kl = kl[..., :6].sum(-1)
    gripper_kl = kl[..., 6] if kl.shape[-1] > 6 else torch.zeros_like(arm_kl)
    return arm_kl, gripper_kl


def _bptt_should_update(step_idx: int, seq_len: int, grad_interval: int) -> bool:
    """True when truncated BPTT should backprop and optimizer-step."""
    if grad_interval <= 0:
        return step_idx == seq_len - 1
    return (step_idx + 1) % grad_interval == 0 or step_idx == seq_len - 1


def optimize_kl_batch(
    policy,
    optimizer: torch.optim.Optimizer,
    obs_batch: torch.Tensor,
    teacher_batch: torch.Tensor,
    teacher_std: torch.Tensor,
    *,
    unroll_steps: int,
    dim_weights: torch.Tensor | None = None,
    grad_interval: int = 20,
    max_grad_norm: float = 20.0,
) -> tuple[float, float, float]:
    """Run truncated BPTT KL optimization. Returns (weighted_total, arm_kl, gripper_kl) unweighted means."""
    del unroll_steps  # sequence length is defined by obs_batch.shape[1]
    if obs_batch.ndim != 3:
        raise ValueError(f"Expected obs_batch shape (B, T, D), got {tuple(obs_batch.shape)}")

    batch_size, seq_len, _ = obs_batch.shape
    hidden = init_rnn_state(batch_size, policy.rnn_hidden_dim, obs_batch.device)
    step_losses: list[torch.Tensor] = []
    arm_kl_losses: list[torch.Tensor] = []
    gripper_kl_losses: list[torch.Tensor] = []
    chunk_losses: list[torch.Tensor] = []

    for step_idx in range(seq_len):
        obs_step = obs_batch[:, step_idx : step_idx + 1, :]
        dist, hidden = policy.policy_dist(obs_step, hidden)
        teacher_step = teacher_batch[:, step_idx]
        step_loss = gaussian_kl_to_teacher(dist, teacher_step, teacher_std, dim_weights=dim_weights).mean()
        step_losses.append(step_loss)
        chunk_losses.append(step_loss)
        arm_kl, gripper_kl = gaussian_kl_arm_gripper_unweighted(dist, teacher_step, teacher_std)
        arm_kl_losses.append(arm_kl.mean())
        gripper_kl_losses.append(gripper_kl.mean())

        if _bptt_should_update(step_idx, seq_len, grad_interval):
            chunk_loss = torch.stack(chunk_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            chunk_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            chunk_losses = []
            hidden = detach_rnn_state(hidden)

    return (
        float(torch.stack(step_losses).mean().item()),
        float(torch.stack(arm_kl_losses).mean().item()),
        float(torch.stack(gripper_kl_losses).mean().item()),
    )


def optimize_kl_sequences(
    policy,
    optimizer: torch.optim.Optimizer,
    buffer: TwistEpisodeReplayBuffer,
    teacher_std: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    optim_steps: int,
    dim_weights: torch.Tensor | None = None,
    grad_interval: int = 20,
    max_grad_norm: float = 20.0,
    grasp_sample_ratio: float = 0.0,
    min_grasp_phase: int = 3,
) -> tuple[float, float, float, float]:
    """Sample episode segments and run multiple BPTT KL updates.

    Returns (total, arm_kl, gripper_kl, grasp_sample_hit_rate).
    """
    total_acc = 0.0
    arm_acc = 0.0
    gripper_acc = 0.0
    grasp_hit_acc = 0.0
    for _ in range(optim_steps):
        obs_batch, teacher_batch, grasp_hit_rate = buffer.sample_sequences(
            batch_size,
            seq_len,
            grasp_sample_ratio=grasp_sample_ratio,
            min_grasp_phase=min_grasp_phase,
        )
        batch_total, batch_arm, batch_gripper = optimize_kl_batch(
            policy,
            optimizer,
            obs_batch,
            teacher_batch,
            teacher_std,
            unroll_steps=seq_len,
            dim_weights=dim_weights,
            grad_interval=grad_interval,
            max_grad_norm=max_grad_norm,
        )
        total_acc += batch_total
        arm_acc += batch_arm
        gripper_acc += batch_gripper
        grasp_hit_acc += grasp_hit_rate
    n = max(optim_steps, 1)
    return total_acc / n, arm_acc / n, gripper_acc / n, grasp_hit_acc / n


def gripper_bce_to_teacher(
    gripper_logit: torch.Tensor,
    teacher_gripper: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy for gripper logit vs teacher labels (+1 open, -1 close). Returns scalar."""
    target = (teacher_gripper + 1.0) * 0.5
    return F.binary_cross_entropy_with_logits(gripper_logit.squeeze(-1), target, reduction="mean")


def optimize_kl_bce_batch(
    policy,
    optimizer: torch.optim.Optimizer,
    obs_batch: torch.Tensor,
    teacher_batch: torch.Tensor,
    teacher_arm_std: torch.Tensor,
    *,
    unroll_steps: int,
    gripper_bce_weight: float = 1.0,
    grad_interval: int = 20,
    max_grad_norm: float = 20.0,
) -> tuple[float, float, float]:
    """Run truncated BPTT with decoupled arm KL and gripper BCE updates.

    Arm KL trains the shared encoder + arm head. Gripper BCE trains the gripper head only
    (encoder forward runs under ``no_grad`` so BCE does not move the representation).
    """
    del unroll_steps
    if obs_batch.ndim != 3:
        raise ValueError(f"Expected obs_batch shape (B, T, D), got {tuple(obs_batch.shape)}")

    if not hasattr(policy, "arm_parameters") or not hasattr(policy, "gripper_parameters"):
        raise TypeError("optimize_kl_bce_batch requires a dual-head binary gripper policy.")

    batch_size, seq_len, _ = obs_batch.shape
    arm_kl_losses: list[torch.Tensor] = []
    gripper_bce_losses: list[torch.Tensor] = []

    # Pass 1: arm KL -> encoder + arm head (truncated BPTT every grad_interval steps)
    hidden = init_rnn_state(batch_size, policy.rnn_hidden_dim, obs_batch.device)
    arm_chunk: list[torch.Tensor] = []
    for step_idx in range(seq_len):
        obs_step = obs_batch[:, step_idx : step_idx + 1, :]
        features, hidden = policy.encode_step(obs_step, hidden)
        arm_dist = policy.arm_dist_from_features(features)
        teacher_arm = teacher_batch[:, step_idx, :6]
        step_arm_kl = gaussian_kl_to_teacher(arm_dist, teacher_arm, teacher_arm_std).mean()
        arm_kl_losses.append(step_arm_kl)
        arm_chunk.append(step_arm_kl)

        if _bptt_should_update(step_idx, seq_len, grad_interval):
            arm_loss = torch.stack(arm_chunk).mean()
            optimizer.zero_grad(set_to_none=True)
            arm_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.arm_parameters(), max_grad_norm)
            optimizer.step()
            arm_chunk = []
            hidden = detach_rnn_state(hidden)

    # Pass 2: gripper BCE -> gripper head only (encoder frozen, truncated BPTT)
    hidden = init_rnn_state(batch_size, policy.rnn_hidden_dim, obs_batch.device)
    gripper_chunk: list[torch.Tensor] = []
    policy.eval()
    for step_idx in range(seq_len):
        obs_step = obs_batch[:, step_idx : step_idx + 1, :]
        teacher_gripper = teacher_batch[:, step_idx, 6]
        with torch.no_grad():
            features, hidden = policy.encode_step(obs_step, hidden)
        gripper_logit = policy.gripper_logit_from_features(features)
        step_bce = gripper_bce_to_teacher(gripper_logit, teacher_gripper)
        gripper_bce_losses.append(step_bce)
        gripper_chunk.append(step_bce)

        if _bptt_should_update(step_idx, seq_len, grad_interval):
            gripper_loss = torch.stack(gripper_chunk).mean() * gripper_bce_weight
            optimizer.zero_grad(set_to_none=True)
            gripper_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.gripper_parameters(), max_grad_norm)
            optimizer.step()
            gripper_chunk = []
            hidden = detach_rnn_state(hidden)
    policy.train()

    mean_arm_kl = float(torch.stack(arm_kl_losses).mean().item())
    mean_gripper_bce = float(torch.stack(gripper_bce_losses).mean().item())
    total = mean_arm_kl + gripper_bce_weight * mean_gripper_bce
    return total, mean_arm_kl, mean_gripper_bce


def optimize_kl_bce_sequences(
    policy,
    optimizer: torch.optim.Optimizer,
    buffer: TwistEpisodeReplayBuffer,
    teacher_arm_std: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    optim_steps: int,
    gripper_bce_weight: float = 1.0,
    grad_interval: int = 20,
    max_grad_norm: float = 20.0,
    grasp_sample_ratio: float = 0.0,
    min_grasp_phase: int = 3,
) -> tuple[float, float, float, float]:
    """Sample episode segments and run multiple decoupled arm-KL + gripper-BCE updates."""
    total_loss = 0.0
    arm_kl_acc = 0.0
    gripper_bce_acc = 0.0
    grasp_hit_acc = 0.0
    for _ in range(optim_steps):
        obs_batch, teacher_batch, grasp_hit_rate = buffer.sample_sequences(
            batch_size,
            seq_len,
            grasp_sample_ratio=grasp_sample_ratio,
            min_grasp_phase=min_grasp_phase,
        )
        batch_total, batch_arm_kl, batch_gripper_bce = optimize_kl_bce_batch(
            policy,
            optimizer,
            obs_batch,
            teacher_batch,
            teacher_arm_std,
            unroll_steps=seq_len,
            gripper_bce_weight=gripper_bce_weight,
            grad_interval=grad_interval,
            max_grad_norm=max_grad_norm,
        )
        total_loss += batch_total
        arm_kl_acc += batch_arm_kl
        gripper_bce_acc += batch_gripper_bce
        grasp_hit_acc += grasp_hit_rate
    n = max(optim_steps, 1)
    return total_loss / n, arm_kl_acc / n, gripper_bce_acc / n, grasp_hit_acc / n
