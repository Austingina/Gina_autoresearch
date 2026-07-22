# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DAgger-style dataset aggregation for lift-cube teacher-student training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

from .kl_trainer import EpisodeData, TwistEpisodeReplayBuffer


def compute_dagger_beta(
    iteration: int,
    *,
    beta_start: float = 1.0,
    beta_end: float = 0.0,
    warmup_iters: int = 0,
    decay_iters: int = 2000,
) -> float:
    """Linear β schedule: expert mixing probability during rollout.

    Returns ``beta_start`` for ``iteration < warmup_iters``, then linearly
    interpolates to ``beta_end`` over ``decay_iters``.
    """
    if iteration < warmup_iters:
        return beta_start
    if decay_iters <= 0:
        return beta_end
    progress = min(1.0, (iteration - warmup_iters) / float(decay_iters))
    return beta_start + (beta_end - beta_start) * progress


def compute_dagger_beta_adaptive(
    iteration: int,
    success_rate: float | None,
    *,
    beta_start: float = 1.0,
    beta_end: float = 0.0,
    warmup_iters: int = 0,
    success_target: float = 0.5,
) -> float:
    """Success-driven β schedule for DAgger rollout mixing.

    Keeps ``beta_start`` during warmup and until the first eval success rate is
    available. Afterwards:

        β = β_end + (β_start - β_end) * max(0, 1 - success / success_target)

    so β stays high while the student eval success is low and approaches
    ``beta_end`` once ``success_target`` is reached.
    """
    if iteration < warmup_iters:
        return beta_start
    if success_rate is None:
        return beta_start
    if success_target <= 0.0:
        return beta_end
    progress = min(1.0, max(0.0, success_rate / success_target))
    beta = beta_end + (beta_start - beta_end) * (1.0 - progress)
    return max(beta_end, min(beta_start, beta))


@dataclass
class GraspGatedBetaState:
    """Mutable state for grasp-gated two-phase β decay."""

    grasp_ema: float | None = None
    grasp_stable: bool = False
    grasp_stable_iteration: int | None = None
    recent_collect_grasp: list[float] = field(default_factory=list)

    def observe_collect_grasp(
        self,
        collect_grasp_rate: float,
        *,
        ema: float = 0.7,
        history_len: int = 12,
    ) -> None:
        rate = float(max(0.0, min(1.0, collect_grasp_rate)))
        if self.grasp_ema is None:
            self.grasp_ema = rate
        else:
            self.grasp_ema = ema * self.grasp_ema + (1.0 - ema) * rate
        self.recent_collect_grasp.append(rate)
        if len(self.recent_collect_grasp) > history_len:
            self.recent_collect_grasp.pop(0)

    def maybe_latch_grasp_stable(
        self,
        iteration: int,
        *,
        warmup_iters: int,
        stable_window: int = 5,
        stable_delta: float = 0.03,
        min_rate: float = 0.30,
        min_improvement: float = 0.05,
    ) -> bool:
        """Return True when grasp EMA has risen and recently plateaued."""
        if self.grasp_stable or iteration < warmup_iters:
            return self.grasp_stable
        if self.grasp_ema is None or len(self.recent_collect_grasp) < stable_window:
            return False
        if self.grasp_ema < min_rate:
            return False
        window = self.recent_collect_grasp[-stable_window:]
        if window[-1] - window[0] < min_improvement:
            return False
        tail = window[-min(3, len(window)) :]
        if len(tail) < 2:
            return False
        tail_deltas = [abs(tail[i] - tail[i - 1]) for i in range(1, len(tail))]
        if max(tail_deltas) > stable_delta:
            return False
        self.grasp_stable = True
        self.grasp_stable_iteration = int(iteration)
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "grasp_ema": self.grasp_ema,
            "grasp_stable": self.grasp_stable,
            "grasp_stable_iteration": self.grasp_stable_iteration,
            "recent_collect_grasp": list(self.recent_collect_grasp),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GraspGatedBetaState:
        if not data:
            return cls()
        return cls(
            grasp_ema=data.get("grasp_ema"),
            grasp_stable=bool(data.get("grasp_stable", False)),
            grasp_stable_iteration=data.get("grasp_stable_iteration"),
            recent_collect_grasp=list(data.get("recent_collect_grasp", [])),
        )


def compute_dagger_beta_grasp_gated(
    iteration: int,
    state: GraspGatedBetaState,
    *,
    beta_start: float = 1.0,
    beta_end: float = 0.0,
    beta_floor: float = 0.6,
    warmup_iters: int = 50,
    phase1_decay_iters: int = 2000,
    grasp_end_iter: int = 2000,
    grasp_target: float = 0.50,
    min_grasp_factor: float = 0.15,
) -> float:
    """Two-phase β: decay to ``beta_floor`` while grasp improves, then decay to 0.

    Phase 1 (before ``state.grasp_stable``):
      - Warmup: β = ``beta_start``.
      - Linear decay toward ``beta_floor``, speed scaled by collect grasp EMA /
        ``grasp_target`` so student share grows faster only when grasp improves.
      - β never drops below ``beta_floor``.

    Phase 2 (after grasp stable):
      - Linear decay from ``beta_floor`` to ``beta_end`` (usually 0) at
        ``grasp_end_iter`` (absolute iteration index).
    """
    beta_floor = max(beta_end, min(beta_start, beta_floor))
    if iteration < warmup_iters:
        return beta_start

    if state.grasp_stable:
        stable_iter = state.grasp_stable_iteration
        if stable_iter is None:
            stable_iter = iteration
        if iteration >= grasp_end_iter:
            return beta_end
        span = max(1, grasp_end_iter - stable_iter)
        progress = min(1.0, max(0.0, (iteration - stable_iter) / float(span)))
        return beta_floor + (beta_end - beta_floor) * progress

    if phase1_decay_iters <= 0:
        return beta_floor

    time_progress = min(1.0, (iteration - warmup_iters) / float(phase1_decay_iters))
    if state.grasp_ema is None:
        grasp_factor = min_grasp_factor
    else:
        grasp_factor = max(min_grasp_factor, min(1.0, state.grasp_ema / max(grasp_target, 1e-6)))
    progress = min(1.0, time_progress * grasp_factor)
    beta = beta_start + (beta_floor - beta_start) * progress
    return max(beta_floor, min(beta_start, beta))


def mix_rollout_twist(
    teacher_twist: torch.Tensor,
    student_twist: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix teacher/student twists for environment execution.

    Labels stored in the aggregate dataset should always be ``teacher_twist``;
    only the executed action is mixed.

    Returns:
        executed_twist: (N, 7)
        teacher_executed: (N,) bool mask — True where teacher action was executed.
    """
    if beta >= 1.0 - 1e-6:
        mask = torch.ones(teacher_twist.shape[0], dtype=torch.bool, device=teacher_twist.device)
        return teacher_twist.clone(), mask
    if beta <= 1e-6:
        mask = torch.zeros(teacher_twist.shape[0], dtype=torch.bool, device=teacher_twist.device)
        return student_twist.clone(), mask

    mask = torch.rand(teacher_twist.shape[0], device=teacher_twist.device) < beta
    executed = torch.where(mask.unsqueeze(-1), teacher_twist, student_twist)
    return executed, mask


@dataclass
class DaggerEpisodeData(EpisodeData):
    """Episode with optional teacher-success tag for reserved buffer slots."""

    teacher_success: bool = False


class DaggerAggregateBuffer(TwistEpisodeReplayBuffer):
    """Growing aggregate buffer for DAgger with reserved teacher-success episodes."""

    def __init__(
        self,
        max_episodes: int,
        device: str | torch.device,
        *,
        success_reserve_ratio: float = 0.35,
    ):
        self.max_episodes = max_episodes
        self.device = device
        self.total_steps = 0
        self.success_reserve_ratio = float(success_reserve_ratio)
        self._success_episodes: list[DaggerEpisodeData] = []
        self._rolling_episodes: list[DaggerEpisodeData] = []

    @property
    def success_cap(self) -> int:
        if self.max_episodes <= 0:
            return 0
        return max(0, int(round(self.max_episodes * self.success_reserve_ratio)))

    @property
    def rolling_cap(self) -> int:
        return max(0, self.max_episodes - self.success_cap)

    @property
    def episodes(self) -> list[EpisodeData]:
        return self._success_episodes + self._rolling_episodes

    @property
    def num_success_episodes(self) -> int:
        return len(self._success_episodes)

    @property
    def num_rolling_episodes(self) -> int:
        return len(self._rolling_episodes)

    def _recompute_total_steps(self) -> None:
        self.total_steps = sum(ep.length for ep in self.episodes)

    def _append_to_pool(
        self,
        pool: list[DaggerEpisodeData],
        episode: DaggerEpisodeData,
        cap: int,
    ) -> None:
        if cap <= 0:
            return
        pool.append(episode)
        while len(pool) > cap:
            pool.pop(0)
        self._recompute_total_steps()

    def add_episode(
        self,
        obs: torch.Tensor,
        teacher: torch.Tensor,
        teacher_phase: torch.Tensor | None = None,
        *,
        teacher_success: bool = False,
    ) -> None:
        """Append one episode to the success-reserved or rolling FIFO pool."""
        if obs.shape[0] == 0:
            return
        phase = None
        if teacher_phase is not None:
            phase = teacher_phase.detach().clone().to(dtype=torch.long)
        episode = DaggerEpisodeData(
            obs=obs.detach().clone(),
            teacher=teacher.detach().clone(),
            teacher_phase=phase,
            teacher_success=teacher_success,
        )
        if teacher_success and self.success_cap > 0:
            self._append_to_pool(self._success_episodes, episode, self.success_cap)
        elif self.rolling_cap > 0:
            self._append_to_pool(self._rolling_episodes, episode, self.rolling_cap)
        elif self.success_cap > 0:
            # Rolling pool disabled; fall back to success pool.
            self._append_to_pool(self._success_episodes, episode, self.success_cap)

    def state_dict(self) -> dict[str, Any]:
        """Serialize all aggregated episodes for checkpointing."""
        packed = []
        for episode in self.episodes:
            packed.append(
                {
                    "obs": episode.obs.detach().cpu(),
                    "teacher": episode.teacher.detach().cpu(),
                    "teacher_phase": (
                        episode.teacher_phase.detach().cpu()
                        if episode.teacher_phase is not None
                        else None
                    ),
                    "teacher_success": bool(getattr(episode, "teacher_success", False)),
                }
            )
        return {
            "max_episodes": self.max_episodes,
            "success_reserve_ratio": self.success_reserve_ratio,
            "total_steps": self.total_steps,
            "episodes": packed,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore aggregated episodes from :meth:`state_dict`."""
        self._success_episodes.clear()
        self._rolling_episodes.clear()
        self.total_steps = 0
        self.max_episodes = int(state.get("max_episodes", self.max_episodes))
        self.success_reserve_ratio = float(
            state.get("success_reserve_ratio", self.success_reserve_ratio)
        )
        for item in state["episodes"]:
            phase = item.get("teacher_phase")
            if phase is not None:
                phase = phase.to(device=self.device, dtype=torch.long)
            self.add_episode(
                item["obs"].to(device=self.device),
                item["teacher"].to(device=self.device),
                teacher_phase=phase,
                teacher_success=bool(item.get("teacher_success", False)),
            )

    def save(self, path: str) -> None:
        """Write aggregate buffer to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str | torch.device) -> DaggerAggregateBuffer:
        """Load aggregate buffer from disk."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        buffer = cls(
            max_episodes=int(state.get("max_episodes", 5000)),
            device=device,
            success_reserve_ratio=float(state.get("success_reserve_ratio", 0.35)),
        )
        buffer.load_state_dict(state)
        return buffer
