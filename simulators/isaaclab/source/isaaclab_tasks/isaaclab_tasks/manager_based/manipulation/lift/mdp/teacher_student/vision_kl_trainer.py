# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision episode buffer and BPTT KL+BCE optimization for DAgger."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch

from .kl_trainer import (
    _bptt_should_update,
    gripper_bce_to_teacher,
    gaussian_kl_to_teacher,
    init_rnn_state,
    detach_rnn_state,
)


@dataclass
class VisionEpisodeData:
    """One vision rollout episode (tensors live on ``storage_device``, usually CPU)."""

    visual: torch.Tensor
    teacher: torch.Tensor
    proprio: torch.Tensor | None = None
    teacher_phase: torch.Tensor | None = None
    cube_ee_delta: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return int(self.visual.shape[0])


def _vision_tensor_to_storage(tensor: torch.Tensor, storage_device: torch.device) -> torch.Tensor:
    return tensor.detach().clone().to(device=storage_device)


class VisionEpisodeReplayBuffer:
    """Episode buffer storing (T, C, H, W) visual tensors."""

    def __init__(
        self,
        max_episodes: int,
        device: str | torch.device,
        *,
        storage_device: str | torch.device = "cpu",
    ):
        self.max_episodes = max_episodes
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device)
        self.episodes: list[VisionEpisodeData] = []
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
        visual: torch.Tensor,
        teacher: torch.Tensor,
        teacher_phase: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        cube_ee_delta: torch.Tensor | None = None,
    ) -> None:
        if visual.shape[0] == 0:
            return
        store = self.storage_device
        phase = (
            _vision_tensor_to_storage(teacher_phase, store).to(dtype=torch.long)
            if teacher_phase is not None
            else None
        )
        prop = _vision_tensor_to_storage(proprio, store) if proprio is not None else None
        delta = _vision_tensor_to_storage(cube_ee_delta, store) if cube_ee_delta is not None else None
        episode = VisionEpisodeData(
            visual=_vision_tensor_to_storage(visual, store),
            teacher=_vision_tensor_to_storage(teacher, store),
            proprio=prop,
            teacher_phase=phase,
            cube_ee_delta=delta,
        )
        self.episodes.append(episode)
        self.total_steps += episode.length
        if len(self.episodes) > self.max_episodes:
            removed = self.episodes.pop(0)
            self.total_steps -= removed.length

    def add_in_progress_episodes(
        self,
        ep_visual: torch.Tensor,
        ep_teacher: torch.Tensor,
        ep_step: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        ep_teacher_phase: torch.Tensor | None = None,
        ep_proprio: torch.Tensor | None = None,
        ep_cube_ee_delta: torch.Tensor | None = None,
    ) -> int:
        if env_ids is None:
            env_ids = torch.arange(ep_step.shape[0], device=ep_step.device)
        added = 0
        for env_id in env_ids.tolist():
            length = int(ep_step[env_id].item())
            if length <= 0:
                continue
            phase = ep_teacher_phase[env_id, :length] if ep_teacher_phase is not None else None
            prop = ep_proprio[env_id, :length] if ep_proprio is not None else None
            delta = ep_cube_ee_delta[env_id, :length] if ep_cube_ee_delta is not None else None
            self.add_episode(
                ep_visual[env_id, :length],
                ep_teacher[env_id, :length],
                teacher_phase=phase,
                proprio=prop,
                cube_ee_delta=delta,
            )
            ep_step[env_id] = 0
            added += 1
        return added

    def _segment_pools(
        self,
        valid_episodes: list[VisionEpisodeData],
        seq_len: int,
        min_grasp_phase: int,
    ) -> tuple[list[tuple[VisionEpisodeData, int]], list[tuple[VisionEpisodeData, int]]]:
        all_segments: list[tuple[VisionEpisodeData, int]] = []
        grasp_segments: list[tuple[VisionEpisodeData, int]] = []
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
        return_teacher_phase: bool = False,
        return_cube_ee_delta: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, float
    ]:
        if len(self.episodes) == 0:
            raise RuntimeError("Vision episode replay buffer is empty.")

        valid_episodes = [ep for ep in self.episodes if ep.length >= seq_len]
        if len(valid_episodes) == 0:
            max_len = max(ep.length for ep in self.episodes)
            seq_len = max(2, min(seq_len, max_len))
            valid_episodes = [ep for ep in self.episodes if ep.length >= seq_len]
        if len(valid_episodes) == 0:
            raise RuntimeError("No vision episode is long enough for sequence sampling.")
        if return_cube_ee_delta:
            aux_episodes = [ep for ep in valid_episodes if ep.cube_ee_delta is not None]
            if aux_episodes:
                valid_episodes = aux_episodes

        all_segments, grasp_segments = self._segment_pools(valid_episodes, seq_len, min_grasp_phase)
        c, h, w = valid_episodes[0].visual.shape[1:]
        action_dim = valid_episodes[0].teacher.shape[-1]
        visual_batch = torch.zeros(batch_size, seq_len, c, h, w, device=self.device)
        teacher_batch = torch.zeros(batch_size, seq_len, action_dim, device=self.device)
        teacher_phase_batch = None
        if return_teacher_phase and valid_episodes[0].teacher_phase is not None:
            teacher_phase_batch = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
        proprio_batch = None
        if valid_episodes[0].proprio is not None:
            proprio_dim = valid_episodes[0].proprio.shape[-1]
            proprio_batch = torch.zeros(batch_size, seq_len, proprio_dim, device=self.device)
        cube_ee_delta_batch = None
        if return_cube_ee_delta and valid_episodes[0].cube_ee_delta is not None:
            cube_ee_delta_batch = torch.zeros(batch_size, seq_len, 3, device=self.device)

        grasp_hits = 0
        use_grasp_bias = grasp_sample_ratio > 0.0 and len(grasp_segments) > 0
        for i in range(batch_size):
            if use_grasp_bias and random.random() < grasp_sample_ratio:
                episode, start = grasp_segments[random.randrange(len(grasp_segments))]
                grasp_hits += 1
            else:
                episode, start = all_segments[random.randrange(len(all_segments))]
            visual_batch[i] = episode.visual[start : start + seq_len].to(self.device, non_blocking=True)
            teacher_batch[i] = episode.teacher[start : start + seq_len].to(self.device, non_blocking=True)
            if teacher_phase_batch is not None and episode.teacher_phase is not None:
                teacher_phase_batch[i] = episode.teacher_phase[start : start + seq_len].to(
                    self.device, non_blocking=True
                )
            if proprio_batch is not None and episode.proprio is not None:
                proprio_batch[i] = episode.proprio[start : start + seq_len].to(self.device, non_blocking=True)
            if cube_ee_delta_batch is not None and episode.cube_ee_delta is not None:
                cube_ee_delta_batch[i] = episode.cube_ee_delta[start : start + seq_len].to(
                    self.device, non_blocking=True
                )

        grasp_hit_rate = grasp_hits / batch_size if use_grasp_bias else 0.0
        if return_teacher_phase and return_cube_ee_delta:
            return visual_batch, teacher_batch, proprio_batch, teacher_phase_batch, cube_ee_delta_batch, grasp_hit_rate
        if return_teacher_phase:
            return visual_batch, teacher_batch, proprio_batch, teacher_phase_batch, grasp_hit_rate
        if return_cube_ee_delta:
            return visual_batch, teacher_batch, proprio_batch, cube_ee_delta_batch, grasp_hit_rate
        return visual_batch, teacher_batch, proprio_batch, grasp_hit_rate


def optimize_kl_bce_batch_vision(
    policy,
    optimizer: torch.optim.Optimizer,
    visual_batch: torch.Tensor,
    teacher_batch: torch.Tensor,
    teacher_arm_std: torch.Tensor,
    *,
    proprio_batch: torch.Tensor | None = None,
    gripper_bce_weight: float = 1.0,
    cube_ee_delta_batch: torch.Tensor | None = None,
    aux_cube_ee_weight: float = 0.0,
    aux_cube_ee_scale: float = 0.25,
    grad_interval: int = 20,
    max_grad_norm: float = 20.0,
) -> tuple[float, float, float, float, float]:
    if visual_batch.ndim != 5:
        raise ValueError(f"Expected visual_batch (B, T, C, H, W), got {tuple(visual_batch.shape)}")

    batch_size, seq_len, _, _, _ = visual_batch.shape
    arm_kl_losses: list[torch.Tensor] = []
    gripper_bce_losses: list[torch.Tensor] = []
    aux_delta_losses: list[torch.Tensor] = []
    aux_delta_l2_m: list[torch.Tensor] = []
    use_aux_delta = cube_ee_delta_batch is not None and aux_cube_ee_weight > 0.0
    aux_scale = max(float(aux_cube_ee_scale), 1.0e-6)

    hidden = init_rnn_state(batch_size, policy.rnn_hidden_dim, visual_batch.device)
    arm_chunk: list[torch.Tensor] = []
    for step_idx in range(seq_len):
        visual_step = visual_batch[:, step_idx : step_idx + 1]
        proprio_step = None if proprio_batch is None else proprio_batch[:, step_idx : step_idx + 1]
        cnn_feat = policy.visual_features(visual_step)
        features, hidden = policy.encode_from_visual_features(cnn_feat, hidden, proprio_step)
        arm_dist = policy.arm_dist_from_features(features)
        teacher_arm = teacher_batch[:, step_idx, :6]
        # Normalize the Gaussian KL by action dimension; gaussian_kl_to_teacher sums over 6 arm dims.
        step_arm_kl = gaussian_kl_to_teacher(arm_dist, teacher_arm, teacher_arm_std).mean() / policy.arm_action_dim
        arm_kl_losses.append(step_arm_kl)
        step_loss = step_arm_kl
        if use_aux_delta:
            pred_delta = policy.aux_delta_from_cnn_features(cnn_feat)[:, -1, :]
            target_delta = cube_ee_delta_batch[:, step_idx, :].to(device=pred_delta.device, dtype=pred_delta.dtype)
            pred_delta_norm = pred_delta / aux_scale
            target_delta_norm = target_delta / aux_scale
            step_aux_delta = torch.nn.functional.huber_loss(pred_delta_norm, target_delta_norm, delta=1.0)
            step_aux_l2_m = torch.linalg.norm(pred_delta.detach() - target_delta.detach(), dim=-1).mean()
            aux_delta_losses.append(step_aux_delta)
            aux_delta_l2_m.append(step_aux_l2_m)
            step_loss = step_loss + aux_cube_ee_weight * step_aux_delta
        arm_chunk.append(step_loss)

        if _bptt_should_update(step_idx, seq_len, grad_interval):
            arm_loss = torch.stack(arm_chunk).mean()
            optimizer.zero_grad(set_to_none=True)
            arm_loss.backward()
            clip_params = policy.arm_parameters()
            if use_aux_delta:
                clip_params += policy.aux_parameters()
            torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm)
            optimizer.step()
            arm_chunk = []
            hidden = detach_rnn_state(hidden)

    hidden = init_rnn_state(batch_size, policy.rnn_hidden_dim, visual_batch.device)
    gripper_chunk: list[torch.Tensor] = []
    policy.eval()
    for step_idx in range(seq_len):
        visual_step = visual_batch[:, step_idx : step_idx + 1]
        proprio_step = None if proprio_batch is None else proprio_batch[:, step_idx : step_idx + 1]
        teacher_gripper = teacher_batch[:, step_idx, 6]
        with torch.no_grad():
            features, hidden = policy.encode_step(visual_step, hidden, proprio_step)
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
    mean_aux_delta = float(torch.stack(aux_delta_losses).mean().item()) if aux_delta_losses else 0.0
    mean_aux_l2_m = float(torch.stack(aux_delta_l2_m).mean().item()) if aux_delta_l2_m else 0.0
    total = mean_arm_kl + gripper_bce_weight * mean_gripper_bce + aux_cube_ee_weight * mean_aux_delta
    return total, mean_arm_kl, mean_gripper_bce, mean_aux_delta, mean_aux_l2_m


def optimize_kl_bce_sequences_vision(
    policy,
    optimizer: torch.optim.Optimizer,
    buffer: VisionEpisodeReplayBuffer,
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
    aux_cube_ee_weight: float = 0.0,
    aux_cube_ee_scale: float = 0.25,
    visual_augment_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor] | None = None,
) -> tuple[float, float, float, float, float, float]:
    total_loss = 0.0
    arm_kl_acc = 0.0
    gripper_bce_acc = 0.0
    aux_delta_acc = 0.0
    aux_l2_m_acc = 0.0
    grasp_hit_acc = 0.0
    request_aux_delta = aux_cube_ee_weight > 0.0
    for _ in range(optim_steps):
        if visual_augment_fn is None:
            if request_aux_delta:
                visual_batch, teacher_batch, proprio_batch, cube_ee_delta_batch, grasp_hit_rate = buffer.sample_sequences(
                    batch_size,
                    seq_len,
                    grasp_sample_ratio=grasp_sample_ratio,
                    min_grasp_phase=min_grasp_phase,
                    return_cube_ee_delta=True,
                )
            else:
                visual_batch, teacher_batch, proprio_batch, grasp_hit_rate = buffer.sample_sequences(
                    batch_size,
                    seq_len,
                    grasp_sample_ratio=grasp_sample_ratio,
                    min_grasp_phase=min_grasp_phase,
                )
                cube_ee_delta_batch = None
        else:
            if request_aux_delta:
                (
                    visual_batch,
                    teacher_batch,
                    proprio_batch,
                    teacher_phase_batch,
                    cube_ee_delta_batch,
                    grasp_hit_rate,
                ) = buffer.sample_sequences(
                    batch_size,
                    seq_len,
                    grasp_sample_ratio=grasp_sample_ratio,
                    min_grasp_phase=min_grasp_phase,
                    return_teacher_phase=True,
                    return_cube_ee_delta=True,
                )
            else:
                visual_batch, teacher_batch, proprio_batch, teacher_phase_batch, grasp_hit_rate = buffer.sample_sequences(
                    batch_size,
                    seq_len,
                    grasp_sample_ratio=grasp_sample_ratio,
                    min_grasp_phase=min_grasp_phase,
                    return_teacher_phase=True,
                )
                cube_ee_delta_batch = None
            visual_batch = visual_augment_fn(visual_batch, teacher_phase_batch)
        batch_total, batch_arm_kl, batch_gripper_bce, batch_aux_delta, batch_aux_l2_m = optimize_kl_bce_batch_vision(
            policy,
            optimizer,
            visual_batch,
            teacher_batch,
            teacher_arm_std,
            proprio_batch=proprio_batch,
            gripper_bce_weight=gripper_bce_weight,
            cube_ee_delta_batch=cube_ee_delta_batch,
            aux_cube_ee_weight=aux_cube_ee_weight,
            aux_cube_ee_scale=aux_cube_ee_scale,
            grad_interval=grad_interval,
            max_grad_norm=max_grad_norm,
        )
        total_loss += batch_total
        arm_kl_acc += batch_arm_kl
        gripper_bce_acc += batch_gripper_bce
        aux_delta_acc += batch_aux_delta
        aux_l2_m_acc += batch_aux_l2_m
        grasp_hit_acc += grasp_hit_rate
    n = max(optim_steps, 1)
    return total_loss / n, arm_kl_acc / n, gripper_bce_acc / n, aux_delta_acc / n, aux_l2_m_acc / n, grasp_hit_acc / n
