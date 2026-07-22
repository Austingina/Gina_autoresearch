# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DAgger aggregate buffer for vision episodes (RAM and disk backends)."""

from __future__ import annotations

import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch

from .kl_trainer import EpisodeData
from .vision_kl_trainer import VisionEpisodeData, VisionEpisodeReplayBuffer, _vision_tensor_to_storage


def _clip_episode_tensors(
    visual: torch.Tensor,
    teacher: torch.Tensor,
    teacher_phase: torch.Tensor | None,
    proprio: torch.Tensor | None,
    cube_ee_delta: torch.Tensor | None,
    *,
    max_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if max_steps <= 0 or visual.shape[0] <= max_steps:
        return visual, teacher, teacher_phase, proprio, cube_ee_delta
    return (
        visual[:max_steps],
        teacher[:max_steps],
        teacher_phase[:max_steps] if teacher_phase is not None else None,
        proprio[:max_steps] if proprio is not None else None,
        cube_ee_delta[:max_steps] if cube_ee_delta is not None else None,
    )


@dataclass
class VisionDaggerEpisodeData(VisionEpisodeData):
    teacher_success: bool = False


class VisionDaggerAggregateBuffer(VisionEpisodeReplayBuffer):
    """Growing aggregate buffer for vision DAgger with success-reserved pools."""

    def __init__(
        self,
        max_episodes: int,
        device: str | torch.device,
        *,
        success_reserve_ratio: float = 0.35,
        storage_device: str | torch.device = "cpu",
        max_store_episode_steps: int = 512,
    ):
        self.max_episodes = max_episodes
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device)
        self.max_store_episode_steps = max(0, int(max_store_episode_steps))
        self.total_steps = 0
        self.success_reserve_ratio = float(success_reserve_ratio)
        self._success_episodes: list[VisionDaggerEpisodeData] = []
        self._rolling_episodes: list[VisionDaggerEpisodeData] = []

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
        pool: list[VisionDaggerEpisodeData],
        episode: VisionDaggerEpisodeData,
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
        visual: torch.Tensor,
        teacher: torch.Tensor,
        teacher_phase: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        cube_ee_delta: torch.Tensor | None = None,
        *,
        teacher_success: bool = False,
    ) -> None:
        if visual.shape[0] == 0:
            return
        visual, teacher, teacher_phase, proprio, cube_ee_delta = _clip_episode_tensors(
            visual,
            teacher,
            teacher_phase,
            proprio,
            cube_ee_delta,
            max_steps=self.max_store_episode_steps,
        )
        store = self.storage_device
        phase = (
            _vision_tensor_to_storage(teacher_phase, store).to(dtype=torch.long)
            if teacher_phase is not None
            else None
        )
        prop = _vision_tensor_to_storage(proprio, store) if proprio is not None else None
        delta = _vision_tensor_to_storage(cube_ee_delta, store) if cube_ee_delta is not None else None
        episode = VisionDaggerEpisodeData(
            visual=_vision_tensor_to_storage(visual, store),
            teacher=_vision_tensor_to_storage(teacher, store),
            proprio=prop,
            teacher_phase=phase,
            cube_ee_delta=delta,
            teacher_success=teacher_success,
        )
        if teacher_success and self.success_cap > 0:
            self._append_to_pool(self._success_episodes, episode, self.success_cap)
        elif self.rolling_cap > 0:
            self._append_to_pool(self._rolling_episodes, episode, self.rolling_cap)
        elif self.success_cap > 0:
            self._append_to_pool(self._success_episodes, episode, self.success_cap)

    def state_dict(self) -> dict[str, Any]:
        packed = []
        for episode in self.episodes:
            item = {
                "visual": episode.visual.detach().cpu(),
                "teacher": episode.teacher.detach().cpu(),
                "teacher_phase": episode.teacher_phase.detach().cpu() if episode.teacher_phase is not None else None,
                "teacher_success": bool(getattr(episode, "teacher_success", False)),
            }
            if episode.proprio is not None:
                item["proprio"] = episode.proprio.detach().cpu()
            if episode.cube_ee_delta is not None:
                item["cube_ee_delta"] = episode.cube_ee_delta.detach().cpu()
            packed.append(item)
        return {
            "backend": "ram",
            "max_episodes": self.max_episodes,
            "success_reserve_ratio": self.success_reserve_ratio,
            "total_steps": self.total_steps,
            "episodes": packed,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._success_episodes.clear()
        self._rolling_episodes.clear()
        self.total_steps = 0
        self.max_episodes = int(state.get("max_episodes", self.max_episodes))
        self.success_reserve_ratio = float(state.get("success_reserve_ratio", self.success_reserve_ratio))
        for item in state["episodes"]:
            phase = item.get("teacher_phase")
            if phase is not None:
                phase = phase.to(device=self.storage_device, dtype=torch.long)
            proprio = item.get("proprio")
            if proprio is not None:
                proprio = proprio.to(device=self.storage_device)
            cube_ee_delta = item.get("cube_ee_delta")
            if cube_ee_delta is not None:
                cube_ee_delta = cube_ee_delta.to(device=self.storage_device)
            self.add_episode(
                item["visual"].to(device=self.storage_device),
                item["teacher"].to(device=self.storage_device),
                teacher_phase=phase,
                proprio=proprio,
                cube_ee_delta=cube_ee_delta,
                teacher_success=bool(item.get("teacher_success", False)),
            )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str | torch.device) -> VisionDaggerAggregateBuffer:
        state = torch.load(path, map_location="cpu", weights_only=False)
        buffer = cls(
            max_episodes=int(state.get("max_episodes", 5000)),
            device=device,
            success_reserve_ratio=float(state.get("success_reserve_ratio", 0.35)),
        )
        buffer.load_state_dict(state)
        return buffer


@dataclass
class VisionDaggerDiskEpisodeMeta:
    """Metadata for one on-disk episode file."""

    path: str
    length: int
    teacher_success: bool = False
    has_teacher_phase: bool = False
    has_proprio: bool = False
    has_cube_ee_delta: bool = False


class VisionDaggerAggregateDiskBuffer(VisionDaggerAggregateBuffer):
    """DAgger aggregate buffer backed by per-episode files on disk."""

    def __init__(
        self,
        max_episodes: int,
        device: str | torch.device,
        *,
        success_reserve_ratio: float = 0.35,
        storage_dir: str,
        cache_size: int = 8,
        visual_storage_dtype: str = "float16",
        max_store_episode_steps: int = 512,
    ):
        self.max_episodes = max_episodes
        self.device = torch.device(device)
        self.storage_device = torch.device("cpu")
        self.max_store_episode_steps = max(0, int(max_store_episode_steps))
        self.total_steps = 0
        self.success_reserve_ratio = float(success_reserve_ratio)
        self._success_episodes: list[VisionDaggerDiskEpisodeMeta] = []
        self._rolling_episodes: list[VisionDaggerDiskEpisodeMeta] = []
        self.storage_dir = os.path.abspath(storage_dir)
        self._episodes_dir = os.path.join(self.storage_dir, "episodes")
        os.makedirs(self._episodes_dir, exist_ok=True)
        self.cache_size = max(0, int(cache_size))
        self._episode_cache: OrderedDict[str, dict[str, torch.Tensor | None]] = OrderedDict()
        self._episode_counter = 0
        if visual_storage_dtype not in ("float16", "float32"):
            raise ValueError(f"Unsupported visual_storage_dtype: {visual_storage_dtype}")
        self.visual_storage_dtype = visual_storage_dtype

    @property
    def episodes(self) -> list[EpisodeData]:
        return self._success_episodes + self._rolling_episodes

    @property
    def max_episode_length(self) -> int:
        if not self.episodes:
            return 0
        return max(ep.length for ep in self.episodes)

    def _recompute_total_steps(self) -> None:
        self.total_steps = sum(ep.length for ep in self.episodes)

    def _next_episode_path(self) -> str:
        self._episode_counter += 1
        return os.path.join(self._episodes_dir, f"ep_{self._episode_counter:09d}.pt")

    def _to_visual_dtype(self, visual: torch.Tensor) -> torch.Tensor:
        if self.visual_storage_dtype == "float16":
            return visual.to(dtype=torch.float16)
        return visual.to(dtype=torch.float32)

    def _clear_cache_entry(self, path: str) -> None:
        if path in self._episode_cache:
            del self._episode_cache[path]

    def _delete_episode_file(self, path: str) -> None:
        self._clear_cache_entry(path)
        if os.path.isfile(path):
            os.remove(path)

    def _append_to_pool(
        self,
        pool: list[VisionDaggerDiskEpisodeMeta],
        episode: VisionDaggerDiskEpisodeMeta,
        cap: int,
    ) -> None:
        if cap <= 0:
            self._delete_episode_file(episode.path)
            return
        pool.append(episode)
        while len(pool) > cap:
            removed = pool.pop(0)
            self._delete_episode_file(removed.path)
        self._recompute_total_steps()

    def add_episode(
        self,
        visual: torch.Tensor,
        teacher: torch.Tensor,
        teacher_phase: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        cube_ee_delta: torch.Tensor | None = None,
        *,
        teacher_success: bool = False,
    ) -> None:
        if visual.shape[0] == 0:
            return
        visual, teacher, teacher_phase, proprio, cube_ee_delta = _clip_episode_tensors(
            visual,
            teacher,
            teacher_phase,
            proprio,
            cube_ee_delta,
            max_steps=self.max_store_episode_steps,
        )
        path = self._next_episode_path()
        payload = {
            "visual": self._to_visual_dtype(_vision_tensor_to_storage(visual, self.storage_device)),
            "teacher": _vision_tensor_to_storage(teacher, self.storage_device).to(dtype=torch.float32),
            "teacher_phase": (
                _vision_tensor_to_storage(teacher_phase, self.storage_device).to(dtype=torch.long)
                if teacher_phase is not None
                else None
            ),
            "proprio": _vision_tensor_to_storage(proprio, self.storage_device).to(dtype=torch.float32)
            if proprio is not None
            else None,
            "cube_ee_delta": _vision_tensor_to_storage(cube_ee_delta, self.storage_device).to(dtype=torch.float32)
            if cube_ee_delta is not None
            else None,
        }
        torch.save(payload, path)
        meta = VisionDaggerDiskEpisodeMeta(
            path=path,
            length=int(visual.shape[0]),
            teacher_success=teacher_success,
            has_teacher_phase=teacher_phase is not None,
            has_proprio=proprio is not None,
            has_cube_ee_delta=cube_ee_delta is not None,
        )
        if teacher_success and self.success_cap > 0:
            self._append_to_pool(self._success_episodes, meta, self.success_cap)
        elif self.rolling_cap > 0:
            self._append_to_pool(self._rolling_episodes, meta, self.rolling_cap)
        elif self.success_cap > 0:
            self._append_to_pool(self._success_episodes, meta, self.success_cap)
        else:
            self._delete_episode_file(path)

    def _load_episode(self, path: str) -> dict[str, torch.Tensor | None]:
        cached = self._episode_cache.get(path)
        if cached is not None:
            self._episode_cache.move_to_end(path)
            return cached
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if self.cache_size > 0:
            self._episode_cache[path] = payload
            self._episode_cache.move_to_end(path)
            while len(self._episode_cache) > self.cache_size:
                self._episode_cache.popitem(last=False)
        return payload

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
            aux_episodes = [ep for ep in valid_episodes if ep.has_cube_ee_delta]
            if aux_episodes:
                valid_episodes = aux_episodes

        ref_payload = self._load_episode(valid_episodes[0].path)
        ref_visual = ref_payload["visual"]
        ref_teacher = ref_payload["teacher"]
        if ref_visual is None or ref_teacher is None:
            raise RuntimeError("Corrupt on-disk episode payload.")
        c, h, w = ref_visual.shape[1:]
        action_dim = ref_teacher.shape[-1]
        visual_batch = torch.zeros(batch_size, seq_len, c, h, w, device=self.device)
        teacher_batch = torch.zeros(batch_size, seq_len, action_dim, device=self.device)
        teacher_phase_batch = None
        if return_teacher_phase and valid_episodes[0].has_teacher_phase:
            teacher_phase_batch = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
        proprio_batch = None
        if valid_episodes[0].has_proprio:
            ref_proprio = ref_payload.get("proprio")
            if ref_proprio is not None:
                proprio_batch = torch.zeros(batch_size, seq_len, ref_proprio.shape[-1], device=self.device)
        cube_ee_delta_batch = None
        if return_cube_ee_delta and valid_episodes[0].has_cube_ee_delta:
            cube_ee_delta_batch = torch.zeros(batch_size, seq_len, 3, device=self.device)

        grasp_hits = 0
        grasp_candidates = [ep for ep in valid_episodes if ep.has_teacher_phase]
        use_grasp_bias = grasp_sample_ratio > 0.0 and len(grasp_candidates) > 0
        for i in range(batch_size):
            selected = random.choice(grasp_candidates if use_grasp_bias and random.random() < grasp_sample_ratio else valid_episodes)
            payload = self._load_episode(selected.path)
            visual = payload["visual"]
            teacher = payload["teacher"]
            if visual is None or teacher is None:
                raise RuntimeError(f"Corrupt on-disk episode payload: {selected.path}")
            start = random.randint(0, selected.length - seq_len)
            if use_grasp_bias and selected.has_teacher_phase and random.random() < grasp_sample_ratio:
                phase = payload.get("teacher_phase")
                if phase is not None and phase.shape[0] >= seq_len:
                    phase_windows = phase.unfold(0, seq_len, 1).max(dim=1).values
                    grasp_starts = (phase_windows >= min_grasp_phase).nonzero(as_tuple=False).squeeze(-1)
                    if grasp_starts.numel() > 0:
                        start = int(grasp_starts[random.randrange(grasp_starts.numel())].item())
                        grasp_hits += 1
            visual_batch[i] = visual[start : start + seq_len].to(device=self.device, dtype=torch.float32, non_blocking=True)
            teacher_batch[i] = teacher[start : start + seq_len].to(self.device, non_blocking=True)
            if teacher_phase_batch is not None:
                phase = payload.get("teacher_phase")
                if phase is not None:
                    teacher_phase_batch[i] = phase[start : start + seq_len].to(self.device, non_blocking=True)
            if proprio_batch is not None:
                prop = payload.get("proprio")
                if prop is not None:
                    proprio_batch[i] = prop[start : start + seq_len].to(self.device, non_blocking=True)
            if cube_ee_delta_batch is not None:
                delta = payload.get("cube_ee_delta")
                if delta is not None:
                    cube_ee_delta_batch[i] = delta[start : start + seq_len].to(self.device, non_blocking=True)

        grasp_hit_rate = grasp_hits / batch_size if use_grasp_bias else 0.0
        if return_teacher_phase and return_cube_ee_delta:
            return visual_batch, teacher_batch, proprio_batch, teacher_phase_batch, cube_ee_delta_batch, grasp_hit_rate
        if return_teacher_phase:
            return visual_batch, teacher_batch, proprio_batch, teacher_phase_batch, grasp_hit_rate
        if return_cube_ee_delta:
            return visual_batch, teacher_batch, proprio_batch, cube_ee_delta_batch, grasp_hit_rate
        return visual_batch, teacher_batch, proprio_batch, grasp_hit_rate

    def state_dict(self) -> dict[str, Any]:
        def _pack(items: list[VisionDaggerDiskEpisodeMeta]) -> list[dict[str, Any]]:
            packed = []
            for item in items:
                packed.append(
                    {
                        "path": os.path.relpath(item.path, self.storage_dir),
                        "length": item.length,
                        "teacher_success": bool(item.teacher_success),
                        "has_teacher_phase": bool(item.has_teacher_phase),
                        "has_proprio": bool(item.has_proprio),
                        "has_cube_ee_delta": bool(item.has_cube_ee_delta),
                    }
                )
            return packed

        return {
            "backend": "disk",
            "max_episodes": self.max_episodes,
            "success_reserve_ratio": self.success_reserve_ratio,
            "total_steps": self.total_steps,
            "storage_dir": self.storage_dir,
            "visual_storage_dtype": self.visual_storage_dtype,
            "episode_counter": self._episode_counter,
            "success_episodes": _pack(self._success_episodes),
            "rolling_episodes": _pack(self._rolling_episodes),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._success_episodes.clear()
        self._rolling_episodes.clear()
        self.total_steps = 0
        self._episode_cache.clear()
        self.max_episodes = int(state.get("max_episodes", self.max_episodes))
        self.success_reserve_ratio = float(state.get("success_reserve_ratio", self.success_reserve_ratio))
        self.visual_storage_dtype = str(state.get("visual_storage_dtype", self.visual_storage_dtype))
        self._episode_counter = int(state.get("episode_counter", 0))
        state_storage_dir = os.path.abspath(str(state.get("storage_dir", self.storage_dir)))
        if state_storage_dir != self.storage_dir:
            # Keep current requested storage_dir, but remap relative paths from saved dir.
            saved_root = state_storage_dir
        else:
            saved_root = self.storage_dir

        def _restore(items: list[dict[str, Any]]) -> list[VisionDaggerDiskEpisodeMeta]:
            restored: list[VisionDaggerDiskEpisodeMeta] = []
            for item in items:
                rel_path = str(item["path"])
                candidate = os.path.join(saved_root, rel_path)
                if not os.path.isfile(candidate):
                    candidate = os.path.join(self.storage_dir, rel_path)
                if not os.path.isfile(candidate):
                    continue
                restored.append(
                    VisionDaggerDiskEpisodeMeta(
                        path=os.path.abspath(candidate),
                        length=int(item["length"]),
                        teacher_success=bool(item.get("teacher_success", False)),
                        has_teacher_phase=bool(item.get("has_teacher_phase", False)),
                        has_proprio=bool(item.get("has_proprio", False)),
                        has_cube_ee_delta=bool(item.get("has_cube_ee_delta", False)),
                    )
                )
            return restored

        self._success_episodes = _restore(state.get("success_episodes", []))
        self._rolling_episodes = _restore(state.get("rolling_episodes", []))
        self._recompute_total_steps()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str,
        device: str | torch.device,
        *,
        storage_dir: str | None = None,
        cache_size: int = 8,
        visual_storage_dtype: str = "float16",
    ) -> VisionDaggerAggregateDiskBuffer:
        state = torch.load(path, map_location="cpu", weights_only=False)
        disk_dir = storage_dir or str(state.get("storage_dir", os.path.join(os.path.dirname(path), "dagger_episodes")))
        buffer = cls(
            max_episodes=int(state.get("max_episodes", 5000)),
            device=device,
            success_reserve_ratio=float(state.get("success_reserve_ratio", 0.35)),
            storage_dir=disk_dir,
            cache_size=cache_size,
            visual_storage_dtype=str(state.get("visual_storage_dtype", visual_storage_dtype)),
        )
        buffer.load_state_dict(state)
        return buffer


def create_vision_dagger_buffer(
    *,
    backend: str,
    max_episodes: int,
    device: str | torch.device,
    success_reserve_ratio: float,
    storage_dir: str,
    cache_size: int,
    visual_storage_dtype: str,
    max_store_episode_steps: int = 512,
) -> VisionDaggerAggregateBuffer:
    if backend == "disk":
        return VisionDaggerAggregateDiskBuffer(
            max_episodes=max_episodes,
            device=device,
            success_reserve_ratio=success_reserve_ratio,
            storage_dir=storage_dir,
            cache_size=cache_size,
            visual_storage_dtype=visual_storage_dtype,
            max_store_episode_steps=max_store_episode_steps,
        )
    return VisionDaggerAggregateBuffer(
        max_episodes=max_episodes,
        device=device,
        success_reserve_ratio=success_reserve_ratio,
        max_store_episode_steps=max_store_episode_steps,
    )


def load_vision_dagger_buffer(
    path: str,
    *,
    device: str | torch.device,
    default_backend: str,
    default_max_episodes: int,
    default_success_reserve_ratio: float,
    storage_dir: str,
    cache_size: int,
    visual_storage_dtype: str,
    max_store_episode_steps: int = 512,
) -> VisionDaggerAggregateBuffer:
    state = torch.load(path, map_location="cpu", weights_only=False)
    backend = str(state.get("backend", default_backend))
    if backend == "disk":
        buffer = VisionDaggerAggregateDiskBuffer(
            max_episodes=int(state.get("max_episodes", default_max_episodes)),
            device=device,
            success_reserve_ratio=float(state.get("success_reserve_ratio", default_success_reserve_ratio)),
            storage_dir=storage_dir or str(state.get("storage_dir", os.path.join(os.path.dirname(path), "dagger_episodes"))),
            cache_size=cache_size,
            visual_storage_dtype=str(state.get("visual_storage_dtype", visual_storage_dtype)),
            max_store_episode_steps=int(state.get("max_store_episode_steps", max_store_episode_steps)),
        )
    else:
        buffer = VisionDaggerAggregateBuffer(
            max_episodes=int(state.get("max_episodes", default_max_episodes)),
            device=device,
            success_reserve_ratio=float(state.get("success_reserve_ratio", default_success_reserve_ratio)),
            max_store_episode_steps=int(state.get("max_store_episode_steps", max_store_episode_steps)),
        )
    buffer.load_state_dict(state)
    return buffer
