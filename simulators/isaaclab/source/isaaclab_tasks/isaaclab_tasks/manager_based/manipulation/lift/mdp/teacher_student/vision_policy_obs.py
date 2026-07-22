# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch helpers for vision DAgger / eval rollouts."""

from __future__ import annotations

import torch
import torch.nn.functional as F

VISUAL_OBS_KEYS = ("ee_mask", "ee_depth", "third_person_mask", "third_person_depth")
MASK_OBS_KEYS = ("ee_mask", "third_person_mask")
VISUAL_LAYOUT_CHANNELS = "channels"
VISUAL_LAYOUT_EE_CHANNELS = "ee_channels"
VISUAL_LAYOUT_MOSAIC = "mosaic"
DEFAULT_MOSAIC_HISTORY_LEN = 1


def mosaic_spatial_shape(image_size: int) -> tuple[int, int]:
    """Return (H, W) for mosaic layout: ee over top, single timestep (no temporal concat)."""
    h = int(image_size)
    return 2 * h, h


def policy_visual_channels(*, visual_layout: str) -> int:
    if visual_layout in (VISUAL_LAYOUT_EE_CHANNELS, VISUAL_LAYOUT_MOSAIC):
        return 4
    if visual_layout == VISUAL_LAYOUT_CHANNELS:
        return 8
    raise ValueError(f"Unsupported visual_layout={visual_layout!r}")


def stack_mask_channels_from_policy_obs_torch(
    obs: dict[str, torch.Tensor],
    *,
    env_ids: slice | torch.Tensor | None = None,
    image_size: int | None = None,
) -> torch.Tensor:
    """Build (N, 6, H, W) mask tensor for guidance-field inference."""
    if env_ids is None:
        env_ids = slice(None)

    parts: list[torch.Tensor] = []
    for key in MASK_OBS_KEYS:
        tensor = obs[key][env_ids]
        if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        parts.append(tensor.float())

    masks = torch.cat(parts, dim=1)
    if image_size is not None and (masks.shape[-1] != image_size or masks.shape[-2] != image_size):
        masks = F.interpolate(masks, size=(image_size, image_size), mode="area", align_corners=None)
    return masks


def stack_visual_from_policy_obs_torch(
    obs: dict[str, torch.Tensor],
    *,
    env_ids: slice | torch.Tensor | None = None,
    image_size: int | None = None,
) -> torch.Tensor:
    """Build (N, C, H, W) visual tensor from env policy obs dict (C=8)."""
    if env_ids is None:
        env_ids = slice(None)

    parts: list[torch.Tensor] = []
    for key in VISUAL_OBS_KEYS:
        tensor = obs[key][env_ids]
        if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        parts.append(tensor.float())

    visual = torch.cat(parts, dim=1)
    if image_size is not None and (visual.shape[-1] != image_size or visual.shape[-2] != image_size):
        visual = F.interpolate(visual, size=(image_size, image_size), mode="area", align_corners=None)
    return visual


def _split_ee_top_mask_depth(frame_8ch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split (..., 8, H, W) into ee/top (..., 4, H, W) mask RGB + depth pairs."""
    if frame_8ch.shape[-3] != 8:
        raise ValueError(f"Expected 8 visual channels, got {frame_8ch.shape[-3]}")
    ee = frame_8ch[..., 0:4, :, :]
    top = frame_8ch[..., 4:8, :, :]
    return ee, top


def build_mosaic_from_frame(frame_8ch: torch.Tensor) -> torch.Tensor:
    """Build (..., 4, 2H, W) mosaic from current (..., 8, H, W) frame."""
    ee, top = _split_ee_top_mask_depth(frame_8ch)
    return torch.cat([ee, top], dim=-2)


def build_mosaic_from_frame_history(
    frame_history: torch.Tensor,
    *,
    history_len: int = DEFAULT_MOSAIC_HISTORY_LEN,
) -> torch.Tensor:
    """Build mosaic from frame history; only the latest frame is used (temporal via LSTM)."""
    del history_len
    if frame_history.ndim < 4:
        raise ValueError(f"Expected frame history (..., T, 8, H, W), got {frame_history.shape}")
    return build_mosaic_from_frame(frame_history[..., -1, :, :, :])


class VisualMosaicHistory:
    """Compatibility wrapper: mosaic uses the current frame only (no spatial history buffer)."""

    def __init__(
        self,
        num_envs: int,
        image_size: int,
        device: str | torch.device,
        *,
        history_len: int = DEFAULT_MOSAIC_HISTORY_LEN,
    ):
        self.num_envs = int(num_envs)
        self.image_size = int(image_size)
        self.history_len = 1
        self.device = torch.device(device)
        if int(history_len) > 1:
            print(
                f"[WARN] mosaic_history_len={history_len} is ignored; "
                "mosaic layout uses a single frame (4,2H,W). Use LSTM for temporal context."
            )

    def reset(self, env_ids: torch.Tensor | slice | list[int]) -> None:
        del env_ids

    def push_and_build(
        self,
        frame_8ch: torch.Tensor,
        env_ids: slice | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build (N, 4, 2H, W) mosaic from current (N, 8, H, W) frame."""
        del env_ids
        return build_mosaic_from_frame(frame_8ch)


def stack_policy_visual_torch(
    obs: dict[str, torch.Tensor],
    *,
    image_size: int | None = None,
    visual_layout: str = VISUAL_LAYOUT_CHANNELS,
    mosaic_history: VisualMosaicHistory | None = None,
    env_ids: slice | torch.Tensor | None = None,
) -> torch.Tensor:
    """Build dual-camera channels, EE-only channels, or dual-camera mosaic input."""
    frame_8ch = stack_visual_from_policy_obs_torch(obs, env_ids=env_ids, image_size=image_size)
    if visual_layout == VISUAL_LAYOUT_CHANNELS:
        return frame_8ch
    if visual_layout == VISUAL_LAYOUT_EE_CHANNELS:
        return frame_8ch[..., 0:4, :, :]
    if visual_layout == VISUAL_LAYOUT_MOSAIC:
        if mosaic_history is not None:
            return mosaic_history.push_and_build(frame_8ch, env_ids=env_ids)
        return build_mosaic_from_frame(frame_8ch)
    raise ValueError(f"Unsupported visual_layout={visual_layout!r}")


def extract_proprio_from_policy_obs_torch(
    obs: dict[str, torch.Tensor],
    *,
    env_ids: slice | torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return (N, D) proprio if ``student_state`` is present."""
    if "student_state" not in obs:
        return None
    if env_ids is None:
        return obs["student_state"].float()
    return obs["student_state"][env_ids].float()
