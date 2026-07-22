# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation utilities shared by offline eval and sim rollout."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def resize_visual_frame(visual_chw: np.ndarray, image_size: int) -> np.ndarray:
    """Resize a single (C, H, W) frame."""
    if visual_chw.shape[-1] == image_size and visual_chw.shape[-2] == image_size:
        return visual_chw.astype(np.float32)
    c = visual_chw.shape[0]
    out = np.empty((c, image_size, image_size), dtype=np.float32)
    for ci in range(c):
        out[ci] = cv2.resize(visual_chw[ci], (image_size, image_size), interpolation=cv2.INTER_AREA)
    return out


def stack_visual_from_policy_obs(
    obs: dict[str, torch.Tensor],
    *,
    env_ids: slice | torch.Tensor | None = None,
    image_size: int | None = None,
) -> np.ndarray:
    """Build (N, C, H, W) visual tensor from env policy obs dict."""
    if env_ids is None:
        env_ids = slice(None)

    ee_mask = obs["ee_mask"][env_ids].detach().cpu().numpy()
    ee_depth = obs["ee_depth"][env_ids].detach().cpu().numpy()
    tp_mask = obs["third_person_mask"][env_ids].detach().cpu().numpy()
    tp_depth = obs["third_person_depth"][env_ids].detach().cpu().numpy()

    if ee_depth.ndim == 4 and ee_depth.shape[-1] == 1:
        ee_depth = ee_depth[..., 0]
    if tp_depth.ndim == 4 and tp_depth.shape[-1] == 1:
        tp_depth = tp_depth[..., 0]
    if ee_depth.ndim == 3:
        ee_depth = ee_depth[..., np.newaxis]
    if tp_depth.ndim == 3:
        tp_depth = tp_depth[..., np.newaxis]

    visual = np.concatenate([ee_mask, ee_depth, tp_mask, tp_depth], axis=-1).astype(np.float32)
    visual = np.transpose(visual, (0, 3, 1, 2))

    if image_size is not None:
        resized = np.empty((visual.shape[0], visual.shape[1], image_size, image_size), dtype=np.float32)
        for i in range(visual.shape[0]):
            resized[i] = resize_visual_frame(visual[i], image_size)
        visual = resized
    return visual


def stack_visual_sequence(frames: list[np.ndarray]) -> np.ndarray:
    """Stack list of (C,H,W) into (T,C,H,W)."""
    return np.stack(frames, axis=0).astype(np.float32)
