# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Save mosaic policy inputs to disk for manual inspection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover
    import imageio as iio  # type: ignore[no-redef]


def _chw01_to_hwc_u8(chw: np.ndarray) -> np.ndarray:
    chw = np.clip(chw, 0.0, 1.0)
    if chw.shape[0] == 1:
        gray = (chw[0] * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    rgb = (chw[:3].transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return rgb


def _tile_mask_depth(four_ch_hw: np.ndarray) -> np.ndarray:
    """Render (4,H,W) mask RGB + depth as a horizontal (H, 2W, 3) RGB tile."""
    mask = _chw01_to_hwc_u8(four_ch_hw[:3])
    depth = _chw01_to_hwc_u8(four_ch_hw[3:4])
    return np.concatenate([mask, depth], axis=1)


def mosaic_tensor_to_panel_np(mosaic: np.ndarray, *, image_size: int) -> np.ndarray:
    """Convert one (4, 2H, W) mosaic into an RGB panel for human inspection.

    Layout: ee mask|depth on top, top mask|depth on bottom (single timestep).
    """
    if mosaic.ndim != 3 or mosaic.shape[0] != 4:
        raise ValueError(f"Expected mosaic (4, 2H, W), got {mosaic.shape}")
    h = int(image_size)
    if mosaic.shape[1] != 2 * h:
        raise ValueError(f"Mosaic height mismatch for image_size={h}: {mosaic.shape}")
    width = mosaic.shape[2]
    if width == 3 * h:
        mosaic = mosaic[:, :, -h:]
    elif width != h:
        raise ValueError(f"Mosaic width mismatch for image_size={h}: {mosaic.shape}")

    ee = mosaic[:, :h, :]
    top = mosaic[:, h : 2 * h, :]
    return np.concatenate([_tile_mask_depth(ee), _tile_mask_depth(top)], axis=0)


class MosaicPolicyInputDumper:
    """Periodically dump mosaic tensors right before policy inference."""

    def __init__(
        self,
        save_dir: str | Path,
        image_size: int,
        *,
        interval: int = 1,
        max_saves: int = 200,
        env_ids: tuple[int, ...] = (0,),
        save_raw_pt: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.image_size = int(image_size)
        self.interval = max(int(interval), 1)
        self.max_saves = max(int(max_saves), 1)
        self.env_ids = tuple(int(i) for i in env_ids)
        self.save_raw_pt = bool(save_raw_pt)
        self._global_step = 0
        self._saved = 0
        self.save_dir.mkdir(parents=True, exist_ok=True)
        readme = self.save_dir / "README.txt"
        if not readme.exists():
            readme.write_text(
                "Mosaic policy inputs saved before policy.act().\n"
                "Panel PNG layout: ee (top) and top camera (bottom); left=mask RGB, right=depth.\n"
                "Raw .pt files store exact float tensors in [0,1] (shape 4,2H,W).\n",
                encoding="utf-8",
            )

    def maybe_save(
        self,
        mosaic: torch.Tensor,
        *,
        iteration: int | None = None,
        phase: str = "collect",
    ) -> None:
        """Save selected env mosaics if interval/max limits allow."""
        if self._saved >= self.max_saves:
            return
        if self._global_step % self.interval != 0:
            self._global_step += 1
            return

        step_tag = self._global_step
        self._global_step += 1

        mosaic_cpu = mosaic.detach().float().cpu()
        for env_id in self.env_ids:
            if env_id < 0 or env_id >= mosaic_cpu.shape[0]:
                continue
            one = mosaic_cpu[env_id].numpy()
            stem_parts = [phase]
            if iteration is not None:
                stem_parts.append(f"iter{int(iteration):05d}")
            stem_parts.extend([f"step{step_tag:06d}", f"env{env_id}"])
            stem = "_".join(stem_parts)

            panel = mosaic_tensor_to_panel_np(one, image_size=self.image_size)
            iio.imwrite(self.save_dir / f"{stem}_panel.png", panel)
            if self.save_raw_pt:
                torch.save(torch.from_numpy(one), self.save_dir / f"{stem}_mosaic.pt")

            self._saved += 1
            if self._saved >= self.max_saves:
                break
