"""Mask-conditioned guidance field predictor (Step 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskFieldCNN(nn.Module):
    """Predict guidance direction from dual-camera mask (+ depth) features.

    out_dim: 2 (legacy x-z) or 3 (Interaction Geometry ℝ³).
    cond_dim: optional ψ conditioning (e.g. 1 = contact_latched) for G-phase.
    """

    def __init__(self, in_channels: int = 8, hidden: int = 64, out_dim: int = 2, cond_dim: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64 + cond_dim, out_dim)

    def forward(self, mask: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """visual: (B, C, H, W) [, cond (B, cond_dim)] -> unit (B, out_dim) direction."""
        x = self.encoder(mask).flatten(1)
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError(f"MaskFieldCNN(cond_dim={self.cond_dim}) requires cond input")
            x = torch.cat([x, cond.to(x.dtype)], dim=-1)
        v = self.head(x)
        return F.normalize(v, dim=-1, eps=1e-6)

    @torch.no_grad()
    def predict_numpy(self, mask_np, cond_np=None) -> tuple[float, ...]:
        import numpy as np

        x = torch.from_numpy(np.asarray(mask_np, dtype=np.float32)).unsqueeze(0)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        device = next(self.parameters()).device
        cond = None
        if self.cond_dim > 0:
            cond = torch.from_numpy(np.asarray(cond_np, dtype=np.float32)).reshape(1, self.cond_dim).to(device)
        v = self.forward(x.to(device), cond).squeeze(0).cpu().numpy()
        return tuple(float(c) for c in v)


def cosine_alignment(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity in [−1, 1]."""
    return F.cosine_similarity(pred, target, dim=-1).mean()


def save_field_checkpoint(
    path: Path,
    model: MaskFieldCNN,
    *,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_class": "MaskFieldCNN",
        "in_channels": model.in_channels,
        "out_dim": model.out_dim,
        "cond_dim": model.cond_dim,
        "meta": meta,
    }
    torch.save(payload, path)


def load_field_checkpoint(path: Path | str, device: str | torch.device = "cpu") -> tuple[MaskFieldCNN, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MaskFieldCNN(
        in_channels=int(ckpt.get("in_channels", 8)),
        out_dim=int(ckpt.get("out_dim", 2)),
        cond_dim=int(ckpt.get("cond_dim", 0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt.get("meta", {})
