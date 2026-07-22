# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small eval helpers with no Isaac Sim imports (unit-testable)."""

from __future__ import annotations

import torch


def early_reset_env_ids(
    *masks: torch.Tensor | None,
    max_env_id: int | None = None,
) -> torch.Tensor:
    """Combine early-done/success bool masks into integer env_ids for ``env.reset``."""
    combined: torch.Tensor | None = None
    for mask in masks:
        if mask is None or not mask.any():
            continue
        combined = mask if combined is None else (combined | mask)
    if combined is None:
        return torch.empty(0, dtype=torch.long)
    env_ids = combined.nonzero(as_tuple=False).squeeze(-1)
    if env_ids.ndim == 0:
        env_ids = env_ids.unsqueeze(0)
    if max_env_id is not None:
        env_ids = env_ids[env_ids < max_env_id]
    return env_ids
