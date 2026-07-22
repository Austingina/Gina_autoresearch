# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint loading helpers for offline CQL."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch

from .agent import CQLSACAgentLSTM
from .dataset import ActionStats


def _register_numpy_unpickle_compat() -> None:
    """Map NumPy 2.x module paths when loading pickles on NumPy 1.x (and vice versa)."""
    try:
        import numpy._core  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    legacy_core = getattr(np, "core", None)
    if legacy_core is None:
        return

    aliases = {
        "numpy._core": legacy_core,
        "numpy._core.multiarray": legacy_core.multiarray,
        "numpy._core.umath": legacy_core.umath,
        "numpy._core._multiarray_umath": getattr(legacy_core, "_multiarray_umath", legacy_core.multiarray),
    }
    for name, module in aliases.items():
        sys.modules.setdefault(name, module)


def pack_checkpoint(
    *,
    epoch: int,
    global_step: int,
    agent: CQLSACAgentLSTM,
    action_stats: ActionStats,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Build a checkpoint dict using plain Python types (avoids NumPy pickle issues)."""
    return {
        "epoch": epoch,
        "global_step": global_step,
        "agent": agent.state_dict(),
        "action_stats": {"low": action_stats.low.tolist(), "high": action_stats.high.tolist()},
        "args": dict(args),
    }


def load_checkpoint(path: str, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    _register_numpy_unpickle_compat()
    return torch.load(path, map_location=map_location, weights_only=False)


def build_action_stats(checkpoint: dict[str, Any]) -> ActionStats:
    stats = checkpoint["action_stats"]
    if isinstance(stats, ActionStats):
        return stats
    low = np.asarray(stats["low"], dtype=np.float32)
    high = np.asarray(stats["high"], dtype=np.float32)
    return ActionStats(low=low, high=high)


def build_agent_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[CQLSACAgentLSTM, ActionStats, dict[str, Any]]:
    """Reconstruct agent and action stats from a training checkpoint."""
    ckpt_args = checkpoint.get("args", {})
    action_stats = build_action_stats(checkpoint)
    action_dim = int(checkpoint["agent"].get("action_dim", len(action_stats.low)))
    obs_mode = str(checkpoint["agent"].get("obs_mode", ckpt_args.get("obs_mode", "vision")))

    proprio_dim = 0
    if obs_mode in ("proprio", "vision_proprio"):
        proprio_dim = int(ckpt_args.get("proprio_dim", 10))

    agent = CQLSACAgentLSTM(
        action_dim=action_dim,
        device=device,
        obs_mode=obs_mode,
        visual_channels=8,
        proprio_dim=proprio_dim,
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        cnn_feat_dim=int(ckpt_args.get("cnn_feat_dim", 128)),
        lstm_hidden_dim=int(ckpt_args.get("lstm_hidden_dim", 128)),
        lstm_layers=int(ckpt_args.get("lstm_layers", 1)),
        learning_rate=float(ckpt_args.get("learning_rate", 3e-5)),
        gamma=float(ckpt_args.get("gamma", 0.99)),
        tau=float(ckpt_args.get("tau", 5e-3)),
        cql_weight=float(ckpt_args.get("cql_weight", 1.0)),
        cql_temp=float(ckpt_args.get("cql_temp", 1.0)),
        target_action_gap=float(ckpt_args.get("target_action_gap", 10.0)),
        with_lagrange=bool(ckpt_args.get("with_lagrange", False)),
        num_random_actions=int(ckpt_args.get("num_random_actions", 10)),
    )
    agent.load_state_dict(checkpoint["agent"])
    agent.encoder.eval()
    agent.actor.eval()
    agent.critic1.eval()
    agent.critic2.eval()
    return agent, action_stats, ckpt_args
