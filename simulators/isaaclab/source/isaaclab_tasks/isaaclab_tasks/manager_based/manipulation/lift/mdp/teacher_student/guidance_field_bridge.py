# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge learned or oracle guidance fields into vision DAgger observations."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from ..data_collection import get_ee_pose_in_robot_root_frame
from ..interaction_geometry import compute_lift_subgoal_vectors_3d
from ..observations import object_position_in_robot_root_frame
from .teacher_phase import infer_teacher_phase
from .vision_policy_obs import (
    stack_mask_channels_from_policy_obs_torch,
    stack_visual_from_policy_obs_torch,
)


def _ensure_autoresearch_path() -> str:
    root = os.environ.get("AUTORESEARCH_ROOT", "/home/l/code/AutoResearch")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


class LearnedGuidanceField:
    """Load MaskFieldCNN checkpoint and append guidance channels to visual obs.

    Supports 2D (legacy x-z) or 3D (Interaction Geometry) output, and optional ψ
    conditioning (G-phase): when ``cond_dim > 0`` the field is conditioned on
    contact_latched derived from the teacher phase.
    """

    GRASP_PHASE_MIN = 3

    def __init__(self, checkpoint: str | Path, device: str | torch.device):
        _ensure_autoresearch_path()
        from goals.field_model import load_field_checkpoint

        self.device = torch.device(device)
        self.model, self.meta = load_field_checkpoint(checkpoint, device=self.device)
        self.model.eval()
        self.enabled = True
        self.input_channels = int(self.model.in_channels)
        self.guidance_channels = int(self.model.out_dim)
        self.cond_dim = int(getattr(self.model, "cond_dim", 0))
        print(
            f"[INFO] Loaded guidance field from {checkpoint} "
            f"(in_channels={self.input_channels}, out_dim={self.guidance_channels}, "
            f"cond_dim={self.cond_dim}, val_align={self.meta.get('field_alignment_val', 'n/a')})"
        )

    @property
    def extra_visual_channels(self) -> int:
        return self.guidance_channels if self.enabled else 0

    def _stack_field_input(self, obs: dict[str, torch.Tensor], image_size: int | None) -> torch.Tensor:
        if self.input_channels >= 8:
            return stack_visual_from_policy_obs_torch(obs, image_size=image_size)
        return stack_mask_channels_from_policy_obs_torch(obs, image_size=image_size)

    def _psi_cond(self, n: int, env, teacher_phase: torch.Tensor | None) -> torch.Tensor | None:
        if self.cond_dim <= 0:
            return None
        if teacher_phase is None:
            if env is None:
                raise ValueError("G-phase field requires env or teacher_phase for ψ")
            teacher_phase = infer_teacher_phase(env)
        psi = (teacher_phase >= self.GRASP_PHASE_MIN).reshape(n, 1).to(self.device, dtype=torch.float32)
        return psi

    @torch.no_grad()
    def predict_vectors(
        self,
        obs: dict[str, torch.Tensor],
        image_size: int | None = None,
        *,
        env=None,
        teacher_phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual = self._stack_field_input(obs, image_size=image_size)
        cond = self._psi_cond(visual.shape[0], env, teacher_phase)
        return self.model(visual.to(self.device), cond)

    @torch.no_grad()
    def alignment_metrics(
        self,
        obs: dict[str, torch.Tensor],
        *,
        env,
        image_size: int | None = None,
        teacher_phase: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compare learned G with runtime GT Interaction Geometry G on current states."""
        if teacher_phase is None:
            teacher_phase = infer_teacher_phase(env)
        pred = self.predict_vectors(obs, image_size=image_size, env=env, teacher_phase=teacher_phase)

        ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
        cube_pos_b = object_position_in_robot_root_frame(env)
        gt3 = compute_lift_subgoal_vectors_3d(
            ee_pos_b,
            cube_pos_b,
            teacher_phase,
            grasp_phase_min=self.GRASP_PHASE_MIN,
        )
        if pred.shape[-1] == 2:
            gt = torch.stack([gt3[:, 0], gt3[:, 2]], dim=-1)
            gt = torch.nn.functional.normalize(gt, dim=-1, eps=1e-6)
        else:
            gt = gt3

        pred = torch.nn.functional.normalize(pred, dim=-1, eps=1e-6)
        cos = torch.nn.functional.cosine_similarity(pred, gt.to(pred.device), dim=-1).clamp(-1.0, 1.0)
        l2 = torch.linalg.norm(pred - gt.to(pred.device), dim=-1)
        angle_deg = torch.rad2deg(torch.acos(cos))
        reach = teacher_phase.to(pred.device) < self.GRASP_PHASE_MIN
        actuate = ~reach

        def _mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> float:
            if mask is not None:
                if not bool(mask.any().item()):
                    return float("nan")
                x = x[mask]
            return float(x.mean().item())

        return {
            "cos": _mean(cos),
            "angle_deg": _mean(angle_deg),
            "l2": _mean(l2),
            "reach_cos": _mean(cos, reach),
            "actuate_cos": _mean(cos, actuate),
            "reach_n": float(reach.sum().item()),
            "actuate_n": float(actuate.sum().item()),
        }

    @torch.no_grad()
    def augment_visual(
        self,
        visual: torch.Tensor,
        obs: dict[str, torch.Tensor],
        *,
        image_size: int | None = None,
        env=None,
        teacher_phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append broadcast G maps (N, C, H, W) -> (N, C+out_dim, H, W)."""
        if not self.enabled:
            return visual
        g = self.predict_vectors(obs, image_size=image_size, env=env, teacher_phase=teacher_phase)
        return append_guidance_channels_nd(visual, g)


class OracleRuleGuidanceField:
    """Oracle GT subgoal field (B3): phase-aware 3D unit vector in robot base frame."""

    GUIDANCE_CHANNELS = 3
    GRASP_PHASE_MIN = 3

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.enabled = True
        print(
            "[INFO] Oracle rule guidance field: GT 3D subgoal "
            "(REACH EE→cube, ACTUATE vertical above cube / anchor_field_rule / B3)"
        )

    @property
    def extra_visual_channels(self) -> int:
        return self.GUIDANCE_CHANNELS if self.enabled else 0

    @staticmethod
    def compute_subgoal_vectors_3d(
        env,
        teacher_phase: torch.Tensor,
        *,
        grasp_phase_min: int = 3,
    ) -> torch.Tensor:
        """Return (N, 3) unit vectors: EE→cube (REACH), +z (ACTUATE / κ=vertical_free)."""
        ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
        cube_pos_b = object_position_in_robot_root_frame(env)
        return compute_lift_subgoal_vectors_3d(
            ee_pos_b, cube_pos_b, teacher_phase, grasp_phase_min=grasp_phase_min
        )

    @torch.no_grad()
    def predict_vectors(self, env, teacher_phase: torch.Tensor | None = None) -> torch.Tensor:
        if teacher_phase is None:
            teacher_phase = infer_teacher_phase(env)
        return self.compute_subgoal_vectors_3d(env, teacher_phase, grasp_phase_min=self.GRASP_PHASE_MIN)

    @torch.no_grad()
    def augment_visual(
        self,
        visual: torch.Tensor,
        obs: dict[str, torch.Tensor],
        *,
        image_size: int | None = None,
        env=None,
        teacher_phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del obs, image_size
        if not self.enabled:
            return visual
        if env is None:
            raise ValueError("OracleRuleGuidanceField.augment_visual requires env=...")
        g = self.predict_vectors(env, teacher_phase=teacher_phase)
        return append_guidance_channels_nd(visual, g)


def append_guidance_channels(visual: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
    """visual (N,C,H,W), guidance (N,2) -> (N,C+2,H,W)."""
    return append_guidance_channels_nd(visual, guidance)


def append_guidance_channels_nd(visual: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
    """visual (N,C,H,W), guidance (N,D) -> (N,C+D,H,W) via per-channel broadcast."""
    n, _, h, w = visual.shape
    d = int(guidance.shape[-1])
    maps = guidance.reshape(n, d, 1, 1).expand(n, d, h, w)
    return torch.cat([visual, maps], dim=1)
