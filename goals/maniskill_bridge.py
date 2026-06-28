"""ManiSkill3 ↔ Interaction Geometry bridge.

Maps a ManiSkill env to the shared (ψ, κ) interface and produces the oracle
guidance field G* used by:
  - B3 (anchor_field_rule) at rollout time, and
  - Step-1 G_θ supervision labels.

Task → κ (constraint geometry), validated at runtime against finite-diff:
  PickCube-v1          → vertical_free  (free lift)
  OpenCabinetDrawer-v1 → slide          (prismatic)
  OpenCabinetDoor-v1   → revolute       (hinge)

Everything is computed in the world frame from privileged sim state; the
student never sees this — it only sees mask+depth (+ broadcast G channels).
"""

from __future__ import annotations

import numpy as np
import torch

from goals.interaction_geometry import (
    ConstraintKappa,
    InteractionPrimitive,
    actuate_direction_3d,
    quat_wxyz_to_axis_x,
    reach_direction_3d,
)

TASK_KAPPA: dict[str, ConstraintKappa] = {
    "PickCube-v1": ConstraintKappa.VERTICAL_FREE,
    "OpenCabinetDrawer-v1": ConstraintKappa.SLIDE,
    "OpenCabinetDoor-v1": ConstraintKappa.REVOLUTE,
}


def _np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def env_kappa(env_id: str) -> ConstraintKappa:
    base_id = env_id.split(":")[-1]
    if base_id not in TASK_KAPPA:
        raise KeyError(f"No κ mapping for env_id={env_id!r}; known: {list(TASK_KAPPA)}")
    return TASK_KAPPA[base_id]


def extract_interaction_state(env, env_id: str) -> dict[str, np.ndarray]:
    """Pull world-frame anchor/contact + κ parameters for every parallel env.

    Returns numpy arrays of shape (B, ...):
      ee_pos (B,3), contact_pos (B,3), and κ-specific:
        slide   : world_axis (B,3)
        revolute: world_axis (B,3), hinge_origin (B,3)
      plus latch signals (B,) where available (grasped / open_frac).
    """
    base = env.unwrapped
    kappa = env_kappa(env_id)
    out: dict[str, np.ndarray] = {}

    ee_pos = _np(base.agent.tcp.pose.p).reshape(-1, 3)
    out["ee_pos"] = ee_pos.astype(np.float32)

    if kappa == ConstraintKappa.VERTICAL_FREE:
        # PickCube: anchor = cube
        cube = getattr(base, "cube", None) or getattr(base, "obj", None)
        contact = _np(cube.pose.p).reshape(-1, 3)
        out["contact_pos"] = contact.astype(np.float32)
        out["anchor_pos"] = contact.astype(np.float32)
        if hasattr(base, "agent") and hasattr(base.agent, "is_grasping"):
            try:
                out["grasped"] = _np(base.agent.is_grasping(base.cube)).reshape(-1).astype(np.float32)
            except Exception:
                pass
        return out

    # Cabinet tasks: anchor = handle, κ params from the target joint pose
    handle_pos = _np(base.handle_link_positions()).reshape(-1, 3)
    out["contact_pos"] = handle_pos.astype(np.float32)
    out["handle_pos"] = handle_pos.astype(np.float32)

    j = base.handle_link.joint
    gp = j.global_pose
    quat = _np(gp.q).reshape(-1, 4)  # wxyz
    origin = _np(gp.p).reshape(-1, 3)
    world_axis = np.stack([quat_wxyz_to_axis_x(quat[i]) for i in range(quat.shape[0])], axis=0)
    out["world_axis"] = world_axis.astype(np.float32)
    out["hinge_origin"] = origin.astype(np.float32)

    # open fraction latch
    qpos = _np(j.qpos).reshape(-1)
    lim = _np(j.limits).reshape(-1, 2)
    span = np.clip(lim[:, 1] - lim[:, 0], 1e-6, None)
    out["open_frac"] = ((qpos - lim[:, 0]) / span).astype(np.float32)
    return out


def oracle_reach_field(state: dict[str, np.ndarray]) -> np.ndarray:
    """G*_REACH per env: unit(contact - ee). (B,3), zeros where degenerate."""
    ee = state["ee_pos"]
    contact = state["contact_pos"]
    b = ee.shape[0]
    g = np.zeros((b, 3), dtype=np.float32)
    for i in range(b):
        d = reach_direction_3d(ee[i], contact[i])
        if d is not None:
            g[i] = d
    return g


def oracle_actuate_field(state: dict[str, np.ndarray], kappa: ConstraintKappa, open_sign: float = 1.0) -> np.ndarray:
    """G*_ACTUATE per env along κ. (B,3)."""
    b = state["ee_pos"].shape[0]
    g = np.zeros((b, 3), dtype=np.float32)
    for i in range(b):
        if kappa == ConstraintKappa.VERTICAL_FREE:
            d = actuate_direction_3d(kappa, anchor_pos=state["anchor_pos"][i])
        elif kappa == ConstraintKappa.SLIDE:
            d = actuate_direction_3d(kappa, world_axis=state["world_axis"][i], open_sign=open_sign)
        elif kappa == ConstraintKappa.REVOLUTE:
            d = actuate_direction_3d(
                kappa,
                world_axis=state["world_axis"][i],
                hinge_origin=state["hinge_origin"][i],
                handle_pos=state["handle_pos"][i],
                open_sign=open_sign,
            )
        else:
            d = None
        if d is not None:
            g[i] = d
    return g


def oracle_field(
    env,
    env_id: str,
    *,
    reach_dist_thresh: float = 0.05,
    open_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Phase-gated oracle G* (B,3) + ψ per env (0=REACH,1=ACTUATE) + raw state.

    ψ latch:
      - lift     : grasped flag (fallback: ee within reach_dist of cube)
      - cabinet  : ee within reach_dist of handle OR joint already opening
    """
    state = extract_interaction_state(env, env_id)
    kappa = env_kappa(env_id)
    ee = state["ee_pos"]
    contact = state["contact_pos"]
    dist = np.linalg.norm(contact - ee, axis=1)

    if "grasped" in state:
        actuate = state["grasped"] > 0.5
    elif "open_frac" in state:
        actuate = (dist < reach_dist_thresh) | (state["open_frac"] > 0.02)
    else:
        actuate = dist < reach_dist_thresh

    reach_g = oracle_reach_field(state)
    act_g = oracle_actuate_field(state, kappa, open_sign=open_sign)
    g = np.where(actuate[:, None], act_g, reach_g).astype(np.float32)
    psi = actuate.astype(np.float32)  # 0 reach, 1 actuate
    return g, psi, state
