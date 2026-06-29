"""Interaction Geometry — shared G conditioning and attractor rules.

Lift is the first task instance:
  ψ=REACH   → EE toward contact site (cube center)
  ψ=ACTUATE → anchor moves along κ (lift: vertical above cube)

See docs/INTERACTION_GEOMETRY.md for cross-task (drawer/door/button) plan.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

# Default lift ACTUATE attractor: cube spawn xy, +Δz in robot base frame (z up).
DEFAULT_LIFT_HEIGHT_ABOVE_CUBE_M = 0.20


class InteractionPrimitive(str, Enum):
    """Phase of interaction — shared across manipulation tasks."""

    REACH = "reach"
    GRASP = "grasp"
    ACTUATE = "actuate"
    HOLD = "hold"


class ConstraintKappa(str, Enum):
    """Local constraint geometry for ACTUATE (task-specific κ)."""

    VERTICAL_FREE = "vertical_free"  # lift: straight up above anchor
    SLIDE = "slide"  # drawer
    REVOLUTE = "revolute"  # door
    NORMAL_PRESS = "normal_press"  # button


def lift_actuate_attractor_3d(
    cube_pos: np.ndarray,
    *,
    height_above: float = DEFAULT_LIFT_HEIGHT_ABOVE_CUBE_M,
) -> np.ndarray:
    """Target point directly above the cube (same x,y; z + height_above)."""
    cube = np.asarray(cube_pos, dtype=np.float32).reshape(3)
    attractor = cube.copy()
    attractor[2] = cube[2] + float(height_above)
    return attractor


def lift_reach_direction_3d(ee_pos: np.ndarray, cube_pos: np.ndarray) -> np.ndarray | None:
    """Unit vector EE → cube (REACH / ψ=reach)."""
    vec = np.asarray(cube_pos, dtype=np.float32).reshape(3) - np.asarray(ee_pos, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-4:
        return None
    return (vec / norm).astype(np.float32)


def lift_actuate_direction_3d(
    cube_pos: np.ndarray,
    *,
    height_above: float = DEFAULT_LIFT_HEIGHT_ABOVE_CUBE_M,
) -> np.ndarray:
    """Unit vector cube → attractor above cube (ACTUATE / κ=vertical_free).

    Pure vertical in base frame when attractor shares cube xy.
    """
    cube = np.asarray(cube_pos, dtype=np.float32).reshape(3)
    attractor = lift_actuate_attractor_3d(cube, height_above=height_above)
    vec = attractor - cube
    norm = float(np.linalg.norm(vec))
    if norm < 1e-4:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def lift_reach_direction_2d_xz(ee_pos: np.ndarray, cube_pos: np.ndarray) -> np.ndarray | None:
    """Legacy 2D G head: (Δx, Δz) in robot base frame."""
    vec = np.array(
        [cube_pos[0] - ee_pos[0], cube_pos[2] - ee_pos[2]],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(vec))
    if norm < 1e-4:
        return None
    return (vec / norm).astype(np.float32)


def lift_actuate_direction_2d_xz() -> np.ndarray:
    """Legacy 2D ACTUATE: vertical in (x, z) encoding — (0, 1)."""
    return np.array([0.0, 1.0], dtype=np.float32)


def subgoal_direction_3d(
    student_state: np.ndarray,
    t: int,
    phase: float,
    *,
    height_above: float = DEFAULT_LIFT_HEIGHT_ABOVE_CUBE_M,
    ee_pose: np.ndarray | None = None,
) -> np.ndarray | None:
    """Phase-aware lift G* for Step 1 (ψ-conditioned attractor labels)."""
    ee = student_state[t, 0:3]
    cube = student_state[t, 4:7]
    if phase < 0.5:
        return lift_reach_direction_3d(ee, cube)
    if ee_pose is not None:
        return lift_actuate_direction_from_ee_forward_3d(ee_pose[t, 3:7])
    return lift_actuate_direction_3d(cube, height_above=height_above)


def subgoal_direction_2d_xz(
    student_state: np.ndarray,
    t: int,
    phase: float,
) -> np.ndarray | None:
    """Phase-aware lift G* for legacy MaskFieldCNN (out_dim=2)."""
    ee = student_state[t, 0:3]
    cube = student_state[t, 4:7]
    if phase < 0.5:
        return lift_reach_direction_2d_xz(ee, cube)
    return lift_actuate_direction_2d_xz()


# ---------------------------------------------------------------------------
# Cross-task Interaction Geometry (ψ, κ) — shared across lift/drawer/door/...
#
# REACH  (ψ=reach):   unit(contact - ee)                      — task-agnostic
# ACTUATE (ψ=actuate): direction the anchor moves along κ:
#     κ=vertical_free : +z above anchor                       (lift)
#     κ=slide         : world prismatic axis toward open      (drawer)
#     κ=revolute      : axis × (handle - hinge), toward open  (door)
#     κ=normal_press  : -surface normal (into the button)     (button)
#
# All inputs are world/base-frame 3D; outputs are unit vectors in the same
# frame. Simulator bridges (e.g. ManiSkill) supply axis/hinge from joint pose.
# ---------------------------------------------------------------------------

_EPS = 1e-6


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return None if fallback is None else np.asarray(fallback, dtype=np.float32).reshape(3)
    return (v / n).astype(np.float32)


def quat_wxyz_to_axis_x(quat_wxyz: np.ndarray) -> np.ndarray:
    """World-frame x-axis of a frame given its (w, x, y, z) quaternion.

    SAPIEN/ManiSkill joint frames place the joint axis along local +x, so this
    returns the world joint axis (rotation matrix first column).
    """
    w, x, y, z = (float(c) for c in np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
    col0 = np.array(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
        dtype=np.float32,
    )
    return _unit(col0, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))


def quat_wxyz_to_axis_z(quat_wxyz: np.ndarray) -> np.ndarray:
    """World/base-frame z-axis of a frame given its (w, x, y, z) quaternion."""
    w, x, y, z = (float(c) for c in np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
    col2 = np.array(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
        dtype=np.float32,
    )
    return _unit(col2, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))


def lift_actuate_direction_from_ee_forward_3d(ee_quat_wxyz: np.ndarray) -> np.ndarray:
    """Lift ACTUATE G*: retreat opposite the current EE forward (+Z) axis."""
    forward = quat_wxyz_to_axis_z(ee_quat_wxyz)
    return _unit(-forward, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))


def reach_direction_3d(ee_pos: np.ndarray, contact_pos: np.ndarray) -> np.ndarray | None:
    """ψ=REACH: unit vector EE → contact site (task-agnostic)."""
    return _unit(np.asarray(contact_pos, np.float32).reshape(3) - np.asarray(ee_pos, np.float32).reshape(3))


def slide_actuate_direction_3d(world_axis: np.ndarray, open_sign: float = 1.0) -> np.ndarray | None:
    """κ=SLIDE (prismatic): unit slide axis oriented toward 'open'.

    `world_axis` = prismatic joint axis in world frame (e.g. from joint pose).
    `open_sign` = +1 if qpos increases toward open along +axis (SAPIEN default).
    """
    return _unit(float(open_sign) * np.asarray(world_axis, np.float32).reshape(3))


def revolute_actuate_direction_3d(
    world_axis: np.ndarray,
    hinge_origin: np.ndarray,
    handle_pos: np.ndarray,
    open_sign: float = 1.0,
) -> np.ndarray | None:
    """κ=REVOLUTE (door): instantaneous tangent of the handle about the hinge.

    tangent = open_sign * (axis × (handle - hinge_origin)), normalized.
    `open_sign` = +1 if qpos increases toward open by right-hand rule (+axis).
    """
    axis = _unit(world_axis)
    if axis is None:
        return None
    r = np.asarray(handle_pos, np.float32).reshape(3) - np.asarray(hinge_origin, np.float32).reshape(3)
    return _unit(float(open_sign) * np.cross(axis, r))


def normal_press_direction_3d(surface_normal: np.ndarray) -> np.ndarray | None:
    """κ=NORMAL_PRESS (button): into the surface = -outward normal."""
    return _unit(-np.asarray(surface_normal, np.float32).reshape(3))


def actuate_direction_3d(
    kappa: "ConstraintKappa | str",
    *,
    anchor_pos: np.ndarray | None = None,
    world_axis: np.ndarray | None = None,
    hinge_origin: np.ndarray | None = None,
    handle_pos: np.ndarray | None = None,
    surface_normal: np.ndarray | None = None,
    open_sign: float = 1.0,
    height_above: float = DEFAULT_LIFT_HEIGHT_ABOVE_CUBE_M,
) -> np.ndarray | None:
    """Dispatch ACTUATE G* by constraint geometry κ (world/base frame)."""
    k = ConstraintKappa(kappa) if not isinstance(kappa, ConstraintKappa) else kappa
    if k == ConstraintKappa.VERTICAL_FREE:
        if anchor_pos is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return lift_actuate_direction_3d(anchor_pos, height_above=height_above)
    if k == ConstraintKappa.SLIDE:
        if world_axis is None:
            raise ValueError("κ=slide needs world_axis")
        return slide_actuate_direction_3d(world_axis, open_sign=open_sign)
    if k == ConstraintKappa.REVOLUTE:
        if world_axis is None or hinge_origin is None or handle_pos is None:
            raise ValueError("κ=revolute needs world_axis, hinge_origin, handle_pos")
        return revolute_actuate_direction_3d(world_axis, hinge_origin, handle_pos, open_sign=open_sign)
    if k == ConstraintKappa.NORMAL_PRESS:
        if surface_normal is None:
            raise ValueError("κ=normal_press needs surface_normal")
        return normal_press_direction_3d(surface_normal)
    raise ValueError(f"Unknown κ={kappa!r}")
