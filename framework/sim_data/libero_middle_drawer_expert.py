"""Pure NumPy online expert logic for LIBERO's middle-drawer task.

This module deliberately has no LIBERO/MuJoCo runtime dependency.  It resolves a
small MuJoCo-shaped geometry view and emits actions only; simulator oracle gates,
not this controller, own any success label.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

import numpy as np

# Collision-handle center from the pinned LIBERO wooden_cabinet.xml at
# commit 8f1084e3132a39270c3a13ebe37270a43ece2a01, expressed in the
# moving ``cabinet_middle`` body frame.
MIDDLE_HANDLE_LOCAL = np.asarray([-0.00278, -0.10161, 0.11041], dtype=np.float64)
MIDDLE_JOINT_RANGE = np.asarray([-0.16, 0.01], dtype=np.float64)
_MJ_JNT_SLIDE = 2


class GeometryError(ValueError):
    """Raised when drawer geometry cannot be resolved unambiguously and safely."""


class State(str, Enum):
    PREGRASP = "PREGRASP"
    APPROACH = "APPROACH"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    PULL = "PULL"
    HOLD = "HOLD"
    FAILED = "FAILED"


@dataclass(frozen=True)
class DrawerGeometry:
    joint_id: int
    body_id: int
    joint_range: np.ndarray
    axis_world: np.ndarray
    handle_world: np.ndarray
    pull_direction: np.ndarray


@dataclass(frozen=True)
class StepResult:
    action: np.ndarray
    state: State
    confirm_count: int
    failed: bool = False
    error: str | None = None


def sanitize_action(action: Iterable[float]) -> np.ndarray:
    """Return an exactly 7-D, finite action clipped to the normalized action box."""
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (7,):
        raise ValueError("action must have shape (7,)")
    if not np.isfinite(value).all():
        raise ValueError("action must contain only finite values")
    return np.clip(value, -1.0, 1.0)


def _finite_vector(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise GeometryError(f"{label} must be a finite array with shape {shape}")
    return array


def resolve_middle_drawer_geometry(
    model: Any,
    data: Any,
    *,
    candidate_joint_ids: Iterable[int],
) -> DrawerGeometry:
    """Resolve the one BDDL-nominated middle-drawer joint, failing closed.

    The future runtime adapter must obtain ``candidate_joint_ids`` from
    ``object_sites_dict['wooden_cabinet_1_middle_region'].joints``.  Global
    uniqueness across the full MuJoCo model is intentionally not assumed.
    """
    try:
        njnt = int(model.njnt)
        nbody = int(model.nbody)
        joint_types = np.asarray(model.jnt_type)
        joint_ranges = np.asarray(model.jnt_range, dtype=np.float64)
        joint_bodies = np.asarray(model.jnt_bodyid)
        joint_axes = np.asarray(model.jnt_axis, dtype=np.float64)
        body_positions = np.asarray(data.xpos, dtype=np.float64)
        body_rotations = np.asarray(data.xmat, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise GeometryError("required model/data geometry is missing") from exc

    if joint_types.shape != (njnt,):
        raise GeometryError("joint type table has an invalid shape")
    try:
        target_ids = tuple(int(value) for value in candidate_joint_ids)
    except (TypeError, ValueError) as exc:
        raise GeometryError("target joint ids are invalid") from exc
    if len(target_ids) != 1:
        raise GeometryError("expected exactly one explicit target joint")
    joint_id = target_ids[0]
    if joint_id < 0 or joint_id >= njnt:
        raise GeometryError("target joint id is out of range")
    if int(joint_types[joint_id]) != _MJ_JNT_SLIDE:
        raise GeometryError("target joint must be a slide joint")

    if joint_ranges.shape != (njnt, 2):
        raise GeometryError("joint range table has an invalid shape")
    joint_range = _finite_vector(joint_ranges[joint_id], (2,), "joint range")
    if not np.allclose(joint_range, MIDDLE_JOINT_RANGE, rtol=0.0, atol=1e-6):
        raise GeometryError("target slide joint range does not match pinned middle drawer")

    if joint_bodies.shape != (njnt,):
        raise GeometryError("joint body table has an invalid shape")
    body_id = int(joint_bodies[joint_id])
    if body_id <= 0 or body_id >= nbody:
        raise GeometryError("slide joint must belong to a valid non-world body")
    try:
        if hasattr(model, "body_id2name"):
            body_name = str(model.body_id2name(body_id))
        else:
            body_names = tuple(model.body_names)
            if len(body_names) != nbody:
                raise GeometryError("body name table has an invalid shape")
            body_name = str(body_names[body_id])
    except (AttributeError, TypeError, IndexError) as exc:
        raise GeometryError("target body name is unavailable") from exc
    if "cabinet_middle" not in body_name:
        raise GeometryError("target joint body is not cabinet_middle")

    if joint_axes.shape != (njnt, 3):
        raise GeometryError("joint axis table has an invalid shape")
    axis_local = _finite_vector(joint_axes[joint_id], (3,), "joint axis")
    axis_norm = float(np.linalg.norm(axis_local))
    if axis_norm <= 1e-12:
        raise GeometryError("slide joint axis must be non-zero")

    if body_positions.shape != (nbody, 3):
        raise GeometryError("body position table has an invalid shape")
    body_position = _finite_vector(body_positions[body_id], (3,), "body position")
    if body_rotations.shape == (nbody, 9):
        body_rotations = body_rotations.reshape(nbody, 3, 3)
    if body_rotations.shape != (nbody, 3, 3):
        raise GeometryError("body rotation table has an invalid shape")
    rotation = _finite_vector(body_rotations[body_id], (3, 3), "body rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0.0:
        raise GeometryError("body rotation must be a proper orthonormal matrix")

    axis_world = rotation @ (axis_local / axis_norm)
    axis_world /= np.linalg.norm(axis_world)
    handle_world = body_position + rotation @ MIDDLE_HANDLE_LOCAL
    pull_direction = -axis_world
    if not np.isfinite(handle_world).all() or float(np.dot(pull_direction, axis_world)) >= 0.0:
        raise GeometryError("resolved drawer geometry is unsafe")

    return DrawerGeometry(
        joint_id=joint_id,
        body_id=body_id,
        joint_range=joint_range.copy(),
        axis_world=axis_world,
        handle_world=handle_world,
        pull_direction=pull_direction,
    )


def projected_joint_delta(geometry: DrawerGeometry, pull_distance: float) -> float:
    """Projected slide-coordinate change for a positive pull command."""
    distance = float(pull_distance)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("pull_distance must be finite and positive")
    return float(np.dot(geometry.pull_direction * distance, geometry.axis_world))


class MiddleDrawerExpert:
    """Small waypoint FSM with consecutive pose/gripper-error confirmation."""

    def __init__(
        self,
        *,
        confirm_steps: int = 2,
        position_tolerance: float = 0.01,
        gripper_tolerance: float = 0.05,
        pregrasp_offset: float = 0.08,
        pull_distance: float = 0.12,
        position_gain: float = 10.0,
    ) -> None:
        if confirm_steps < 1:
            raise ValueError("confirm_steps must be positive")
        scalars = (position_tolerance, gripper_tolerance, pregrasp_offset, pull_distance, position_gain)
        if not np.isfinite(scalars).all() or min(scalars) <= 0.0:
            raise ValueError("controller scales must be finite and positive")
        self.confirm_steps = int(confirm_steps)
        self.position_tolerance = float(position_tolerance)
        self.gripper_tolerance = float(gripper_tolerance)
        self.pregrasp_offset = float(pregrasp_offset)
        self.pull_distance = float(pull_distance)
        self.position_gain = float(position_gain)
        self.state = State.PREGRASP
        self._confirm_count = 0
        self._pull_target: np.ndarray | None = None

    def _fail(self, error: str) -> StepResult:
        self.state = State.FAILED
        self._confirm_count = 0
        return StepResult(np.zeros(7, dtype=np.float64), self.state, 0, True, error)

    def _confirmed(self, condition: bool) -> bool:
        self._confirm_count = self._confirm_count + 1 if condition else 0
        if self._confirm_count >= self.confirm_steps:
            self._confirm_count = 0
            return True
        return False

    @staticmethod
    def _geometry_is_safe(geometry: Any) -> bool:
        if not isinstance(geometry, DrawerGeometry):
            return False
        vectors = (geometry.axis_world, geometry.handle_world, geometry.pull_direction)
        return all(np.asarray(vector).shape == (3,) and np.isfinite(vector).all() for vector in vectors)

    def _target(self, geometry: DrawerGeometry) -> np.ndarray:
        if self.state == State.PREGRASP:
            return geometry.handle_world + geometry.pull_direction * self.pregrasp_offset
        if self.state in (State.APPROACH, State.CLOSE_GRIPPER):
            return geometry.handle_world
        if self._pull_target is not None:
            return self._pull_target.copy()
        return geometry.handle_world + geometry.pull_direction * self.pull_distance

    def step(
        self,
        ee_position: Iterable[float],
        geometry: DrawerGeometry | None,
        *,
        gripper_error: float = 1.0,
    ) -> StepResult:
        """Advance the FSM from observable end-effector and gripper errors only."""
        if self.state == State.FAILED:
            return self._fail("controller is latched failed")
        ee = np.asarray(ee_position, dtype=np.float64)
        if ee.shape != (3,) or not np.isfinite(ee).all():
            return self._fail("end-effector position must be a finite 3-vector")
        if not self._geometry_is_safe(geometry):
            return self._fail("drawer geometry is missing or non-finite")
        assert geometry is not None
        if not np.isfinite(gripper_error):
            return self._fail("gripper error must be finite")

        target = self._target(geometry)
        if self.state in (State.PREGRASP, State.APPROACH, State.PULL):
            error_ok = float(np.linalg.norm(target - ee)) <= self.position_tolerance
            if self._confirmed(error_ok):
                self.state = {
                    State.PREGRASP: State.APPROACH,
                    State.APPROACH: State.CLOSE_GRIPPER,
                    State.PULL: State.HOLD,
                }[self.state]
        elif self.state == State.CLOSE_GRIPPER:
            if self._confirmed(abs(float(gripper_error)) <= self.gripper_tolerance):
                self._pull_target = (
                    geometry.handle_world + geometry.pull_direction * self.pull_distance
                ).copy()
                self.state = State.PULL

        target = self._target(geometry)
        if self.state == State.HOLD:
            delta = np.zeros(3, dtype=np.float64)
        else:
            delta = (target - ee) * self.position_gain
        # Provisional robosuite convention.  Runtime integration must verify
        # this sign with a one-step gripper-qpos smoke before any rollout.
        gripper_command = -1.0 if self.state in (State.PREGRASP, State.APPROACH) else 1.0
        action = sanitize_action([*delta, 0.0, 0.0, 0.0, gripper_command])
        return StepResult(action, self.state, self._confirm_count)
