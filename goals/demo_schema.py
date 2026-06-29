"""Shared demonstration schema for IsaacLab Interaction Geometry tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from goals.interaction_geometry import ConstraintKappa, InteractionPrimitive


class IsaacTaskName(str, Enum):
    LIFT_CUBE = "lift_cube"
    OPEN_DRAWER = "open_drawer"
    OPEN_SIDE_DOOR = "open_side_door"


class DemoKeys:
    """Canonical HDF5 paths under each ``data/demo_*`` group."""

    EE_RGB = "obs/ee_rgb"
    EE_DEPTH = "obs/ee_depth"
    EE_MASK = "obs/ee_mask"
    THIRD_PERSON_RGB = "obs/third_person_rgb"
    THIRD_PERSON_DEPTH = "obs/third_person_depth"
    THIRD_PERSON_MASK = "obs/third_person_mask"
    STUDENT_STATE = "obs/student_state"
    TEACHER_ACTION = "actions/teacher_action"
    PRIMITIVE = "goals/interaction_primitive"
    KAPPA = "goals/constraint_kappa"
    ANCHOR_MASK_ID = "goals/anchor_mask_id"
    GUIDANCE_FIELD = "goals/guidance_field"
    PHASE = "goals/phase"
    SUCCESS = "success"


REQUIRED_OBSERVATION_KEYS: tuple[str, ...] = (
    DemoKeys.EE_DEPTH,
    DemoKeys.EE_MASK,
    DemoKeys.THIRD_PERSON_DEPTH,
    DemoKeys.THIRD_PERSON_MASK,
    DemoKeys.STUDENT_STATE,
)

REQUIRED_LABEL_KEYS: tuple[str, ...] = (
    DemoKeys.TEACHER_ACTION,
    DemoKeys.PRIMITIVE,
    DemoKeys.KAPPA,
    DemoKeys.ANCHOR_MASK_ID,
    DemoKeys.GUIDANCE_FIELD,
    DemoKeys.PHASE,
)


@dataclass(frozen=True)
class TaskSchema:
    task: IsaacTaskName
    kappa: ConstraintKappa
    anchor: str
    primitive_sequence: tuple[InteractionPrimitive, ...]
    success_metric: str


TASK_SCHEMAS: dict[IsaacTaskName, TaskSchema] = {
    IsaacTaskName.LIFT_CUBE: TaskSchema(
        task=IsaacTaskName.LIFT_CUBE,
        kappa=ConstraintKappa.VERTICAL_FREE,
        anchor="cube",
        primitive_sequence=(
            InteractionPrimitive.REACH,
            InteractionPrimitive.GRASP,
            InteractionPrimitive.ACTUATE,
            InteractionPrimitive.HOLD,
        ),
        success_metric="cube_height_gain",
    ),
    IsaacTaskName.OPEN_DRAWER: TaskSchema(
        task=IsaacTaskName.OPEN_DRAWER,
        kappa=ConstraintKappa.SLIDE,
        anchor="drawer_handle",
        primitive_sequence=(
            InteractionPrimitive.REACH,
            InteractionPrimitive.GRASP,
            InteractionPrimitive.ACTUATE,
            InteractionPrimitive.HOLD,
        ),
        success_metric="drawer_joint_open",
    ),
    IsaacTaskName.OPEN_SIDE_DOOR: TaskSchema(
        task=IsaacTaskName.OPEN_SIDE_DOOR,
        kappa=ConstraintKappa.REVOLUTE,
        anchor="door_knob",
        primitive_sequence=(
            InteractionPrimitive.REACH,
            InteractionPrimitive.GRASP,
            InteractionPrimitive.ACTUATE,
            InteractionPrimitive.HOLD,
        ),
        success_metric="door_joint_open",
    ),
}


def required_demo_keys(*, include_rgb: bool = False) -> tuple[str, ...]:
    """Return canonical required keys for a collected demo."""
    rgb_keys = (DemoKeys.EE_RGB, DemoKeys.THIRD_PERSON_RGB) if include_rgb else ()
    return (*REQUIRED_OBSERVATION_KEYS, *rgb_keys, *REQUIRED_LABEL_KEYS)


def task_schema(task: str | IsaacTaskName) -> TaskSchema:
    """Look up the canonical schema for a task name."""
    return TASK_SCHEMAS[IsaacTaskName(task)]
