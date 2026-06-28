"""Policy-facing visual goal schema — immutable interface for autoresearch.

Core representation:
    g_t = (M_t, G_t)
      M_t = anchor mask   — what entity to act on
      G_t = guidance field — how that entity should visually evolve

WM / world models are optional upstream field learners, NOT part of this interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class GuidanceMode(str, Enum):
    """How policy receives visual goal guidance (fixed per experiment baseline)."""

    NONE = "none"  # B0: obs only, no explicit goal channels
    ANCHOR_ONLY = "anchor_only"  # B1: single mask
    ANCHOR_FIELD_RULE = "anchor_field_rule"  # B3: oracle / handcrafted field
    ANCHOR_FIELD_DEMO = "anchor_field_demo"  # B7: learned from demonstrations
    ANCHOR_FIELD_WM = "anchor_field_wm"  # B8: optional WM-derived field (Phase 2)
    ANCHOR_TARGET_MASK = "anchor_target_mask"  # B2: counterfactual target mask baseline


class FieldType(str, Enum):
    """Encoding of guidance field G_t fed to policy."""

    OBJECT_VECTOR = "object_vector"  # (du, dv) or (dx, dy, dz) on anchor centroid
    DENSE_2D = "dense_2d"  # H×W vector field inside anchor support
    REGION_PULL = "region_pull"  # pull anchor toward target region (HIEVT bbox-like)


class TaskType(str, Enum):
    TRACK = "track"
    APPROACH = "approach"
    LIFT = "lift"
    PUSH = "push"
    PLACE = "place"
    NAVIGATE = "navigate"


@dataclass(frozen=True)
class GuidanceField:
    """Visual guidance: desired change of anchor entity, action-agnostic."""

    field_type: FieldType
    vector: tuple[float, ...] | None = None  # object-level: 2D or 3D direction + optional magnitude
    dense_shape: tuple[int, ...] | None = None  # for dense_2d placeholder metadata
    source: str = "rule"  # rule | demo | wm | oracle
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["field_type"] = self.field_type.value
        return d


@dataclass(frozen=True)
class VisualPolicyGoal:
    """Policy input goal representation: anchor mask + optional guidance."""

    anchor_mask: dict[str, Any]  # ref to mask tensor metadata (shape, source)
    guidance: GuidanceField | None = None
    task: TaskType = TaskType.LIFT
    instruction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_mask": self.anchor_mask,
            "guidance": self.guidance.to_dict() if self.guidance else None,
            "task": self.task.value,
            "instruction": self.instruction,
        }


# Task → default guidance direction (sim oracle hints for rule-based field)
TASK_DEFAULT_FIELD: dict[str, tuple[float, ...]] = {
    "track": (0.0, 0.0),  # toward view region center — filled per episode
    "approach": (0.0, 0.0),  # toward gripper-centered region
    "lift": (0.0, 1.0, 0.0),  # upward in image/robot frame (task-specific)
    "pick": (0.0, 0.0, 0.0),  # pre-grasp: approach dominates
    "push": (1.0, 0.0, 0.0),
    "pull": (-1.0, 0.0, 0.0),
    "place": (0.0, 0.0, -0.5),
}

# Legacy alias (deprecated)
GoalRepresentation = VisualPolicyGoal
InteractionType = TaskType
TASK_INTERACTIONS = {k: TaskType.LIFT for k in ("pick", "lift")}  # minimal compat
