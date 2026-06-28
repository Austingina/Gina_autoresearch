#!/usr/bin/env python3
"""
Editable experiment pipeline — THE ONLY FILE the autoresearch agent modifies.

Policy-facing representation:
    a_t = pi(o_t, M_t, G_t)

    M_t = anchor mask      (what entity)
    G_t = guidance field   (how it should visually evolve)

The agent tunes field predictor architecture and training hyperparameters.
WM is NOT required — demo-learned or rule fields are the default path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from goals.schema import (
    FieldType,
    GuidanceField,
    GuidanceMode,
    TaskType,
    VisualPolicyGoal,
    TASK_DEFAULT_FIELD,
)


@dataclass
class ExperimentConfig:
    """Knobs for guidance field learning + policy training. Agent may edit."""

    # --- Representation (primary) ---
    # Step 1 default: learn G from demos. Use anchor_field_rule only as oracle upper bound.
    guidance_mode: str = GuidanceMode.ANCHOR_FIELD_DEMO.value
    field_type: str = FieldType.OBJECT_VECTOR.value

    # Field predictor (Step 1 — learn potential / guidance field)
    field_hidden_dim: int = 128
    field_learning_rate: float = 1e-4
    use_demo_supervision: bool = True
    field_train_epochs: int = 50
    field_checkpoint: str = ""  # path after learn_guidance_field.py

    # Anchor extraction (VFM / sim GT)
    use_vfm_anchor: bool = False
    vfm_model: str = "dinov2_vitb14"  # dinov2 | sam2 | deva

    # Policy training (Isaac DAgger / RL)
    hidden_dim: int = 128
    learning_rate: float = 1e-4
    max_dagger_iterations: int = 200
    collect_steps: int = 64
    batch_size: int = 16
    image_size: int = 64

    # Field magnitude scaling (rule oracle)
    field_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnchorExtractor:
    """obs (+ optional instruction) -> anchor mask metadata."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def forward(self, task: str = "lift", instruction: str | None = None) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "shape": [self.config.image_size, self.config.image_size],
            "source": self.config.vfm_model if self.config.use_vfm_anchor else "sim_gt",
            "task": task,
        }
        if instruction:
            meta["instruction"] = instruction
        return meta


class FieldPredictor:
    """Predict guidance field G_t conditioned on anchor mask (+ obs).

    Phase 1: rule oracle (TASK_DEFAULT_FIELD * field_scale).
    Phase 2: demo-learned MLP/CNN (agent implements forward).
    Phase 2 optional: WM-derived field (not default).
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def forward(
        self,
        anchor_mask: dict[str, Any],
        task: str = "lift",
        phase: str = "post_grasp",
    ) -> GuidanceField | None:
        mode = GuidanceMode(self.config.guidance_mode)
        if mode in (GuidanceMode.NONE, GuidanceMode.ANCHOR_ONLY):
            return None

        ft = FieldType(self.config.field_type)
        source = "rule"
        if mode == GuidanceMode.ANCHOR_FIELD_DEMO:
            source = "demo"
        elif mode == GuidanceMode.ANCHOR_FIELD_WM:
            source = "wm"
        elif mode == GuidanceMode.ANCHOR_TARGET_MASK:
            source = "target_mask"

        vec = TASK_DEFAULT_FIELD.get(task, (0.0, 1.0, 0.0))
        scale = self.config.field_scale
        if len(vec) == 2:
            vector = (vec[0] * scale, vec[1] * scale)
        else:
            vector = tuple(v * scale for v in vec)

        # Lift: zero field pre-grasp, upward post-grasp (minimal phase stub)
        if task in ("lift", "pick") and phase == "pre_grasp":
            vector = (0.0, 0.0, 0.0) if len(vector) == 3 else (0.0, 0.0)

        return GuidanceField(
            field_type=ft,
            vector=vector,
            source=source,
            metadata={"phase": phase, "predictor_hidden": self.config.field_hidden_dim},
        )


def build_policy_goal(
    config: ExperimentConfig | None = None,
    task: str = "lift",
    phase: str = "post_grasp",
    instruction: str | None = None,
) -> VisualPolicyGoal:
    cfg = config or ExperimentConfig()
    anchor = AnchorExtractor(cfg).forward(task=task, instruction=instruction)
    guidance = FieldPredictor(cfg).forward(anchor, task=task, phase=phase)
    try:
        task_enum = TaskType(task)
    except ValueError:
        task_enum = TaskType.LIFT
    return VisualPolicyGoal(
        anchor_mask=anchor,
        guidance=guidance,
        task=task_enum,
        instruction=instruction,
    )


def build_pipeline(
    config: ExperimentConfig | None = None,
) -> tuple[AnchorExtractor, FieldPredictor, ExperimentConfig]:
    cfg = config or ExperimentConfig()
    return AnchorExtractor(cfg), FieldPredictor(cfg), cfg


def policy_training_hint(isaac_root: str, cfg: ExperimentConfig, task: str) -> str:
    """Reference command for external IsaacLab DAgger (human/agent)."""
    if task in ("lift", "pick"):
        return (
            f"# Inject g_t=(M,G) into student obs before training:\n"
            f"{isaac_root}/isaaclab.sh -p "
            f"scripts/reinforcement_learning/train_lift_cube_dagger_vision_binary_gripper.py "
            f"--max_iterations {cfg.max_dagger_iterations} "
            f"--collect_steps {cfg.collect_steps} "
            f"--learning_rate {cfg.learning_rate} "
            f"# guidance_mode={cfg.guidance_mode} field_type={cfg.field_type}"
        )
    return f"# No wired Isaac command for task={task}"


if __name__ == "__main__":
    for mode in ("anchor_only", "anchor_field_rule"):
        cfg = ExperimentConfig(guidance_mode=mode)
        g = build_policy_goal(cfg, task="lift", phase="post_grasp")
        print(f"mode={mode} goal={g.to_dict()}")
