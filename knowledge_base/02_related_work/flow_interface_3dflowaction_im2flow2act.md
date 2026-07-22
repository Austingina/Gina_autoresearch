# Flow Interface: 3DFlowAction and Im2Flow2Act

**Date added:** 2026-07-06

## Papers

- **3DFlowAction: Learning Cross-Embodiment Manipulation from 3D Flow World Model**
  - arXiv: https://arxiv.org/abs/2506.06199
  - Date: 2025-06-06
  - Core idea: learn a language-conditioned 3D object-flow world model from
    robot and human videos, then use predicted 3D object flow as constraints
    for action optimization.
- **Flow as the Cross-domain Manipulation Interface / Im2Flow2Act**
  - Project: https://im-flow-act.github.io/
  - CoRL 2024 / PMLR 2025
  - Core idea: use object flow as the manipulation interface connecting
    actionless human videos and simulated robot play data.

## What They Establish

These papers are strong evidence for the claim that **object motion flow can be
an embodiment-agnostic manipulation interface**. The important details are:

- the flow is object-centric, excluding background and embodiment motion;
- the target is how the manipulated object should move, not the full future
  image or full scene state;
- large-scale video data can supervise the flow generator without robot action
  labels;
- the downstream executor can be either a flow-conditioned policy
  (Im2Flow2Act) or an optimization policy using 3D flow as constraints
  (3DFlowAction).

This directly supports our intuition that `G` should encode **task-relevant
local motion**, not a dense latent future state.

## Overlap With Our Current Direction

There is substantial overlap if our story becomes:

> learn a 3D object-flow world model and use it to guide manipulation.

That would collide with 3DFlowAction/Im2Flow2Act. The competitive version of
our story must therefore be narrower and more structural:

> learn or specify **typed interaction geometry tokens** that can include local
> flow, but also encode active entity, affordance frame, relation/progress, and
> contact mode, and use them as a policy-facing interface.

In other words, flow should be one field inside `G`, not the whole definition
of `G`.

## Implications For CALVIN Experiments

The current CALVIN v5/v6/v7 experiments can be interpreted as a tiny analytic
version of object-flow guidance:

- `v5_potential`: target-local flow plus a non-target repulsion shortcut.
- `v6_interaction_field`: target-local attraction and correction terms.
- `v7_unified_field`: a single target-local field that merges attraction,
  corridor correction, and backstop correction.

3DFlowAction suggests that the more scalable target should be:

```text
G_flow = {
  active_object_or_part_mask,
  query_points_on_active_part,
  3D_displacement_trajectory_or_endpoint_flow,
  visibility_or_confidence,
}
```

For articulation tasks, this maps naturally to handle/slider/drawer-part point
flow. For object transfer tasks, it maps to object point flow toward a target
region. However, our recent failures show that flow alone can be ambiguous in
contact-rich control, especially for slider-left vs slider-right and
open/close variants.

Therefore, the next useful `G` should be:

```text
G = {
  entity: active object/part mask or slot,
  affordance: contact/handle point and local task frame,
  flow: local object/part motion field,
  progress: scalar residual or phase,
  contact: desired contact mode or gripper/contact hint,
}
```

## Practical Borrowing Points

- Use moving-object / active-part filtering before learning `G`; avoid gripper
  and background flow.
- Use side-camera plus depth to lift 2D tracks into 3D flow targets.
- Treat side camera and ee camera as complementary: side camera gives object
  motion, ee camera gives approach/contact evidence.
- Evaluate learned `G` by downstream task success and per-task transfer, not
  only flow MSE.
- Keep no-G and oracle-G baselines, because strong object-flow papers raise the
  bar: the gain must appear in closed-loop success, data efficiency, or
  cross-task reuse.

## Research Positioning

Use these papers as positive evidence and a boundary:

- Positive evidence: object-centric flow is a scalable, cross-embodiment
  representation.
- Boundary: "flow world model for action" is already occupied.
- Our correction: `Interaction Geometry G` is a **typed policy interface** that
  can absorb flow as one reusable component while preserving explicit
  affordance, relation, progress, and contact structure.

