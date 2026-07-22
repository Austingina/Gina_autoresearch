# Task-Relevant Representation Survey V1

**Date:** 2026-07-03

Core takeaway:

> G should become typed task-family interaction geometry, not a dense future
> scene-state vector.

Useful representation primitives from recent and classic work:

- Mask bottlenecks: active object/part masks and future masks
- Affordance frames: contact point, handle point, axis, approach direction
- Relational keypoint constraints: cost functions over eef/object/target
- Progress: phase and continuous subtask progress
- Local flow: future motion of selected object/part query points
- Contact state: should-contact, contact normal, gripper/contact phase

Most relevant papers:

- PALM: progress-aware policy learning with interaction-centric affordance
  representations
- Mask World Model: sparse future masks as task-relevant bottleneck
- ReKep: relational keypoint constraints
- Where2Act: localized actionability for articulated objects
- 3D Flow Diffusion Policy: local scene-level 3D flow as policy condition
- 3DFlowAction / Im2Flow2Act: object-centric flow as a cross-embodiment
  manipulation interface
- CLIPort / PerAct / DP3: spatial representations and action maps
- UniVLA / ReinFlow: important boundary cases; not the current G story

Current recommended G schema:

```text
G = {
  entity,
  affordance,
  relation,
  progress,
  flow,
  contact,
}
```

Immediate experiment:

- continue articulation branch;
- add `phase`, `axis`, and `part_point` to the current rule-oracle
  articulation G;
- inject with FiLM/gating, not concat;
- verify with balanced task success before training a learned G predictor.

Full memo:

- `docs/TASK_RELEVANT_REPRESENTATION_SURVEY_V1.md`
- `knowledge_base/02_related_work/flow_interface_3dflowaction_im2flow2act.md`
