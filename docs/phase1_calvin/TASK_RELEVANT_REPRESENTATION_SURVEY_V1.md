# Task-Relevant Representation Survey for Interaction Geometry G

**Date:** 2026-07-03

## Working Diagnosis

Our recent CALVIN evidence suggests that the current G failure is less about
policy capacity and more about representation/interface:

- full scene delta G improves one-step MSE but hurts task success;
- concat G is consistently unsafe;
- narrow articulation G as FiLM modulation gives the first positive task-level
  signal on balanced articulation eval.

The literature points in the same direction: useful task representations are
usually typed, sparse, local, and tied to interaction structure. They are rarely
a single dense future-state vector.

## Paper Signals

| Paper | Effective representation | Why it matters for us |
| --- | --- | --- |
| [PALM](https://arxiv.org/abs/2601.07060) | interaction-centric affordance cues: object relevance, contact geometry, spatial placement, motion dynamics, and continuous subtask progress | Closest to our revised story. It explicitly says long-horizon failures come from missing interaction cues and progress tracking. |
| [Mask World Model](https://arxiv.org/abs/2604.19683) | future object masks as a sparse task-relevant bottleneck | Supports replacing full future-state prediction with "what matters" masks. Good for object-centric G and visual validation. |
| [ReKep](https://arxiv.org/abs/2409.01652) | relational keypoint constraints as Python cost functions over 3D keypoints | Strong evidence that tasks can be represented as relations/constraints, not latent action vectors. Useful for defining G targets and diagnostics. |
| [CLIPort](https://arxiv.org/abs/2109.12098) | semantic "what" pathway plus spatial "where" pathway | A classic reminder that semantics and spatial precision should be separated. Our G should provide "where/how" anchors, not absorb all task semantics. |
| [PerAct](https://arxiv.org/abs/2209.05451) | voxelized 3D observation/action space; next-best-voxel action | 3D spatial structure is a strong prior for low-data manipulation. Relevant for future CALVIN vision/3D policy. |
| [3D Diffusion Policy](https://arxiv.org/abs/2403.03954) | compact sparse point-cloud representation feeding diffusion policy | Useful baseline direction if we move from state-BC to flow/diffusion with 3D input. |
| [D3Fields](https://arxiv.org/abs/2309.16118) | dynamic 3D descriptor fields with semantic features and instance masks | Shows object-level semantic geometry can be reusable across instances and goals. Good candidate for a future learned G perception layer. |
| [3D Flow Diffusion Policy](https://arxiv.org/abs/2509.18676) | scene-level 3D flow over sampled query points, used as an intermediate representation | Very relevant if we want G to encode local interaction dynamics rather than target scene deltas. |
| [3DFlowAction](https://arxiv.org/abs/2506.06199) / [Im2Flow2Act](https://im-flow-act.github.io/) | object-centric 3D/2D flow as a cross-embodiment manipulation interface | Strong positive evidence for local object/part flow, but also a boundary: our story should not become just another flow world model. Flow should be one typed component inside Interaction Geometry G. |
| [Where2Act](https://arxiv.org/abs/2101.02692) | localized actionability over articulated parts, e.g. where to push/pull and what part moves | Directly useful for drawer/slider/door tasks: handle/actionable point + action direction should be part of articulation G. |
| [UniVLA](https://arxiv.org/abs/2505.06111) | task-centric latent actions learned from videos | Important boundary: this is a scalable action-representation story. We should not make G another latent action tokenizer. |
| [ReinFlow](https://arxiv.org/abs/2505.22094) | online RL fine-tuning method for flow-matching policies | Useful after a representation works. It should not drive current G design. |

## Representation Primitives Worth Borrowing

### 1. Entity/Mask Bottleneck

From Mask World Model, D3Fields, and object-centric VLA work:

```text
G_entity = {
  active_entity_id,
  entity_mask_or_slot,
  entity_pose_or_centroid,
  task_relevant_part_mask,
}
```

This is the replacement for full scene state. It asks "what object/part matters?"
before asking "what future state should happen?"

### 2. Affordance Point or Frame

From PALM, Where2Act, CLIPort, and affordance-centric methods:

```text
G_affordance = {
  contact_point_or_handle_point,
  contact_normal_or_pull_axis,
  approach_direction,
  local task frame,
}
```

For CALVIN articulation, this means drawer/slider should not just provide a
joint scalar. It should expose "where to interact" and "which direction the
interaction should move."

### 3. Relational Keypoint Constraint

From ReKep:

```text
G_relation = {
  keypoints: [eef, object, handle, target],
  desired relation: cost(keypoints),
  constraint type: distance / alignment / containment / ordering,
}
```

This is a useful unifying format because many tasks become small constraints:

- reach: distance(eef, handle) -> 0
- place: distance(object, target) -> 0
- open drawer: drawer_progress -> target and eef aligned with pull axis
- stack: object_top aligned with target_top

### 4. Progress / Phase

From PALM and our own Isaac/DAgger failure modes:

```text
G_progress = {
  phase: pre_contact / contact / post_contact / done,
  scalar_progress: 0..1,
  remaining_delta,
}
```

This is likely missing from our G. A single target delta is ambiguous before and
after contact. The same "open drawer" G should mean different things during
approach, contact, and pull.

### 5. Local Motion / Flow

From Mask World Model and 3D Flow Diffusion Policy:

```text
G_flow = {
  selected_query_points,
  local_future_displacements,
  confidence_or_mask,
}
```

This is a better future prediction target than full scene delta because it is
localized to the moving object/part. For articulation, this can be the expected
handle/slider/drawer motion; for object tasks, object point flow toward target.

3DFlowAction and Im2Flow2Act sharpen this point: object flow is likely
scalable because it can be supervised from human/robot videos without robot
action labels and can bridge embodiment differences. The danger is that "G =
object flow" is now an occupied story. Our safer formulation is:

```text
G_flow is one token group inside Interaction Geometry G,
not the full representation.
```

For CALVIN, this means v7-style unified fields should be paired with explicit
entity/affordance/progress/contact tokens, then judged by closed-loop task
success and transfer rather than flow MSE alone.

### 6. Contact State

From PALM's contact geometry and contact-rich manipulation work:

```text
G_contact = {
  should_contact,
  contact_part,
  contact_normal,
  gripper_open_close_state,
  contact_confidence,
}
```

This is especially important because many policy failures are phase/contact
errors rather than goal-direction errors.

## Composable G Proposal

Instead of one dense G vector, represent G as typed interaction tokens:

```text
G = {
  entity:      active object/part slot,
  affordance: contact/keypoint/frame,
  relation:   desired geometric constraint,
  progress:   phase + scalar progress,
  flow:       local future motion for selected points,
  contact:    expected contact state,
}
```

For a lightweight state-policy implementation, this can be flattened after
typing:

```text
base_state -> trunk
G_entity/G_affordance/G_relation/G_progress -> FiLM or gated residual
G_flow -> auxiliary loss or separate flow head
```

For a VLA/vision implementation:

```text
image tokens + entity tokens + G tokens
G tokens produce attention bias / cross-attention routing
policy head predicts action chunk or flow action
```

## Candidate Task-Family G Schemas

### Articulation Family: drawer / slider / door

Minimum next version:

```text
G_articulation = {
  active_part: drawer_handle / slider_knob / door_handle,
  part_point: 3D point or 2D mask centroid,
  axis: normalized pull/push/slide direction,
  current_joint,
  target_joint,
  joint_delta,
  progress: normalized joint progress,
  phase: approach / contact / actuate / done,
}
```

Current positive result only used:

```text
current_joint, target_joint, delta, sign(delta), task_one_hot
```

So the next plausible improvement is adding `active_part`, `axis`, and `phase`.

### Object Transfer Family: lift / place / push / stack

```text
G_object = {
  active_object_mask_or_slot,
  object_pose_or_centroid,
  target_pose_or_region,
  eef_to_object,
  object_to_target,
  grasp_or_contact_progress,
  local_object_flow,
}
```

This is closer to Mask World Model, CLIPort, DP3, and 3D FDP.

### Toggle/Button Family

```text
G_toggle = {
  active_button_or_switch_point,
  activation_axis,
  current_binary_state,
  target_binary_state,
  eef_to_button,
  contact_phase,
}
```

This should be much narrower than scene delta and should not include unrelated
block states.

## What To Avoid

1. Do not train a generic full-scene future G as the main representation.
   It collides with world models and has already failed our task metric.
2. Do not concat G by default. Our experiments and Geo-VLA-style evidence both
   suggest that raw feature injection can corrupt closed-loop behavior.
3. Do not make G a latent action tokenizer. That would collide with UniVLA/FAST
   style work and weaken our story.
4. Do not evaluate only by MSE. The main metric must remain task success,
   especially balanced per-task success.

## Revised Research Scope

The competitive story should be:

> Learn or specify reusable, task-family interaction geometry tokens that
> modulate policy learning and improve data efficiency/cross-task robustness.

This distinguishes us from:

- world models: we do not predict the whole future world;
- latent action models: we do not tokenize actions;
- plain VLA scaling: we add typed interaction structure;
- simple geometry concat: we use modulation/routing and task-family schemas.

## Immediate Experimental Plan

1. Confirm the articulation FiLM signal with 3 seeds and 25 or 50 balanced
   first-task eval sequences per task.
2. Add `phase` to articulation G:
   - approach if eef far from handle/part;
   - contact/actuate if close and joint still far from target;
   - done if joint near target.
3. Add `axis` and `part_point`:
   - for CALVIN state version, use simple approximate constants or scene
     heuristics;
   - for vision version, derive from masks/segmentation later.
4. Keep concat as a negative baseline.
5. Only after oracle typed G works robustly should we train a G predictor.

## Flow-Interface Note

Detailed memo:

- `knowledge_base/02_related_work/flow_interface_3dflowaction_im2flow2act.md`
