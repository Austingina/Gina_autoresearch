# CALVIN behavior-G gate before scaling

Date: 2026-08-13

## Decision

Do not start large-scale learned-G pretraining merely because the visual stack
and a large converted dataset are available. First pass a matched-definition
behavior-G gate: the G consumed online must have the same semantics and scale
as the G used to train the executor.

Keep the interface coarse:

```text
G = [attraction_to_interaction_region(3), desired_object_flow(3)]
```

Language supplies task context and the intended signed object flow upstream.
The low-level executor consumes visual/proprioceptive state and G. Do not add a
stage one-hot, active-joint id, task-specific controller, or privileged action
correction to G.

## Evidence behind the gate

- The accepted official ABC-to-D visual experiment gives `32/200` for oracle
  G and `1/200` for both zero and sign-flipped G. G is causally useful, but
  absolute scene/composition transfer is still weak.
- Source-scene health is `72/100`, so the large D drop is not explained by an
  executor that cannot perform the source tasks.
- In the official debug push trajectory, EEF is already within roughly
  6--7 mm of the block and tracks it from the beginning. The older generated G
  places a contact anchor 5 cm from the object along the task axis. That offset
  is a hand-built convention, not demonstrated interaction geometry.
- A held-out RGB reconstruction diagnostic selects the target color component
  reliably, but direct monocular anchor regression has approximately 1.4 cm
  median and 5.9 cm p90 3-D error. This is too inaccurate to assume a learned
  contact anchor is ready for close-loop scaling.
- The short-window LeRobot conversion contains exact proprioception/actions
  and both RGB streams but no `scene_obs`. It is useful for video-derived G and
  policy probes, not oracle-G benchmark replacement.

## M3A: matched behavior-G method gate

1. Build behavior G from source demonstrations. Detect contact onset from
   target motion plus EEF proximity. Before contact, attraction points to the
   demonstrated interaction region; after contact, attraction decays to zero.
   Language determines the coarse signed object-flow vector.
2. Train on four source color-direction cells and hold out at least two
   recombined cells. Use task-balanced episode sampling and three seeds.
3. At evaluation, construct the same field definition. For the method gate it
   is acceptable to use scene state only to provide the matched oracle
   interaction region; scene state must not enter the executor observation.
4. Evaluate normal, zero, and sign-flipped G on identical initial states. Also
   report source tasks, held-out combinations, contact-entry rate, object
   displacement along G, and task success.
5. Repeat with language upstream and no direct executor language. A direct
   language executor remains a baseline, not part of the reusable interface.

The current 52-episode probe is a pipeline smoke test. It cannot pass this gate
because its training behavior G and online scene-generated G use different
contact-anchor definitions.

## M3A acceptance criteria

- Normal G beats both causal controls on matched starts with paired evidence.
- Every held-out combination is non-zero; no result is carried by one color or
  one direction.
- Normal G improves contact-entry and signed object displacement, not only a
  final threshold accident.
- The conclusion repeats across at least three training seeds.
- Source-task performance does not materially regress relative to the same
  executor without the held-out split.

Absolute thresholds should be set after a full-data seed-0 run. A result that
only raises offline action MSE, only improves approach, or only works with a
different online G definition does not pass.

## M3B: learn G from scalable observations

Begin this stage only after M3A passes.

- Contact/attraction: learn an object-centric interaction-region heatmap from
  dense frozen visual features and temporal target motion. Use EEF proximity
  as self-supervision when robot trajectories are available; do not regress a
  single 3-D anchor from global RGB alone.
- Flow: infer task-conditioned object flow from language and video motion. It
  remains one 3-D vector, not a list of task-specific terms.
- Pretraining order: frozen DINOv2 diagnostic, then a trainable dense visual
  adapter only if the contact-region error is the limiting variable.
- Evaluate G quality through contact entry, signed displacement, and policy
  success. MSE is a diagnostic, not the target metric.

## Scaling decision

Scaling is justified when a learned G closes a meaningful fraction of the
matched oracle behavior-G gap on held-out compositions and new initial states.
If matched oracle behavior G does not improve the executor, redesign policy-G
integration before collecting more video. If oracle works but learned G does
not, scale the G learner and its temporal supervision rather than the policy.
