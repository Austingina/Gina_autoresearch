# Compact G as a Residual Flow-Policy Prior

Date: 2026-07-15
Status: next-stage design after the D0--D4 direction-coverage matrix

## Evidence Boundary

The current matrix supports a narrow but important claim:

```text
Compact G=[A,F] contains enough held-out interaction information to solve
close_drawer when the policy uses it as a closed-loop directional prior.
```

D1 and D2 use identical data and G and have nearly identical first-action
validation MSE (`0.026389` versus `0.026304`), but held-out close success is
`0.4%` versus `85.2%`. D2 also falls to `0%` under zero G, `0.6%` under flow
sign flip, and `16.8%` under wrong-open G. The gain is therefore caused by how G
enters closed-loop control, not by better offline regression fit.

The stricter D4 control matches D2's data, G, H16 chunks, first-action weight
16, and replan-1 inference but uses ordinary FiLM. It obtains lower offline
first-action MSE (`0.024973`) and only `2.2%` close success versus D2's `85.2%`.
This isolates field-coupled action parameterization from the chunk/objective
change. D4 also reaches only `639/1250 = 51.1%` on the common seen tasks versus
D2's `803/1250 = 64.2%`; field coupling improves both seen execution and
held-out transfer rather than trading one for the other.

The matrix does not yet establish that:

- the current fixed-offset A builder is universal;
- ordinary FiLM can learn the physical G semantics;
- G can already be predicted from video at scale;
- D2's hard positive projection is the final policy architecture.

D2 reaches only `64.2%` on seen translational tasks. Its articulation family is
healthy (`94.4%`), while free-object pushes reach `51.3%`. The contact audit
shows that one fixed `-0.05 m` block offset and a `0.10 m` readiness radius do
not match demonstrated contact transitions.

## Representation Decision

Keep the interface compact:

```text
G = [A(3), F(3)]
A = executable interaction anchor - current EEF
F = desired object-motion direction
```

Do not add task stage, contact one-hot, active-joint ID, future EEF trajectory,
or per-task behavior labels to G. These are execution variables and would make
the interface harder to learn from broad video.

Improve A without changing the contract. For free objects, estimate an
executable support-surface point opposite F from depth/geometry, followed by a
single embodiment/tool offset learned across tasks. For articulated objects,
retain the geometry-fitted handle/part anchor. Use object-motion onset and
hand/EEF proximity in demonstrations as weak supervision for the contact
manifold.

## Language Contract

Language has two non-interchangeable roles:

1. select the entity, relation, and desired motion used to construct G;
2. condition the executor on task/contact context that G intentionally omits.

The scalable policy should therefore keep language available even when G is
valid. The current task-one-hot masking is useful for proving that oracle G can
carry held-out task intent, but it overburdens G in mixed push/grasp/place data
and is not the final integration rule.

Required baselines remain:

```text
language only
language + raw conditional G
language + G field-prior residual policy
```

## Policy Design

Let a small policy-side module compute a low-frequency base field from current
observation history and G:

```text
b_G = c_psi(o[t-k:t], G) * d_G(o_t, G)
```

`d_G` blends attraction toward A and object flow F. `c_psi` predicts a soft
readiness/gain from observation history. Readiness is not another G token. It
is embodiment- and state-dependent execution logic learned by the policy.

A standard conditional flow-matching policy predicts a full action residual
chunk:

```text
r* = action_chunk* - repeat(b_G, H)
v_theta = FM_velocity(noisy_r, time, image/state, language, G, b_G)
action_chunk = repeat(b_G, H) + sampled_residual_chunk
```

The residual remains full rank and may override the prior. There is no hard
orthogonal cap, positive final-action projection, or first-window loss weight.
Receding-horizon inference recomputes G and b_G after one or a small number of
executed actions. D2 remains a mechanism control that motivates this softer
parameterization.

Contact/readiness pseudo-labels can be obtained without simulator-only force
signals:

- negatives: pre-contact frames with large hand/object separation;
- transition: closest pre-motion frames before coherent object motion onset;
- positives: sustained object motion aligned with F while proximity persists.

This label is used to train the policy gate, not appended to G.

## Experiment Ladder

### E1: Repair the oracle builder

Keep D2 and its data fixed. Compare current fixed-offset A with
support-surface/contact-transition A on five seeds.

Primary metrics:

```text
10-task seen success
free-object push success
held-out close success
pre-motion readiness lead time
anchor error at contact proxy
```

Do not proceed if better anchor diagnostics fail to improve task success.

### E2: Remove the hard action projection

On the corrected builder, compare:

```text
deterministic residual BC
small conditional FM with raw G conditioning
small conditional FM with G base field + full residual
```

Use the same H16 chunks, segment split, seeds, and rollouts. Evaluate execution
horizon 1 and a small multi-action horizon without retraining a different loss
for the first action.

The field-prior FM must retain a nonzero worst-seed close result, beat
language-only/raw-G FM, degrade under zero/sign-flip/wrong-task G, and avoid a
material seen-task regression relative to the corrected deterministic model.

### E3: Visual G learning and scaling

Only after E1/E2 pass, replace oracle G with a language-conditioned visual
predictor. Train A from contact-manifold pseudo-labels and F from task-relevant
object flow. Report oracle-G, predicted-G, no-G, and corrupted-G task metrics;
G-space MSE is diagnostic only.

## Paper Claim If E1/E2 Pass

The competitive claim is not that object flow itself is new. It is:

```text
A language-grounded, reusable interaction field becomes transferable control
only when represented as an executable anchor/object-flow interface and used as
a recomputed low-frequency prior inside a learned residual action policy.
```

This separates the work from flow-only world models, raw condition tokens, and
action-side intent tokenization while preserving a realistic scaling path from
video.
