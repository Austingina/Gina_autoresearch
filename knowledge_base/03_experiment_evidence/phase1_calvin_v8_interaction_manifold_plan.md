# Phase1 CALVIN V8 Interaction-Manifold Field Plan

Date: 2026-07-07

## Why V8 Is Needed

The current CALVIN articulation experiments have reached a useful boundary:

- Pure no-G policies can solve some same-task settings, but fail held-out task
  transfer when task one-hot is removed.
- V5/V6/V7 field variants can condition behavior, but the best high-SR
  same-task result either used a non-scalable shortcut (`v5_potential`) or did
  not transfer to held-out close (`v7`).
- V7 FiLM shows a small qualitative transfer signal on
  `open/left/right -> close`, but only 2/25 rollouts enter the effective
  contact region.

Therefore, the missing piece is not another cleaner object-flow field. The
missing piece is a policy-facing interaction-manifold field.

## Current Experimental State

### Same-task articulation

Known checkpoint-level evidence:

| Variant | Best Signal | Main Issue |
| --- | --- | --- |
| `no_g` | Can solve parts of same-task control | Uses task/action priors; weak transfer |
| `v5_potential_concat` | Best same-task SR observed, around 0.85 | Non-target repulsion shortcut, not scalable |
| `v5_attraction_only_concat` | Clean field, around 0.79 | Lower than flawed v5 |
| `v6_interaction_field_concat` | Clean target-local field; left can improve later | Trades off right/open |
| `v7_unified_field_concat` | Clean object-local unified field, best around 0.78 | Does not transfer |

### Held-out transfer

Split:

```text
train: open_drawer + move_slider_left + move_slider_right
test: close_drawer
task one-hot: zeroed
```

Result:

| Variant | Held-out Close |
| --- | ---: |
| `no_g` | 0/25 |
| `v7_concat` | 0/25 |
| `v7_film` | 1/25 |

Behavior diagnosis:

| Variant | Entered Near Part | Root Bucket |
| --- | ---: | --- |
| `no_g_e50` | 0/25 | `no_effective_contact`: 25 |
| `v7_concat_e50` | 0/25 | `no_effective_contact`: 25 |
| `v7_film_e50` | 2/25 | `success`: 1, `contact_but_insufficient_progress`: 1, `no_effective_contact`: 23 |

## Root Cause

V7 is object-motion-centric:

```text
token = [anchor_delta, object_motion_field, zeros, field_weight, 0]
```

Its anchors are placed around the articulated part point. The field says how
the target part should move, but it does not robustly define:

- where the EEF can apply force,
- which side/direction is effective,
- when approach should become force application,
- and how to maintain contact while applying force.

This explains why:

- `v7_concat` can get closer to the target part without moving the joint;
- `v7_film` sometimes discovers the right interaction mode, but only
  intermittently;
- failures are dominated by `no_effective_contact`, not just by wrong
  object-flow direction.

## V8 Definition

V8 should be an interaction-manifold field:

```text
G_t = { (p_i - e_t, F_i, w_i) }_{i=1..K}
```

where each token has the same semantics:

- `p_i - e_t`: EEF-relative point on or near a valid interaction manifold;
- `F_i`: local EEF-space vector field that first attracts the EEF toward the
  valid manifold and, near the manifold, aligns with the task-effective action
  tangent;
- `w_i`: soft validity / residual weight.

Important constraints:

- No task one-hot.
- No `active_joint_id` one-hot.
- No hard stage one-hot.
- No close-specific rule.
- Same token semantics for drawer open, drawer close, slider left, slider
  right.
- Keep tokens homogeneous so the design can scale to learned G later.

The key conceptual shift:

```text
V7: object should move this way
V8: EEF should enter and follow this reusable interaction manifold
```

## Candidate Analytic Oracle V8

For each articulation joint, construct candidate EEF interaction tokens around
the part/contact region.

For a target part point `p`, desired object/joint direction `d`, and current
EEF position `e`:

1. Define a soft manifold around valid contactable offsets from `p`.
2. For each token point `p_i`, compute an approach component:

```text
F_approach = normalize(p_i - e)
```

3. Compute an effective tangent/force component:

```text
F_tangent = desired_joint_direction
```

4. Blend continuously by distance-to-manifold:

```text
c = exp(-||e - p_i||^2 / sigma^2)
F_i = (1 - c) * F_approach + c * F_tangent
w_i = residual_weight * validity_weight
```

This is still a field, not a symbolic phase label. The policy sees a smooth
vector field whose behavior changes continuously as it approaches contact.

## Why This Should Be Less Degenerate

V8 is not tuned for one case. It targets the failure bucket shared across
tasks: failure to enter and stay on an effective interaction manifold.

Expected effects:

- For no-contact failures: increase approach correctness and near-part entry.
- For contact-but-insufficient-progress failures: increase contact-conditioned
  action along the effective tangent.
- For already-successful cases: residual/validity weights should decay near
  completion, reducing unnecessary disturbance.

The field should be conservative enough that it does not force a single
strategy. For example, slider-left can still be solved by pushing rather than
grasping, as long as the EEF follows a valid interaction manifold.

## Evaluation Criteria

V8 should not be judged only by MSE.

Primary transfer test:

```text
train: open_drawer + move_slider_left + move_slider_right
test: close_drawer
metric: held-out close SR
```

A meaningful result:

```text
v8_interaction_manifold > v7_film ceiling of 1/25 close success
```

Required diagnostics:

- `near_part_entry`: fraction of rollouts entering effective contact region;
- `joint_progress_after_contact`: q progress after first near-part step;
- `approach_cosine`: action direction toward valid interaction manifold before
  contact;
- `contact_tangent_action`: action projection along effective tangent while in
  contact;
- same-task non-regression on close/left/right/open.

## Minimal Experiment Matrix

1. Visualize oracle V8 on CALVIN render frames for all four tasks:
   `close_drawer`, `open_drawer`, `move_slider_left`, `move_slider_right`.
2. Train state-BC transfer split:
   - `no_g`
   - `v7_film`
   - `v8_interaction_manifold_concat`
   - `v8_interaction_manifold_film`
3. Evaluate checkpoints `4, 8, 16, 32, 50`, `n=25`, horizon `160`.
4. Run trajectory diagnostics and selected rollout videos.
5. Run same-task non-regression check on all four tasks.

## Learned-G Implication

The oracle V8 test is an interface test, not the final method.

If oracle V8 improves transfer without hurting same-task behavior, the learned
G direction should estimate interaction-manifold tokens from demonstration
videos:

- infer likely contact windows from EEF/part proximity and object progress;
- estimate approach-to-contact fields from pre-contact EEF trajectories;
- estimate tangent fields from contact-conditioned object/EEF motion;
- train a visual/geometry predictor to output the same homogeneous token set.

This keeps the paper story aligned with reusable interaction geometry rather
than a hand-designed per-task feature bundle.
