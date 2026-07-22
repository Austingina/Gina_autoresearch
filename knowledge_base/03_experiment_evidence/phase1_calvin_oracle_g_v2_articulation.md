# Phase 1 CALVIN Oracle-G V2 Articulation

**Date:** 2026-07-04

## Result

Oracle G v2 compact was constructed from demo-estimated articulation geometry:

```text
G_v2 = target_affordance + residual_motion + interaction_stage
```

Geometry estimation is stable: with 100 demos per task, axis cosine is near
1.0 and midpoint drift is under 1 mm for both slider and drawer.

Balanced first-task eval with 25 sequences per task:

| Mode | Balanced SR@1 |
| --- | ---: |
| `no_g` | 0.56 |
| `articulation_g_concat` | 0.33 |
| `articulation_g_film` | **0.66** |
| `articulation_g_v2_concat` | 0.65 |
| `articulation_g_v2_film` | 0.57 |
| `articulation_g_v2_split_concat` | **0.70** |
| `articulation_g_v2_split_film` | 0.61 |

Per-task highlights:

- `v2_concat` improves `open_drawer` from `2/25` to `15/25`.
- `v2_concat` preserves `move_slider_right` at `25/25`.
- `v2_concat` collapses `move_slider_left` to `1/25`.
- `v2_split_concat` recovers `move_slider_left` to `6/25`, keeps
  `move_slider_right` at `25/25`, and reaches the current best seed-0 balanced
  score (`0.70`).

## Takeaway

Better G can make concat useful, so concat should remain in future comparisons.
Splitting `axis_unit` from the signed residual scalar improves the compact G
interface and makes concat the current best seed-0 mode. However,
`move_slider_left` is only recovered to no-G level, not solved. Do not train
learned G yet; first run causal oracle-G ablations and inspect the videos.

## Left/Right Diagnostic Update

The follow-up demo diagnostic suggests that the fitted v2 geometry is mostly
symmetric between `move_slider_left` and `move_slider_right`:

- normalized residual mean is `+0.782` for left and `-0.778` for right;
- affordance norm medians are close (`0.024 m` vs `0.027 m`);
- stage fractions are close (`approach/actuate/done` about
  `0.41/0.56/0.03` vs `0.44/0.52/0.04`).

The asymmetry is mainly in how the signed residual-motion vector interacts with
the action distribution. `move_slider_left` has negative-x residual motion and
a negative median x action, while `move_slider_right` has positive-x residual
motion but a much heavier positive-x action tail. This points to a
representation/interface issue: `axis * residual` may be too prescriptive for
concat/FiLM and should be compared against a split representation such as
`[axis_unit, signed_residual_scalar, affordance, stage]`.

## Split-G And Videos

The split representation was tested:

```text
G_v2_split = target_affordance + axis_unit + signed_residual_scalar + interaction_stage
```

Balanced result:

- `v2_split_concat`: `25/25` close, `6/25` left, `25/25` right, `14/25` open,
  total `0.70`.
- `v2_split_film`: `25/25` close, `0/25` left, `25/25` right, `11/25` open,
  total `0.61`.

Qualitative videos were exported locally under:

```text
artifacts/server_sync/oracle_g_v2_articulation_2026-07-04/2026-07-04_state_bc_articulation_v1_v2_video_rollouts
```

## No-G 50 Epoch Audit

The no-G baseline was retrained for 50 epochs to test whether the poor
`move_slider_left` result was simply undertraining.

Result:

- final val MSE: `0.02361` vs epoch-8 `0.02899`;
- last-10 epoch MSE band: `0.02349-0.02390`, close to plateau;
- balanced SR: `0.59`;
- per-task: `25/25` close, `7/25` left, `18/25` right, `9/25` open.

This improves MSE and `open_drawer`, but `move_slider_left` only changes from
`6/25` to `7/25`. The left-slider failure is therefore not explained by no-G
undertraining alone.

## V2 50 Epoch Audit

The v2 modes were also retrained for 50 epochs. Offline MSE improves for every
mode, but task success drops:

| Mode | e8 SR | e50 SR | e50 per-task |
| --- | ---: | ---: | --- |
| `v2_concat` | 0.65 | 0.58 | `25/25` close, `0/25` left, `17/25` right, `16/25` open |
| `v2_film` | 0.57 | 0.33 | `21/25` close, `0/25` left, `10/25` right, `2/25` open |
| `v2_split_concat` | 0.70 | 0.61 | `25/25` close, `8/25` left, `12/25` right, `16/25` open |
| `v2_split_film` | 0.61 | 0.47 | `23/25` close, `0/25` left, `19/25` right, `5/25` open |

Conclusion: v2's earlier weaknesses are not caused by short training. Longer
BC training lowers MSE but can hurt closed-loop success, especially
`move_slider_right`. Future comparisons should either evaluate checkpoints
across training or select epochs by task rollouts rather than final MSE.

## Checkpoint Sweep Evidence

The three most important policy interfaces were swept at epochs 4/8/16/32/50
with the same balanced n25 protocol:

| Mode | Best Epoch | Best SR | close | left | right | open |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_g` | 32 | 0.61 | 24/25 | 8/25 | 21/25 | 8/25 |
| `v2_concat` | 4 | **0.70** | 25/25 | 7/25 | 25/25 | 13/25 |
| `v2_split_concat` | 8 | **0.70** | 25/25 | 6/25 | 25/25 | 14/25 |

Full checkpoint table:

| Mode | Epoch | SR | close | left | right | open | Val MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_g` | 4 | 0.41 | 25/25 | 7/25 | 7/25 | 2/25 | 0.03351 |
| `no_g` | 8 | 0.51 | 25/25 | 5/25 | 19/25 | 2/25 | 0.02899 |
| `no_g` | 16 | 0.54 | 25/25 | 8/25 | 21/25 | 0/25 | 0.02624 |
| `no_g` | 32 | 0.61 | 24/25 | 8/25 | 21/25 | 8/25 | 0.02441 |
| `no_g` | 50 | 0.59 | 25/25 | 7/25 | 18/25 | 9/25 | 0.02361 |
| `v2_concat` | 4 | 0.70 | 25/25 | 7/25 | 25/25 | 13/25 | 0.03109 |
| `v2_concat` | 8 | 0.62 | 25/25 | 0/25 | 25/25 | 12/25 | 0.02782 |
| `v2_concat` | 16 | 0.68 | 25/25 | 1/25 | 23/25 | 19/25 | 0.02534 |
| `v2_concat` | 32 | 0.66 | 24/25 | 2/25 | 22/25 | 18/25 | 0.02375 |
| `v2_concat` | 50 | 0.58 | 25/25 | 0/25 | 17/25 | 16/25 | 0.02295 |
| `v2_split_concat` | 4 | 0.57 | 25/25 | 4/25 | 25/25 | 3/25 | 0.03132 |
| `v2_split_concat` | 8 | 0.70 | 25/25 | 6/25 | 25/25 | 14/25 | 0.02778 |
| `v2_split_concat` | 16 | 0.67 | 25/25 | 1/25 | 24/25 | 17/25 | 0.02531 |
| `v2_split_concat` | 32 | 0.67 | 25/25 | 1/25 | 24/25 | 17/25 | 0.02378 |
| `v2_split_concat` | 50 | 0.61 | 25/25 | 8/25 | 12/25 | 16/25 | 0.02312 |

Main evidence update:

- Oracle G is useful under rollout-selected checkpoints: both concat variants
  reach `0.70`, above the no-G best of `0.61`.
- The benefit is task-skewed. Current G is reliable for right-slider and drawer
  opening, but does not robustly solve left-slider.
- Split G is a reasonable interface but not a strict dominance result; compact
  G at epoch 4 ties split G at epoch 8.
- The next scientific question is not whether MSE converges, but whether a G
  representation can improve closed-loop success and transfer without harming
  alternative but valid contact strategies.

## Left Dynamics Diagnostic

Best checkpoints were replayed on the same first 25 left and first 25 right
validation states with per-step state/action diagnostics:

| Variant | Task | SR | Failure mode | Failed max-progress p50 | Failed min-dist p50 | Contact action-axis p50 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `no_g@32` | left | 0.28 | mixed approach + insufficient motion | 0.043 | 0.043 m | 0.093 |
| `no_g@32` | right | 0.84 | rare approach/insufficient motion | 0.237 | 0.045 m | -0.166 |
| `v2_concat@4` | left | 0.28 | insufficient motion after contact | 0.000 | 0.014 m | 0.146 |
| `v2_concat@4` | right | 1.00 | none | n/a | n/a | -0.353 |
| `v2_split_concat@8` | left | 0.24 | insufficient motion after contact | 0.000 | 0.010 m | 0.115 |
| `v2_split_concat@8` | right | 1.00 | none | n/a | n/a | -0.166 |

Interpretation:

- G policies reach the fitted slider affordance for left and move along the
  correct residual direction, but their failed rollouts do not move the slider.
- Therefore left is not primarily an approach failure, sign failure, or
  undertraining issue.
- The current fitted-point affordance is not a sufficient contact-mode oracle
  for left. It lacks the side-of-handle and push-vs-pull variables needed to
  transmit force into the joint.
- Future oracle G should encode contact mode explicitly: side contact point,
  contact normal/side, force direction, and an actuation-stage condition based
  on actual joint progress or sustained contact rather than only distance to a
  fitted point.

## Left Success Video Set

Successful `move_slider_left` videos were saved locally:

```text
artifacts/server_sync/oracle_g_v2_articulation_2026-07-05/2026-07-05_move_slider_left_success_rollouts
```

Saved cases:

- no-G epoch 8: rollouts `1`, `13`, `14`;
- old v2 concat: rollouts `32`, `52`, `72` from a first-100 candidate scan;
- v2 split concat: rollouts `0`, `1`, `13`;
- old v2 FiLM: no success in first 100 candidates.

Full record:

- `docs/PHASE1_CALVIN_ORACLE_G_V2_ARTICULATION.md`
