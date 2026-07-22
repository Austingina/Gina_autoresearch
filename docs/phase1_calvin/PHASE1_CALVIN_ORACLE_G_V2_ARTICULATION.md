# Phase 1 CALVIN Oracle-G V2 Articulation

**Date:** 2026-07-04

## Goal

Build and validate a compact oracle Interaction Geometry representation for
CALVIN articulation tasks, then compare concat vs FiLM under the same oracle G.

This run explicitly avoids judging from very small eval sets. The final policy
comparison uses balanced first-task eval with 25 sequences per task, 100 rollout
attempts per mode.

## Tasks

- `open_drawer`
- `close_drawer`
- `move_slider_left`
- `move_slider_right`

## Oracle G V2 Definition

`G_v2` is a compact 9D representation:

```text
G_v2 = [
  target_affordance,  # 3D: fitted part/contact point - eef position
  residual_motion,    # 3D: fitted motion axis * normalized remaining joint residual
  interaction_stage,  # 3D one-hot: approach / actuate / done
]
```

No `active_joint_id`, task-id shortcut, full target scene, block deltas, or
unrelated scene state is included.

## Oracle Geometry Estimation

A per-joint contact trajectory model is fitted from demos:

```text
tcp_contact_point(q) ~= intercept + slope * q
```

where `q` is slider or drawer joint state. Fitting uses frames where the joint
is moving.

Full model samples:

| Joint | Actuation Samples | Axis | Fit Mean Residual | Contact Threshold |
| --- | ---: | --- | ---: | ---: |
| slider | 8487 | `[-0.9998, 0.0003, 0.0191]` | 0.0123 m | 0.0400 m |
| drawer | 7855 | `[0.0425, -0.9960, 0.0783]` | 0.0266 m | 0.0627 m |

## Geometry Stability

Geometry estimation was tested with different demo counts per task and 3 seeds.
Compared to the full model:

| Segments / Task | Joint | Axis Cosine Range | Midpoint Delta Range |
| ---: | --- | ---: | ---: |
| 20 | slider | 0.99966-0.999998 | 2.55-3.46 mm |
| 20 | drawer | 0.99933-0.99964 | 1.65-4.64 mm |
| 50 | slider | 0.99994-0.99999 | 0.94-1.78 mm |
| 50 | drawer | 0.99976-0.99998 | 1.73-3.57 mm |
| 100 | slider | 0.999998-0.9999996 | 0.71-0.80 mm |
| 100 | drawer | 0.99983-0.99999 | 0.45-0.90 mm |
| 150 | slider | ~1.00000 | 0.02-0.10 mm |
| 150 | drawer | 0.999997-1.00000 | 0.06-0.12 mm |

Conclusion: the fitted affordance/motion geometry is stable enough for oracle G
construction. Using the full model is justified.

## Policy Comparison

Setup:

- train samples: 60000
- stride: 2
- epochs: 8
- hidden: 256
- layers: 3
- eval: `balanced_first_task`
- eval sequences: 25 per task, 100 total attempts per mode
- `ep_len`: 160
- seed: 0

| Mode | G | Interface | Val MSE | Balanced SR@1 |
| --- | --- | --- | ---: | ---: |
| `no_g` | none | none | 0.02899 | 0.56 |
| `articulation_g_concat` | v1 narrow joint residual | concat | 0.02835 | 0.33 |
| `articulation_g_film` | v1 narrow joint residual | FiLM | 0.02891 | **0.66** |
| `articulation_g_v2_concat` | v2 compact affordance/residual/stage | concat | 0.02783 | 0.65 |
| `articulation_g_v2_film` | v2 compact affordance/residual/stage | FiLM | **0.02776** | 0.57 |
| `articulation_g_v2_split_concat` | v2 split axis/scalar residual | concat | 0.02778 | **0.70** |
| `articulation_g_v2_split_film` | v2 split axis/scalar residual | FiLM | 0.02789 | 0.61 |

Per-task SR:

| Mode | close_drawer | move_slider_left | move_slider_right | open_drawer |
| --- | ---: | ---: | ---: | ---: |
| `no_g` | 25/25 | 6/25 | 23/25 | 2/25 |
| `articulation_g_concat` | 25/25 | 3/25 | 3/25 | 2/25 |
| `articulation_g_film` | 25/25 | **9/25** | 18/25 | 14/25 |
| `articulation_g_v2_concat` | 24/25 | 1/25 | **25/25** | **15/25** |
| `articulation_g_v2_film` | 25/25 | 0/25 | 24/25 | 8/25 |
| `articulation_g_v2_split_concat` | 25/25 | 6/25 | **25/25** | 14/25 |
| `articulation_g_v2_split_film` | 25/25 | 0/25 | **25/25** | 11/25 |

## Interpretation

1. The earlier concern was correct: concat should not be globally dismissed.
   With a better compact G, `v2_concat` improves from the bad `v1_concat`
   result (`0.33 -> 0.65`) and beats no-G (`0.56`).
2. FiLM is not universally better. After splitting `axis_unit` and
   `signed_residual_scalar`, `v2_split_concat` becomes the best seed-0 mode
   (`0.70`), while `v2_split_film` still lags (`0.61`).
3. `v2_concat` is more selective:
   - preserves `move_slider_right` perfectly (`25/25`);
   - strongly improves `open_drawer` (`2/25 -> 15/25`);
   - but collapses `move_slider_left` (`6/25 -> 1/25`).
   Splitting axis and scalar residual recovers `move_slider_left` to no-G
   level (`6/25`) while preserving `move_slider_right` (`25/25`) and keeping
   `open_drawer` high (`14/25`).
4. Offline MSE is not predictive of task success:
   - `v2_film` has the best MSE but only `0.57` SR;
   - `v2_split_concat` and `v2_split_film` have similar MSE, but task success
     differs by 9 points (`0.70` vs `0.61`).

## Slider Left/Right Diagnostic

A follow-up diagnostic was run on the CALVIN training demos with the same fitted
v2 geometry. The goal was to check whether `move_slider_left` and
`move_slider_right` are symmetric in the variables that enter `G_v2`.

Frame-level demo statistics:

| Quantity | `move_slider_left` | `move_slider_right` |
| --- | ---: | ---: |
| Frames | 4907 | 5001 |
| `q` mean / p50 | 0.061 / 0.020 | 0.218 / 0.255 |
| normalized residual mean / p50 | 0.782 / 0.930 | -0.778 / -0.911 |
| affordance norm mean / p50 / p90 | 0.061 / 0.024 / 0.166 | 0.065 / 0.027 / 0.168 |
| residual motion mean | `[-0.782, 0.000, 0.015]` | `[0.778, -0.000, -0.015]` |
| stage fractions approach / actuate / done | 0.414 / 0.558 / 0.028 | 0.441 / 0.523 / 0.037 |
| action xyz mean | `[-0.164, 0.072, 0.004]` | `[0.053, 0.064, -0.010]` |
| action x p50 / p90 | -0.162 / 0.270 | 0.019 / 0.574 |

Interpretation:

- The v2 geometric state is mostly symmetric: residual magnitude, affordance
  distance, and interaction-stage distribution are close between left and
  right.
- The residual sign is intentionally opposite. With the fitted slider axis,
  `move_slider_left` produces mostly negative-x residual motion and
  `move_slider_right` produces mostly positive-x residual motion.
- The demo action distribution is not a clean mirror. `move_slider_left` has a
  more negative median x action, while `move_slider_right` has a much larger
  positive p90 x action. Therefore a policy that treats `axis * residual` as a
  strong motion prior can become direction-biased.
- Since no-G already solves `move_slider_right` well and weakly solves
  `move_slider_left`, the `v2_concat` collapse on left is likely an interface
  or representation issue rather than lack of oracle information.

Immediate implication: test a less prescriptive slider representation before
training a learned G, e.g. split `axis_unit` and `residual_scalar`, or include
both `axis_unit` and signed scalar residual instead of only the product
`axis * residual`.

## Split Axis/Residual Follow-up

The split representation was implemented as:

```text
G_v2_split = [
  target_affordance,       # 3D
  axis_unit,               # 3D
  signed_residual_scalar,  # 1D
  interaction_stage,       # 3D
]
```

It was trained and evaluated with the same n25 balanced protocol.

Result:

- `v2_split_concat`: `0.70` SR, current best seed-0 balanced score.
- `v2_split_film`: `0.61` SR, better than old `v2_film` (`0.57`) but still
  fails on `move_slider_left`.

Interpretation:

- The split representation confirms that old `axis * residual` was too
  prescriptive for concat.
- The improvement is mainly from recovering `move_slider_left`
  (`1/25 -> 6/25`) without losing `move_slider_right`.
- The remaining left performance is only back to no-G level, not solved.
  Therefore the representation is improved but the left-slider interaction
  still needs a better phase/contact or action-conditioning design.

## Qualitative Videos

Rollout videos were exported for one fixed validation first-task initial state
per task, using the same initial state across:

- `no_g`
- `articulation_g_v2_concat`
- `articulation_g_v2_film`

Single-rollout success labels:

| Mode | close_drawer | move_slider_left | move_slider_right | open_drawer |
| --- | ---: | ---: | ---: | ---: |
| `no_g` | success @64 | fail | success @73 | fail |
| `articulation_g_v2_concat` | success @66 | fail | success @70 | success @142 |
| `articulation_g_v2_film` | success @61 | fail | success @76 | fail |

These videos are for qualitative failure analysis only; they should not replace
the n25 task metrics.

## Decision

Keep both branches alive:

- `v2_split_concat`: current best seed-0 balanced task success.
- `v1_film`: still a strong modulation baseline and has the best
  `move_slider_left` count among current modes.
- `v2_concat`: validates compact oracle G but its old `axis * residual`
  product is too brittle on `move_slider_left`.

Do not train a learned G predictor yet. First run ablations that prove the
oracle G is being used causally in the control loop.

## Important Protocol Note

The first n25 run was interrupted because eval appeared slow. That was a bad
call: 100 rollouts per mode are slow but necessary to reduce eval bias. The run
was restarted and completed. Future method comparisons should not be shortened
for convenience once launched, unless there is evidence of a true hang or crash.

## Next Steps

1. Run zero-G, shuffled-G, and sign-flipped-G eval ablations for
   `v2_split_concat`.
2. Visualize `move_slider_left` failures from the exported videos and compare
   approach/contact/action phase against successful demos.
3. Run seed 1 and seed 2 for `v2_split_concat`, `v1_film`, and no-G.
4. Keep n25 balanced eval as the minimum credible policy metric for this branch.

## Artifacts

Remote:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-04_state_bc_articulation_v1_v2_g_balanced_n25_h160_seed0_rerun
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-04_state_bc_articulation_v2_split_g_balanced_n25_h160_seed0_rerun
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-04_state_bc_articulation_v1_v2_video_rollouts
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-04_oracle_g_v2_left_right_diagnostics
/mnt/nvme33n1/wuk_code/AutoResearch/logs/oracle_g_v2_articulation_geometry_stability.log
/mnt/nvme33n1/wuk_code/AutoResearch/logs/state_bc_articulation_v1_v2_g_balanced_n25_h160_seed0_rerun.log
/mnt/nvme33n1/wuk_code/AutoResearch/logs/state_bc_articulation_v2_split_g_balanced_n25_h160_seed0_rerun.log
/mnt/nvme33n1/wuk_code/AutoResearch/logs/oracle_g_v2_left_right_diagnostics.log
```

Local:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/oracle_g_v2_articulation_2026-07-04
```
