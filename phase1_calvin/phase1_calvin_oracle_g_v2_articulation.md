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

Full record:

- `docs/PHASE1_CALVIN_ORACLE_G_V2_ARTICULATION.md`
