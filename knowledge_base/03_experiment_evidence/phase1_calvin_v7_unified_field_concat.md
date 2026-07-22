# Phase 1 CALVIN V7 Unified Field Concat

**Date:** 2026-07-06

## Goal

Test whether merging v6 attraction/correction into a single target-local field
improves the articulation policy input.

`v7_unified_field` keeps the v5/v6 token shape for compatibility, but uses only
one field channel:

```text
token = [anchor_delta(3), unified_field(3), zeros(3), field_weight, 0]
```

All anchors are target-local, centered around the active drawer/slider part.
This removes the old v5 non-target-object repulsion shortcut.

## Setup

- Dataset: CALVIN `task_D_D`
- Tasks: `close_drawer`, `move_slider_left`, `move_slider_right`, `open_drawer`
- Mode: `articulation_g_v7_unified_field_concat`
- Training: 50 epochs, checkpoint eval at 4/8/16/32/50
- Eval: 25 balanced first-task rollouts per task, horizon 160
- GPU: `CUDA_VISIBLE_DEVICES=2`

Command script:

```text
framework/scripts/policy_state_bc/run_v7_unified_field_concat_seed0.sh
```

## Results

| Epoch | Val MSE | SR | Close | Left | Right | Open |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.03033 | 0.77 | 25/25 | 2/25 | 25/25 | 25/25 |
| 8 | 0.02673 | 0.74 | 25/25 | 0/25 | 25/25 | 24/25 |
| 16 | 0.02464 | 0.74 | 25/25 | 0/25 | 24/25 | 25/25 |
| 32 | 0.02350 | **0.78** | 24/25 | **9/25** | 24/25 | 21/25 |
| 50 | **0.02249** | 0.68 | 23/25 | 1/25 | 25/25 | 19/25 |

## Interpretation

V7 fixes the visible semantic bug in v5: the field no longer uses non-target
objects as repulsion anchors. However, removing that shortcut does not improve
closed-loop task success. The best checkpoint is `e32` at `0.78`, below the
old v5 peak (`0.85`) and only slightly above v6 clean-field variants.

The key failure remains `move_slider_left`. More training lowers action MSE but
does not monotonically improve rollout success; `e50` has the best MSE and a
worse SR. This is additional evidence that task success is not governed by
offline action MSE or raw field cleanliness alone.

Current conclusion:

- A clean local flow/field is a good visualizable primitive.
- It is not sufficient as the full `G`.
- The next `G` should combine local flow with explicit entity, affordance
  frame, progress/residual, and contact/phase information.

This agrees with the 3DFlowAction / Im2Flow2Act literature boundary: object
flow is useful, but our competitive story should be typed Interaction Geometry
tokens rather than "another flow world model."

## Artifacts

Remote:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-06_state_bc_v7_unified_field_concat_e4_8_16_32_50_n25_h160_seed0
/mnt/nvme33n1/wuk_code/AutoResearch/logs/state_bc_v7_unified_field_concat_e4_8_16_32_50_n25_h160_seed0.log
```

Local:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/oracle_g_v7_unified_field_2026-07-06/state_bc_eval/2026-07-06_state_bc_v7_unified_field_concat_e4_8_16_32_50_n25_h160_seed0
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/oracle_g_v7_unified_field_2026-07-06/render_overlay_static/2026-07-06_g_v7_unified_field_render_overlay_static
```

