# Phase 1 CALVIN Transfer: Open/Left/Right -> Close

**Date:** 2026-07-06

## Goal

Test the core transfer claim:

> Can a field-conditioned policy trained on seen interaction tasks reuse its
> control skill on a held-out task when only the field changes?

This is the first clean held-out-task split after v7. We intentionally do not
use `move_slider_left` as the held-out task because it is already a known hard
case. Instead, we test `close_drawer`, which shares the drawer mechanism with
`open_drawer` but requires the opposite direction.

## Setup

- Train tasks: `open_drawer`, `move_slider_left`, `move_slider_right`
- Test task: `close_drawer`
- Task one-hot: zeroed for both train and eval via `--zero-task-onehot`
- Eval: 25 balanced first-task close rollouts, horizon 160
- Checkpoints: 4, 8, 16, 32, 50
- Modes:
  - `no_g`
  - `articulation_g_v7_unified_field_concat`
  - `articulation_g_v7_unified_field_film`

Command script:

```text
framework/scripts/policy_state_bc/run_transfer_open_left_right_to_close_v7_seed0.sh
```

## Results

| Mode | Best Checkpoint | Best Close SR |
| --- | ---: | ---: |
| `no_g` | any | 0/25 |
| `articulation_g_v7_unified_field_concat` | any | 0/25 |
| `articulation_g_v7_unified_field_film` | 50 | 1/25 |

Full checkpoint table:

```text
no_g: e04 0/25, e08 0/25, e16 0/25, e32 0/25, e50 0/25
v7_concat: e04 0/25, e08 0/25, e16 0/25, e32 0/25, e50 0/25
v7_film: e04 0/25, e08 0/25, e16 0/25, e32 0/25, e50 1/25
```

## Interpretation

This split is useful because no-G cannot solve held-out close without task id:
it scores `0/25` at every checkpoint. That means the eval is sensitive to
whether `G` can actually carry transferable task information.

The current v7 field does not pass this test. Even with close-specific v7
field input at eval time, the policy trained on open/left/right does not learn
to close the drawer. FiLM does not fix the issue: the best FiLM checkpoint is
only `1/25`.

This is stronger evidence than same-task SR:

- v7 can fit actions offline (`val_mse ~0.022-0.023`);
- v7 can condition behavior in seen-task settings;
- but v7 does not provide a reusable interaction interface for new tasks.

Therefore, the next G should not merely be a cleaner target-motion field. It
must be a reusable interaction field whose semantics are shared across tasks,
so a policy can learn "approach/contact/apply local field" from seen tasks and
reuse that control pattern on held-out tasks.

## Design Consequence For V8

V8 should be evaluated first on this split. The target criterion is not
same-task average SR, but:

```text
train: open_drawer + move_slider_left + move_slider_right
test: close_drawer
metric: held-out close SR
```

A meaningful v8 result should move close above the v7 ceiling:

```text
v8_interaction_field >> 1/25 close success
```

## Artifacts

Remote:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-06_transfer_train_open_left_right_test_close_no_task_v7_e4_8_16_32_50_n25_h160_seed0
/mnt/nvme33n1/wuk_code/AutoResearch/logs/transfer_train_open_left_right_test_close_no_task_v7_e4_8_16_32_50_n25_h160_seed0.log
```

Local:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/transfer_open_left_right_to_close_2026-07-06/state_bc_eval/2026-07-06_transfer_train_open_left_right_test_close_no_task_v7_e4_8_16_32_50_n25_h160_seed0
```

