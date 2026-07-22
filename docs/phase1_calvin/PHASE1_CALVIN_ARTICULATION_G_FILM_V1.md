# Phase 1 CALVIN Articulation-G FiLM V1

**Date:** 2026-07-03

## Goal

Follow up on the negative direct oracle-G results by testing whether a narrower
task-family interaction geometry can help state-BC policies on CALVIN
articulation tasks.

The tested family is:

- `open_drawer`
- `close_drawer`
- `move_slider_left`
- `move_slider_right`

## Motivation

Earlier state-BC rollout experiments showed that raw dense oracle-G
(`target_scene - current_scene`) and broad structured G can improve offline
action MSE while hurting closed-loop task success. This experiment tests two
corrections:

1. keep the original absolute state input so ID skills such as slider-right are
   not destroyed by premature canonicalization;
2. use a narrow articulation-family G instead of a full scene target.

## Articulation G

The articulation G is a small rule-oracle vector computed from current slider
and drawer state:

- current controls: `scene_obs[:2]`
- target controls:
  - `move_slider_left`: slider target `0.28`
  - `move_slider_right`: slider target `0.0`
  - `open_drawer`: drawer target `0.22`
  - `close_drawer`: drawer target `0.0`
- delta, absolute delta, sign(delta)
- task one-hot

This is intentionally much narrower than full scene-delta G.

## Variants

| Mode | Base input | G interface |
| --- | --- | --- |
| `no_g` | `robot_obs + scene_obs + task one-hot` | none |
| `articulation_g_concat` | absolute base | concat narrow articulation G |
| `articulation_g_film` | absolute base | FiLM modulation from narrow articulation G |

All variants use the same fixed train/val split and model initialization seed.

## Training Setup

- dataset: CALVIN `task_D_D`
- training tasks: articulation family only
- max samples: 20000
- stride: 2
- epochs: 8
- hidden: 256
- layers: 3
- device: CPU
- seed: 0

Important implementation fix:

- `train_variant` now uses a per-variant deterministic seed for train/val split,
  model initialization, and dataloader shuffling. This avoids mode-order effects.

## First-Task Eval

Initial filtered eval used the first 20 official CALVIN sequences whose first
task was in the articulation family.

| Mode | Val MSE | SR@1 |
| --- | ---: | ---: |
| `no_g` | 0.02899 | **0.60** |
| `articulation_g_concat` | **0.02835** | 0.20 |
| `articulation_g_film` | 0.02891 | **0.60** |

Per-task:

| Mode | close_drawer | move_slider_left | move_slider_right | open_drawer |
| --- | ---: | ---: | ---: | ---: |
| `no_g` | 1/1 | 1/4 | **8/8** | 2/7 |
| `articulation_g_concat` | 1/1 | 0/4 | 2/8 | 1/7 |
| `articulation_g_film` | 1/1 | 1/4 | 6/8 | **4/7** |

This showed that FiLM improved drawer opening but the unbalanced sequence set
hid the benefit.

## Balanced First-Task Eval

Eval-only rerun using the saved checkpoints:

- eval mode: `balanced_first_task`
- 10 official first-task sequences per articulation task
- total: 40 attempts per mode
- `ep_len`: 160

| Mode | Val MSE | Balanced SR@1 |
| --- | ---: | ---: |
| `no_g` | 0.02899 | 0.55 |
| `articulation_g_concat` | **0.02835** | 0.35 |
| `articulation_g_film` | 0.02891 | **0.675** |

Per-task balanced results:

| Mode | close_drawer | move_slider_left | move_slider_right | open_drawer |
| --- | ---: | ---: | ---: | ---: |
| `no_g` | 10/10 | 1/10 | **10/10** | 1/10 |
| `articulation_g_concat` | 10/10 | 1/10 | 2/10 | 1/10 |
| `articulation_g_film` | 10/10 | **3/10** | 8/10 | **6/10** |

## Interpretation

This is the first policy-level positive signal for interaction geometry in the
CALVIN task-metric loop:

1. Raw dense G remains a bad policy interface.
2. Concatenating G remains unsafe: it improves one-step MSE but damages
   closed-loop task success.
3. Narrow task-family G with FiLM modulation can improve balanced articulation
   success: `0.55 -> 0.675`.
4. The gain comes mostly from harder underfit tasks (`open_drawer`,
   `move_slider_left`), while it slightly hurts an already-solved task
   (`move_slider_right`).

This supports the revised story: G should be a reusable task-family interaction
geometry that modulates policy learning, not a raw future-state vector appended
to the action policy.

## Decision

Keep the articulation-G FiLM branch alive.

Next steps:

1. Repeat balanced eval with more sequences, e.g. 25 or 50 per task.
2. Run two or three seeds for `no_g` vs `articulation_g_film`.
3. Add a gated FiLM variant that only modulates the relevant task head or
   hidden block, to reduce the `move_slider_right` regression.
4. Treat concat G as a negative baseline rather than the main method.

## Artifacts

Remote:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_state_bc_articulation_abs_rule_g_seedfixed_first_task_n20_h160
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_state_bc_articulation_abs_rule_g_balanced_evalonly_n10_h160
/mnt/nvme33n1/wuk_code/AutoResearch/logs/state_bc_articulation_abs_rule_g_seedfixed_first_task_n20_h160.log
/mnt/nvme33n1/wuk_code/AutoResearch/logs/state_bc_articulation_abs_rule_g_balanced_evalonly_n10_h160.log
```

Local:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/phase1_calvin_articulation_g_2026-07-03
```
