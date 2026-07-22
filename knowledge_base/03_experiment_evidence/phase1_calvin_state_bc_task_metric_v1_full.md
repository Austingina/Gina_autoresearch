# Phase 1 CALVIN State-BC Task Metric V1

**Date:** 2026-07-03

## Goal

Run the first real rollout/task-metric comparison between a no-G policy and an
oracle-G policy. This is a lightweight state-BC bridge experiment, not the final
vision/flow policy stack.

## Setup

- dataset: CALVIN `task_D_D`
- policy family: state-conditioned behavior cloning MLP
- base input: `robot_obs`, `scene_obs`, task one-hot
- no-G: base input only
- oracle-G: base input plus demo-derived oracle geometry
- oracle-G definition: for each subtask, retrieve the nearest training demo
  start scene and use `target_scene - current_scene` as privileged G
- train samples: 47350 collected transitions
- train segments: 3000
- stride: 4
- epochs: 8
- eval: 10 CALVIN long-horizon sequences, `EP_LEN=120`

Command:

```bash
CUDA_VISIBLE_DEVICES=5 \
PYTHONPATH=/mnt/nvme33n1/wuk_code/AutoResearch/phase1_calvin/calvin_code/calvin_models:/mnt/nvme33n1/wuk_code/AutoResearch/phase1_calvin/calvin_code/calvin_env:${PYTHONPATH:-} \
/mnt/nvme33n1/wuk_code/AutoResearch/phase1_calvin/envs/calvin_h200/bin/python \
  framework/scripts/policy_state_bc/train_eval_state_bc_task_policy.py \
  --calvin-code /mnt/nvme33n1/wuk_code/AutoResearch/phase1_calvin/calvin_code \
  --data-root /mnt/nvme33n1/wuk_code/AutoResearch/phase1_calvin/data/task_D_D \
  --out-dir /mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_state_bc_task_D_D_no_g_vs_oracle_g_n10_h120 \
  --max-samples 100000 \
  --max-segments 3000 \
  --stride 4 \
  --epochs 8 \
  --batch-size 1024 \
  --hidden 256 \
  --layers 3 \
  --num-sequences 10 \
  --ep-len 120 \
  --device cuda:0 \
  --seed 0
```

## Results

| Mode | Val MSE | Gripper Acc | Avg Seq Len | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_g` | 0.03746 | 0.9787 | **0.20** | **0.10** | **0.10** | 0.00 | 0.00 | 0.00 |
| `oracle_g` | **0.03655** | 0.9776 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

No-G per-task successes:

```text
move_slider_left: 1 / 1
open_drawer: 1 / 3
all other observed tasks: 0 success
```

Oracle-G per-task successes:

```text
all observed tasks: 0 success
```

## Interpretation

This result is important because it separates offline action fit from closed-loop
task success:

1. Oracle-G slightly improves offline action MSE.
2. Oracle-G does not improve rollout task metrics in this first state-BC policy.
3. The current demo-target `target_scene - current_scene` G is not yet a safe
   direct policy input.

The likely issue is interface/alignment, not necessarily that interaction
geometry is useless. A single dense scene delta mixes object, articulated
mechanism, and irrelevant state dimensions; it can improve one-step regression
while destabilizing closed-loop action selection.

## Decision

Do not promote this oracle-G design as evidence of task-level improvement.

Next policy-level oracle-G attempt should use a more structured interface:

- task/family-specific G heads for object, articulation, and toggle tasks
- phase-gated G input, especially after contact/grasp
- auxiliary G loss or FiLM modulation instead of raw concatenation
- per-task success diagnostics with the same official evaluator

## Artifacts

Remote:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_state_bc_task_D_D_no_g_vs_oracle_g_n10_h120
/mnt/nvme33n1/wuk_code/AutoResearch/logs/state_bc_task_D_D_n10_h120_launcher.log
```

Local:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/state_bc_task_D_D_no_g_vs_oracle_g_n10_h120_2026-07-03
```
