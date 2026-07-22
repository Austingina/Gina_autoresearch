# Phase1 CALVIN Gated / Scale-Augmented V7 Plan

Date: 2026-07-08

## Motivation

The latest v7 scale sweep changed the interpretation of the held-out close
failure.

Observed evidence:

```text
chunk_bc_v7 e50, train open/left/right, test close:
g=1 close_drawer: 0/25
g=2 close_drawer: 19/25
g=4 close_drawer: 0/25
```

But the same global `g=2` destroys seen-task behavior:

```text
train tasks at g=1: 71/75
train tasks at g=2: 5/75
```

Therefore v7 G is not useless, but the policy interface is fragile. The next
question is not whether to add more hand-engineered terms to G. The next
question is whether the policy can learn how strongly to use G under
distribution shift.

## Hypothesis

```text
A calibrated G interface can preserve seen-task behavior while using stronger
G guidance on held-out close_drawer.
```

The goal is to make G usage adaptive rather than manually choosing an
inference-time `g_scale`.

## Variants

### 1. `chunk_bc_v7_scale_aug`

Architecture:

```text
MLP([state, v7_G]) -> action_chunk
```

Training-time augmentation:

```text
g_scale sampled from {0, 0.5, 1, 2, 4}
G dropout probability = 0.15
```

Purpose:

```text
Test whether ordinary BC becomes robust to G magnitude once it sees multiple
G strengths during training.
```

### 2. `chunk_bc_v7_gated`

Architecture:

```text
base_action = base_branch(state)
g_residual = residual_branch(state, G)
lambda = sigmoid(gate_branch(state, G)) * gate_max
action_chunk = base_action + lambda * g_residual
```

Training-time augmentation:

```text
g_scale sampled from {0, 0.5, 1, 2, 4}
G dropout probability = 0.15
gate_max = 2.0
```

Purpose:

```text
Test whether separating no-G behavior from G residual correction prevents G
from destroying already learned train-task behavior.
```

## Primary Transfer Split

```text
train tasks: open_drawer, move_slider_left, move_slider_right
held-out task: close_drawer
task one-hot: zeroed
chunk size: 16
stride: 2
checkpoints: 4, 8, 16, 32, 50
eval: 25 close_drawer first-task rollouts, ep_len 160
```

Run script:

```text
framework/scripts/policy_state_bc/run_transfer_gated_scale_aug_v7_seed0.sh
```

Remote output:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-08_transfer_gated_scale_aug_v7_e4_8_16_32_50_n25_h160_seed0
```

## Required Follow-Up Evaluation

After the close-transfer sweep:

1. Evaluate the best checkpoints on train tasks:

```text
open_drawer, move_slider_left, move_slider_right
```

2. Run G causal controls:

```text
g_scale = 0
g_scale = 0.5
g_scale = 1
g_scale = 2
g_scale = 4
```

3. Run axis/contact diagnostics for the best close checkpoint:

```text
near_part_entry
min_dist_to_part
min_dist_to_axis_line
joint_progress_after_contact
weird_axis_satisfied_steps
```

4. Save selected close videos for success and failure cases.

## Decision Criteria

Positive signal:

```text
held-out close_drawer improves above v7 default while train tasks do not
collapse.
```

Concrete first-pass target:

```text
close_drawer >= 10/25 at g=1
train open/left/right >= 55/75 at g=1
zero/shuffle/low-scale G should reduce close performance
```

Strong signal:

```text
close_drawer >= 15/25 at g=1
train open/left/right >= 65/75 at g=1
best checkpoint remains useful under n=100 close rollouts
```

Negative signal:

```text
close improves only by manual eval-time g_scale and train tasks collapse, or
gated/scale-aug policies do not outperform default v7 on close.
```

## Link To Mainstream FM / Diffusion Policies

This experiment is the interface gate before adapting mainstream policies.

If gated/scale-aug v7 succeeds, then G can be inserted into stronger methods as
an extra structured condition:

```text
action_chunk ~ p(a_{t:t+H} | image, state, language, M, G)
```

Recommended integration order:

1. LeRobot Diffusion Policy with a G token encoder.
2. DP3 / FlowPolicy-style 3D policy with G tokens attached to point features.
3. OpenPI/pi0-style VLA adapter only after the LeRobot/DP3-level result is
   positive.

Do not claim that current small FM is a mainstream flow baseline. It is only a
diagnostic decoder. The mainstream baseline should be started after the G
interface passes this gate.

## First Run Result

Date: 2026-07-08

Remote artifact:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-08_transfer_gated_scale_aug_v7_e4_8_16_32_50_n25_h160_seed0
```

Local synced artifact:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-07_transfer_chunk_bc_vs_fm_v7_h16/2026-07-08_transfer_gated_scale_aug_v7_e4_8_16_32_50_n25_h160_seed0
```

Close transfer result:

| Mode | e4 | e8 | e16 | e32 | e50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chunk_bc_v7_scale_aug` | 0/25 | 0/25 | 0/25 | 0/25 | 0/25 |
| `chunk_bc_v7_gated` | 0/25 | 0/25 | 0/25 | 0/25 | 0/25 |

Gated model diagnostics:

| Epoch | Val chunk MSE | Mean gate |
| ---: | ---: | ---: |
| 1 | 0.0689 | 1.06 |
| 4 | 0.0441 | 1.12 |
| 8 | 0.0403 | 1.24 |
| 16 | 0.0381 | 1.31 |
| 32 | 0.0401 | 1.61 |
| 50 | 0.0415 | 1.87 |

Interpretation:

```text
This first gated/scale-aug attempt does not pass the interface gate.

Scale augmentation alone does not teach the ordinary MLP to reuse v7 G for
held-out close. The residual-gated model learns to assign a high scalar gate,
eventually close to gate_max=2.0, but still gets 0/25 close success. Therefore
the failure is not just that the model refuses to use G. It may be using the
G residual in a non-task-conditioned or behaviorally wrong way.
```

Immediate implication:

```text
Do not proceed to mainstream FM/VLA adaptation based on this result alone.
The current interface has not yet shown automatic closed-loop transfer.
```

Next diagnostic before another architecture change:

```text
Evaluate the trained scale_aug/gated checkpoints on train tasks and g_scale
sweeps. This will tell whether the new policies preserved seen-task behavior
and whether manual eval-time g=2 can still rescue them. If train tasks are also
weak, the augmentation/gate damaged policy learning. If train tasks are strong
but close remains 0 even at g=2, then the learned interface suppressed or
misused the transferable v7 close signal.
```

## Follow-Up Result: Train Non-Regression And Close G Sweep

Date: 2026-07-08

Remote train-task artifact:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-08_gated_scale_aug_v7_train_task_eval_e50_g1_2_n25_h160
```

Remote close-sweep artifact:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-08_gated_scale_aug_v7_close_gscale_sweep_e50_n25_h160
```

Local synced parent:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-07_transfer_chunk_bc_vs_fm_v7_h16
```

Train-task result at epoch 50:

| Mode | g scale | Left | Right | Open | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chunk_bc_v7_scale_aug` | 1 | 22/25 | 23/25 | 24/25 | 69/75 |
| `chunk_bc_v7_scale_aug` | 2 | 25/25 | 25/25 | 25/25 | 75/75 |
| `chunk_bc_v7_gated` | 1 | 22/25 | 24/25 | 25/25 | 71/75 |
| `chunk_bc_v7_gated` | 2 | 23/25 | 23/25 | 25/25 | 71/75 |

Held-out close result at epoch 50:

| Mode | g=0 | g=0.5 | g=1 | g=2 | g=4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chunk_bc_v7_scale_aug` | 0/25 | 0/25 | 0/25 | 0/25 | 0/25 |
| `chunk_bc_v7_gated` | 0/25 | 0/25 | 0/25 | 0/25 | 0/25 |

Conclusion:

```text
The negative result is now sharper. Gated/scale-aug does not simply damage
the policy: seen-task behavior is preserved, and scale_aug is even robust at
g=2. However, the held-out close transfer signal disappears completely.
```

Interpretation:

```text
The original non-augmented chunk_bc_v7 g=2 close success was likely a narrow
extrapolation effect of the raw concat MLP interface. Scale augmentation made
the policy invariant or robust to G magnitude on the training tasks, but did
not teach it to reinterpret the same reusable field as a new task objective.

Therefore the bottleneck is not only G strength. It is the semantic interface
between reusable field and action decoder. The current decoder can preserve
known behaviors under G perturbation, but it does not bind unseen close_drawer
to the desired object-motion direction.
```

Actionable next step:

```text
Do not continue by only sweeping global G scale or gate_max. The next useful
diagnostic is to compare trajectories against the original non-augmented
bc_e50 g=2 close-success cases:

1. EEF-to-handle minimum distance and contact timing.
2. Object joint progress after near-contact.
3. Whether the action component along drawer axis appears before contact.
4. Whether scale_aug/gated treats G as a harmless perturbation rather than a
   task-switching condition.

If this diagnosis confirms that G is ignored as task semantics, the next model
change should expose G as an explicit goal token or energy/field query target,
not only as a concatenated continuous feature or scalar residual gate.
```
