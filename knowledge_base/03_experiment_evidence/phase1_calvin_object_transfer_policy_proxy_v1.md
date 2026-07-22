# Phase 1 CALVIN Object-Transfer Policy Proxy V1

**Date:** 2026-07-03  
**Goal:** test whether the current reusable-G has action-predictive signal before modifying the full CALVIN policy stack.

## Setup

This is an offline behavior-cloning proxy, not a CALVIN rollout. It trains the same small MLP action predictor on source records and evaluates on held-out target records:

- split: `object_transfer_red_blue_to_pink`
- source: red/blue object-transfer records
- target: pink object-transfer records
- train records: 8000
- target validation records: 4000
- base feature: `feature_mode=canonical_nearest`
- action target: `rel_actions`
- G source: G-v2 `canonical_nearest` feature + `canonical_nearest_with_scene` target, `clip=0.99`
- epochs: 60

Variants:

- `no_g`
- `zero_g`
- `source_mean_g`
- `shuffled_pred_g`
- `predicted_g`
- `oracle_g`

## Result

| Variant | Val MSE | Val MAE | Pos MSE | Rot MSE | Gripper Acc | vs No-G |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle_g` | **0.05462** | **0.12285** | **0.00831** | **0.08976** | **0.9790** | **-29.75%** |
| `source_mean_g` | 0.07639 | 0.16548 | 0.04075 | 0.10273 | 0.9730 | -1.74% |
| `zero_g` | 0.07684 | 0.16593 | 0.04078 | 0.10334 | 0.9730 | -1.16% |
| `no_g` | 0.07774 | 0.16683 | 0.04160 | 0.10376 | 0.9728 | 0.00% |
| `shuffled_pred_g` | 0.08046 | 0.17272 | 0.04395 | 0.10720 | 0.9740 | +3.50% |
| `predicted_g` | 0.08288 | 0.17236 | 0.04703 | 0.10892 | 0.9725 | +6.62% |

## Interpretation

This is the clearest policy-side diagnostic so far:

1. **G is not meaningless.** Oracle G gives a large improvement over no-G, especially on position/action components. The G representation contains policy-relevant information.
2. **Predicted G is not ready for direct policy training.** Predicted G is worse than no-G, zero-G, source-mean-G, and shuffled predicted G in this proxy. Feeding the current predicted G into a policy is likely to hurt.
3. **The bottleneck is prediction quality/alignment, not the existence of useful G.** The gap between `oracle_g` and `predicted_g` is large enough that policy training should not yet rely on predicted G as a normal input.

## Decision

Do **not** start full predicted-G policy training yet.

Next step should be one of:

1. Improve G prediction/alignment first, then repeat this proxy.
2. Train a policy with **oracle-G** as an upper-bound diagnostic only.
3. Use predicted G as an auxiliary loss/regularizer, not as a direct observation channel.

The most informative next experiment is:

```text
object-transfer policy smoke:
  no_g vs oracle_g first
  only add predicted_g after it passes this proxy
```

## Artifacts

Remote:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_object_transfer_policy_proxy_v1
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/object_transfer_policy_proxy_v1_2026-07-03.tar.gz
```

Local:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/object_transfer_policy_proxy_v1_2026-07-03
```
