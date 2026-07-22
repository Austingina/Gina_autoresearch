# Phase 1 CALVIN Candidate-G Predictor Smoke

Date: 2026-07-02

## Goal

Validate the first reusable-G pretraining node after candidate G labels are extracted.

This is not yet the paper model. It is a tiny MLP smoke test to check whether future interaction-geometry targets are well-formed, learnable, and easy to visualize.

## Script

```text
framework/scripts/g_models/train_candidate_g_predictor.py
```

## Input

```text
artifacts/g_labels/debug_training_all_schema_aware
```

The input contains 72 records from all 9 annotated training sequences in the CALVIN debug dataset.

## Command

```bash
cd /mnt/nvme33n1/wuk_code/AutoResearch
PY=/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python

$PY framework/scripts/g_models/train_candidate_g_predictor.py \
  --g-dir artifacts/g_labels/debug_training_all_schema_aware \
  --horizon 8 \
  --epochs 150 \
  --batch-size 16 \
  --hidden 128 \
  --depth 2 \
  --out-dir artifacts/g_predictors/debug_training_h8_mlp_smoke
```

## Output

```text
artifacts/g_predictors/debug_training_h8_mlp_smoke
```

Files:

- `checkpoint.pt`
- `metrics.json`
- `predictions.npz`
- `loss_curve.png`
- `prediction_preview.png`
- `summary.md`

## Metrics

```text
records: 72
train_records: 58
val_records: 14
feature_dim: 46
target_dim: 12
horizon: 8
final_train_mse_standardized: 0.01846
final_val_mse_standardized: 0.22729
best_val_mse_standardized: 0.22311
```

## Interpretation

The tiny predictor overfits the debug dataset, which is expected because the dataset is very small. The useful result is that the full G-pretraining path now runs:

```text
CALVIN frames -> schema-aware G labels -> future-G targets -> predictor checkpoint -> loss/prediction visualizations
```

The prediction preview shows plausible tracking for EEF future deltas. Some object-delta dimensions are near-zero in this small subset, so they should not be overinterpreted.

## Next

1. Repeat this on full CALVIN data.
2. Add per-task/family validation splits instead of random record splits.
3. Add a sequence model or temporal encoder after the MLP smoke path is stable.
4. Use this predictor as the first learned-G candidate for hard/auxiliary/gated policy ablations only after full-data validation.
