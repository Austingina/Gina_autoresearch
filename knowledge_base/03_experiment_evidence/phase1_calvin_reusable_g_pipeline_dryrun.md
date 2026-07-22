# Phase 1 CALVIN Reusable-G Pipeline Dry Run

Date: 2026-07-02

## Goal

Prepare the formal full-CALVIN reusable interaction geometry pipeline and validate it end-to-end on the current debug dataset.

## Current Data Availability

The server currently has only:

```text
/mnt/nvme33n1/wuk_code/calvin_g/data/calvin_debug_dataset
```

Full CALVIN data is not yet present under:

```text
/mnt/nvme33n1/wuk_code/calvin_g/data
/mnt/nvme33n1/wuk_code/AutoResearch/data/phase1_calvin
```

## New Pipeline Script

```text
framework/scripts/pipelines/run_reusable_g_validation_v1.sh
```

The script runs:

1. dataset schema summary;
2. task split manifest;
3. raw RGB/RGBD visual checks;
4. source G extraction;
5. target G extraction;
6. source G overlay visualization;
7. source-to-target candidate-G predictor.

## Dry-Run Command

```bash
cd /mnt/nvme33n1/wuk_code/AutoResearch

DATA_ROOT=/mnt/nvme33n1/wuk_code/calvin_g/data/calvin_debug_dataset \
TAG=debug_dryrun \
RECIPE=object_transfer_red_blue_to_pink \
RAW_NUM_SEQUENCES=4 \
RAW_MAX_FRAMES=4 \
G_MAX_FRAMES_PER_SEQUENCE=8 \
G_VIS_NUM_SEQUENCES=4 \
G_VIS_MAX_FRAMES=6 \
PREDICTOR_EPOCHS=80 \
framework/scripts/pipelines/run_reusable_g_validation_v1.sh
```

## Dry-Run Output

```text
experiments/phase1_calvin/debug_dryrun_object_transfer_red_blue_to_pink
```

The dry-run completed successfully.

Debug split size:

```text
source sequences: 3
target sequences: 1
source G records: 24
target G records: 8
```

## Predictor Result

The source-to-target predictor ran in held-out mode:

```text
split_mode: heldout_g_dir
horizon: 8
feature_dim: 46
target_dim: 12
train_records: 24
val_records: 8
```

Metrics from `g_predictor_object_transfer_red_blue_to_pink_h8_v3`:

```text
train_rmse_raw: 0.00845
val_rmse_raw: 36.17960
zero_train_rmse_raw: 0.01807
zero_val_rmse_raw: 0.02614
```

Interpretation:

- The predictor fits the tiny source set.
- It fails badly on the tiny held-out target set.
- The zero-delta baseline is far better on debug target.
- This is not a scientific conclusion about reusable G; the debug dataset is too small and target coverage is too sparse.
- This result is useful because it forced us to add zero-delta baseline metrics and confirms that held-out evaluation is wired correctly.

## Full CALVIN Data Requirement

The next formal run requires at least CALVIN split D:

```text
task_D_D.zip, about 166 GB
```

CALVIN also provides:

```text
task_ABC_D.zip, about 517 GB
task_ABCD_D.zip, about 656 GB
```

Recommended first full-data step:

```bash
cd /mnt/nvme33n1/wuk_code/calvin_g/data
bash /mnt/nvme33n1/wuk_code/calvin_g/code/calvin/dataset/download_data.sh D
```

After download, the expected dataset root is:

```text
/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D
```

Then run:

```bash
cd /mnt/nvme33n1/wuk_code/AutoResearch

DATA_ROOT=/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D \
TAG=task_D_D_reusable_g_v1 \
RECIPE=object_transfer_red_blue_to_pink \
framework/scripts/pipelines/run_reusable_g_validation_v1.sh
```

Repeat with:

```text
primitive_transfer_push_lift_rotate_to_place_stack
articulation_transfer_slider_to_drawer
articulation_transfer_drawer_to_slider
```

## Next Decision Point

The first real decision should be based on full data:

> Does source-trained G dynamics beat zero-delta and simple baselines on held-out target tasks?

If yes, proceed to GT/proxy-G policy upper bound.

If no, improve G dynamics formulation before injecting learned G into policy.
