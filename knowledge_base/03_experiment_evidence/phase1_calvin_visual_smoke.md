# Phase 1 CALVIN Visual Smoke

Date: 2026-07-02

## Goal

Start the reusable interaction geometry pipeline with manual verification before training new models.

## Implemented Scripts

Location:

```text
framework/scripts/visual_checks/
```

Scripts:

- `visualize_calvin_episode.py`
- `make_calvin_task_split_manifest.py`
- `extract_candidate_g_labels.py`
- `visualize_interaction_geometry.py`
- `calvin_vis_utils.py`

## Smoke Dataset

Dataset:

```text
/mnt/nvme33n1/wuk_code/calvin_g/data/calvin_debug_dataset
```

Confirmed per-frame keys:

- `rgb_static`
- `rgb_gripper`
- `depth_static`
- `depth_gripper`
- `robot_obs`
- `scene_obs`
- `actions`
- `rel_actions`

## Smoke Commands

```bash
cd /mnt/nvme33n1/wuk_code/AutoResearch
PY=/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python
DATA=/mnt/nvme33n1/wuk_code/calvin_g/data/calvin_debug_dataset

$PY framework/scripts/visual_checks/visualize_calvin_episode.py \
  --dataset-root "$DATA" \
  --num-sequences 2 \
  --max-frames 4 \
  --seed 7 \
  --out-dir artifacts/visual_checks/smoke_raw_episode

$PY framework/scripts/visual_checks/make_calvin_task_split_manifest.py \
  --dataset-root "$DATA" \
  --recipe object_transfer_red_blue_to_pink \
  --out artifacts/task_splits/object_transfer_red_blue_to_pink.json

$PY framework/scripts/visual_checks/extract_candidate_g_labels.py \
  --dataset-root "$DATA" \
  --split-manifest artifacts/task_splits/object_transfer_red_blue_to_pink.json \
  --partition source \
  --max-sequences 4 \
  --max-frames-per-sequence 6 \
  --out-dir artifacts/g_labels/smoke_object_transfer_source

$PY framework/scripts/visual_checks/visualize_interaction_geometry.py \
  --dataset-root "$DATA" \
  --g-dir artifacts/g_labels/smoke_object_transfer_source \
  --num-sequences 4 \
  --max-frames 6 \
  --out-dir artifacts/visual_checks/smoke_interaction_geometry_v2
```

## Outputs

Raw episode visualizations:

```text
artifacts/visual_checks/smoke_raw_episode
```

Task split manifest:

```text
artifacts/task_splits/object_transfer_red_blue_to_pink.json
```

Candidate G labels:

```text
artifacts/g_labels/smoke_object_transfer_source
```

Interaction geometry overlays:

```text
artifacts/visual_checks/smoke_interaction_geometry_v2
```

## Smoke Result

The first end-to-end visual check passed.

- raw RGB/RGBD panels render correctly;
- task and language annotations are readable;
- candidate G labels are extracted from `robot_obs` and `scene_obs`;
- object-to-EEF distance curves show plausible contact approach;
- interaction geometry overlays are generated successfully.

The debug split is tiny for object transfer in this smoke run: source has 3 annotated sequences and target has 1 annotated sequence. This is enough for code validation, not for scientific evaluation.

## Immediate Issues

1. Overlay text is dense and should be made easier to inspect.
2. The `scene_obs` schema should be verified against CALVIN source/docs before being treated as final GT-G.
3. First full-data run should generate at least 20 raw episode checks and 20 G overlay checks.
4. Split recipes need leakage checks once full CALVIN data is available.
