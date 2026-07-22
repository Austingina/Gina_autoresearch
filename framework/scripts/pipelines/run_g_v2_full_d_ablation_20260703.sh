#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
DATA_ROOT="${DATA_ROOT:-/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D}"
TAG_PREFIX="${TAG_PREFIX:-2026-07-03_task_D_D_g_v2}"
PREDICTOR_EPOCHS="${PREDICTOR_EPOCHS:-100}"
G_TARGET_CLIP_QUANTILE="${G_TARGET_CLIP_QUANTILE:-0.99}"

cd "$ROOT"

recipes=(
  object_transfer_red_blue_to_pink
  primitive_transfer_push_lift_rotate_to_place_stack
  articulation_transfer_slider_to_drawer
  articulation_transfer_drawer_to_slider
)

target_modes=(
  canonical_nearest_with_scene
  raw_with_scene
)

echo "[start] $(date -Iseconds)"
echo "[root] $ROOT"
echo "[data_root] $DATA_ROOT"
echo "[tag_prefix] $TAG_PREFIX"
echo "[predictor_epochs] $PREDICTOR_EPOCHS"
echo "[target_clip_quantile] $G_TARGET_CLIP_QUANTILE"

for mode in "${target_modes[@]}"; do
  for recipe in "${recipes[@]}"; do
    echo "[run] mode=$mode recipe=$recipe"
    TAG="${TAG_PREFIX}_${mode}_clip${G_TARGET_CLIP_QUANTILE}" \
    DATA_ROOT="$DATA_ROOT" \
    RECIPE="$recipe" \
    G_TARGET_MODE="$mode" \
    G_TARGET_CLIP_QUANTILE="$G_TARGET_CLIP_QUANTILE" \
    RAW_NUM_SEQUENCES="${RAW_NUM_SEQUENCES:-8}" \
    RAW_MAX_FRAMES="${RAW_MAX_FRAMES:-4}" \
    G_MAX_SEQUENCES="${G_MAX_SEQUENCES:-0}" \
    G_MAX_FRAMES_PER_SEQUENCE="${G_MAX_FRAMES_PER_SEQUENCE:-16}" \
    G_VIS_NUM_SEQUENCES="${G_VIS_NUM_SEQUENCES:-8}" \
    G_VIS_MAX_FRAMES="${G_VIS_MAX_FRAMES:-6}" \
    PREDICTOR_EPOCHS="$PREDICTOR_EPOCHS" \
    framework/scripts/pipelines/run_reusable_g_validation_v1.sh
    echo "[done-run] mode=$mode recipe=$recipe"
  done
done

echo "[done] $(date -Iseconds)"
