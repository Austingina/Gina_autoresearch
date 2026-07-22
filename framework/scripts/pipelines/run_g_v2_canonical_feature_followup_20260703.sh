#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PY="${PY:-/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python}"
TAG_PREFIX="${TAG_PREFIX:-2026-07-03_task_D_D_g_v2}"
TARGET_MODE="${TARGET_MODE:-canonical_nearest_with_scene}"
TARGET_CLIP="${TARGET_CLIP:-0.99}"
PREDICTOR_EPOCHS="${PREDICTOR_EPOCHS:-100}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$ROOT"

recipes=(
  object_transfer_red_blue_to_pink
  primitive_transfer_push_lift_rotate_to_place_stack
  articulation_transfer_slider_to_drawer
  articulation_transfer_drawer_to_slider
)

echo "[start] $(date -Iseconds)"
echo "[target_mode] $TARGET_MODE"
echo "[target_clip] $TARGET_CLIP"
echo "[predictor_epochs] $PREDICTOR_EPOCHS"

for recipe in "${recipes[@]}"; do
  run_dir="$ROOT/experiments/phase1_calvin/${TAG_PREFIX}_${TARGET_MODE}_clip${TARGET_CLIP}_${recipe}"
  source_dir="$run_dir/artifacts/g_${recipe}_source"
  target_dir="$run_dir/artifacts/g_${recipe}_target"
  out_dir="$run_dir/artifacts/g_predictor_${recipe}_featcanonical_targetcanonical_scene_clip${TARGET_CLIP}_h8"

  echo "[wait] recipe=$recipe"
  until [ -f "$source_dir/candidate_g_labels.npz" ] && [ -f "$target_dir/candidate_g_labels.npz" ]; do
    sleep "$WAIT_SECONDS"
  done

  if [ -f "$out_dir/metrics.json" ]; then
    echo "[skip-existing] $out_dir"
    continue
  fi

  echo "[train] recipe=$recipe out=$out_dir"
  "$PY" framework/scripts/g_models/train_candidate_g_predictor.py \
    --g-dir "$source_dir" \
    --val-g-dir "$target_dir" \
    --horizon 8 \
    --feature-mode canonical_nearest \
    --target-mode "$TARGET_MODE" \
    --target-clip-quantile "$TARGET_CLIP" \
    --epochs "$PREDICTOR_EPOCHS" \
    --batch-size 64 \
    --hidden 128 \
    --depth 2 \
    --out-dir "$out_dir"
  echo "[done-train] recipe=$recipe"
done

echo "[done] $(date -Iseconds)"
