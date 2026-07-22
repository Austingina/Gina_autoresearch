#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/nvme33n1/wuk_code/AutoResearch
PY=/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python
BASE=$ROOT/experiments/phase1_calvin
recipes=(articulation_transfer_slider_to_drawer articulation_transfer_drawer_to_slider)
features=(raw canonical_nearest)
for recipe in "${recipes[@]}"; do
  RUN=$BASE/2026-07-03_task_D_D_g_v2_canonical_nearest_with_scene_clip0.99_${recipe}
  SRC=$RUN/artifacts/g_${recipe}_source
  TGT=$RUN/artifacts/g_${recipe}_target
  for feat in "${features[@]}"; do
    OUT=$RUN/artifacts/g_predictor_${recipe}_feat${feat}_targetscene_only_clip0.99_h8
    echo "[train] recipe=$recipe feature=$feat out=$OUT"
    "$PY" $ROOT/framework/scripts/g_models/train_candidate_g_predictor.py \
      --g-dir "$SRC" \
      --val-g-dir "$TGT" \
      --horizon 8 \
      --feature-mode "$feat" \
      --target-mode scene_only \
      --target-clip-quantile 0.99 \
      --epochs 100 \
      --batch-size 64 \
      --hidden 128 \
      --depth 2 \
      --out-dir "$OUT"
  done
done
