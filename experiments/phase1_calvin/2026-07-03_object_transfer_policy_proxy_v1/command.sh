#!/usr/bin/env bash
set -euo pipefail
cd /mnt/nvme33n1/wuk_code/AutoResearch
PY=/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python
SRC=/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_task_D_D_g_v2_canonical_nearest_with_scene_clip0.99_object_transfer_red_blue_to_pink/artifacts/g_object_transfer_red_blue_to_pink_source
TGT=/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_task_D_D_g_v2_canonical_nearest_with_scene_clip0.99_object_transfer_red_blue_to_pink/artifacts/g_object_transfer_red_blue_to_pink_target
PRED=/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_task_D_D_g_v2_canonical_nearest_with_scene_clip0.99_object_transfer_red_blue_to_pink/artifacts/g_predictor_object_transfer_red_blue_to_pink_featcanonical_targetcanonical_scene_clip0.99_h8/predictions.npz
CUDA_VISIBLE_DEVICES=2 "$PY" framework/scripts/policy_proxy/train_g_conditioned_action_proxy.py \
  --dataset-root /mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D \
  --source-g-dir "$SRC" \
  --target-g-dir "$TGT" \
  --g-predictions "$PRED" \
  --out-dir /mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_object_transfer_policy_proxy_v1 \
  --feature-mode canonical_nearest \
  --action-key rel_actions \
  --max-train-records 8000 \
  --max-val-records 4000 \
  --epochs 60 \
  --batch-size 256 \
  --hidden 256 \
  --depth 2 \
  --seed 0 \
  --device auto \
  --torch-threads 8
