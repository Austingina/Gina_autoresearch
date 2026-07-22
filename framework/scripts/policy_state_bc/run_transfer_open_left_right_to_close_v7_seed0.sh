#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/nvme33n1/wuk_code/AutoResearch
PY=${ROOT}/phase1_calvin/envs/calvin_h200/bin/python
SCRIPT=${ROOT}/framework/scripts/policy_state_bc/train_eval_state_bc_task_policy.py
CALVIN_CODE=${ROOT}/phase1_calvin/calvin_code
DATA_ROOT=${ROOT}/phase1_calvin/data/task_D_D
GEOM_JSON=${ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json
OUT_DIR=${ROOT}/experiments/phase1_calvin/2026-07-06_transfer_train_open_left_right_test_close_no_task_v7_e4_8_16_32_50_n25_h160_seed0

mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES=2 "${PY}" "${SCRIPT}" \
  --calvin-code "${CALVIN_CODE}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --modes no_g articulation_g_v7_unified_field_concat articulation_g_v7_unified_field_film \
  --include-tasks open_drawer move_slider_left move_slider_right \
  --eval-task-filter close_drawer \
  --zero-task-onehot \
  --articulation-geometry-json "${GEOM_JSON}" \
  --max-samples 60000 \
  --stride 2 \
  --epochs 50 \
  --checkpoint-epochs 4 8 16 32 50 \
  --eval-checkpoints \
  --batch-size 512 \
  --hidden 256 \
  --layers 3 \
  --lr 3e-4 \
  --num-sequences 25 \
  --ep-len 160 \
  --eval-mode balanced_first_task \
  --eval-scan-sequences 5000 \
  --device cuda:0 \
  --seed 0
