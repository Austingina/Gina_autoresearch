#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}
CALVIN_CODE=${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}
DATA_ROOT=${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}
GEOM_JSON=${GEOM_JSON:-${ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json}
OUT_DIR=${OUT_DIR:-${ROOT}/experiments/phase1_calvin/2026-07-07_transfer_chunk_bc_vs_fm_v7_h16_e4_8_16_32_50_n25_h160_seed0}
PYTHON=${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}

cd "${ROOT}"

"${PYTHON}" framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py \
  --calvin-code "${CALVIN_CODE}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --articulation-geometry-json "${GEOM_JSON}" \
  --modes chunk_bc_v7 chunk_fm_v7 \
  --include-tasks open_drawer move_slider_left move_slider_right \
  --zero-task-onehot \
  --chunk-size 16 \
  --stride 2 \
  --epochs 50 \
  --batch-size 512 \
  --hidden 512 \
  --layers 4 \
  --lr 1e-3 \
  --seed 0 \
  --checkpoint-epochs 4 8 16 32 50 \
  --eval-checkpoints \
  --eval-mode balanced_first_task \
  --eval-task-filter close_drawer \
  --eval-scan-sequences 5000 \
  --num-sequences 25 \
  --ep-len 160 \
  --fm-steps 10 \
  --device cuda:0
