#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
GPU="${GPU:-5}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
PYTHON="/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python"
RUN_DIR="${ROOT}/experiments/phase1_calvin/2026-07-16_E0_g_token_bc_fm_matched_d4_seed${SEED}"
OUT_DIR="${ROOT}/experiments/phase1_calvin/2026-07-16_E0_g_token_close_controls_seed${SEED}"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS="1"

exec "${PYTHON}" framework/scripts/policy_state_bc/eval_chunk_flow_checkpoints.py \
  --calvin-code /mnt/nvme33n1/wuk_code/calvin_g/code/calvin \
  --data-root /mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D \
  --out-dir "${OUT_DIR}" \
  --policies \
    "token_bc=${RUN_DIR}/chunk_bc_minimal_g_tokens/policy.pt" \
    "token_fm=${RUN_DIR}/chunk_fm_minimal_g_tokens/policy.pt" \
  --eval-task-filter close_drawer \
  --num-sequences 100 \
  --ep-len 160 \
  --eval-scan-sequences 50000 \
  --eval-sequence-offset 0 \
  --device cuda:0 \
  --g-scales 1.0 \
  --query-g-controls normal zero sign_flip task:open_drawer \
  --language-controls normal
