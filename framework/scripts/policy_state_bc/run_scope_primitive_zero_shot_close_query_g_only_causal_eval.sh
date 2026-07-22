#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}"
SEED="${SEED:-0}"
TRAIN_DIR="${TRAIN_DIR:-${ROOT}/experiments/phase1_calvin/2026-07-11_scope_primitive_zero_shot_close_query_g_only_paired_e50_n100_h160_seed${SEED}}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/phase1_calvin/2026-07-11_scope_primitive_zero_shot_close_query_g_only_causal_val_n100_seed${SEED}}"

cd "${ROOT}"

"${PYTHON}" framework/scripts/policy_state_bc/eval_chunk_flow_checkpoints.py \
  --calvin-code "${CALVIN_CODE}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --policies "seed${SEED}_query_g=${TRAIN_DIR}/chunk_bc_query_g/policy.pt" \
  --eval-task-filter close_drawer \
  --num-sequences 100 \
  --ep-len 160 \
  --eval-scan-sequences 50000 \
  --eval-sequence-offset 0 \
  --query-g-controls normal zero sign_flip task:open_drawer \
  --language-controls normal \
  --device "${DEVICE:-cuda:0}"
