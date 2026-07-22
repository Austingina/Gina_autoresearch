#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-7}"
WAIT_PID="${WAIT_PID:-0}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
CALVIN_CODE="/mnt/nvme33n1/wuk_code/calvin_g/code/calvin"
DATA_ROOT="/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D"
TRAIN_DIR="${ROOT}/experiments/phase1_calvin/2026-07-16_E2_g_token_reverse_open_seed0"
CKPT="${TRAIN_DIR}/chunk_fm_minimal_g_tokens/policy.pt"
RUN_META="${TRAIN_DIR}/run_meta.json"
RESUMABLE="${ROOT}/framework/scripts/run_resumable_state_policy_eval.sh"

cd "${ROOT}"
if [[ "${WAIT_PID}" -gt 0 ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    echo "[E2-horizon] waiting for pid ${WAIT_PID}" >&2
    sleep 30
  done
fi

run_eval() {
  local label="$1"
  local horizon="$2"
  local ensemble="$3"
  local eval_dir="${ROOT}/experiments/phase1_calvin/2026-07-16_E2_g_token_fm_close_${label}_stateonly_seed0"
  bash "${RESUMABLE}" "${GPU}" "${eval_dir}" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${CKPT}" \
    --run-meta "${RUN_META}" \
    --num-sequences 100 \
    --ep-len 160 \
    --eval-scan-sequences 5000 \
    --eval-task-filter close_drawer \
    --action-execution-horizon "${horizon}" \
    --temporal-ensemble "${ensemble}" \
    --temporal-ensemble-decay 0.25 \
    --query-g-control normal \
    --policy-sample-seed 17 \
    --fm-reseed-each-episode 1 \
    --device cuda:0 \
    --state-only-observation
}

run_eval h2 2 0
run_eval h4 4 0
run_eval h8 8 0
run_eval h1_temporal_ensemble 1 1
