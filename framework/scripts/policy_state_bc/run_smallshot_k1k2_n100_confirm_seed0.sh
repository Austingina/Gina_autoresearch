#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}"
BASE_OUT_DIR="${BASE_OUT_DIR:-${ROOT}/experiments/phase1_calvin/2026-07-08_smallshot_language_query_g_e50_n25_h160_seed0}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/phase1_calvin/2026-07-08_smallshot_k1k2_confirm_n100_h160_seed0}"
K_VALUES=(${K_VALUES:-1 2})
MEM_LIMIT_MB="${MEM_LIMIT_MB:-30000}"
UTIL_LIMIT="${UTIL_LIMIT:-30}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, -v mem_limit="${MEM_LIMIT_MB}" -v util_limit="${UTIL_LIMIT}" '
      {
        gsub(/ /, "", $1);
        gsub(/ /, "", $2);
        gsub(/ /, "", $3);
        if (first == "" && $2 < mem_limit && $3 < util_limit) {
          first = $1;
        }
      }
      END {
        if (first != "") {
          print first;
        }
      }'
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  while true; do
    GPU="$(pick_gpu)"
    if [[ -n "${GPU}" ]]; then
      export CUDA_VISIBLE_DEVICES="${GPU}"
      export DEVICE="cuda:0"
      echo "[$(date '+%F %T')] launch n100 confirm on gpu ${GPU}"
      break
    fi
    echo "[$(date '+%F %T')] no free gpu yet; waiting ${SLEEP_SECONDS}s"
    sleep "${SLEEP_SECONDS}"
  done
fi

cd "${ROOT}"
for K in "${K_VALUES[@]}"; do
  STATE_CKPT="${BASE_OUT_DIR}/close_k${K}/chunk_bc_state/checkpoints/epoch_0050/policy.pt"
  LANG_CKPT="${BASE_OUT_DIR}/close_k${K}/chunk_bc_lang/checkpoints/epoch_0050/policy.pt"
  QUERY_G_CKPT="${BASE_OUT_DIR}/close_k${K}/chunk_bc_query_g/checkpoints/epoch_0050/policy.pt"
  LANG_QUERY_G_CKPT="${BASE_OUT_DIR}/close_k${K}/chunk_bc_lang_query_g/checkpoints/epoch_0050/policy.pt"
  "${PYTHON}" framework/scripts/policy_state_bc/eval_chunk_flow_checkpoints.py \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${DATA_ROOT}" \
    --out-dir "${OUT_DIR}/normal_close_k${K}" \
    --policies \
      "state_k${K}=${STATE_CKPT}" \
      "lang_k${K}=${LANG_CKPT}" \
      "query_g_k${K}=${QUERY_G_CKPT}" \
      "lang_query_g_k${K}=${LANG_QUERY_G_CKPT}" \
    --eval-task-filter close_drawer \
    --num-sequences 100 \
    --ep-len 160 \
    --eval-scan-sequences 10000 \
    --device "${DEVICE:-cuda:0}" \
    --g-scales 1 \
    --query-g-controls normal

  "${PYTHON}" framework/scripts/policy_state_bc/eval_chunk_flow_checkpoints.py \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${DATA_ROOT}" \
    --out-dir "${OUT_DIR}/corruption_close_k${K}" \
    --policies \
      "query_g_k${K}=${QUERY_G_CKPT}" \
      "lang_query_g_k${K}=${LANG_QUERY_G_CKPT}" \
    --eval-task-filter close_drawer \
    --num-sequences 100 \
    --ep-len 160 \
    --eval-scan-sequences 10000 \
    --device "${DEVICE:-cuda:0}" \
    --g-scales 1 \
    --query-g-controls normal zero sign_flip task:open_drawer
done
