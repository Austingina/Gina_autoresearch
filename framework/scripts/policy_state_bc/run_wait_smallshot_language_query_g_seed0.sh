#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
RUN_SCRIPT="${ROOT}/framework/scripts/policy_state_bc/run_smallshot_language_query_g_seed0.sh"
BASE_OUT_DIR="${BASE_OUT_DIR:-${ROOT}/experiments/phase1_calvin/2026-07-08_smallshot_language_query_g_e50_n25_h160_seed0}"
K_VALUES="${K_VALUES:-5 10 25}"
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

while true; do
  gpu="$(pick_gpu)"
  ts="$(date '+%F %T')"
  if [[ -n "${gpu}" ]]; then
    echo "[${ts}] launch smallshot on gpu ${gpu}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export DEVICE="cuda:0"
    export BASE_OUT_DIR
    export K_VALUES
    bash "${RUN_SCRIPT}"
    echo "[$(date '+%F %T')] finished"
    exit 0
  fi
  echo "[${ts}] no free gpu yet; waiting ${SLEEP_SECONDS}s"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  sleep "${SLEEP_SECONDS}"
done
