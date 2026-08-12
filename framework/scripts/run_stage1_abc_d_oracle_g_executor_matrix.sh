#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${REPO_ROOT}/.env.server" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.server"
fi
ROOT="${AUTORESEARCH_ROOT:-${REPO_ROOT}}"
LOG_ROOT="${ROOT}/logs/M1_abc_d_scene_gate_none"
mkdir -p "${LOG_ROOT}"
IFS=',' read -r -a GPUS <<< "${CALVIN_GPU_LIST:-0}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "CALVIN_GPU_LIST must contain at least one CUDA device" >&2
  exit 2
fi

index=0
for state_scope in full proprio; do
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  index=$((index + 1))
  train_log="${LOG_ROOT}/train_seed${SEED}_${state_scope}_with_g.log"
  queue_log="${LOG_ROOT}/queue_seed${SEED}_${state_scope}_with_g.log"

  nohup bash "${ROOT}/framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_train.sh" \
    "${SEED}" "${gpu}" "${state_scope}" with_g none \
    > "${train_log}" 2>&1 < /dev/null &
  train_pid=$!

  nohup bash "${ROOT}/framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_eval_queue.sh" \
    "${SEED}" "${gpu}" "${state_scope}" with_g none \
    > "${queue_log}" 2>&1 < /dev/null &
  queue_pid=$!

  printf 'seed=%s scope=%s gpu=%s train_pid=%s queue_pid=%s\n' \
    "${SEED}" "${state_scope}" "${gpu}" "${train_pid}" "${queue_pid}"
done
