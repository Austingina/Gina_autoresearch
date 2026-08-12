#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
LANGUAGE_MODE="${2:-minilm}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${REPO_ROOT}/.env.server" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.server"
fi
ROOT="${AUTORESEARCH_ROOT:-${REPO_ROOT}}"
LOG_ROOT="${ROOT}/logs/M1_abc_d_scene_gate"
if [[ "${LANGUAGE_MODE}" != "onehot" ]]; then
  LOG_ROOT="${LOG_ROOT}_${LANGUAGE_MODE}"
fi
mkdir -p "${LOG_ROOT}"
IFS=',' read -r -a GPUS <<< "${CALVIN_GPU_LIST:-0}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "CALVIN_GPU_LIST must contain at least one CUDA device" >&2
  exit 2
fi

CONDITIONS=(
  "full with_g"
  "full language_only"
  "proprio with_g"
  "proprio language_only"
)

index=0
for condition in "${CONDITIONS[@]}"; do
  read -r state_scope variant <<< "${condition}"
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  index=$((index + 1))
  train_log="${LOG_ROOT}/train_seed${SEED}_${state_scope}_${variant}.log"
  queue_log="${LOG_ROOT}/queue_seed${SEED}_${state_scope}_${variant}.log"

  nohup bash "${ROOT}/framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_train.sh" \
    "${SEED}" "${gpu}" "${state_scope}" "${variant}" "${LANGUAGE_MODE}" \
    > "${train_log}" 2>&1 < /dev/null &
  train_pid=$!

  nohup bash "${ROOT}/framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_eval_queue.sh" \
    "${SEED}" "${gpu}" "${state_scope}" "${variant}" "${LANGUAGE_MODE}" \
    > "${queue_log}" 2>&1 < /dev/null &
  queue_pid=$!

  printf 'seed=%s language=%s scope=%s variant=%s gpu=%s train_pid=%s queue_pid=%s\n' \
    "${SEED}" "${LANGUAGE_MODE}" "${state_scope}" "${variant}" "${gpu}" \
    "${train_pid}" "${queue_pid}"
done
