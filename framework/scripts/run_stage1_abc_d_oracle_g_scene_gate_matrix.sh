#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
LANGUAGE_MODE="${2:-minilm}"

ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
LOG_ROOT="${ROOT}/logs/M1_abc_d_scene_gate"
if [[ "${LANGUAGE_MODE}" != "onehot" ]]; then
  LOG_ROOT="${LOG_ROOT}_${LANGUAGE_MODE}"
fi
mkdir -p "${LOG_ROOT}"

CONDITIONS=(
  "full with_g 0"
  "full language_only 1"
  "proprio with_g 2"
  "proprio language_only 6"
)

for condition in "${CONDITIONS[@]}"; do
  read -r state_scope variant gpu <<< "${condition}"
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
