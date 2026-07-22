#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
LOG_ROOT="${ROOT}/logs/M1_abc_d_scene_gate_none"
mkdir -p "${LOG_ROOT}"

for condition in "full 1" "proprio 6"; do
  read -r state_scope gpu <<< "${condition}"
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
