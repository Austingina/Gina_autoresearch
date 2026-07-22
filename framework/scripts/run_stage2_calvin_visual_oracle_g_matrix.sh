#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
FEATURE_MANIFEST="/mnt/nvme33n1/wuk_code/calvin_g/data/task_ABC_D_compact_blocks_dinov2_small/manifest.json"
LOG_ROOT="${ROOT}/logs/M2_abc_d_visual_dinov2"
mkdir -p "${LOG_ROOT}"

until [[ -f "${FEATURE_MANIFEST}" ]]; do
  sleep 30
done

CONDITIONS=(
  "with_g none 0"
  "language_only minilm 2"
)

for condition in "${CONDITIONS[@]}"; do
  read -r variant language_mode gpu <<< "${condition}"
  train_log="${LOG_ROOT}/train_seed${SEED}_${variant}_${language_mode}.log"
  queue_log="${LOG_ROOT}/queue_seed${SEED}_${variant}_${language_mode}.log"

  nohup bash "${ROOT}/framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_train.sh" \
    "${SEED}" "${gpu}" visual "${variant}" "${language_mode}" \
    > "${train_log}" 2>&1 < /dev/null &
  train_pid=$!

  nohup bash "${ROOT}/framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_eval_queue.sh" \
    "${SEED}" "${gpu}" visual "${variant}" "${language_mode}" \
    > "${queue_log}" 2>&1 < /dev/null &
  queue_pid=$!

  printf 'seed=%s variant=%s language=%s gpu=%s train_pid=%s queue_pid=%s\n' \
    "${SEED}" "${variant}" "${language_mode}" "${gpu}" "${train_pid}" "${queue_pid}"
done
