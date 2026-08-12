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
CALVIN_G_ROOT="${CALVIN_G_ROOT:-$(cd "${ROOT}/.." && pwd)/calvin_g}"
CALVIN_DATA="${CALVIN_DATA:-${CALVIN_G_ROOT}/data}"
FEATURE_MANIFEST="${CALVIN_DINOV2_FEATURE_ROOT:-${CALVIN_DATA}/task_ABC_D_compact_blocks_dinov2_small}/manifest.json"
LOG_ROOT="${ROOT}/logs/M2_abc_d_visual_dinov2"
mkdir -p "${LOG_ROOT}"
IFS=',' read -r -a GPUS <<< "${CALVIN_GPU_LIST:-0}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "CALVIN_GPU_LIST must contain at least one CUDA device" >&2
  exit 2
fi

until [[ -f "${FEATURE_MANIFEST}" ]]; do
  sleep 30
done

CONDITIONS=(
  "with_g none"
  "language_only minilm"
)

index=0
for condition in "${CONDITIONS[@]}"; do
  read -r variant language_mode <<< "${condition}"
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  index=$((index + 1))
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
