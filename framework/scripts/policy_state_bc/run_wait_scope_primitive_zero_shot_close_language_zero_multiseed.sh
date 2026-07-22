#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
RUN_SCRIPT="${ROOT}/framework/scripts/policy_state_bc/run_scope_primitive_zero_shot_close_language_zero_seed.sh"
LOG_ROOT="${ROOT}/logs/2026-07-11_scope_primitive_zero_shot_close_language_zero"
STATUS_ROOT="${LOG_ROOT}/status"
SEEDS=(${SEEDS:-0 1 2 3 4})
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MAX_USED_MEM_MB="${MAX_USED_MEM_MB:-20000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-20}"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "${LOG_ROOT}" "${STATUS_ROOT}"

declare -A JOB_PIDS=()
declare -A JOB_GPUS=()
pending=("${SEEDS[@]}")

timestamp() {
  date '+%F %T'
}

gpu_is_free() {
  local gpu="$1"
  local used util
  IFS=, read -r used util < <(
    nvidia-smi --id="${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
  )
  used="${used// /}"
  util="${util// /}"
  [[ "${used}" -le "${MAX_USED_MEM_MB}" && "${util}" -le "${MAX_UTIL_PCT}" ]]
}

gpu_is_assigned() {
  local gpu="$1"
  local seed
  for seed in "${!JOB_GPUS[@]}"; do
    [[ "${JOB_GPUS[${seed}]}" == "${gpu}" ]] && return 0
  done
  return 1
}

launch_seed() {
  local seed="$1"
  local gpu="$2"
  local log="${LOG_ROOT}/seed${seed}.log"
  local status="${STATUS_ROOT}/seed${seed}.status"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export DEVICE="cuda:0"
    export SEED="${seed}"
    if bash "${RUN_SCRIPT}" >"${log}" 2>&1; then
      printf 'complete\n' >"${status}"
    else
      printf 'failed\n' >"${status}"
      exit 1
    fi
  ) &
  JOB_PIDS["${seed}"]=$!
  JOB_GPUS["${seed}"]="${gpu}"
  printf '[%s] launched seed=%s gpu=%s pid=%s log=%s\n' \
    "$(timestamp)" "${seed}" "${gpu}" "${JOB_PIDS[${seed}]}" "${log}"
}

while [[ "${#pending[@]}" -gt 0 || "${#JOB_PIDS[@]}" -gt 0 ]]; do
  for seed in "${!JOB_PIDS[@]}"; do
    pid="${JOB_PIDS[${seed}]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then
        printf '[%s] completed seed=%s gpu=%s\n' "$(timestamp)" "${seed}" "${JOB_GPUS[${seed}]}"
      else
        printf '[%s] failed seed=%s gpu=%s\n' "$(timestamp)" "${seed}" "${JOB_GPUS[${seed}]}"
      fi
      unset 'JOB_PIDS['"${seed}"']'
      unset 'JOB_GPUS['"${seed}"']'
    fi
  done

  while [[ "${#pending[@]}" -gt 0 && "${#JOB_PIDS[@]}" -lt "${MAX_PARALLEL}" ]]; do
    selected_gpu=""
    for gpu in $(seq 0 7); do
      if ! gpu_is_assigned "${gpu}" && gpu_is_free "${gpu}"; then
        selected_gpu="${gpu}"
        break
      fi
    done
    [[ -z "${selected_gpu}" ]] && break
    seed="${pending[0]}"
    pending=("${pending[@]:1}")
    launch_seed "${seed}" "${selected_gpu}"
  done

  [[ "${#pending[@]}" -gt 0 || "${#JOB_PIDS[@]}" -gt 0 ]] && sleep "${POLL_SECONDS}"
done

printf '[%s] all requested seeds finished\n' "$(timestamp)"
