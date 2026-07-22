#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
TRAIN_SCRIPT="${ROOT}/framework/scripts/policy_state_bc/run_minimal_g_direction_coverage_condition.sh"
EVAL_SCRIPT="${ROOT}/framework/scripts/policy_state_bc/run_minimal_g_direction_coverage_eval.sh"
LOG_ROOT="${ROOT}/logs/2026-07-15_minimal_g_direction_coverage_matrix"
STATUS_ROOT="${LOG_ROOT}/status"
read -r -a CONDITION_LIST <<< "${CONDITIONS:-D0 D1 D2 D3}"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4}"
read -r -a GPU_LIST <<< "${GPU_IDS:-0 2}"
MAX_USED_MEM_MB="${MAX_USED_MEM_MB:-20000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-20}"
POLL_SECONDS="${POLL_SECONDS:-30}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
read -r -a WAIT_PIDS <<< "${WAIT_FOR_PIDS:-}"

mkdir -p "${LOG_ROOT}" "${STATUS_ROOT}"

for wait_pid in "${WAIT_PIDS[@]}"; do
  [[ -z "${wait_pid}" ]] && continue
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
done

declare -A JOB_PIDS=()
declare -A JOB_GPUS=()
pending=()
for condition in "${CONDITION_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    pending+=("${condition}:${seed}")
  done
done

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
  local job
  for job in "${!JOB_GPUS[@]}"; do
    [[ "${JOB_GPUS[${job}]}" == "${gpu}" ]] && return 0
  done
  return 1
}

launch_job() {
  local job="$1"
  local gpu="$2"
  local condition="${job%%:*}"
  local seed="${job##*:}"
  local label="${condition}_seed${seed}"
  local log="${LOG_ROOT}/${label}.log"
  local status="${STATUS_ROOT}/${label}.status"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export EGL_VISIBLE_DEVICES="${gpu}"
    export DEVICE="cuda:0"
    export CONDITION="${condition}"
    export SEED="${seed}"
    if [[ "${RUN_TRAIN}" == "0" ]] || bash "${TRAIN_SCRIPT}" >"${log}" 2>&1; then
      if [[ "${RUN_EVAL}" == "1" ]]; then
        bash "${EVAL_SCRIPT}" >>"${log}" 2>&1
      fi
      printf 'complete\n' >"${status}"
    else
      printf 'failed\n' >"${status}"
      exit 1
    fi
  ) &
  JOB_PIDS["${job}"]=$!
  JOB_GPUS["${job}"]="${gpu}"
  printf '[launch] job=%s gpu=%s pid=%s log=%s\n' \
    "${job}" "${gpu}" "${JOB_PIDS[${job}]}" "${log}"
}

while [[ "${#pending[@]}" -gt 0 || "${#JOB_PIDS[@]}" -gt 0 ]]; do
  for job in "${!JOB_PIDS[@]}"; do
    pid="${JOB_PIDS[${job}]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" || true
      unset 'JOB_PIDS['"${job}"']'
      unset 'JOB_GPUS['"${job}"']'
    fi
  done

  while [[ "${#pending[@]}" -gt 0 ]]; do
    selected_gpu=""
    for gpu in "${GPU_LIST[@]}"; do
      if ! gpu_is_assigned "${gpu}" && gpu_is_free "${gpu}"; then
        selected_gpu="${gpu}"
        break
      fi
    done
    [[ -z "${selected_gpu}" ]] && break
    job="${pending[0]}"
    pending=("${pending[@]:1}")
    launch_job "${job}" "${selected_gpu}"
  done

  [[ "${#pending[@]}" -gt 0 || "${#JOB_PIDS[@]}" -gt 0 ]] && sleep "${POLL_SECONDS}"
done

printf '[complete] all requested jobs finished\n'
