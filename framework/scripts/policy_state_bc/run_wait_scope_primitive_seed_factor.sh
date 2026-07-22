#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
RUN_SCRIPT="${ROOT}/framework/scripts/policy_state_bc/run_scope_primitive_seed_factor_combo.sh"
LOG_ROOT="${ROOT}/logs/2026-07-11_scope_primitive_seed_factor"
STATUS_ROOT="${LOG_ROOT}/status"
COMBOS=(${COMBOS:-0:0 1:0 2:0 3:0 4:0 0:1 0:2 0:3 0:4})
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MAX_USED_MEM_MB="${MAX_USED_MEM_MB:-132000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-90}"
POLL_SECONDS="${POLL_SECONDS:-15}"

mkdir -p "${LOG_ROOT}" "${STATUS_ROOT}"

declare -A JOB_PIDS=()
declare -A JOB_GPUS=()
pending=("${COMBOS[@]}")

timestamp() {
  date '+%F %T'
}

combo_label() {
  local combo="$1"
  printf 'i%s_l%s' "${combo%%:*}" "${combo##*:}"
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
  local label
  for label in "${!JOB_GPUS[@]}"; do
    [[ "${JOB_GPUS[${label}]}" == "${gpu}" ]] && return 0
  done
  return 1
}

launch_combo() {
  local combo="$1"
  local gpu="$2"
  local init_seed="${combo%%:*}"
  local loader_seed="${combo##*:}"
  local label
  label="$(combo_label "${combo}")"
  local log="${LOG_ROOT}/${label}.log"
  local status="${STATUS_ROOT}/${label}.status"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export DEVICE="cuda:0"
    export INIT_SEED="${init_seed}"
    export LOADER_SEED="${loader_seed}"
    if bash "${RUN_SCRIPT}" >"${log}" 2>&1; then
      printf 'complete\n' >"${status}"
    else
      printf 'failed\n' >"${status}"
      exit 1
    fi
  ) &
  JOB_PIDS["${label}"]=$!
  JOB_GPUS["${label}"]="${gpu}"
  printf '[%s] launched combo=%s gpu=%s pid=%s log=%s\n' \
    "$(timestamp)" "${label}" "${gpu}" "${JOB_PIDS[${label}]}" "${log}"
}

while [[ "${#pending[@]}" -gt 0 || "${#JOB_PIDS[@]}" -gt 0 ]]; do
  for label in "${!JOB_PIDS[@]}"; do
    pid="${JOB_PIDS[${label}]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then
        printf '[%s] completed combo=%s gpu=%s\n' "$(timestamp)" "${label}" "${JOB_GPUS[${label}]}"
      else
        printf '[%s] failed combo=%s gpu=%s\n' "$(timestamp)" "${label}" "${JOB_GPUS[${label}]}"
      fi
      unset 'JOB_PIDS['"${label}"']'
      unset 'JOB_GPUS['"${label}"']'
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
    combo="${pending[0]}"
    pending=("${pending[@]:1}")
    launch_combo "${combo}" "${selected_gpu}"
  done

  [[ "${#pending[@]}" -gt 0 || "${#JOB_PIDS[@]}" -gt 0 ]] && sleep "${POLL_SECONDS}"
done

printf '[%s] all requested seed-factor combos finished\n' "$(timestamp)"
