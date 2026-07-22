#!/usr/bin/env bash
# Monitor S1 B1 training; restart with lower collect_steps if OOM-killed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${1:-${ROOT}/runs/s1_B1_visible_ee_forward_20260630_204236}"
LOG="${RUN_DIR}/launch.log"
POLL_SEC="${POLL_SEC:-30}"

log() { echo "[monitor $(date '+%H:%M:%S')] $*"; }

get_train_pid() {
  pgrep -f "train_lift_cube_dagger_vision_binary_gripper.py.*${RUN_DIR##*/}" 2>/dev/null | head -1 || true
}

wait_for_start() {
  local pid="$1"
  local i
  for i in $(seq 1 120); do
    [[ -n "$(get_train_pid)" ]] && return 0
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    sleep 2
  done
  return 0
}

restart_fallback() {
  log "OOM/kill detected — restarting with NUM_ENVS=8 COLLECT_STEPS=256"
  cd "${ROOT}"
  NUM_ENVS=8 COLLECT_STEPS=256 \
    AGGREGATE_BACKEND=disk AGGREGATE_CACHE_SIZE=4 \
    bash scripts/run_s1_b1_fresh.sh >> "${ROOT}/runs/b1_oom_restart_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
  local wrapper_pid=$!
  sleep 5
  local new_run
  new_run="$(ls -td "${ROOT}"/runs/s1_B1_visible_ee_forward_* 2>/dev/null | head -1)"
  log "Restart launched (wrapper pid=${wrapper_pid}); new run dir likely: ${new_run}"
  echo "${new_run}" > "${ROOT}/runs/.b1_monitor_active_run"
}

log "Watching run: ${RUN_DIR}"
log "Log file: ${LOG}"

while true; do
  pid="$(get_train_pid)"
  if [[ -n "${pid}" ]]; then
    mem_avail="$(free -g | awk 'NR==2{print $7}')"
    swap_used="$(free -g | awk 'NR==3{print $3}')"
    iter="$(grep -oP '\[DAgger \K[0-9]+' "${LOG}" 2>/dev/null | tail -1 || echo '?')"
    log "alive pid=${pid} iter≈${iter} avail_mem=${mem_avail}Gi swap_used=${swap_used}Gi"
  else
  if [[ -f "${LOG}" ]] && tail -20 "${LOG}" | grep -qE 'Killed|killed|OOM|out of memory'; then
      log "Process gone + kill/OOM signature in log tail"
      restart_fallback
      exit 0
    fi
    # Wrapper may still be up briefly; check parent bash too
    if pgrep -f "run_s1_b1_fresh.sh" >/dev/null 2>&1; then
      log "train python gone but launcher still running — waiting"
    else
      if [[ -f "${LOG}" ]] && grep -qE 'Killed|killed' "${LOG}"; then
        log "Training ended with Killed in log"
        restart_fallback
        exit 0
      fi
      log "Training process ended (no OOM signature) — monitor exiting"
      exit 0
    fi
  fi
  sleep "${POLL_SEC}"
done
