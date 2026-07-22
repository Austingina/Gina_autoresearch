#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
SEED_ORDER="${SEEDS:-0 1 2 3 4}"

if [[ -n "${WAIT_FOR_PID}" ]]; then
  echo "[master] waiting for PID ${WAIT_FOR_PID}"
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    sleep 30
  done
fi

cd "${ROOT}"
for horizon in h16 h1; do
  echo "[master] starting ${horizon} seeds=${SEED_ORDER}"
  HORIZON_MODE="${horizon}" \
  SEEDS="${SEED_ORDER}" \
  DEVICE="${DEVICE:-cuda:0}" \
  EGL_DEVICE="${EGL_DEVICE:?Set EGL_DEVICE to the physical rendering GPU}" \
  MAX_RETRIES="${MAX_RETRIES:-50}" \
  EVAL_SCAN_SEQUENCES="${EVAL_SCAN_SEQUENCES:-5000}" \
    bash framework/scripts/policy_state_bc/run_learned_g_executor_eval_queue.sh
done
echo "[master] all learned-G executor evaluations complete"
