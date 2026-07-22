#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 GPU EVAL_DIR [evaluator args...]" >&2
  exit 2
fi

GPU="$1"
EVAL_DIR="$2"
shift 2

ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
PYTHON="/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python"
EVALUATOR="framework/scripts/policy_state_bc/eval_state_chunk_policy_ckpt.py"
ATTEMPT_TIMEOUT_SECONDS="${ATTEMPT_TIMEOUT_SECONDS:-3600}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-30}"

cd "${ROOT}"
mkdir -p "${EVAL_DIR}"

attempt=0
while [[ ! -f "${EVAL_DIR}/state_bc_summary.json" ]]; do
  attempt=$((attempt + 1))
  echo "[resumable-eval] attempt=${attempt} eval_dir=${EVAL_DIR}" >&2
  CUDA_VISIBLE_DEVICES="${GPU}" OMP_NUM_THREADS=1 timeout "${ATTEMPT_TIMEOUT_SECONDS}" \
    "${PYTHON}" "${EVALUATOR}" --eval-dir "${EVAL_DIR}" "$@"
  status=$?
  if [[ -f "${EVAL_DIR}/state_bc_summary.json" ]]; then
    break
  fi
  echo "[resumable-eval] status=${status}; retrying in ${RETRY_DELAY_SECONDS}s" >&2
  sleep "${RETRY_DELAY_SECONDS}"
done

echo "[resumable-eval] complete: ${EVAL_DIR}/state_bc_summary.json" >&2
