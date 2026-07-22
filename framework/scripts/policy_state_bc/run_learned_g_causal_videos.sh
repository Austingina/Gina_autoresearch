#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}"
EGL_DEVICE="${EGL_DEVICE:?Set EGL_DEVICE to the physical rendering GPU index}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
export EGL_VISIBLE_DEVICES="${EGL_DEVICE}"

if [[ -n "${WAIT_FOR_PID}" ]]; then
  echo "[videos] waiting for PID ${WAIT_FOR_PID}"
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    sleep 30
  done
fi

CHECKPOINT="${ROOT}/experiments/phase1_calvin/2026-07-14_learned_g_executor_chunk1_exec1_e50_n100_h160_seed3/chunk_bc_minimal_g_film/policy.pt"
OUT_ROOT="${ROOT}/experiments/phase1_calvin/2026-07-14_learned_g_executor_h1_film_causal_videos_seed3"
read -r -a ROLLOUT_INDICES <<< "${ROLLOUT_INDICES:-1 6 10 13 18 19}"

cd "${ROOT}"
for control in normal sign_flip; do
  "${PYTHON}" framework/scripts/policy_state_bc/render_chunk_flow_rollouts.py \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${DATA_ROOT}" \
    --out-dir "${OUT_ROOT}/${control}" \
    --policies "seed3_${control}=${CHECKPOINT}" \
    --tasks close_drawer \
    --rollout-indices "${ROLLOUT_INDICES[@]}" \
    --candidate-rollouts-per-task 100 \
    --ep-len 160 \
    --scan-sequences 5000 \
    --render-stride 2 \
    --post-success-steps 12 \
    --query-g-control "${control}" \
    --action-execution-horizon 1 \
    --device "${DEVICE:-cuda:0}"
done
