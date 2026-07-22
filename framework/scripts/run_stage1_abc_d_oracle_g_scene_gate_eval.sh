#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: $0 SEED GPU STATE_SCOPE VARIANT QUERY_G_CONTROL [LANGUAGE_CONTROL] [LANGUAGE_MODE] [EVAL_TAG]}"
GPU="${2:?usage: $0 SEED GPU STATE_SCOPE VARIANT QUERY_G_CONTROL [LANGUAGE_CONTROL] [LANGUAGE_MODE] [EVAL_TAG]}"
STATE_SCOPE="${3:?usage: $0 SEED GPU STATE_SCOPE VARIANT QUERY_G_CONTROL [LANGUAGE_CONTROL] [LANGUAGE_MODE] [EVAL_TAG]}"
VARIANT="${4:?usage: $0 SEED GPU STATE_SCOPE VARIANT QUERY_G_CONTROL [LANGUAGE_CONTROL] [LANGUAGE_MODE] [EVAL_TAG]}"
QUERY_G_CONTROL="${5:?usage: $0 SEED GPU STATE_SCOPE VARIANT QUERY_G_CONTROL [LANGUAGE_CONTROL] [LANGUAGE_MODE] [EVAL_TAG]}"
LANGUAGE_CONTROL="${6:-normal}"
LANGUAGE_MODE="${7:-onehot}"
EVAL_TAG="${8:-}"
EGL_RENDER_DEVICE="${EGL_RENDER_DEVICE:-${GPU}}"

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
CALVIN_G_ROOT="${CALVIN_G_ROOT:-/mnt/nvme33n1/wuk_code/calvin_g}"
PYTHON="${CALVIN_PYTHON:-${CALVIN_G_ROOT}/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${CALVIN_G_ROOT}/code/calvin}"
CALVIN_DATA="${CALVIN_DATA:-${CALVIN_G_ROOT}/data}"
EVAL_DATA_ROOT="${CALVIN_DATA}/task_D_D"
LANGUAGE_TAG=""
if [[ "${LANGUAGE_MODE}" != "onehot" ]]; then
  LANGUAGE_TAG="_${LANGUAGE_MODE}"
fi
RUN_ROOT="${ROOT}/experiments/phase1_calvin/2026-07-19_M1_abc_to_d_oracle_g${LANGUAGE_TAG}_${STATE_SCOPE}_seed${SEED}/${VARIANT}"
CONTROL_LABEL="${QUERY_G_CONTROL//:/_}"
EVAL_SUFFIX=""
if [[ -n "${EVAL_TAG}" ]]; then
  EVAL_SUFFIX="_${EVAL_TAG}"
fi
EVAL_DIR="${RUN_ROOT}/chunk_bc_minimal_g_tokens/task_eval_D_${CONTROL_LABEL}_lang_${LANGUAGE_CONTROL}${EVAL_SUFFIX}"
OBSERVATION_ARGS=()
if [[ "${STATE_SCOPE}" != "visual" ]]; then
  OBSERVATION_ARGS+=(--state-only-observation)
fi

cd "${ROOT}"
unset CUDA_VISIBLE_DEVICES
export EGL_VISIBLE_DEVICES="${EGL_RENDER_DEVICE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

"${PYTHON}" framework/scripts/policy_state_bc/eval_state_chunk_policy_ckpt.py \
  --calvin-code "${CALVIN_CODE}" \
  --data-root "${EVAL_DATA_ROOT}" \
  --checkpoint "${RUN_ROOT}/chunk_bc_minimal_g_tokens/policy.pt" \
  --run-meta "${RUN_ROOT}/run_meta.json" \
  --eval-dir "${EVAL_DIR}" \
  --num-sequences 100 \
  --ep-len 160 \
  --eval-scan-sequences 10000 \
  --eval-task-filter push_red_block_right push_blue_block_left \
  --action-execution-horizon 1 \
  --query-g-control "${QUERY_G_CONTROL}" \
  --language-control "${LANGUAGE_CONTROL}" \
  --device "cuda:${GPU}" \
  "${OBSERVATION_ARGS[@]}"
