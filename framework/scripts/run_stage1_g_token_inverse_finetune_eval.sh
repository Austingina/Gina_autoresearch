#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:?usage: $0 LABEL GPU [ACTION_HORIZON] [G_SCALE] [TEMPORAL_ENSEMBLE] [ENSEMBLE_DECAY] [NUM_SEQUENCES] [QUERY_G_CONTROL]}"
GPU="${2:?usage: $0 LABEL GPU [ACTION_HORIZON] [G_SCALE] [TEMPORAL_ENSEMBLE] [ENSEMBLE_DECAY] [NUM_SEQUENCES] [QUERY_G_CONTROL]}"
ACTION_HORIZON="${3:-1}"
G_SCALE="${4:-1.0}"
TEMPORAL_ENSEMBLE="${5:-0}"
ENSEMBLE_DECAY="${6:-0.25}"
NUM_SEQUENCES="${7:-100}"
QUERY_G_CONTROL="${8:-normal}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
CALVIN_CODE="/mnt/nvme33n1/wuk_code/calvin_g/code/calvin"
DATA_ROOT="/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D"
RUN_ROOT="${ROOT}/experiments/phase1_calvin/2026-07-17_E4_inverse_finetune_${LABEL}_seed0"
RESUMABLE="${ROOT}/framework/scripts/run_resumable_state_policy_eval.sh"

HELD_OUT_TASKS=(
  close_drawer
  move_slider_left
  push_red_block_right
  push_blue_block_left
)

cd "${ROOT}"

EVAL_SUFFIX=""
if [[ "${ACTION_HORIZON}" != "1" ]]; then
  EVAL_SUFFIX="_h${ACTION_HORIZON}"
fi
if [[ "${G_SCALE}" != "1.0" && "${G_SCALE}" != "1" ]]; then
  EVAL_SUFFIX="${EVAL_SUFFIX}_gs${G_SCALE}"
fi
if [[ "${TEMPORAL_ENSEMBLE}" == "1" ]]; then
  EVAL_SUFFIX="${EVAL_SUFFIX}_te${ENSEMBLE_DECAY}"
fi
if [[ "${NUM_SEQUENCES}" != "100" ]]; then
  EVAL_SUFFIX="${EVAL_SUFFIX}_n${NUM_SEQUENCES}"
fi

for task in "${HELD_OUT_TASKS[@]}"; do
  bash "${RESUMABLE}" "${GPU}" "${RUN_ROOT}/eval/with_g_${QUERY_G_CONTROL}_${task}${EVAL_SUFFIX}" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${RUN_ROOT}/with_g/chunk_bc_minimal_g_tokens/policy.pt" \
    --run-meta "${RUN_ROOT}/with_g/run_meta.json" \
    --num-sequences "${NUM_SEQUENCES}" \
    --ep-len 160 \
    --eval-scan-sequences 10000 \
    --eval-task-filter "${task}" \
    --action-execution-horizon "${ACTION_HORIZON}" \
    --g-scale "${G_SCALE}" \
    --temporal-ensemble "${TEMPORAL_ENSEMBLE}" \
    --temporal-ensemble-decay "${ENSEMBLE_DECAY}" \
    --query-g-control "${QUERY_G_CONTROL}" \
    --language-control zero \
    --device cuda:0 \
    --state-only-observation
done
