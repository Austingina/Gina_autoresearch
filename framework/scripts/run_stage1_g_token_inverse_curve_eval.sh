#!/usr/bin/env bash
set -euo pipefail

K="${1:?usage: $0 REVERSE_SEGMENTS GPU}"
GPU="${2:?usage: $0 REVERSE_SEGMENTS GPU}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
CALVIN_CODE="/mnt/nvme33n1/wuk_code/calvin_g/code/calvin"
DATA_ROOT="/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D"
RUN_ROOT="${ROOT}/experiments/phase1_calvin/2026-07-17_E4_inverse_curve_k${K}_seed0"
RESUMABLE="${ROOT}/framework/scripts/run_resumable_state_policy_eval.sh"

HELD_OUT_TASKS=(
  close_drawer
  move_slider_left
  push_red_block_right
  push_blue_block_left
)

cd "${ROOT}"

for task in "${HELD_OUT_TASKS[@]}"; do
  bash "${RESUMABLE}" "${GPU}" "${RUN_ROOT}/eval/with_g_normal_${task}" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${RUN_ROOT}/with_g/chunk_bc_minimal_g_tokens/policy.pt" \
    --run-meta "${RUN_ROOT}/with_g/run_meta.json" \
    --num-sequences 100 \
    --ep-len 160 \
    --eval-scan-sequences 10000 \
    --eval-task-filter "${task}" \
    --action-execution-horizon 1 \
    --query-g-control normal \
    --language-control zero \
    --device cuda:0 \
    --state-only-observation
done

