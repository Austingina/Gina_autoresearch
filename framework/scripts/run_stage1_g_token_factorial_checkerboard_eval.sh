#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
GPU="${GPU:-7}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
CALVIN_CODE="/mnt/nvme33n1/wuk_code/calvin_g/code/calvin"
DATA_ROOT="/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D"
RUN_ROOT="${ROOT}/experiments/phase1_calvin/2026-07-16_E3_factorial_checkerboard_seed${SEED}"
RESUMABLE="${ROOT}/framework/scripts/run_resumable_state_policy_eval.sh"

HELD_OUT_TASKS=(
  close_drawer
  move_slider_left
  push_red_block_right
  push_blue_block_left
)

SOURCE_TASKS=(
  open_drawer
  move_slider_right
  push_red_block_left
  push_blue_block_right
  push_pink_block_left
  push_pink_block_right
  push_into_drawer
  place_in_slider
  turn_on_lightbulb
  turn_off_lightbulb
  turn_on_led
  turn_off_led
)

cd "${ROOT}"

run_eval() {
  local label="$1"
  local train_variant="$2"
  local control="$3"
  local language_control="$4"
  local task="$5"
  local num_sequences="$6"
  local checkpoint="${RUN_ROOT}/${train_variant}/chunk_bc_minimal_g_tokens/policy.pt"
  local run_meta="${RUN_ROOT}/${train_variant}/run_meta.json"
  local eval_dir="${RUN_ROOT}/eval/${label}_${task}"

  bash "${RESUMABLE}" "${GPU}" "${eval_dir}" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${checkpoint}" \
    --run-meta "${run_meta}" \
    --num-sequences "${num_sequences}" \
    --ep-len 160 \
    --eval-scan-sequences 10000 \
    --eval-task-filter "${task}" \
    --action-execution-horizon 1 \
    --query-g-control "${control}" \
    --language-control "${language_control}" \
    --device cuda:0 \
    --state-only-observation
}

for task in "${HELD_OUT_TASKS[@]}"; do
  run_eval with_g_normal with_g normal zero "${task}" 100
done

for task in "${HELD_OUT_TASKS[@]}"; do
  run_eval matched_no_g no_g zero zero "${task}" 100
done

for task in "${HELD_OUT_TASKS[@]}"; do
  run_eval with_g_sign_flip with_g sign_flip zero "${task}" 100
done

run_source_health() {
  local label="$1"
  local train_variant="$2"
  local control="$3"
  local checkpoint="${RUN_ROOT}/${train_variant}/chunk_bc_minimal_g_tokens/policy.pt"
  local run_meta="${RUN_ROOT}/${train_variant}/run_meta.json"
  local eval_dir="${RUN_ROOT}/eval/${label}_source12"

  bash "${RESUMABLE}" "${GPU}" "${eval_dir}" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${checkpoint}" \
    --run-meta "${run_meta}" \
    --num-sequences 25 \
    --ep-len 160 \
    --eval-scan-sequences 10000 \
    --eval-task-filter "${SOURCE_TASKS[@]}" \
    --action-execution-horizon 1 \
    --query-g-control "${control}" \
    --language-control zero \
    --device cuda:0 \
    --state-only-observation
}

run_source_health with_g with_g normal
run_source_health matched_no_g no_g zero
