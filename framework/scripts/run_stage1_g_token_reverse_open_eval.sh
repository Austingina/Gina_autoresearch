#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-7}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
CALVIN_CODE="/mnt/nvme33n1/wuk_code/calvin_g/code/calvin"
DATA_ROOT="/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D"
TRAIN_DIR="${ROOT}/experiments/phase1_calvin/2026-07-16_E2_g_token_reverse_open_seed0"
RUN_META="${TRAIN_DIR}/run_meta.json"
RESUMABLE="${ROOT}/framework/scripts/run_resumable_state_policy_eval.sh"

BC_CKPT="${TRAIN_DIR}/chunk_bc_minimal_g_tokens/policy.pt"
FM_CKPT="${TRAIN_DIR}/chunk_fm_minimal_g_tokens/policy.pt"

cd "${ROOT}"
while [[ ! -f "${FM_CKPT}" ]]; do
  echo "[E2-eval] waiting for FM checkpoint" >&2
  sleep 30
done

run_close_eval() {
  local label="$1"
  local checkpoint="$2"
  local horizon="$3"
  local control="$4"
  local eval_dir="${ROOT}/experiments/phase1_calvin/2026-07-16_E2_g_token_${label}_close_stateonly_seed0"
  bash "${RESUMABLE}" "${GPU}" "${eval_dir}" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${checkpoint}" \
    --run-meta "${RUN_META}" \
    --num-sequences 100 \
    --ep-len 160 \
    --eval-scan-sequences 5000 \
    --eval-task-filter close_drawer \
    --action-execution-horizon "${horizon}" \
    --query-g-control "${control}" \
    --device cuda:0 \
    --state-only-observation
}

run_seen_eval() {
  local label="$1"
  local checkpoint="$2"
  local horizon="$3"
  local eval_dir="${ROOT}/experiments/phase1_calvin/2026-07-16_E2_g_token_${label}_seen14_n25_stateonly_seed0"
  bash "${RESUMABLE}" "${GPU}" "${eval_dir}" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${checkpoint}" \
    --run-meta "${RUN_META}" \
    --num-sequences 25 \
    --ep-len 160 \
    --eval-scan-sequences 10000 \
    --eval-task-filter \
      open_drawer move_slider_left move_slider_right \
      push_into_drawer \
      push_blue_block_left push_blue_block_right \
      push_pink_block_left push_pink_block_right \
      push_red_block_left push_red_block_right \
      turn_on_lightbulb turn_off_lightbulb turn_on_led turn_off_led \
    --action-execution-horizon "${horizon}" \
    --query-g-control normal \
    --device cuda:0 \
    --state-only-observation
}

run_close_eval "bc_normal" "${BC_CKPT}" 1 normal
run_close_eval "bc_sign_flip" "${BC_CKPT}" 1 sign_flip
run_close_eval "fm_normal_h16" "${FM_CKPT}" 16 normal
run_close_eval "fm_sign_flip_h16" "${FM_CKPT}" 16 sign_flip
run_seen_eval "bc" "${BC_CKPT}" 1
run_seen_eval "fm_h16" "${FM_CKPT}" 16
