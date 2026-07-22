#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-7}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
CALVIN_CODE="/mnt/nvme33n1/wuk_code/calvin_g/code/calvin"
DATA_ROOT="/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D"
TRAIN_DIR="${ROOT}/experiments/phase1_calvin/2026-07-16_E2_g_token_reverse_open_seed0"
CKPT="${TRAIN_DIR}/chunk_fm_minimal_g_tokens/policy.pt"
RUN_META="${TRAIN_DIR}/run_meta.json"
RESUMABLE="${ROOT}/framework/scripts/run_resumable_state_policy_eval.sh"

cd "${ROOT}"

run_eval() {
  local eval_dir="$1"
  local count="$2"
  local sample_seed="$3"
  local reseed="$4"
  bash "${RESUMABLE}" "${GPU}" "${eval_dir}" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${CKPT}" \
    --run-meta "${RUN_META}" \
    --num-sequences "${count}" \
    --ep-len 160 \
    --eval-scan-sequences 5000 \
    --eval-task-filter close_drawer \
    --action-execution-horizon 1 \
    --query-g-control normal \
    --policy-sample-seed "${sample_seed}" \
    --fm-reseed-each-episode "${reseed}" \
    --device cuda:0 \
    --state-only-observation
}

for seed in 18 19 20 21; do
  run_eval \
    "${ROOT}/experiments/phase1_calvin/2026-07-16_E2_g_token_fm_close_h1_fixed_noise${seed}_n25" \
    25 "${seed}" 1
done

run_eval \
  "${ROOT}/experiments/phase1_calvin/2026-07-16_E2_g_token_fm_close_h1_noise_stream_seed17_n100" \
  100 17 0
