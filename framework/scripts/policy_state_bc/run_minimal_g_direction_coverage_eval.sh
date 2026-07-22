#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}"
CONDITION="${CONDITION:?Set CONDITION to D0, D1, D2, D3, or D4}"
SEED="${SEED:-0}"
read -r -a CONTROL_LIST <<< "${QUERY_G_CONTROLS:-normal zero sign_flip task:open_drawer}"
RUN_SEEN_TASKS="${RUN_SEEN_TASKS:-1}"
MAX_RETRIES="${MAX_RETRIES:-50}"
SEEN_TASKS=(
  open_drawer move_slider_left move_slider_right
  push_into_drawer
  push_blue_block_left push_blue_block_right
  push_pink_block_left push_pink_block_right
  push_red_block_left push_red_block_right
)

case "${CONDITION}" in
  D0)
    MODE="chunk_bc_minimal_g_film"
    HORIZON=1
    STEM="2026-07-15_D0_sparse_direction_coverage_film_h1_seed"
    ;;
  D1)
    MODE="chunk_bc_minimal_g_film"
    HORIZON=1
    STEM="2026-07-15_D1_expanded_direction_coverage_film_h1_seed"
    ;;
  D2)
    MODE="chunk_bc_minimal_g_bounded_projected"
    HORIZON=1
    STEM="2026-07-15_D2_expanded_direction_coverage_field_coupled_h16_seed"
    ;;
  D3)
    MODE="chunk_bc_minimal_g_film"
    HORIZON=1
    STEM="2026-07-15_D3_signed_direction_coverage_film_h1_seed"
    SEEN_TASKS+=(place_in_slider)
    ;;
  D4)
    MODE="chunk_bc_minimal_g_film"
    HORIZON=1
    STEM="2026-07-15_D4_expanded_direction_coverage_film_h16_seed"
    ;;
  *)
    echo "Unknown CONDITION=${CONDITION}; expected D0, D1, D2, D3, or D4" >&2
    exit 2
    ;;
esac

OUT_DIR="${OUT_DIR:-${ROOT}/experiments/phase1_calvin/${STEM}${SEED}}"
CHECKPOINT="${OUT_DIR}/${MODE}/policy.pt"
RUN_META="${OUT_DIR}/run_meta.json"
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 3; }

cd "${ROOT}"
for control in "${CONTROL_LIST[@]}"; do
  label="${control//:/_}"
  eval_dir="${OUT_DIR}/${MODE}/eval_close_${label}_n100"
  summary="${eval_dir}/state_bc_summary.json"
  [[ -f "${summary}" ]] && continue
  attempt=1
  while (( attempt <= MAX_RETRIES )); do
    echo "[direction-eval] condition=${CONDITION} seed=${SEED} control=${control} attempt=${attempt}"
    if "${PYTHON}" framework/scripts/policy_state_bc/eval_state_chunk_policy_ckpt.py \
      --calvin-code "${CALVIN_CODE}" \
      --data-root "${DATA_ROOT}" \
      --checkpoint "${CHECKPOINT}" \
      --run-meta "${RUN_META}" \
      --eval-dir "${eval_dir}" \
      --num-sequences 100 \
      --ep-len 160 \
      --eval-scan-sequences 50000 \
      --eval-task-filter close_drawer \
      --action-execution-horizon "${HORIZON}" \
      --query-g-control "${control}" \
      --language-control normal \
      --device "${DEVICE:-cuda:0}"; then
      break
    fi
    [[ -f "${summary}" ]] && break
    attempt=$((attempt + 1))
    sleep 10
  done
  [[ -f "${summary}" ]] || { echo "Close evaluation failed after ${MAX_RETRIES} attempts" >&2; exit 4; }
done

seen_dir="${OUT_DIR}/${MODE}/eval_seen_translational_n25"
if [[ "${RUN_SEEN_TASKS}" == "1" && ! -f "${seen_dir}/state_bc_summary.json" ]]; then
  attempt=1
  while (( attempt <= MAX_RETRIES )); do
    if "${PYTHON}" framework/scripts/policy_state_bc/eval_state_chunk_policy_ckpt.py \
      --calvin-code "${CALVIN_CODE}" \
      --data-root "${DATA_ROOT}" \
      --checkpoint "${CHECKPOINT}" \
      --run-meta "${RUN_META}" \
      --eval-dir "${seen_dir}" \
      --num-sequences 25 \
      --ep-len 160 \
      --eval-scan-sequences 50000 \
      --eval-task-filter "${SEEN_TASKS[@]}" \
      --action-execution-horizon "${HORIZON}" \
      --query-g-control normal \
      --language-control normal \
      --device "${DEVICE:-cuda:0}"; then
      break
    fi
    [[ -f "${seen_dir}/state_bc_summary.json" ]] && break
    attempt=$((attempt + 1))
    sleep 10
  done
  [[ -f "${seen_dir}/state_bc_summary.json" ]] || {
    echo "Seen evaluation failed after ${MAX_RETRIES} attempts" >&2
    exit 5
  }
fi
