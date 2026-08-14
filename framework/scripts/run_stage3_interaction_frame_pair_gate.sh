#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${REPO_ROOT}/.env.server" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.server"
fi

CALVIN_G_ROOT="${CALVIN_G_ROOT:?Set CALVIN_G_ROOT or provide .env.server}"
CALVIN_CODE="${CALVIN_CODE:-${CALVIN_G_ROOT}/code/calvin}"
CALVIN_PYTHON="${CALVIN_PYTHON:-${CALVIN_G_ROOT}/envs/calvin/bin/python}"
ABC_SUBSET_ROOT="${CALVIN_VOLUME_G_SUBSET_ROOT:-${CALVIN_G_ROOT}/data/task_ABC_D_volume_g_scale_subset}"
PACKED_STATE_ARRAYS_NAME="${FRAME_PAIR_PACKED_STATE_ARRAYS_NAME:-state_only_arrays.npz}"
SIM_DATA_ROOT="${CALVIN_SIM_DATA_ROOT:-${CALVIN_G_ROOT}/data/calvin_debug_dataset}"
CALIBRATION_REPORT="${CALVIN_VOLUME_G_CALIBRATION_REPORT:-${ABC_SUBSET_ROOT}/surface_g_calibration.json}"
SEED="${1:-0}"
GPU_ID="${2:-${GPU_ID:-0}}"
STATE_SCOPE="${3:-proprio}"
TARGET_PAIR_WEIGHT="${FRAME_PAIR_TARGET_LOSS_WEIGHT:-0.2}"
INVERSE_PAIR_WEIGHT="${FRAME_PAIR_INVERSE_LOSS_WEIGHT:-0}"
INVERSE_PAIR_SCOPE="${FRAME_PAIR_INVERSE_ACTION_SCOPE:-first_local_xyz}"
FRAME_CONDITION_SCALE="${FRAME_PAIR_CONDITION_SCALE:-0.1}"
EPOCHS="${FRAME_PAIR_EPOCHS:-50}"
DATE_LABEL="${FRAME_PAIR_DATE_LABEL:-2026-08-14}"
case "${STATE_SCOPE}" in
  proprio) STATE_TAG=proprio ;;
  eef_orientation_gripper) STATE_TAG=relative ;;
  *)
    echo "State scope must be proprio or eef_orientation_gripper" >&2
    exit 2
    ;;
esac
RUN_LABEL="${FRAME_PAIR_RUN_LABEL:-M4_frame_pair_${STATE_TAG}}"
RUN_ROOT="${REPO_ROOT}/experiments/phase1_calvin/${DATE_LABEL}_${RUN_LABEL}_seed${SEED}"
LOG_ROOT="${CALVIN_G_ROOT}/logs/${RUN_LABEL}_seed${SEED}"
GEOMETRY_JSON="${REPO_ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json"
CHECKPOINT="${RUN_ROOT}/chunk_bc_minimal_g_frame/checkpoints/epoch_$(printf '%04d' "${EPOCHS}")/policy.pt"
SOURCE_TASKS=(
  push_red_block_left
  push_blue_block_right
  push_pink_block_left
  push_pink_block_right
)
HELDOUT_TASKS=(push_blue_block_left push_red_block_right)
read -r -a HELDOUT_CONTROLS <<< "${FRAME_PAIR_HELDOUT_CONTROLS:-normal}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

TOOL_CLEARANCE="$(${CALVIN_PYTHON} - "${CALIBRATION_REPORT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print(float(report["recommended_interaction_volume_clearance"]))
PY
)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${ABC_SUBSET_ROOT}" \
    --packed-state-arrays-name "${PACKED_STATE_ARRAYS_NAME}" \
    --out-dir "${RUN_ROOT}" \
    --modes chunk_bc_minimal_g_frame \
    --include-tasks "${SOURCE_TASKS[@]}" \
    --stride 2 \
    --chunk-size 1 \
    --epochs "${EPOCHS}" \
    --checkpoint-epochs "${EPOCHS}" \
    --batch-size 128 \
    --hidden 256 \
    --layers 3 \
    --lr 3e-4 \
    --seed "${SEED}" \
    --init-seed "${SEED}" \
    --loader-seed "${SEED}" \
    --data-seed "${SEED}" \
    --split-seed "${SEED}" \
    --device "cuda:${GPU_ID}" \
    --skip-eval 1 \
    --articulation-geometry-json "${GEOMETRY_JSON}" \
    --minimal-g-scope translational \
    --executor-state-scope "${STATE_SCOPE}" \
    --executor-language-mode none \
    --mask-executor-language-with-g 0 \
    --g-scale-aug-values 1.0 \
    --g-dropout-prob 0.0 \
    --frame-condition-scale "${FRAME_CONDITION_SCALE}" \
    --first-action-loss-weight 1 \
    --inverse-pair-loss-weight "${INVERSE_PAIR_WEIGHT}" \
    --inverse-pair-action-scope "${INVERSE_PAIR_SCOPE}" \
    --target-pair-loss-weight "${TARGET_PAIR_WEIGHT}" \
    --target-pair-action-scope first_xyz \
    2>&1 | tee "${LOG_ROOT}/train.log"
fi

evaluate() {
  local split="$1"
  local control="$2"
  local per_task="$3"
  shift 3
  local -a tasks=("$@")
  local eval_dir="${RUN_ROOT}/chunk_bc_minimal_g_frame/eval_D_${split}_${control}_epoch$(printf '%04d' "${EPOCHS}")"
  local sequence_cache="${CALVIN_G_ROOT}/cache/calvin_standard_sequences/D_${split}_${per_task}_offset0.pkl"
  if [[ -f "${eval_dir}/state_bc_summary.json" ]]; then
    return
  fi
  mkdir -p "${eval_dir}"
  printf '%s\n' "${CHECKPOINT}" > "${eval_dir}/checkpoint_path.txt"
  env -u EGL_VISIBLE_DEVICES xvfb-run -a "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/policy_state_bc/eval_state_chunk_policy_ckpt.py" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${SIM_DATA_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --run-meta "${RUN_ROOT}/run_meta.json" \
    --eval-dir "${eval_dir}" \
    --num-sequences "${per_task}" \
    --ep-len 160 \
    --eval-scan-sequences 10000 \
    --eval-sequence-workers 8 \
    --eval-sequence-cache "${sequence_cache}" \
    --eval-task-filter "${tasks[@]}" \
    --action-execution-horizon 1 \
    --query-g-control "${control}" \
    --online-minimal-g-mode behavior_block_volume \
    --behavior-tool-clearance "${TOOL_CLEARANCE}" \
    --behavior-contact-tolerance 0 \
    --behavior-motion-threshold 0.002 \
    --language-control normal \
    --state-only-observation \
    --trace-rollouts 5 \
    --device "cuda:${GPU_ID}" \
    2>&1 | tee "${LOG_ROOT}/eval_${split}_${control}_epoch$(printf '%04d' "${EPOCHS}").log"
}

evaluate source normal 25 "${SOURCE_TASKS[@]}"
for control in "${HELDOUT_CONTROLS[@]}"; do
  evaluate heldout "${control}" 50 "${HELDOUT_TASKS[@]}"
done

date -Is > "${RUN_ROOT}/.interaction_frame_pair_complete"
