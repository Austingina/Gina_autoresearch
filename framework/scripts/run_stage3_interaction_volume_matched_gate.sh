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
ABC_SUBSET_ROOT="${CALVIN_VOLUME_G_SUBSET_ROOT:-${CALVIN_G_ROOT}/data/task_ABC_D_volume_g_subset}"
SIM_DATA_ROOT="${CALVIN_SIM_DATA_ROOT:-${CALVIN_G_ROOT}/data/calvin_debug_dataset}"
CALIBRATION_REPORT="${CALVIN_VOLUME_G_CALIBRATION_REPORT:-${ABC_SUBSET_ROOT}/surface_g_calibration.json}"
SEED="${1:-0}"
GPU_ID="${2:-${GPU_ID:-0}}"
RUN_LABEL="${VOLUME_GATE_RUN_LABEL:-M3_interaction_volume_matched}"
EVAL_FAMILY="${VOLUME_GATE_EVAL_FAMILY:-all}"
CHUNK_SIZE="${VOLUME_GATE_CHUNK_SIZE:-16}"
FIRST_ACTION_LOSS_WEIGHT="${VOLUME_GATE_FIRST_ACTION_LOSS_WEIGHT:-1}"
ACTION_EXECUTION_HORIZON="${VOLUME_GATE_ACTION_EXECUTION_HORIZON:-1}"
INCLUDE_FACTORIZED="${VOLUME_GATE_INCLUDE_FACTORIZED:-1}"
INCLUDE_FRAME="${VOLUME_GATE_INCLUDE_FRAME:-0}"
EXECUTOR_STATE_SCOPE="${VOLUME_GATE_EXECUTOR_STATE_SCOPE:-proprio}"
TARGET_PAIR_LOSS_WEIGHT="${VOLUME_GATE_TARGET_PAIR_LOSS_WEIGHT:-0}"
TARGET_PAIR_ACTION_SCOPE="${VOLUME_GATE_TARGET_PAIR_ACTION_SCOPE:-all}"
RUN_ROOT="${REPO_ROOT}/experiments/phase1_calvin/2026-08-14_${RUN_LABEL}_seed${SEED}"
LOG_ROOT="${CALVIN_G_ROOT}/logs/${RUN_LABEL}_seed${SEED}"
GEOMETRY_JSON="${REPO_ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json"
TOOL_CLEARANCE="${VOLUME_GATE_TOOL_CLEARANCE:-}"
MOTION_THRESHOLD="${VOLUME_GATE_MOTION_THRESHOLD:-0.002}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

case "${EVAL_FAMILY}" in
  all|core|token|factorized|frame|none) ;;
  *)
    echo "VOLUME_GATE_EVAL_FAMILY must be all, core, token, factorized, frame, or none" >&2
    exit 2
    ;;
esac
if (( CHUNK_SIZE < 1 )); then
  echo "VOLUME_GATE_CHUNK_SIZE must be positive" >&2
  exit 2
fi
if (( ACTION_EXECUTION_HORIZON < 1 || ACTION_EXECUTION_HORIZON > CHUNK_SIZE )); then
  echo "VOLUME_GATE_ACTION_EXECUTION_HORIZON must be in [1, chunk size]" >&2
  exit 2
fi
case "${INCLUDE_FACTORIZED}" in
  0|1) ;;
  *)
    echo "VOLUME_GATE_INCLUDE_FACTORIZED must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${INCLUDE_FRAME}" in
  0|1) ;;
  *)
    echo "VOLUME_GATE_INCLUDE_FRAME must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ "${EVAL_FAMILY}" == "frame" && "${INCLUDE_FRAME}" != "1" ]]; then
  echo "VOLUME_GATE_EVAL_FAMILY=frame requires VOLUME_GATE_INCLUDE_FRAME=1" >&2
  exit 2
fi

SOURCE_TASKS=(
  push_red_block_left
  push_blue_block_right
  push_pink_block_left
  push_pink_block_right
)
HELDOUT_TASKS=(push_blue_block_left push_red_block_right)
HELDOUT_CONTROLS=(normal zero_attraction zero_flow zero sign_flip)

if [[ ! -f "${CALIBRATION_REPORT}" ]]; then
  "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/data/calibrate_calvin_block_surface_g.py" \
    --data-root "${ABC_SUBSET_ROOT}" \
    --calvin-code "${CALVIN_CODE}" \
    --output "${CALIBRATION_REPORT}" \
    --motion-threshold "${MOTION_THRESHOLD}"
fi

if [[ -z "${TOOL_CLEARANCE}" ]]; then
  TOOL_CLEARANCE="$("${CALVIN_PYTHON}" - "${CALIBRATION_REPORT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print(float(report["recommended_interaction_volume_clearance"]))
PY
)"
fi

if [[ ! -f "${ABC_SUBSET_ROOT}/training/state_only_arrays.npz" ]]; then
  "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/data/build_calvin_interaction_volume_cache.py" \
    --data-root "${ABC_SUBSET_ROOT}" \
    --calvin-code "${CALVIN_CODE}" \
    --calibration-report "${CALIBRATION_REPORT}" \
    --motion-threshold "${MOTION_THRESHOLD}"
fi

"${CALVIN_PYTHON}" - "${ABC_SUBSET_ROOT}" "${CALIBRATION_REPORT}" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1])
report = json.load(open(sys.argv[2], encoding="utf-8"))
manifest = json.load(open(root / "training/interaction_volume_cache_manifest.json", encoding="utf-8"))
selection = json.load(open(root / "calibration_subset_manifest.json", encoding="utf-8"))
expected = {
    "push_red_block_left",
    "push_blue_block_right",
    "push_pink_block_left",
    "push_pink_block_right",
}
assert set(manifest["task_counts"]) == expected, manifest["task_counts"]
expected_task_counts = {
    task: int(selection["task_counts"][task]) for task in sorted(expected)
}
assert manifest["task_counts"] == expected_task_counts, (
    manifest["task_counts"],
    expected_task_counts,
)
source_per_scene = int(selection["source_per_task_scene"])
expected_scene_count = len(expected) * source_per_scene
assert manifest["scene_counts"] == {
    "calvin_scene_A": expected_scene_count,
    "calvin_scene_B": expected_scene_count,
    "calvin_scene_C": expected_scene_count,
}, manifest["scene_counts"]
assert manifest["frames"] > 0, manifest["frames"]
assert manifest["heldout_tasks_written"] is False
assert report["protocol"]["heldout_used_for_fitting"] is False
assert report["interface_audit"]["interaction_volume_contact_coverage_gate_pass"] is True
with np.load(root / "training/state_only_arrays.npz", allow_pickle=False) as cache:
    assert cache["minimal_g"].shape == (int(manifest["frames"]), 6)
    assert np.isfinite(cache["minimal_g"]).all()
print("matched_volume_data_gate_ok", manifest["task_counts"], manifest["scene_counts"])
PY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

TRAIN_MODES=(chunk_bc_state chunk_bc_minimal_g_tokens)
if [[ "${INCLUDE_FACTORIZED}" == "1" ]]; then
  TRAIN_MODES+=(chunk_bc_minimal_g_factorized_tokens)
fi
if [[ "${INCLUDE_FRAME}" == "1" ]]; then
  TRAIN_MODES+=(chunk_bc_minimal_g_frame)
fi
MISSING_TRAIN_MODES=()
for mode in "${TRAIN_MODES[@]}"; do
  if [[ ! -f "${RUN_ROOT}/${mode}/policy_best_val.pt" ]]; then
    MISSING_TRAIN_MODES+=("${mode}")
  fi
done
if (( ${#MISSING_TRAIN_MODES[@]} > 0 )); then
  "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${ABC_SUBSET_ROOT}" \
    --out-dir "${RUN_ROOT}" \
    --modes "${MISSING_TRAIN_MODES[@]}" \
    --include-tasks "${SOURCE_TASKS[@]}" \
    --stride 2 \
    --chunk-size "${CHUNK_SIZE}" \
    --epochs 100 \
    --checkpoint-epochs 50 100 \
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
    --executor-state-scope "${EXECUTOR_STATE_SCOPE}" \
    --executor-language-mode none \
    --mask-executor-language-with-g 0 \
    --g-scale-aug-values 1.0 \
    --g-dropout-prob 0.0 \
    --first-action-loss-weight "${FIRST_ACTION_LOSS_WEIGHT}" \
    --target-pair-loss-weight "${TARGET_PAIR_LOSS_WEIGHT}" \
    --target-pair-action-scope "${TARGET_PAIR_ACTION_SCOPE}" \
    2>&1 | tee "${LOG_ROOT}/train.log"
fi

if [[ ! -f "${RUN_ROOT}/interaction_volume_offline_audit.json" ]]; then
  "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/data/audit_interaction_volume_policy_offline.py" \
    --data-root "${ABC_SUBSET_ROOT}" \
    --calvin-code "${CALVIN_CODE}" \
    --calibration-report "${CALIBRATION_REPORT}" \
    --no-g-checkpoint "${RUN_ROOT}/chunk_bc_state/policy_best_val.pt" \
    --g-checkpoint "${RUN_ROOT}/chunk_bc_minimal_g_tokens/policy_best_val.pt" \
    --output "${RUN_ROOT}/interaction_volume_offline_audit.json" \
    --motion-threshold "${MOTION_THRESHOLD}" \
    --stride 2 \
    --device "cuda:${GPU_ID}" \
    2>&1 | tee "${LOG_ROOT}/offline_audit.log"
fi

if [[ "${INCLUDE_FACTORIZED}" == "1" && \
      ! -f "${RUN_ROOT}/interaction_volume_factorized_offline_audit.json" ]]; then
  "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/data/audit_interaction_volume_policy_offline.py" \
    --data-root "${ABC_SUBSET_ROOT}" \
    --calvin-code "${CALVIN_CODE}" \
    --calibration-report "${CALIBRATION_REPORT}" \
    --no-g-checkpoint "${RUN_ROOT}/chunk_bc_state/policy_best_val.pt" \
    --g-checkpoint \
      "${RUN_ROOT}/chunk_bc_minimal_g_factorized_tokens/policy_best_val.pt" \
    --output "${RUN_ROOT}/interaction_volume_factorized_offline_audit.json" \
    --motion-threshold "${MOTION_THRESHOLD}" \
    --stride 2 \
    --device "cuda:${GPU_ID}" \
    2>&1 | tee "${LOG_ROOT}/offline_factorized_audit.log"
fi

if [[ "${INCLUDE_FRAME}" == "1" && \
      ! -f "${RUN_ROOT}/interaction_volume_frame_offline_audit.json" ]]; then
  "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/data/audit_interaction_volume_policy_offline.py" \
    --data-root "${ABC_SUBSET_ROOT}" \
    --calvin-code "${CALVIN_CODE}" \
    --calibration-report "${CALIBRATION_REPORT}" \
    --no-g-checkpoint "${RUN_ROOT}/chunk_bc_state/policy_best_val.pt" \
    --g-checkpoint "${RUN_ROOT}/chunk_bc_minimal_g_frame/policy_best_val.pt" \
    --output "${RUN_ROOT}/interaction_volume_frame_offline_audit.json" \
    --motion-threshold "${MOTION_THRESHOLD}" \
    --stride 2 \
    --device "cuda:${GPU_ID}" \
    2>&1 | tee "${LOG_ROOT}/offline_frame_audit.log"
fi

evaluate() {
  local mode="$1"
  local split="$2"
  local control="$3"
  local per_task="$4"
  shift 4
  local -a tasks=("$@")
  local checkpoint="${RUN_ROOT}/${mode}/policy_best_val.pt"
  local eval_dir="${RUN_ROOT}/${mode}/eval_D_${split}_${control}"
  if [[ -f "${eval_dir}/state_bc_summary.json" ]]; then
    return
  fi
  mkdir -p "${eval_dir}"
  env -u EGL_VISIBLE_DEVICES xvfb-run -a "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/policy_state_bc/eval_state_chunk_policy_ckpt.py" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${SIM_DATA_ROOT}" \
    --checkpoint "${checkpoint}" \
    --run-meta "${RUN_ROOT}/run_meta.json" \
    --eval-dir "${eval_dir}" \
    --num-sequences "${per_task}" \
    --ep-len 160 \
    --eval-scan-sequences 10000 \
    --eval-sequence-workers 8 \
    --eval-task-filter "${tasks[@]}" \
    --action-execution-horizon "${ACTION_EXECUTION_HORIZON}" \
    --query-g-control "${control}" \
    --online-minimal-g-mode behavior_block_volume \
    --behavior-tool-clearance "${TOOL_CLEARANCE}" \
    --behavior-contact-tolerance 0 \
    --behavior-motion-threshold "${MOTION_THRESHOLD}" \
    --language-control normal \
    --state-only-observation \
    --trace-rollouts 5 \
    --device "cuda:${GPU_ID}" \
    2>&1 | tee "${LOG_ROOT}/eval_${mode}_${split}_${control}.log"
}

if [[ "${EVAL_FAMILY}" == "all" || "${EVAL_FAMILY}" == "core" ]]; then
  # Source-task success checks policy fit and A/B/C-to-D scene transfer.
  evaluate chunk_bc_state source normal 25 "${SOURCE_TASKS[@]}"
  evaluate chunk_bc_minimal_g_tokens source normal 25 "${SOURCE_TASKS[@]}"

  # Held-out color-direction recombination is the method gate. Each condition
  # uses 50 standard initial states per task.
  evaluate chunk_bc_state heldout normal 50 "${HELDOUT_TASKS[@]}"
  for control in "${HELDOUT_CONTROLS[@]}"; do
    evaluate chunk_bc_minimal_g_tokens heldout "${control}" 50 "${HELDOUT_TASKS[@]}"
  done
fi

if [[ "${EVAL_FAMILY}" == "token" ]]; then
  evaluate chunk_bc_minimal_g_tokens source normal 25 "${SOURCE_TASKS[@]}"
  for control in "${HELDOUT_CONTROLS[@]}"; do
    evaluate chunk_bc_minimal_g_tokens heldout "${control}" 50 "${HELDOUT_TASKS[@]}"
  done
fi

if [[ "${EVAL_FAMILY}" == "all" || "${EVAL_FAMILY}" == "factorized" ]]; then
  evaluate chunk_bc_minimal_g_factorized_tokens source normal 25 "${SOURCE_TASKS[@]}"
  for control in "${HELDOUT_CONTROLS[@]}"; do
    evaluate chunk_bc_minimal_g_factorized_tokens heldout "${control}" 50 \
      "${HELDOUT_TASKS[@]}"
  done
fi

if [[ "${EVAL_FAMILY}" == "frame" || \
      ( "${EVAL_FAMILY}" == "all" && "${INCLUDE_FRAME}" == "1" ) ]]; then
  evaluate chunk_bc_minimal_g_frame source normal 25 "${SOURCE_TASKS[@]}"
  for control in "${HELDOUT_CONTROLS[@]}"; do
    evaluate chunk_bc_minimal_g_frame heldout "${control}" 50 \
      "${HELDOUT_TASKS[@]}"
  done
fi

date -Is > "${RUN_ROOT}/.matched_volume_gate_${EVAL_FAMILY}_complete"
