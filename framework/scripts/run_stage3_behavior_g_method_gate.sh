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
CALVIN_DATA="${CALVIN_DATA:-${CALVIN_G_ROOT}/data}"
CALVIN_PYTHON="${CALVIN_PYTHON:-${CALVIN_G_ROOT}/envs/calvin_h200/bin/python}"
COMPACT_ROOT="${MODELSCOPE_CALVIN_COMPACT:-${CALVIN_DATA}/lerobot_blocks_behavior_g_semantic}"
FEATURE_ROOT="${MODELSCOPE_CALVIN_DINO:-${CALVIN_DATA}/lerobot_blocks_behavior_g_semantic_dinov2_small}"
SIM_DATA_ROOT="${CALVIN_SIM_DATA_ROOT:-${CALVIN_DATA}/calvin_debug_dataset}"
SEED="${1:-0}"
GPU_ID="${2:-${GPU_ID:-0}}"
RUN_LABEL="${METHOD_GATE_RUN_LABEL:-M3_full_behavior_g_semantic}"
TRAIN_LR="${METHOD_GATE_LR:-1e-3}"
CHECKPOINT_BASENAME="${METHOD_GATE_CHECKPOINT_BASENAME:-policy.pt}"
EVAL_TAG="${METHOD_GATE_EVAL_TAG:-behavior}"
BEHAVIOR_CONTACT_RADIUS="${METHOD_GATE_CONTACT_RADIUS:-0.01}"
ONLINE_G_MODE="${METHOD_GATE_ONLINE_G_MODE:-behavior_block_center}"
BEHAVIOR_TOOL_CLEARANCE="${METHOD_GATE_TOOL_CLEARANCE:-0.01}"
BEHAVIOR_CONTACT_TOLERANCE="${METHOD_GATE_CONTACT_TOLERANCE:-0.01}"
RUN_ROOT="${REPO_ROOT}/experiments/phase1_calvin/2026-08-13_${RUN_LABEL}_seed${SEED}"
LOG_ROOT="${CALVIN_G_ROOT}/logs/${RUN_LABEL}_seed${SEED}"
GEOMETRY_JSON="${REPO_ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

while [[ ! -f "${FEATURE_ROOT}/.pipeline_complete" ]]; do
  printf '%s waiting for data pipeline\n' "$(date -Is)"
  sleep 60
done

"${CALVIN_PYTHON}" - "${COMPACT_ROOT}/conversion_manifest.json" "${FEATURE_ROOT}/manifest.json" <<'PY'
import json
import sys

import numpy as np

conversion = json.load(open(sys.argv[1], encoding="utf-8"))
features = json.load(open(sys.argv[2], encoding="utf-8"))
expected = {
    "push_red_block_left",
    "push_red_block_right",
    "push_blue_block_left",
    "push_blue_block_right",
    "push_pink_block_left",
    "push_pink_block_right",
}
counts = conversion["task_counts"]
assert set(counts) == expected, (set(counts), expected)
assert min(counts.values()) >= 400, counts
assert conversion["frames"] >= 100_000, conversion["frames"]
assert features["num_frames"] == conversion["frames"], (features["num_frames"], conversion["frames"])
assert features["feature_dim"] == 384
records = conversion["episode_records"]
assert len(records) == sum(counts.values()), (len(records), sum(counts.values()))
pixel_distances = [
    row["target_eef_pixel_distance_median"]
    for row in records
    if row["target_eef_pixel_distance_median"] is not None
]
assert len(pixel_distances) >= 0.98 * len(records), (len(pixel_distances), len(records))
pixel_p90 = float(np.quantile(pixel_distances, 0.90))
late_contact_fraction = float(
    np.mean([row["contact_index"] >= row["frames"] - 3 for row in records])
)
min_flow_cosine = min(row["flow_semantic_cosine"] for row in records)
assert pixel_p90 <= 25.0, pixel_p90
assert late_contact_fraction <= 0.15, late_contact_fraction
assert min_flow_cosine >= 0.99, min_flow_cosine
print(
    "data_gate_ok",
    conversion["frames"],
    counts,
    {"pixel_p90": pixel_p90, "late_contact_fraction": late_contact_fraction},
)
PY

unset TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-${CALVIN_G_ROOT}/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

CHECKPOINT="${RUN_ROOT}/chunk_bc_minimal_g_tokens/${CHECKPOINT_BASENAME}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${COMPACT_ROOT}" \
    --out-dir "${RUN_ROOT}" \
    --modes chunk_bc_minimal_g_tokens \
    --include-tasks \
      push_red_block_left push_blue_block_right \
      push_pink_block_left push_pink_block_right \
    --stride 2 \
    --chunk-size 16 \
    --epochs 50 \
    --batch-size 512 \
    --hidden 512 \
    --layers 4 \
    --lr "${TRAIN_LR}" \
    --seed "${SEED}" \
    --init-seed "${SEED}" \
    --loader-seed "${SEED}" \
    --data-seed "${SEED}" \
    --split-seed "${SEED}" \
    --device "cuda:${GPU_ID}" \
    --skip-eval 1 \
    --articulation-geometry-json "${GEOMETRY_JSON}" \
    --minimal-g-scope translational \
    --executor-state-scope visual \
    --executor-language-mode none \
    --visual-feature-dir "${FEATURE_ROOT}" \
    --visual-encoder-name facebook/dinov2-small \
    --mask-executor-language-with-g 0 \
    --g-scale-aug-values 1.0 \
    --g-dropout-prob 0.0 \
    --first-action-loss-weight 16 \
    2>&1 | tee "${LOG_ROOT}/train.log"
fi

if [[ "${METHOD_GATE_TRAIN_ONLY:-0}" == "1" ]]; then
  printf '%s seed=%s training complete; METHOD_GATE_TRAIN_ONLY=1\n' "$(date -Is)" "${SEED}"
  exit 0
fi

for control in normal zero sign_flip; do
  eval_dir="${RUN_ROOT}/chunk_bc_minimal_g_tokens/task_eval_D_${EVAL_TAG}_${control}"
  if [[ -f "${eval_dir}/state_bc_summary.json" ]]; then
    continue
  fi
  mkdir -p "${eval_dir}"
  env -u EGL_VISIBLE_DEVICES xvfb-run -a "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/policy_state_bc/eval_state_chunk_policy_ckpt.py" \
    --calvin-code "${CALVIN_CODE}" \
    --data-root "${SIM_DATA_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --run-meta "${RUN_ROOT}/run_meta.json" \
    --eval-dir "${eval_dir}" \
    --num-sequences 50 \
    --ep-len 160 \
    --eval-scan-sequences 10000 \
    --eval-sequence-workers 8 \
    --eval-task-filter push_red_block_right push_blue_block_left \
    --action-execution-horizon 1 \
    --query-g-control "${control}" \
    --online-minimal-g-mode "${ONLINE_G_MODE}" \
    --behavior-contact-radius "${BEHAVIOR_CONTACT_RADIUS}" \
    --behavior-tool-clearance "${BEHAVIOR_TOOL_CLEARANCE}" \
    --behavior-contact-tolerance "${BEHAVIOR_CONTACT_TOLERANCE}" \
    --language-control normal \
    --device "cuda:${GPU_ID}" \
    2>&1 | tee "${LOG_ROOT}/eval_${control}.log"
done

date -Is > "${RUN_ROOT}/.method_gate_complete"
