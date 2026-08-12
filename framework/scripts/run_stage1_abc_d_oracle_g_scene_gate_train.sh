#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: $0 SEED GPU STATE_SCOPE VARIANT [LANGUAGE_MODE]}"
GPU="${2:?usage: $0 SEED GPU STATE_SCOPE VARIANT [LANGUAGE_MODE]}"
STATE_SCOPE="${3:?usage: $0 SEED GPU STATE_SCOPE VARIANT [LANGUAGE_MODE]}"
VARIANT="${4:?usage: $0 SEED GPU STATE_SCOPE VARIANT [LANGUAGE_MODE]}"
LANGUAGE_MODE="${5:-onehot}"

case "${STATE_SCOPE}" in
  full|proprio|visual) ;;
  *) echo "STATE_SCOPE must be full, proprio, or visual" >&2; exit 2 ;;
esac
case "${VARIANT}" in
  with_g) G_DROPOUT=0.0 ;;
  language_only) G_DROPOUT=1.0 ;;
  *) echo "VARIANT must be with_g or language_only" >&2; exit 2 ;;
esac
case "${LANGUAGE_MODE}" in
  none|onehot|minilm) ;;
  *) echo "LANGUAGE_MODE must be none, onehot, or minilm" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${REPO_ROOT}/.env.server" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.server"
fi
ROOT="${AUTORESEARCH_ROOT:-${REPO_ROOT}}"
CALVIN_G_ROOT="${CALVIN_G_ROOT:-$(cd "${ROOT}/.." && pwd)/calvin_g}"
PYTHON="${CALVIN_PYTHON:-${CALVIN_G_ROOT}/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${CALVIN_G_ROOT}/code/calvin}"
CALVIN_DATA="${CALVIN_DATA:-${CALVIN_G_ROOT}/data}"
DATA_ROOT="${CALVIN_COMPACT_DATA_ROOT:-${CALVIN_DATA}/task_ABC_D_compact_blocks}"
GEOMETRY_JSON="${ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json"
LANGUAGE_TAG=""
if [[ "${LANGUAGE_MODE}" != "onehot" ]]; then
  LANGUAGE_TAG="_${LANGUAGE_MODE}"
fi
OUT_ROOT="${ROOT}/experiments/phase1_calvin/2026-07-19_M1_abc_to_d_oracle_g${LANGUAGE_TAG}_${STATE_SCOPE}_seed${SEED}/${VARIANT}"

SOURCE_TASKS=(
  push_red_block_left
  push_blue_block_right
  push_pink_block_left
  push_pink_block_right
)
VISUAL_ARGS=()
if [[ "${STATE_SCOPE}" == "visual" ]]; then
  VISUAL_ARGS+=(
    --visual-feature-dir "${CALVIN_DINOV2_FEATURE_ROOT:-${CALVIN_DATA}/task_ABC_D_compact_blocks_dinov2_small}"
    --visual-encoder-name facebook/dinov2-small
  )
fi

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=1

"${PYTHON}" framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py \
  --calvin-code "${CALVIN_CODE}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_ROOT}" \
  --modes chunk_bc_minimal_g_tokens \
  --include-tasks "${SOURCE_TASKS[@]}" \
  --stride 2 \
  --chunk-size 16 \
  --epochs 50 \
  --batch-size 512 \
  --hidden 512 \
  --layers 4 \
  --lr 1e-3 \
  --seed "${SEED}" \
  --init-seed "${SEED}" \
  --loader-seed 0 \
  --data-seed 0 \
  --split-seed 0 \
  --device cuda:0 \
  --skip-eval 1 \
  --articulation-geometry-json "${GEOMETRY_JSON}" \
  --minimal-g-scope translational \
  --executor-state-scope "${STATE_SCOPE}" \
  --executor-language-mode "${LANGUAGE_MODE}" \
  "${VISUAL_ARGS[@]}" \
  --mask-executor-language-with-g 0 \
  --g-scale-aug-values 1.0 \
  --g-dropout-prob "${G_DROPOUT}" \
  --first-action-loss-weight 16
