#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}"
GEOM_JSON="${GEOM_JSON:-${ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json}"
CONDITION="${CONDITION:?Set CONDITION to D0, D1, D2, D3, or D4}"
SEED="${SEED:-0}"
TRAIN_TASKS=(
  open_drawer move_slider_left move_slider_right
  push_into_drawer
  push_blue_block_left push_blue_block_right
  push_pink_block_left push_pink_block_right
  push_red_block_left push_red_block_right
)

case "${CONDITION}" in
  D0)
    SCOPE="articulation"
    MODE="chunk_bc_minimal_g_film"
    CHUNK_SIZE=1
    FIRST_ACTION_WEIGHT=1
    STEM="2026-07-15_D0_sparse_direction_coverage_film_h1_seed"
    ;;
  D1)
    SCOPE="translational"
    MODE="chunk_bc_minimal_g_film"
    CHUNK_SIZE=1
    FIRST_ACTION_WEIGHT=1
    STEM="2026-07-15_D1_expanded_direction_coverage_film_h1_seed"
    ;;
  D2)
    SCOPE="translational"
    MODE="chunk_bc_minimal_g_bounded_projected"
    CHUNK_SIZE=16
    FIRST_ACTION_WEIGHT=16
    STEM="2026-07-15_D2_expanded_direction_coverage_field_coupled_h16_seed"
    ;;
  D3)
    SCOPE="translational"
    MODE="chunk_bc_minimal_g_film"
    CHUNK_SIZE=1
    FIRST_ACTION_WEIGHT=1
    STEM="2026-07-15_D3_signed_direction_coverage_film_h1_seed"
    TRAIN_TASKS+=(place_in_slider)
    ;;
  D4)
    SCOPE="translational"
    MODE="chunk_bc_minimal_g_film"
    CHUNK_SIZE=16
    FIRST_ACTION_WEIGHT=16
    STEM="2026-07-15_D4_expanded_direction_coverage_film_h16_seed"
    ;;
  *)
    echo "Unknown CONDITION=${CONDITION}; expected D0, D1, D2, D3, or D4" >&2
    exit 2
    ;;
esac

OUT_DIR="${OUT_DIR:-${ROOT}/experiments/phase1_calvin/${STEM}${SEED}}"
mkdir -p "${OUT_DIR}"
cd "${ROOT}"

"${PYTHON}" framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py \
  --calvin-code "${CALVIN_CODE}" \
  --data-root "${DATA_ROOT}" \
  --articulation-geometry-json "${GEOM_JSON}" \
  --minimal-g-scope "${SCOPE}" \
  --out-dir "${OUT_DIR}" \
  --modes "${MODE}" \
  --include-tasks "${TRAIN_TASKS[@]}" \
  --chunk-size "${CHUNK_SIZE}" \
  --stride "${STRIDE:-2}" \
  --epochs "${EPOCHS:-50}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --hidden "${HIDDEN:-512}" \
  --layers "${LAYERS:-4}" \
  --lr "${LR:-1e-3}" \
  --orthogonal-ratio-max "${ORTHOGONAL_RATIO_MAX:-2.0}" \
  --first-action-loss-weight "${FIRST_ACTION_WEIGHT}" \
  --seed "${SEED}" \
  --init-seed "${SEED}" \
  --loader-seed "${SEED}" \
  --data-seed 0 \
  --split-seed 0 \
  --skip-eval 1 \
  --device "${DEVICE:-cuda:0}"
