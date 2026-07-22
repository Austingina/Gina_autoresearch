#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}"
GEOM_JSON="${GEOM_JSON:-${ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/phase1_calvin/2026-07-13_minimal_g_concat_structured_e50_n100_h160_seed${SEED}}"
VARIANT="${VARIANT:-both}"

case "${VARIANT}" in
  state)
    MODES=(chunk_bc_state)
    ;;
  lang)
    MODES=(chunk_bc_lang)
    ;;
  learned_executor_triplet)
    MODES=(chunk_bc_lang chunk_bc_minimal_g_concat chunk_bc_minimal_g_film)
    ;;
  both)
    MODES=(chunk_bc_lang_minimal_g_concat chunk_bc_lang_minimal_g_structured)
    ;;
  concat)
    MODES=(chunk_bc_lang_minimal_g_concat)
    ;;
  minimal_concat_nolang)
    MODES=(chunk_bc_minimal_g_concat)
    ;;
  minimal_film_nolang)
    MODES=(chunk_bc_minimal_g_film)
    ;;
  minimal_semantic_nolang)
    MODES=(chunk_bc_minimal_g_semantic)
    ;;
  minimal_frame_nolang)
    MODES=(chunk_bc_minimal_g_frame)
    ;;
  structured)
    MODES=(chunk_bc_lang_minimal_g_structured)
    ;;
  projected)
    MODES=(chunk_bc_lang_minimal_g_projected)
    ;;
  projected_nolang)
    MODES=(chunk_bc_minimal_g_projected)
    ;;
  bounded_projected)
    MODES=(chunk_bc_lang_minimal_g_bounded_projected)
    ;;
  bounded_projected_nolang)
    MODES=(chunk_bc_minimal_g_bounded_projected)
    ;;
  invariant_bounded_projected)
    MODES=(chunk_bc_lang_minimal_g_invariant_bounded_projected)
    ;;
  invariant_bounded_projected_nolang)
    MODES=(chunk_bc_minimal_g_invariant_bounded_projected)
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 2
    ;;
esac

cd "${ROOT}"

"${PYTHON}" framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py \
  --calvin-code "${CALVIN_CODE}" \
  --data-root "${DATA_ROOT}" \
  --articulation-geometry-json "${GEOM_JSON}" \
  --out-dir "${OUT_DIR}" \
  --modes "${MODES[@]}" \
  --include-tasks \
    open_drawer move_slider_left move_slider_right \
    push_into_drawer \
    push_blue_block_left push_blue_block_right \
    push_pink_block_left push_pink_block_right \
    push_red_block_left push_red_block_right \
    turn_on_lightbulb turn_off_lightbulb \
    turn_on_led turn_off_led \
  --chunk-size "${CHUNK_SIZE:-16}" \
  --stride "${STRIDE:-2}" \
  --epochs "${EPOCHS:-50}" \
  --batch-size 512 \
  --hidden 512 \
  --layers 4 \
  --lr 1e-3 \
  --field-alignment-weight "${FIELD_ALIGNMENT_WEIGHT:-0.05}" \
  --orthogonal-ratio-max "${ORTHOGONAL_RATIO_MAX:-2.0}" \
  --first-action-loss-weight "${FIRST_ACTION_LOSS_WEIGHT:-1.0}" \
  --action-execution-horizon "${ACTION_EXECUTION_HORIZON:-1}" \
  --eval-dir-suffix "${EVAL_DIR_SUFFIX:-}" \
  --temporal-ensemble "${TEMPORAL_ENSEMBLE:-0}" \
  --temporal-ensemble-decay "${TEMPORAL_ENSEMBLE_DECAY:-0.25}" \
  --seed "${SEED}" \
  --data-seed 0 \
  --split-seed 0 \
  --checkpoint-epochs 10 20 30 40 50 \
  --skip-eval "${SKIP_EVAL:-0}" \
  --eval-mode balanced_first_task \
  --eval-task-filter close_drawer \
  --num-sequences "${NUM_SEQUENCES:-100}" \
  --ep-len "${EP_LEN:-160}" \
  --eval-scan-sequences 50000 \
  --eval-sequence-offset 0 \
  --device "${DEVICE:-cuda:0}"
