#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}"
GEOM_JSON="${GEOM_JSON:-${ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/phase1_calvin/2026-07-11_scope_primitive_zero_shot_close_query_g_only_paired_e50_n100_h160_seed${SEED}}"

cd "${ROOT}"

"${PYTHON}" framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py \
  --calvin-code "${CALVIN_CODE}" \
  --data-root "${DATA_ROOT}" \
  --articulation-geometry-json "${GEOM_JSON}" \
  --out-dir "${OUT_DIR}" \
  --modes chunk_bc_query_g \
  --include-tasks \
    open_drawer move_slider_left move_slider_right \
    push_into_drawer \
    push_blue_block_left push_blue_block_right \
    push_pink_block_left push_pink_block_right \
    push_red_block_left push_red_block_right \
    turn_on_lightbulb turn_off_lightbulb \
    turn_on_led turn_off_led \
  --chunk-size 16 \
  --stride 2 \
  --epochs 50 \
  --batch-size 512 \
  --hidden 512 \
  --layers 4 \
  --lr 1e-3 \
  --seed "${SEED}" \
  --data-seed 0 \
  --split-seed 0 \
  --checkpoint-epochs 10 20 30 40 50 \
  --eval-mode balanced_first_task \
  --eval-task-filter close_drawer \
  --num-sequences 100 \
  --ep-len 160 \
  --eval-scan-sequences 50000 \
  --eval-sequence-offset 0 \
  --device "${DEVICE:-cuda:0}"
