#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"
PAIR_WEIGHT="${2:-0.5}"
GPU="${GPU:-5}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
PYTHON="/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python"
CALVIN_CODE="/mnt/nvme33n1/wuk_code/calvin_g/code/calvin"
DATA_ROOT="/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D"
GEOMETRY_JSON="${ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json"
WEIGHT_TAG="${PAIR_WEIGHT/./p}"
OUT_DIR="${ROOT}/experiments/phase1_calvin/2026-07-16_E1_g_token_inverse_pair_w${WEIGHT_TAG}_seed${SEED}"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS="1"

exec "${PYTHON}" framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py \
  --calvin-code "${CALVIN_CODE}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --modes chunk_bc_minimal_g_tokens chunk_fm_minimal_g_tokens \
  --include-tasks \
    open_drawer move_slider_left move_slider_right \
    push_into_drawer \
    push_blue_block_left push_blue_block_right \
    push_pink_block_left push_pink_block_right \
    push_red_block_left push_red_block_right \
    turn_on_lightbulb turn_off_lightbulb turn_on_led turn_off_led \
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
  --checkpoint-epochs 10 20 30 40 50 \
  --articulation-geometry-json "${GEOMETRY_JSON}" \
  --minimal-g-scope translational \
  --g-scale-aug-values 1.0 \
  --g-dropout-prob 0.0 \
  --first-action-loss-weight 16 \
  --inverse-pair-loss-weight "${PAIR_WEIGHT}" \
  --inverse-pairs-per-batch 128 \
  --inverse-pair-max-eef-distance-m 0.05 \
  --inverse-pair-max-mechanism-fraction 0.10
