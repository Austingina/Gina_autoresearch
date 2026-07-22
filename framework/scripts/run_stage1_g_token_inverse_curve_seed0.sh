#!/usr/bin/env bash
set -euo pipefail

K="${1:?usage: $0 REVERSE_SEGMENTS GPU}"
GPU="${2:?usage: $0 REVERSE_SEGMENTS GPU}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
PYTHON="/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python"
CALVIN_CODE="/mnt/nvme33n1/wuk_code/calvin_g/code/calvin"
DATA_ROOT="/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D"
GEOMETRY_JSON="${ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json"
OUT_ROOT="${ROOT}/experiments/phase1_calvin/2026-07-17_E4_inverse_curve_k${K}_seed0"

SOURCE_TASKS=(
  open_drawer
  move_slider_right
  push_red_block_left
  push_blue_block_right
  push_pink_block_left
  push_pink_block_right
  push_into_drawer
  place_in_slider
  turn_on_lightbulb
  turn_off_lightbulb
  turn_on_led
  turn_off_led
)

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=1

"${PYTHON}" framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py \
  --calvin-code "${CALVIN_CODE}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_ROOT}/with_g" \
  --modes chunk_bc_minimal_g_tokens \
  --include-tasks "${SOURCE_TASKS[@]}" \
  --reverse-task-pairs \
    open_drawer=close_drawer \
    move_slider_right=move_slider_left \
  --reverse-pair-max-segments "${K}" \
  --stride 2 \
  --chunk-size 16 \
  --epochs 50 \
  --batch-size 512 \
  --hidden 512 \
  --layers 4 \
  --lr 1e-3 \
  --seed 0 \
  --init-seed 0 \
  --loader-seed 0 \
  --data-seed 0 \
  --split-seed 0 \
  --device cuda:0 \
  --skip-eval 1 \
  --articulation-geometry-json "${GEOMETRY_JSON}" \
  --minimal-g-scope translational \
  --g-scale-aug-values 1.0 \
  --g-dropout-prob 0.0 \
  --first-action-loss-weight 16 \
  --drop-query-g-token-type

