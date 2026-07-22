#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:?usage: $0 LABEL SOURCE_WEIGHT REVERSE_WEIGHT ADAPTER_HIDDEN EPOCHS LR GPU [SLIDER_WEIGHT] [ROUTED] [ROUTER_WEIGHT]}"
SOURCE_WEIGHT="${2:?usage: $0 LABEL SOURCE_WEIGHT REVERSE_WEIGHT ADAPTER_HIDDEN EPOCHS LR GPU [SLIDER_WEIGHT]}"
REVERSE_WEIGHT="${3:?usage: $0 LABEL SOURCE_WEIGHT REVERSE_WEIGHT ADAPTER_HIDDEN EPOCHS LR GPU [SLIDER_WEIGHT]}"
ADAPTER_HIDDEN="${4:?usage: $0 LABEL SOURCE_WEIGHT REVERSE_WEIGHT ADAPTER_HIDDEN EPOCHS LR GPU [SLIDER_WEIGHT]}"
EPOCHS="${5:?usage: $0 LABEL SOURCE_WEIGHT REVERSE_WEIGHT ADAPTER_HIDDEN EPOCHS LR GPU [SLIDER_WEIGHT]}"
LR="${6:?usage: $0 LABEL SOURCE_WEIGHT REVERSE_WEIGHT ADAPTER_HIDDEN EPOCHS LR GPU [SLIDER_WEIGHT]}"
GPU="${7:?usage: $0 LABEL SOURCE_WEIGHT REVERSE_WEIGHT ADAPTER_HIDDEN EPOCHS LR GPU [SLIDER_WEIGHT]}"
SLIDER_WEIGHT="${8:-1.0}"
ROUTED="${9:-0}"
ROUTER_WEIGHT="${10:-1.0}"
SOURCE_EXPERT="${11:-0}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
PYTHON="/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python"
CALVIN_CODE="/mnt/nvme33n1/wuk_code/calvin_g/code/calvin"
DATA_ROOT="/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D"
GEOMETRY_JSON="${ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json"
INIT_CHECKPOINT="${ROOT}/experiments/phase1_calvin/2026-07-16_E3_factorial_checkerboard_seed0/with_g/chunk_bc_minimal_g_tokens/policy.pt"
OUT_ROOT="${ROOT}/experiments/phase1_calvin/2026-07-17_E4_inverse_finetune_${LABEL}_seed0"

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

ROUTED_ARGS=()
if [[ "${ROUTED}" == "1" ]]; then
  ROUTED_ARGS=(
    --residual-adapter-expert-tasks close_drawer move_slider_left
    --residual-router-loss-weight "${ROUTER_WEIGHT}"
  )
  if [[ "${SOURCE_EXPERT}" == "1" ]]; then
    ROUTED_ARGS+=(--residual-adapter-source-expert)
  fi
fi

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=1

"${PYTHON}" framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py \
  --calvin-code "${CALVIN_CODE}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_ROOT}/with_g" \
  --modes chunk_bc_minimal_g_tokens \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --include-tasks "${SOURCE_TASKS[@]}" \
  --reverse-task-pairs \
    open_drawer=close_drawer \
    move_slider_right=move_slider_left \
  --reverse-pair-max-segments 100 \
  --stride 2 \
  --chunk-size 16 \
  --epochs "${EPOCHS}" \
  --batch-size 512 \
  --hidden 512 \
  --layers 4 \
  --lr "${LR}" \
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
  --source-distill-weight "${SOURCE_WEIGHT}" \
  --residual-adapter \
  --residual-adapter-hidden "${ADAPTER_HIDDEN}" \
  --reverse-source-loss-weight "${REVERSE_WEIGHT}" \
  --reverse-task-loss-weights move_slider_left="${SLIDER_WEIGHT}" \
  "${ROUTED_ARGS[@]}" \
  --drop-query-g-token-type
