#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
GPU="${GPU:-2}"

export CUDA_VISIBLE_DEVICES="${GPU}"

"${PYTHON}" "${ROOT}/framework/scripts/policy_state_bc/train_eval_state_bc_task_policy.py" \
  --calvin-code "${ROOT}/phase1_calvin/calvin_code" \
  --data-root "${ROOT}/phase1_calvin/data/task_D_D" \
  --out-dir "${ROOT}/experiments/phase1_calvin/2026-07-06_state_bc_v5_attr_only_v6_interaction_concat_e4_8_16_32_50_n25_h160_seed0" \
  --modes articulation_g_v5_attraction_only_concat articulation_g_v6_interaction_field_concat \
  --include-tasks open_drawer close_drawer move_slider_left move_slider_right \
  --articulation-geometry-json "${ROOT}/experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json" \
  --max-samples 60000 \
  --stride 2 \
  --epochs 50 \
  --checkpoint-epochs 4 8 16 32 50 \
  --eval-checkpoints \
  --batch-size 512 \
  --hidden 256 \
  --layers 3 \
  --lr 3e-4 \
  --num-sequences 25 \
  --ep-len 160 \
  --eval-mode balanced_first_task \
  --eval-task-filter open_drawer close_drawer move_slider_left move_slider_right \
  --eval-scan-sequences 5000 \
  --device cuda:0 \
  --seed 0
