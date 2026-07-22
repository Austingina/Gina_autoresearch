#!/usr/bin/env bash
# Reproduce the original s0_B1_restart training setup on the current codebase.
# Later-added features (aux head, vertical goal, partial flushing) are disabled.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/simulators/resolve_paths.sh"
ISAACLAB_DIR="${ISAACLAB_ROOT}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
RUN_TAG="${RUN_TAG:-exact_repro_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/s0_B1_${RUN_TAG}}"

mkdir -p "${LOG_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm-256color
fi
export PYTHONUNBUFFERED=1
export AUTORESEARCH_ROOT="${ROOT}"
cd "${ISAACLAB_DIR}"

exec ./isaaclab.sh -p scripts/reinforcement_learning/train_lift_cube_dagger_vision_binary_gripper.py \
  --headless \
  --num_envs 16 \
  --max_iterations 100 \
  --collect_steps 512 \
  --min_buffer_episodes 8 \
  --max_aggregate_episodes 500 \
  --episode_length_s 25 \
  --eval_episodes 20 \
  --eval_interval 50 \
  --save_interval 100 \
  --seed 42 \
  --fixed_lift_goal \
  --lift_goal_x 0.5 \
  --lift_goal_y 0.0 \
  --lift_goal_z 0.375 \
  --no-lift_goal_above_cube \
  --no-randomize_cube \
  --no-randomize_cube_z_only \
  --no-randomize_lift_goal \
  --image_size 64 \
  --camera_height 64 \
  --camera_width 64 \
  --no-topdown_close \
  --visual_layout channels \
  --visual_channels 8 \
  --guidance_mode anchor_only \
  --mask_palette_mode random \
  --aux_cube_ee_weight 0 \
  --teacher_min_lin_speed 0.10 \
  --teacher_min_ang_speed 0.20 \
  --teacher_max_lin_vel 0.5 \
  --teacher_max_ang_vel 1.0 \
  --teacher_grasp_hold_steps 15 \
  --teacher_above_offset_z 0.10 \
  --teacher_lin_std 0.05 \
  --teacher_ang_std 0.10 \
  --arm_scale 0.5 \
  --goal_threshold 0.02 \
  --no-success_early_done \
  --no-buffer_success_only \
  --partial_rollout_cap 0 \
  --max_store_episode_steps 1250 \
  --aggregate_backend disk \
  --aggregate_cache_size 8 \
  --aggregate_visual_storage_dtype float16 \
  --beta_start 1.0 \
  --beta_end 0.0 \
  --beta_warmup_iters 100 \
  --beta_decay_iters 2000 \
  --beta_adaptive \
  --beta_success_target 0.5 \
  --beta_success_ema 0.7 \
  --beta_eval_metric success \
  --autoresearch_root "${ROOT}" \
  --log_dir "${LOG_DIR}" \
  2>&1 | tee "${LOG_DIR}/launch.log"
