#!/usr/bin/env bash
# B1 ablation: fixed cube, EE-only mask RGB + depth channel stack, no aux head.
#
# Keeps the current vertical goal and fast teacher, while restoring the original
# S0 B1 adaptive-beta policy and retaining failed rollout segments in the buffer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_TAG="${RUN_TAG:-ee4ch_fast_teacher_s0beta_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/s1_B1_${RUN_TAG}}"

exec env \
  RUN_TAG="${RUN_TAG}" \
  LOG_DIR="${LOG_DIR}" \
  RANDOMIZE_CUBE=0 \
  RANDOMIZE_CUBE_Z_ONLY=0 \
  VISUAL_LAYOUT=ee_channels \
  MASK_PALETTE_MODE=fixed \
  AUX_CUBE_EE_WEIGHT=0 \
  BUFFER_SUCCESS_ONLY=0 \
  PARTIAL_ROLLOUT_CAP=256 \
  MAX_STORE_EPISODE_STEPS=256 \
  SUCCESS_EARLY_DONE=1 \
  LIFT_GOAL_ABOVE_CUBE=1 \
  LIFT_HEIGHT_ABOVE_CUBE=0.20 \
  BETA_ADAPTIVE=1 \
  BETA_GRASP_GATED=0 \
  BETA_START=1.0 \
  BETA_END=0.0 \
  BETA_WARMUP_ITERS=100 \
  BETA_DECAY_ITERS=2000 \
  BETA_SUCCESS_TARGET=0.5 \
  BETA_SUCCESS_EMA=0.7 \
  BETA_EVAL_METRIC=success \
  TEACHER_LIN_STD=0.05 \
  TEACHER_ANG_STD=0.10 \
  bash "${ROOT}/scripts/run_s1_b1_mosaic_fast_teacher.sh"
