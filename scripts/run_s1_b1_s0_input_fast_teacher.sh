#!/usr/bin/env bash
# Controlled B1 retrain:
# - old S0 network input: dual-camera 8ch channel stack at 64 px
# - old S0 buffer semantics: failures included, no partial segmentation, 500 ep cap
# - current geometry: robot x offset / joint6 bias and goal above fixed cube
# - current fast teacher, but original S0 KL std and adaptive-beta policy
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_TAG="${RUN_TAG:-s0input8ch_fast_teacher_currentbias_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/s1_B1_${RUN_TAG}}"

exec env \
  RUN_TAG="${RUN_TAG}" \
  LOG_DIR="${LOG_DIR}" \
  MAX_ITERS="${MAX_ITERS:-2000}" \
  NUM_ENVS="${NUM_ENVS:-8}" \
  COLLECT_STEPS="${COLLECT_STEPS:-256}" \
  MAX_AGGREGATE_EPISODES="${MAX_AGGREGATE_EPISODES:-500}" \
  IMAGE_SIZE=64 \
  CAMERA_SIZE=64 \
  TOPDOWN_CLOSE=0 \
  RANDOMIZE_CUBE=0 \
  RANDOMIZE_CUBE_Z_ONLY=0 \
  ROBOT_X_OFFSET=-0.10 \
  PANDA_JOINT6=2.7 \
  VISUAL_LAYOUT=channels \
  MASK_PALETTE_MODE=fixed \
  AUX_CUBE_EE_WEIGHT=0 \
  BUFFER_SUCCESS_ONLY=0 \
  PARTIAL_ROLLOUT_CAP=0 \
  MAX_STORE_EPISODE_STEPS=1250 \
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
