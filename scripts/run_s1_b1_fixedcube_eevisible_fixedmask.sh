#!/usr/bin/env bash
# Long B1 retrain with the geometry / appearance settings that match play:
#   1) robot_x_offset=-0.10, panda_joint6=2.7  (train + eval same env)
#   2) fixed cube spawn
#   3) fixed mask palette (no random colors)
#
# Network: S0-style dual-camera 8ch @ 64px. Fast teacher + vertical goal above cube.
#
# Beta: linear DAgger (warmup 100, decay 800) — guaranteed student mixing.
#
# Gripper fixes (vs prior linbeta run that reached but never closed):
#   2) gripper_bce_weight=50  (was 1; BCE was ~0.1% of arm_kl)
#   3) teacher_grasp_hold_steps=15  (was 2; more CLOSE labels)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_TAG="${RUN_TAG:-fixedcube_eevis_gripw50_hold15_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/s1_B1_${RUN_TAG}}"

exec env \
  RUN_TAG="${RUN_TAG}" \
  LOG_DIR="${LOG_DIR}" \
  MAX_ITERS="${MAX_ITERS:-2000}" \
  NUM_ENVS="${NUM_ENVS:-8}" \
  COLLECT_STEPS="${COLLECT_STEPS:-256}" \
  MAX_AGGREGATE_EPISODES="${MAX_AGGREGATE_EPISODES:-500}" \
  EVAL_INTERVAL="${EVAL_INTERVAL:-50}" \
  EVAL_EPISODES="${EVAL_EPISODES:-20}" \
  NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-4}" \
  IMAGE_SIZE=64 \
  CAMERA_SIZE=64 \
  TOPDOWN_CLOSE=0 \
  RANDOMIZE_CUBE=0 \
  RANDOMIZE_CUBE_Z_ONLY=0 \
  ROBOT_X_OFFSET=-0.10 \
  PANDA_JOINT6=2.7 \
  VISUAL_LAYOUT=channels \
  MASK_PALETTE_MODE=fixed \
  MASK_ROBOT_COLOR="${MASK_ROBOT_COLOR:-0.10 0.35 0.95}" \
  MASK_TABLE_COLOR="${MASK_TABLE_COLOR:-0.55 0.35 0.18}" \
  MASK_GROUND_COLOR="${MASK_GROUND_COLOR:-0.10 0.10 0.10}" \
  AUX_CUBE_EE_WEIGHT=0 \
  BUFFER_SUCCESS_ONLY=0 \
  PARTIAL_ROLLOUT_CAP=0 \
  MAX_STORE_EPISODE_STEPS=1250 \
  SUCCESS_EARLY_DONE=1 \
  LIFT_GOAL_ABOVE_CUBE=1 \
  LIFT_HEIGHT_ABOVE_CUBE=0.20 \
  BETA_ADAPTIVE=0 \
  BETA_GRASP_GATED=0 \
  BETA_START=1.0 \
  BETA_END=0.0 \
  BETA_WARMUP_ITERS=100 \
  BETA_DECAY_ITERS=800 \
  TEACHER_LIN_STD=0.05 \
  TEACHER_ANG_STD=0.10 \
  TEACHER_GRASP_HOLD_STEPS=15 \
  GRIPPER_BCE_WEIGHT=50 \
  bash "${ROOT}/scripts/run_s1_b1_mosaic_fast_teacher.sh"
