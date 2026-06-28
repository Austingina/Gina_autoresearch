#!/usr/bin/env bash
# Fresh S1 B3: oracle GT 3D G (anchor_field_rule), same safe defaults as B7 fresh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_TAG="${RUN_TAG:-cam240_xy_gt3d_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/runs/s1_B3_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

exec env -i \
  HOME="${HOME}" \
  USER="${USER:-$(id -un)}" \
  PATH="${PATH}" \
  SHELL="${SHELL:-/bin/bash}" \
  TERM="${TERM:-xterm-256color}" \
  ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/l/code/IsaacLab}" \
  CONDA_ENV="${CONDA_ENV:-env_isaaclab}" \
  AUTORESEARCH_ROOT="${ROOT}" \
  RUN_TAG="${RUN_TAG}" \
  LOG_DIR="${LOG_DIR}" \
  MAX_ITERS=2000 \
  NUM_ENVS=4 \
  COLLECT_STEPS=128 \
  MAX_AGGREGATE_EPISODES=250 \
  IMAGE_SIZE=240 \
  CAMERA_SIZE=240 \
  TOPDOWN_CLOSE=1 \
  RANDOMIZE_CUBE_Z_ONLY=0 \
  BETA_ADAPTIVE=1 \
  BETA_WARMUP_ITERS=50 \
  BETA_SUCCESS_TARGET=0.5 \
  EVAL_INTERVAL=25 \
  EVAL_EPISODES=3 \
  NUM_EVAL_ENVS=2 \
  NO_RESUME_BUFFER=1 \
  BUFFER_SAVE_INTERVAL=0 \
  AGGREGATE_BACKEND=disk \
  AGGREGATE_CACHE_SIZE=4 \
  AGGREGATE_VISUAL_STORAGE_DTYPE=float16 \
  RESUME_CHECKPOINT="" \
  RESUME_BUFFER="" \
  bash "${ROOT}/scripts/run_s1_dagger_smoke.sh" B3 \
  2>&1 | tee "${LOG_DIR}/launch.log"
