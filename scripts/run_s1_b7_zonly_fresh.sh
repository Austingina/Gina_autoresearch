#!/usr/bin/env bash
# Fresh S1 B7: z-only cube DR, 240px camera (G_θ aligned), adaptive β, 2000 iters.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/simulators/resolve_paths.sh"
RUN_TAG="${RUN_TAG:-cam240_zonly_adapt_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/runs/s1_B7_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

exec env -i \
  HOME="${HOME}" \
  USER="${USER:-$(id -un)}" \
  PATH="${PATH}" \
  SHELL="${SHELL:-/bin/bash}" \
  TERM="${TERM:-xterm-256color}" \
  ISAACLAB_ROOT="${ISAACLAB_ROOT}" \
  CONDA_ENV="${CONDA_ENV:-env_isaaclab}" \
  AUTORESEARCH_ROOT="${ROOT}" \
  RUN_TAG="${RUN_TAG}" \
  LOG_DIR="${LOG_DIR}" \
  MAX_ITERS=2000 \
  NUM_ENVS=8 \
  COLLECT_STEPS=256 \
  MAX_AGGREGATE_EPISODES=250 \
  IMAGE_SIZE=240 \
  CAMERA_SIZE=240 \
  TOPDOWN_CLOSE=1 \
  RANDOMIZE_CUBE_Z_ONLY=1 \
  CUBE_Z_MIN=0.0 \
  CUBE_Z_MAX=0.0 \
  BETA_ADAPTIVE=1 \
  BETA_WARMUP_ITERS=50 \
  BETA_SUCCESS_TARGET=0.5 \
  EVAL_INTERVAL=25 \
  EVAL_EPISODES=5 \
  NUM_EVAL_ENVS=4 \
  NO_RESUME_BUFFER=1 \
  BUFFER_SAVE_INTERVAL=0 \
  RESUME_CHECKPOINT="" \
  RESUME_BUFFER="" \
  FIELD_CKPT="${ROOT}/checkpoints/lift/guidance_field_s1_random_cube.pt" \
  bash "${ROOT}/scripts/run_s1_dagger_smoke.sh" B7 \
  2>&1 | tee "${LOG_DIR}/launch.log"
