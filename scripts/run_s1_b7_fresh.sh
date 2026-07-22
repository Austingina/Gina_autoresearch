#!/usr/bin/env bash
# Fresh S1 B7 (safe defaults): anti-OOM profile with lighter rollout/eval load.
#
# Why this profile exists (see findings.md OOM incident log):
# - AGGREGATE_BACKEND=disk  (not ram — 11ch×240² episodes OOM in RAM)
# - NUM_ENVS=4, COLLECT_STEPS=128, NUM_EVAL_ENVS=2, EVAL_EPISODES=3
# - EE-visible init: robot_x_offset=-0.10, panda_joint6=2.7
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/simulators/resolve_paths.sh"
RUN_TAG="${RUN_TAG:-visible_ee_forward_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/runs/s1_B7_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

exec env -i \
  HOME="${HOME}" \
  USER="${USER:-$(id -un)}" \
  PATH="${PATH}" \
  SHELL="${SHELL:-/bin/bash}" \
  TERM="${TERM:-xterm-256color}" \
  OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}" \
  ISAACLAB_ROOT="${ISAACLAB_ROOT}" \
  CONDA_ENV="${CONDA_ENV:-env_isaaclab}" \
  AUTORESEARCH_ROOT="${ROOT}" \
  RUN_TAG="${RUN_TAG}" \
  LOG_DIR="${LOG_DIR}" \
  MAX_ITERS="${MAX_ITERS:-2000}" \
  NUM_ENVS="${NUM_ENVS:-4}" \
  COLLECT_STEPS="${COLLECT_STEPS:-128}" \
  MAX_AGGREGATE_EPISODES="${MAX_AGGREGATE_EPISODES:-250}" \
  IMAGE_SIZE="${IMAGE_SIZE:-240}" \
  CAMERA_SIZE="${CAMERA_SIZE:-240}" \
  TOPDOWN_CLOSE="${TOPDOWN_CLOSE:-1}" \
  RANDOMIZE_CUBE_Z_ONLY="${RANDOMIZE_CUBE_Z_ONLY:-0}" \
  ROBOT_X_OFFSET="${ROBOT_X_OFFSET:--0.10}" \
  PANDA_JOINT6="${PANDA_JOINT6:-2.7}" \
  BETA_ADAPTIVE="${BETA_ADAPTIVE:-1}" \
  BETA_WARMUP_ITERS="${BETA_WARMUP_ITERS:-50}" \
  BETA_SUCCESS_TARGET="${BETA_SUCCESS_TARGET:-0.5}" \
  BETA_EVAL_METRIC="${BETA_EVAL_METRIC:-success_5cm}" \
  G_AUG_NOISE_STD="${G_AUG_NOISE_STD:-0.15}" \
  G_AUG_DROPOUT_PROB="${G_AUG_DROPOUT_PROB:-0.10}" \
  G_AUG_BLEND_GT_PROB="${G_AUG_BLEND_GT_PROB:-0.10}" \
  G_AUG_REACH_ONLY="${G_AUG_REACH_ONLY:-1}" \
  MAX_STORE_EPISODE_STEPS="${MAX_STORE_EPISODE_STEPS:-512}" \
  EVAL_PHASE_STALL_STEPS="${EVAL_PHASE_STALL_STEPS:-60}" \
  EVAL_INTERVAL="${EVAL_INTERVAL:-10}" \
  EVAL_EPISODES="${EVAL_EPISODES:-3}" \
  NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-2}" \
  NO_RESUME_BUFFER="${NO_RESUME_BUFFER:-1}" \
  BUFFER_SAVE_INTERVAL="${BUFFER_SAVE_INTERVAL:-0}" \
  AGGREGATE_BACKEND="${AGGREGATE_BACKEND:-disk}" \
  AGGREGATE_CACHE_SIZE="${AGGREGATE_CACHE_SIZE:-4}" \
  AGGREGATE_VISUAL_STORAGE_DTYPE="${AGGREGATE_VISUAL_STORAGE_DTYPE:-float16}" \
  RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}" \
  RESUME_BUFFER="${RESUME_BUFFER:-}" \
  FIELD_CKPT="${FIELD_CKPT:-${ROOT}/checkpoints/lift/guidance_field_s1_random_cube_visible_ee_forward_12k.pt}" \
  bash "${ROOT}/scripts/run_s1_dagger_smoke.sh" B7 \
  2>&1 | tee "${LOG_DIR}/launch.log"
