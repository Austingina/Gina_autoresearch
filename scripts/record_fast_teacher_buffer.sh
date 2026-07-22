#!/usr/bin/env bash
# Record fast-teacher DAgger buffer episodes (no partial flush, <=60 steps) for OpenCV review.
#
# Pure teacher rollout (β=1), higher EE speeds, one contiguous episode per success/timeout.
# After recording:
#   BUFFER_DIR=<log_dir>/dagger_episodes bash scripts/play_dagger_buffer.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_TAG="${RUN_TAG:-fast_teacher_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/s1_B1_${RUN_TAG}}"
mkdir -p "${LOG_DIR}"

# Fast teacher: ~3-4x default min speeds, higher velocity cap.
TEACHER_MIN_LIN_SPEED="${TEACHER_MIN_LIN_SPEED:-0.35}"
TEACHER_MIN_ANG_SPEED="${TEACHER_MIN_ANG_SPEED:-0.50}"
TEACHER_MAX_LIN_VEL="${TEACHER_MAX_LIN_VEL:-0.85}"
TEACHER_MAX_ANG_VEL="${TEACHER_MAX_ANG_VEL:-1.5}"
ARM_SCALE="${ARM_SCALE:-0.5}"

# One whole episode per buffer entry (no 256/512-step segment flush).
PARTIAL_ROLLOUT_CAP="${PARTIAL_ROLLOUT_CAP:-0}"
MAX_STORE_EPISODE_STEPS="${MAX_STORE_EPISODE_STEPS:-60}"

# Short collect pass: teacher-only labels, skip eval.
MAX_ITERS="${MAX_ITERS:-8}"
MIN_BUFFER_EPISODES="${MIN_BUFFER_EPISODES:-4}"
NUM_ENVS="${NUM_ENVS:-8}"
COLLECT_STEPS="${COLLECT_STEPS:-128}"
MAX_AGGREGATE_EPISODES="${MAX_AGGREGATE_EPISODES:-64}"
BETA_ADAPTIVE="${BETA_ADAPTIVE:-0}"
BETA_WARMUP_ITERS="${BETA_WARMUP_ITERS:-0}"
BETA_DECAY_ITERS="${BETA_DECAY_ITERS:-0}"
BETA_START="${BETA_START:-1.0}"
BETA_END="${BETA_END:-1.0}"
NO_EVAL="${NO_EVAL:-1}"
NO_RESUME_BUFFER="${NO_RESUME_BUFFER:-1}"
AGGREGATE_BACKEND="${AGGREGATE_BACKEND:-disk}"
AGGREGATE_CACHE_SIZE="${AGGREGATE_CACHE_SIZE:-4}"
VISUAL_LAYOUT="${VISUAL_LAYOUT:-mosaic}"
MOSAIC_HISTORY_LEN="${MOSAIC_HISTORY_LEN:-3}"
IMAGE_SIZE="${IMAGE_SIZE:-240}"
CAMERA_SIZE="${CAMERA_SIZE:-240}"
TOPDOWN_CLOSE="${TOPDOWN_CLOSE:-1}"
ROBOT_X_OFFSET="${ROBOT_X_OFFSET:--0.10}"
PANDA_JOINT6="${PANDA_JOINT6:-2.7}"

echo "=== Record fast-teacher DAgger buffer ==="
echo "  log_dir: ${LOG_DIR}"
echo "  teacher: min_lin=${TEACHER_MIN_LIN_SPEED} min_ang=${TEACHER_MIN_ANG_SPEED} max_lin=${TEACHER_MAX_LIN_VEL}"
echo "  buffer: max_store=${MAX_STORE_EPISODE_STEPS} partial_cap=${PARTIAL_ROLLOUT_CAP} (0=off)"
echo "  collect: iters=${MAX_ITERS} envs=${NUM_ENVS} steps/iter=${COLLECT_STEPS} layout=${VISUAL_LAYOUT}"
echo ""
echo "  Visualize when done:"
echo "    BUFFER_DIR=${LOG_DIR}/dagger_episodes bash scripts/play_dagger_buffer.sh"

exec env -i \
  HOME="${HOME}" \
  USER="${USER:-$(id -un)}" \
  PATH="${PATH}" \
  SHELL="${SHELL:-/bin/bash}" \
  TERM="${TERM:-xterm-256color}" \
  OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}" \
  ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/code/IsaacLab}" \
  CONDA_ENV="${CONDA_ENV:-env_isaaclab}" \
  AUTORESEARCH_ROOT="${ROOT}" \
  RUN_TAG="${RUN_TAG}" \
  LOG_DIR="${LOG_DIR}" \
  TEACHER_MIN_LIN_SPEED="${TEACHER_MIN_LIN_SPEED}" \
  TEACHER_MIN_ANG_SPEED="${TEACHER_MIN_ANG_SPEED}" \
  TEACHER_MAX_LIN_VEL="${TEACHER_MAX_LIN_VEL}" \
  TEACHER_MAX_ANG_VEL="${TEACHER_MAX_ANG_VEL}" \
  ARM_SCALE="${ARM_SCALE}" \
  PARTIAL_ROLLOUT_CAP="${PARTIAL_ROLLOUT_CAP}" \
  MAX_STORE_EPISODE_STEPS="${MAX_STORE_EPISODE_STEPS}" \
  MAX_ITERS="${MAX_ITERS}" \
  MIN_BUFFER_EPISODES="${MIN_BUFFER_EPISODES}" \
  NUM_ENVS="${NUM_ENVS}" \
  COLLECT_STEPS="${COLLECT_STEPS}" \
  MAX_AGGREGATE_EPISODES="${MAX_AGGREGATE_EPISODES}" \
  BETA_ADAPTIVE="${BETA_ADAPTIVE}" \
  BETA_WARMUP_ITERS="${BETA_WARMUP_ITERS}" \
  BETA_DECAY_ITERS="${BETA_DECAY_ITERS}" \
  BETA_START="${BETA_START}" \
  BETA_END="${BETA_END}" \
  NO_EVAL="${NO_EVAL}" \
  NO_RESUME_BUFFER="${NO_RESUME_BUFFER}" \
  AGGREGATE_BACKEND="${AGGREGATE_BACKEND}" \
  AGGREGATE_CACHE_SIZE="${AGGREGATE_CACHE_SIZE}" \
  VISUAL_LAYOUT="${VISUAL_LAYOUT}" \
  MOSAIC_HISTORY_LEN="${MOSAIC_HISTORY_LEN}" \
  IMAGE_SIZE="${IMAGE_SIZE}" \
  CAMERA_SIZE="${CAMERA_SIZE}" \
  TOPDOWN_CLOSE="${TOPDOWN_CLOSE}" \
  ROBOT_X_OFFSET="${ROBOT_X_OFFSET}" \
  PANDA_JOINT6="${PANDA_JOINT6}" \
  bash "${ROOT}/scripts/run_s1_dagger_smoke.sh" B1 \
  2>&1 | tee "${LOG_DIR}/launch.log"
