#!/usr/bin/env bash
# Resume S1 B1 from iter-500 checkpoint with eval50 x 10ep profile.
#
# Source run: s1_B1_visible_ee_forward_20260630_094041 (segment-continue buffer on disk).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_RUN="${SRC_RUN:-${ROOT}/runs/s1_B1_visible_ee_forward_20260630_094041}"
RESUME_CKPT="${RESUME_CKPT:-${SRC_RUN}/student_iter_500.pt}"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "Missing resume checkpoint: ${RESUME_CKPT}"
  exit 1
fi

RUN_TAG="${RUN_TAG:-resume500_eval50_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/s1_B1_${RUN_TAG}}"
mkdir -p "${LOG_DIR}"

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
  MAX_ITERS="${MAX_ITERS:-1500}" \
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
  G_AUG_NOISE_STD="${G_AUG_NOISE_STD:-0}" \
  G_AUG_DROPOUT_PROB="${G_AUG_DROPOUT_PROB:-0}" \
  G_AUG_BLEND_GT_PROB="${G_AUG_BLEND_GT_PROB:-0}" \
  G_AUG_REACH_ONLY="${G_AUG_REACH_ONLY:-1}" \
  MAX_STORE_EPISODE_STEPS="${MAX_STORE_EPISODE_STEPS:-512}" \
  EVAL_PHASE_STALL_STEPS="${EVAL_PHASE_STALL_STEPS:-60}" \
  EVAL_INTERVAL="${EVAL_INTERVAL:-50}" \
  EVAL_EPISODES="${EVAL_EPISODES:-10}" \
  NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-2}" \
  NO_RESUME_BUFFER="${NO_RESUME_BUFFER:-0}" \
  RESUME_CHECKPOINT="${RESUME_CKPT}" \
  RESUME_BUFFER="${RESUME_BUFFER:-${SRC_RUN}/dagger_aggregate.pt}" \
  AGGREGATE_STORAGE_DIR="${AGGREGATE_STORAGE_DIR:-${SRC_RUN}/dagger_episodes}" \
  BUFFER_SAVE_INTERVAL="${BUFFER_SAVE_INTERVAL:-0}" \
  AGGREGATE_BACKEND="${AGGREGATE_BACKEND:-disk}" \
  AGGREGATE_CACHE_SIZE="${AGGREGATE_CACHE_SIZE:-4}" \
  AGGREGATE_VISUAL_STORAGE_DTYPE="${AGGREGATE_VISUAL_STORAGE_DTYPE:-float16}" \
  bash "${ROOT}/scripts/run_s1_dagger_smoke.sh" B1 \
  2>&1 | tee "${LOG_DIR}/launch.log"
