#!/usr/bin/env bash
# S1 B1 mosaic + fast teacher (aligned with play_teacher.sh).
#
# - Teacher speeds / geometry match scripts/play_teacher.sh
# - success_early_done: episode ends on lift success (~80-100 steps)
# - buffer_success_only: whole successful episodes only (no partial segments)
# - partial_rollout_cap=0: disable mid-rollout segment flush
#
# Compare against prior mosaic run:
#   STAGE=s1 CHECKPOINT=runs/s1_B1_mosaic_*/student_best.pt bash scripts/play_student.sh B1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/simulators/resolve_paths.sh"
RUN_TAG="${RUN_TAG:-mosaic_fast_teacher_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/s1_B1_${RUN_TAG}}"
mkdir -p "${LOG_DIR}"

# Resume-friendly defaults (override by exporting env vars before launch).
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  BETA_ADAPTIVE="${BETA_ADAPTIVE:-0}"
  BETA_GRASP_GATED="${BETA_GRASP_GATED:-1}"
  NO_RESUME_BUFFER="${NO_RESUME_BUFFER:-0}"
  _resume_ckpt="${RESUME_CHECKPOINT}"
  [[ "${_resume_ckpt}" != /* ]] && _resume_ckpt="${ROOT}/${_resume_ckpt#./}"
  RESUME_CHECKPOINT="${_resume_ckpt}"
  if [[ -z "${RESUME_BUFFER:-}" && "${NO_RESUME_BUFFER}" != "1" ]]; then
    if [[ -f "$(dirname "${_resume_ckpt}")/dagger_aggregate.pt" ]]; then
      RESUME_BUFFER="$(dirname "${_resume_ckpt}")/dagger_aggregate.pt"
    fi
  fi
fi

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
  NUM_ENVS="${NUM_ENVS:-8}" \
  COLLECT_STEPS="${COLLECT_STEPS:-256}" \
  MAX_AGGREGATE_EPISODES="${MAX_AGGREGATE_EPISODES:-1500}" \
  IMAGE_SIZE="${IMAGE_SIZE:-240}" \
  CAMERA_SIZE="${CAMERA_SIZE:-240}" \
  TOPDOWN_CLOSE="${TOPDOWN_CLOSE:-1}" \
  RANDOMIZE_CUBE="${RANDOMIZE_CUBE:-1}" \
  RANDOMIZE_CUBE_Z_ONLY="${RANDOMIZE_CUBE_Z_ONLY:-0}" \
  CUBE_X_MIN="${CUBE_X_MIN:--0.1}" \
  CUBE_X_MAX="${CUBE_X_MAX:-0.1}" \
  CUBE_Y_MIN="${CUBE_Y_MIN:--0.25}" \
  CUBE_Y_MAX="${CUBE_Y_MAX:-0.25}" \
  CUBE_Z_MIN="${CUBE_Z_MIN:-0.0}" \
  CUBE_Z_MAX="${CUBE_Z_MAX:-0.0}" \
  LIFT_GOAL_ABOVE_CUBE="${LIFT_GOAL_ABOVE_CUBE:-1}" \
  LIFT_HEIGHT_ABOVE_CUBE="${LIFT_HEIGHT_ABOVE_CUBE:-0.20}" \
  ROBOT_X_OFFSET="${ROBOT_X_OFFSET:--0.10}" \
  PANDA_JOINT6="${PANDA_JOINT6:-2.7}" \
  VISUAL_LAYOUT="${VISUAL_LAYOUT:-mosaic}" \
  MOSAIC_HISTORY_LEN="${MOSAIC_HISTORY_LEN:-1}" \
  MASK_PALETTE_MODE="${MASK_PALETTE_MODE:-random}" \
  MASK_ROBOT_COLOR="${MASK_ROBOT_COLOR:-0.10 0.35 0.95}" \
  MASK_TABLE_COLOR="${MASK_TABLE_COLOR:-0.55 0.35 0.18}" \
  MASK_GROUND_COLOR="${MASK_GROUND_COLOR:-0.10 0.10 0.10}" \
  DEBUG_SAVE_MOSAIC_DIR="${DEBUG_SAVE_MOSAIC_DIR:-${LOG_DIR}/mosaic_debug}" \
  DEBUG_SAVE_MOSAIC_INTERVAL="${DEBUG_SAVE_MOSAIC_INTERVAL:-5}" \
  DEBUG_SAVE_MOSAIC_MAX="${DEBUG_SAVE_MOSAIC_MAX:-100}" \
  BETA_ADAPTIVE="${BETA_ADAPTIVE:-1}" \
  BETA_GRASP_GATED="${BETA_GRASP_GATED:-0}" \
  BETA_GRASP_FLOOR="${BETA_GRASP_FLOOR:-0.6}" \
  BETA_GRASP_END_ITER="${BETA_GRASP_END_ITER:-2000}" \
  BETA_GRASP_TARGET="${BETA_GRASP_TARGET:-0.50}" \
  BETA_PHASE1_DECAY_ITERS="${BETA_PHASE1_DECAY_ITERS:-2000}" \
  BETA_WARMUP_ITERS="${BETA_WARMUP_ITERS:-50}" \
  BETA_DECAY_ITERS="${BETA_DECAY_ITERS:-450}" \
  BETA_SUCCESS_TARGET="${BETA_SUCCESS_TARGET:-0.5}" \
  BETA_SUCCESS_EMA="${BETA_SUCCESS_EMA:-0.7}" \
  BETA_EVAL_METRIC="${BETA_EVAL_METRIC:-success_5cm}" \
  G_AUG_NOISE_STD="${G_AUG_NOISE_STD:-0}" \
  G_AUG_DROPOUT_PROB="${G_AUG_DROPOUT_PROB:-0}" \
  G_AUG_BLEND_GT_PROB="${G_AUG_BLEND_GT_PROB:-0}" \
  G_AUG_REACH_ONLY="${G_AUG_REACH_ONLY:-1}" \
  MAX_STORE_EPISODE_STEPS="${MAX_STORE_EPISODE_STEPS:-256}" \
  PARTIAL_ROLLOUT_CAP="${PARTIAL_ROLLOUT_CAP:-0}" \
  AUX_CUBE_EE_WEIGHT="${AUX_CUBE_EE_WEIGHT:-0.4}" \
  AUX_CUBE_EE_SCALE="${AUX_CUBE_EE_SCALE:-0.25}" \
  TEACHER_MIN_LIN_SPEED="${TEACHER_MIN_LIN_SPEED:-2.5}" \
  TEACHER_MIN_ANG_SPEED="${TEACHER_MIN_ANG_SPEED:-2.5}" \
  TEACHER_LIN_STD="${TEACHER_LIN_STD:-0.05}" \
  TEACHER_ANG_STD="${TEACHER_ANG_STD:-0.10}" \
  TEACHER_MAX_LIN_VEL="${TEACHER_MAX_LIN_VEL:-4.5}" \
  TEACHER_MAX_ANG_VEL="${TEACHER_MAX_ANG_VEL:-4.0}" \
  TEACHER_GRASP_HOLD_STEPS="${TEACHER_GRASP_HOLD_STEPS:-2}" \
  TEACHER_ABOVE_OFFSET_Z="${TEACHER_ABOVE_OFFSET_Z:-0.04}" \
  GRIPPER_BCE_WEIGHT="${GRIPPER_BCE_WEIGHT:-1.0}" \
  ARM_SCALE="${ARM_SCALE:-1.0}" \
  GOAL_THRESHOLD="${GOAL_THRESHOLD:-0.025}" \
  SUCCESS_EARLY_DONE="${SUCCESS_EARLY_DONE:-1}" \
  BUFFER_SUCCESS_ONLY="${BUFFER_SUCCESS_ONLY:-1}" \
  BETA_START="${BETA_START:-1.0}" \
  BETA_END="${BETA_END:-0.0}" \
  NO_EVAL="${NO_EVAL:-0}" \
  MIN_BUFFER_EPISODES="${MIN_BUFFER_EPISODES:-8}" \
  EVAL_PHASE_STALL_STEPS="${EVAL_PHASE_STALL_STEPS:-60}" \
  EVAL_INTERVAL="${EVAL_INTERVAL:-50}" \
  EVAL_EPISODES="${EVAL_EPISODES:-20}" \
  NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-4}" \
  NO_RESUME_BUFFER="${NO_RESUME_BUFFER:-1}" \
  BUFFER_SAVE_INTERVAL="${BUFFER_SAVE_INTERVAL:-0}" \
  AGGREGATE_BACKEND="${AGGREGATE_BACKEND:-disk}" \
  AGGREGATE_CACHE_SIZE="${AGGREGATE_CACHE_SIZE:-4}" \
  AGGREGATE_VISUAL_STORAGE_DTYPE="${AGGREGATE_VISUAL_STORAGE_DTYPE:-float16}" \
  RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}" \
  RESUME_BUFFER="${RESUME_BUFFER:-}" \
  bash "${ROOT}/scripts/run_s1_dagger_smoke.sh" B1 \
  2>&1 | tee "${LOG_DIR}/launch.log"
