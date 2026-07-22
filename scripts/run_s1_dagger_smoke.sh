#!/usr/bin/env bash
# S1 DAgger — IsaacLab vertical-free lift, 240px camera, adaptive β.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_ROOT:-$HOME/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

MODE="${1:-B1}"  # B1, B3, or B7
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/s1_${MODE}_${RUN_TAG}}"
[[ "${LOG_DIR}" != /* ]] && LOG_DIR="${ROOT}/${LOG_DIR#./}"
FIELD_CKPT="${FIELD_CKPT:-$ROOT/checkpoints/lift/guidance_field_s1_random_cube.pt}"

NUM_ENVS="${NUM_ENVS:-8}"
COLLECT_STEPS="${COLLECT_STEPS:-256}"
MIN_BUFFER_EPISODES="${MIN_BUFFER_EPISODES:-8}"
MAX_AGGREGATE_EPISODES="${MAX_AGGREGATE_EPISODES:-250}"
MAX_ITERS="${MAX_ITERS:-2000}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-25}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OPTIM_STEPS="${OPTIM_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
AUX_CUBE_EE_WEIGHT="${AUX_CUBE_EE_WEIGHT:-0.4}"
AUX_CUBE_EE_SCALE="${AUX_CUBE_EE_SCALE:-0.25}"
IMAGE_SIZE="${IMAGE_SIZE:-240}"
CAMERA_SIZE="${CAMERA_SIZE:-240}"
TOPDOWN_CLOSE="${TOPDOWN_CLOSE:-1}"
ROBOT_X_OFFSET="${ROBOT_X_OFFSET:--0.10}"
PANDA_JOINT6="${PANDA_JOINT6:-2.7}"
RANDOMIZE_CUBE="${RANDOMIZE_CUBE:-0}"
RANDOMIZE_CUBE_Z_ONLY="${RANDOMIZE_CUBE_Z_ONLY:-0}"
CUBE_Z_MIN="${CUBE_Z_MIN:-0.0}"
CUBE_Z_MAX="${CUBE_Z_MAX:-0.0}"
CUBE_X_MIN="${CUBE_X_MIN:--0.1}"
CUBE_X_MAX="${CUBE_X_MAX:-0.1}"
CUBE_Y_MIN="${CUBE_Y_MIN:--0.25}"
CUBE_Y_MAX="${CUBE_Y_MAX:-0.25}"
LIFT_GOAL_ABOVE_CUBE="${LIFT_GOAL_ABOVE_CUBE:-1}"
LIFT_HEIGHT_ABOVE_CUBE="${LIFT_HEIGHT_ABOVE_CUBE:-0.20}"
BUFFER_SAVE_INTERVAL="${BUFFER_SAVE_INTERVAL:-0}"
AGGREGATE_BACKEND="${AGGREGATE_BACKEND:-ram}"
AGGREGATE_STORAGE_DIR="${AGGREGATE_STORAGE_DIR:-}"
AGGREGATE_CACHE_SIZE="${AGGREGATE_CACHE_SIZE:-8}"
AGGREGATE_VISUAL_STORAGE_DTYPE="${AGGREGATE_VISUAL_STORAGE_DTYPE:-float16}"
BETA_WARMUP_ITERS="${BETA_WARMUP_ITERS:-50}"
BETA_DECAY_ITERS="${BETA_DECAY_ITERS:-450}"
BETA_ADAPTIVE="${BETA_ADAPTIVE:-1}"
BETA_SUCCESS_TARGET="${BETA_SUCCESS_TARGET:-0.5}"
BETA_SUCCESS_EMA="${BETA_SUCCESS_EMA:-0.7}"
BETA_EVAL_METRIC="${BETA_EVAL_METRIC:-success_5cm}"
G_AUG_NOISE_STD="${G_AUG_NOISE_STD:-0.0}"
G_AUG_DROPOUT_PROB="${G_AUG_DROPOUT_PROB:-0.0}"
G_AUG_BLEND_GT_PROB="${G_AUG_BLEND_GT_PROB:-0.0}"
G_AUG_REACH_ONLY="${G_AUG_REACH_ONLY:-1}"
MAX_STORE_EPISODE_STEPS="${MAX_STORE_EPISODE_STEPS:-512}"
PARTIAL_ROLLOUT_CAP="${PARTIAL_ROLLOUT_CAP:--1}"
TEACHER_MIN_LIN_SPEED="${TEACHER_MIN_LIN_SPEED:-0.10}"
TEACHER_MIN_ANG_SPEED="${TEACHER_MIN_ANG_SPEED:-0.20}"
TEACHER_LIN_STD="${TEACHER_LIN_STD:-0.05}"
TEACHER_ANG_STD="${TEACHER_ANG_STD:-0.10}"
TEACHER_MAX_LIN_VEL="${TEACHER_MAX_LIN_VEL:-0.5}"
TEACHER_MAX_ANG_VEL="${TEACHER_MAX_ANG_VEL:-1.0}"
TEACHER_GRASP_HOLD_STEPS="${TEACHER_GRASP_HOLD_STEPS:-15}"
TEACHER_ABOVE_OFFSET_Z="${TEACHER_ABOVE_OFFSET_Z:-0.10}"
GRIPPER_BCE_WEIGHT="${GRIPPER_BCE_WEIGHT:-1.0}"
ARM_SCALE="${ARM_SCALE:-0.5}"
GOAL_THRESHOLD="${GOAL_THRESHOLD:-0.02}"
SUCCESS_EARLY_DONE="${SUCCESS_EARLY_DONE:-0}"
BUFFER_SUCCESS_ONLY="${BUFFER_SUCCESS_ONLY:-0}"
BETA_START="${BETA_START:-1.0}"
BETA_END="${BETA_END:-0.0}"
BETA_GRASP_GATED="${BETA_GRASP_GATED:-0}"
BETA_GRASP_FLOOR="${BETA_GRASP_FLOOR:-0.6}"
BETA_GRASP_END_ITER="${BETA_GRASP_END_ITER:-2000}"
BETA_GRASP_TARGET="${BETA_GRASP_TARGET:-0.50}"
BETA_PHASE1_DECAY_ITERS="${BETA_PHASE1_DECAY_ITERS:-2000}"
NO_EVAL="${NO_EVAL:-0}"
EVAL_PHASE_STALL_STEPS="${EVAL_PHASE_STALL_STEPS:-60}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-4}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RESUME_BUFFER="${RESUME_BUFFER:-}"
VISUAL_LAYOUT="${VISUAL_LAYOUT:-channels}"
MOSAIC_HISTORY_LEN="${MOSAIC_HISTORY_LEN:-1}"
DEBUG_SAVE_MOSAIC_DIR="${DEBUG_SAVE_MOSAIC_DIR:-}"
DEBUG_SAVE_MOSAIC_INTERVAL="${DEBUG_SAVE_MOSAIC_INTERVAL:-10}"
DEBUG_SAVE_MOSAIC_MAX="${DEBUG_SAVE_MOSAIC_MAX:-200}"
DEBUG_SAVE_MOSAIC_ENV_ID="${DEBUG_SAVE_MOSAIC_ENV_ID:-0}"
MASK_PALETTE_MODE="${MASK_PALETTE_MODE:-fixed}"
MASK_ROBOT_COLOR="${MASK_ROBOT_COLOR:-0.10 0.35 0.95}"
MASK_TABLE_COLOR="${MASK_TABLE_COLOR:-0.55 0.35 0.18}"
MASK_GROUND_COLOR="${MASK_GROUND_COLOR:-0.10 0.10 0.10}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export TERM="${TERM:-xterm-256color}"
export PYTHONUNBUFFERED=1
export AUTORESEARCH_ROOT="${ROOT}"
cd "${ISAACLAB_DIR}"
if [[ "${TERM:-}" == dumb ]]; then
  export TERM=xterm-256color
fi

common=(
  --headless
  --num_envs "${NUM_ENVS}"
  --max_iterations "${MAX_ITERS}"
  --collect_steps "${COLLECT_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --optim_steps "${OPTIM_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --aux_cube_ee_weight "${AUX_CUBE_EE_WEIGHT}"
  --aux_cube_ee_scale "${AUX_CUBE_EE_SCALE}"
  --min_buffer_episodes "${MIN_BUFFER_EPISODES}"
  --max_aggregate_episodes "${MAX_AGGREGATE_EPISODES}"
  --episode_length_s "${EPISODE_LENGTH_S}"
  --eval_episodes "${EVAL_EPISODES}"
  --eval_interval "${EVAL_INTERVAL}"
  --num_eval_envs "${NUM_EVAL_ENVS}"
  --save_interval 100
  --buffer_save_interval "${BUFFER_SAVE_INTERVAL}"
  --aggregate_backend "${AGGREGATE_BACKEND}"
  --aggregate_cache_size "${AGGREGATE_CACHE_SIZE}"
  --aggregate_visual_storage_dtype "${AGGREGATE_VISUAL_STORAGE_DTYPE}"
  --max_store_episode_steps "${MAX_STORE_EPISODE_STEPS}"
  --partial_rollout_cap "${PARTIAL_ROLLOUT_CAP}"
  --teacher_min_lin_speed "${TEACHER_MIN_LIN_SPEED}"
  --teacher_min_ang_speed "${TEACHER_MIN_ANG_SPEED}"
  --teacher_lin_std "${TEACHER_LIN_STD}"
  --teacher_ang_std "${TEACHER_ANG_STD}"
  --teacher_max_lin_vel "${TEACHER_MAX_LIN_VEL}"
  --teacher_max_ang_vel "${TEACHER_MAX_ANG_VEL}"
  --teacher_grasp_hold_steps "${TEACHER_GRASP_HOLD_STEPS}"
  --teacher_above_offset_z "${TEACHER_ABOVE_OFFSET_Z}"
  --gripper_bce_weight "${GRIPPER_BCE_WEIGHT}"
  --goal_threshold "${GOAL_THRESHOLD}"
  --arm_scale "${ARM_SCALE}"
  --beta_start "${BETA_START}"
  --beta_end "${BETA_END}"
  --eval_phase_stall_steps "${EVAL_PHASE_STALL_STEPS}"
  --seed 42
  --fixed_lift_goal
  --no-randomize_lift_goal
  --lift_height_above_cube "${LIFT_HEIGHT_ABOVE_CUBE}"
  --no-randomize_cube
  --image_size "${IMAGE_SIZE}"
  --camera_height "${CAMERA_SIZE}"
  --camera_width "${CAMERA_SIZE}"
  --robot_x_offset "${ROBOT_X_OFFSET}"
  --panda_joint6 "${PANDA_JOINT6}"
)
if [[ "${TOPDOWN_CLOSE}" == "1" ]]; then
  common+=(--topdown_close)
else
  common+=(--no-topdown_close)
fi
if [[ "${RANDOMIZE_CUBE_Z_ONLY}" == "1" ]]; then
  common+=(--randomize_cube_z_only --cube_z_min "${CUBE_Z_MIN}" --cube_z_max "${CUBE_Z_MAX}")
elif [[ "${RANDOMIZE_CUBE}" == "1" ]]; then
  common+=(
    --no-randomize_cube_z_only
    --randomize_cube
    --cube_x_min "${CUBE_X_MIN}"
    --cube_x_max "${CUBE_X_MAX}"
    --cube_y_min "${CUBE_Y_MIN}"
    --cube_y_max "${CUBE_Y_MAX}"
  )
else
  common+=(--no-randomize_cube_z_only --no-randomize_cube)
fi
if [[ "${LIFT_GOAL_ABOVE_CUBE}" == "1" ]]; then
  common+=(--lift_goal_above_cube)
else
  common+=(--no-lift_goal_above_cube)
fi
common+=(
  --autoresearch_root "${ROOT}"
  --log_dir "${LOG_DIR}"
  --beta_warmup_iters "${BETA_WARMUP_ITERS}"
  --beta_decay_iters "${BETA_DECAY_ITERS}"
  --beta_success_target "${BETA_SUCCESS_TARGET}"
  --beta_success_ema "${BETA_SUCCESS_EMA}"
  --beta_eval_metric "${BETA_EVAL_METRIC}"
  --g_aug_noise_std "${G_AUG_NOISE_STD}"
  --g_aug_dropout_prob "${G_AUG_DROPOUT_PROB}"
  --g_aug_blend_gt_prob "${G_AUG_BLEND_GT_PROB}"
)
if [[ "${G_AUG_REACH_ONLY}" == "1" ]]; then
  common+=(--g_aug_reach_only)
else
  common+=(--no-g_aug_reach_only)
fi
common+=(
  --visual_layout "${VISUAL_LAYOUT}"
  --mosaic_history_len "${MOSAIC_HISTORY_LEN}"
  --mask_palette_mode "${MASK_PALETTE_MODE}"
  --mask_robot_color ${MASK_ROBOT_COLOR}
  --mask_table_color ${MASK_TABLE_COLOR}
  --mask_ground_color ${MASK_GROUND_COLOR}
)
if [[ -n "${DEBUG_SAVE_MOSAIC_DIR}" ]]; then
  common+=(
    --debug_save_mosaic_dir "${DEBUG_SAVE_MOSAIC_DIR}"
    --debug_save_mosaic_interval "${DEBUG_SAVE_MOSAIC_INTERVAL}"
    --debug_save_mosaic_max "${DEBUG_SAVE_MOSAIC_MAX}"
    --debug_save_mosaic_env_id "${DEBUG_SAVE_MOSAIC_ENV_ID}"
  )
fi
if [[ -n "${AGGREGATE_STORAGE_DIR}" ]]; then
  common+=(--aggregate_storage_dir "${AGGREGATE_STORAGE_DIR}")
fi
if [[ "${NO_EVAL}" == "1" ]]; then
  common+=(--no_eval)
fi
if [[ "${SUCCESS_EARLY_DONE}" == "1" ]]; then
  common+=(--success_early_done)
else
  common+=(--no-success_early_done)
fi
if [[ "${BUFFER_SUCCESS_ONLY}" == "1" ]]; then
  common+=(--buffer_success_only)
else
  common+=(--no-buffer_success_only)
fi

if [[ "${BETA_ADAPTIVE}" == "1" ]]; then
  beta_flags=(--beta_adaptive)
else
  beta_flags=(--no-beta_adaptive)
fi
if [[ "${BETA_GRASP_GATED}" == "1" ]]; then
  beta_flags=(
    --no-beta_adaptive
    --beta_grasp_gated
    --beta_grasp_floor "${BETA_GRASP_FLOOR}"
    --beta_grasp_end_iter "${BETA_GRASP_END_ITER}"
    --beta_grasp_target "${BETA_GRASP_TARGET}"
    --beta_phase1_decay_iters "${BETA_PHASE1_DECAY_ITERS}"
  )
fi

resume_flags=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  _resume_ckpt="${RESUME_CHECKPOINT}"
  [[ "${_resume_ckpt}" != /* ]] && _resume_ckpt="${ROOT}/${_resume_ckpt#./}"
  RESUME_CHECKPOINT="${_resume_ckpt}"
  resume_flags+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
  if [[ -n "${RESUME_BUFFER}" ]]; then
    _resume_buf="${RESUME_BUFFER}"
    [[ "${_resume_buf}" != /* ]] && _resume_buf="${ROOT}/${_resume_buf#./}"
    RESUME_BUFFER="${_resume_buf}"
    resume_flags+=(--resume_buffer "${RESUME_BUFFER}")
  elif [[ "${NO_RESUME_BUFFER}" == "1" ]]; then
    resume_flags+=(--no_resume_buffer)
  fi
fi

cube_dr_msg="off (fixed spawn)"
[[ "${RANDOMIZE_CUBE}" == "1" ]] && cube_dr_msg="xy[x=${CUBE_X_MIN}..${CUBE_X_MAX},y=${CUBE_Y_MIN}..${CUBE_Y_MAX},z=0]"
[[ "${RANDOMIZE_CUBE_Z_ONLY}" == "1" ]] && cube_dr_msg="z_only[${CUBE_Z_MIN},${CUBE_Z_MAX}]"

if [[ "${MODE}" == "B1" ]]; then
  echo "=== S1 B1 vision-only (${VISUAL_LAYOUT}) -> ${LOG_DIR} ==="
  echo "    iters=${MAX_ITERS} envs=${NUM_ENVS} collect=${COLLECT_STEPS} buf_cap=${MAX_AGGREGATE_EPISODES}"
  echo "    visual_layout=${VISUAL_LAYOUT} mosaic_history_len=${MOSAIC_HISTORY_LEN}"
  echo "    mask_palette=${MASK_PALETTE_MODE} robot=(${MASK_ROBOT_COLOR}) table=(${MASK_TABLE_COLOR}) ground=(${MASK_GROUND_COLOR})"
  echo "    cube_dr=${cube_dr_msg} camera=${CAMERA_SIZE}px topdown_close=${TOPDOWN_CLOSE} image_size=${IMAGE_SIZE} robot_x_offset=${ROBOT_X_OFFSET} panda_joint6=${PANDA_JOINT6}"
  echo "    eval: every ${EVAL_INTERVAL} iters x ${EVAL_EPISODES} eps (${NUM_EVAL_ENVS} envs)"
  echo "    beta: warmup=${BETA_WARMUP_ITERS} adaptive=${BETA_ADAPTIVE} target=${BETA_SUCCESS_TARGET}"
  echo "    aux: cube_ee_weight=${AUX_CUBE_EE_WEIGHT} cube_ee_scale=${AUX_CUBE_EE_SCALE}m"
  echo "    g_aug: noise=${G_AUG_NOISE_STD} dropout=${G_AUG_DROPOUT_PROB} blend_gt=${G_AUG_BLEND_GT_PROB} reach_only=${G_AUG_REACH_ONLY}"
  [[ -n "${RESUME_CHECKPOINT}" ]] && echo "    resume=${RESUME_CHECKPOINT} (no_buffer=${NO_RESUME_BUFFER})"
  ./isaaclab.sh -p scripts/reinforcement_learning/train_lift_cube_dagger_vision_binary_gripper.py \
    "${common[@]}" \
    "${beta_flags[@]}" \
    "${resume_flags[@]}" \
    --guidance_mode anchor_only \
    --visual_channels 8
elif [[ "${MODE}" == "B3" ]]; then
  echo "=== S1 B3 vision+GT G (11ch) -> ${LOG_DIR} ==="
  echo "    guidance=anchor_field_rule (oracle 3D subgoal, no G_θ checkpoint)"
  echo "    mask_palette=${MASK_PALETTE_MODE} robot=(${MASK_ROBOT_COLOR}) table=(${MASK_TABLE_COLOR}) ground=(${MASK_GROUND_COLOR})"
  echo "    iters=${MAX_ITERS} envs=${NUM_ENVS} collect=${COLLECT_STEPS} buf_cap=${MAX_AGGREGATE_EPISODES}"
  echo "    cube_dr=${cube_dr_msg} camera=${CAMERA_SIZE}px topdown_close=${TOPDOWN_CLOSE} image_size=${IMAGE_SIZE} robot_x_offset=${ROBOT_X_OFFSET} panda_joint6=${PANDA_JOINT6}"
  echo "    eval: every ${EVAL_INTERVAL} iters x ${EVAL_EPISODES} eps (${NUM_EVAL_ENVS} envs)"
  echo "    beta: warmup=${BETA_WARMUP_ITERS} adaptive=${BETA_ADAPTIVE} target=${BETA_SUCCESS_TARGET}"
  echo "    aux: cube_ee_weight=${AUX_CUBE_EE_WEIGHT} cube_ee_scale=${AUX_CUBE_EE_SCALE}m"
  echo "    g_aug: noise=${G_AUG_NOISE_STD} dropout=${G_AUG_DROPOUT_PROB} blend_gt=${G_AUG_BLEND_GT_PROB} reach_only=${G_AUG_REACH_ONLY}"
  [[ -n "${RESUME_CHECKPOINT}" ]] && echo "    resume=${RESUME_CHECKPOINT} (no_buffer=${NO_RESUME_BUFFER})"
  ./isaaclab.sh -p scripts/reinforcement_learning/train_lift_cube_dagger_vision_binary_gripper.py \
    "${common[@]}" \
    "${beta_flags[@]}" \
    "${resume_flags[@]}" \
    --guidance_mode anchor_field_rule
elif [[ "${MODE}" == "B7" ]]; then
  echo "=== S1 B7 vision+G_theta -> ${LOG_DIR} ==="
  echo "    field=${FIELD_CKPT}"
  echo "    mask_palette=${MASK_PALETTE_MODE} robot=(${MASK_ROBOT_COLOR}) table=(${MASK_TABLE_COLOR}) ground=(${MASK_GROUND_COLOR})"
  echo "    iters=${MAX_ITERS} envs=${NUM_ENVS} collect=${COLLECT_STEPS} buf_cap=${MAX_AGGREGATE_EPISODES}"
  echo "    cube_dr=${cube_dr_msg} camera=${CAMERA_SIZE}px topdown_close=${TOPDOWN_CLOSE} image_size=${IMAGE_SIZE} robot_x_offset=${ROBOT_X_OFFSET} panda_joint6=${PANDA_JOINT6}"
  echo "    eval: every ${EVAL_INTERVAL} iters x ${EVAL_EPISODES} eps (${NUM_EVAL_ENVS} envs)"
  echo "    beta: warmup=${BETA_WARMUP_ITERS} adaptive=${BETA_ADAPTIVE} target=${BETA_SUCCESS_TARGET}"
  echo "    aux: cube_ee_weight=${AUX_CUBE_EE_WEIGHT} cube_ee_scale=${AUX_CUBE_EE_SCALE}m"
  echo "    g_aug: noise=${G_AUG_NOISE_STD} dropout=${G_AUG_DROPOUT_PROB} blend_gt=${G_AUG_BLEND_GT_PROB} reach_only=${G_AUG_REACH_ONLY}"
  [[ -n "${RESUME_CHECKPOINT}" ]] && echo "    resume=${RESUME_CHECKPOINT} (no_buffer=${NO_RESUME_BUFFER})"
  [[ -f "${FIELD_CKPT}" ]] || { echo "Missing ${FIELD_CKPT} — run: python scripts/learn_guidance_field.py --stage s1_random_cube"; exit 1; }
  ./isaaclab.sh -p scripts/reinforcement_learning/train_lift_cube_dagger_vision_binary_gripper.py \
    "${common[@]}" \
    "${beta_flags[@]}" \
    "${resume_flags[@]}" \
    --guidance_mode anchor_field_demo \
    --field_checkpoint "${FIELD_CKPT}"
else
  echo "Usage: $0 B1|B3|B7"
  exit 1
fi
