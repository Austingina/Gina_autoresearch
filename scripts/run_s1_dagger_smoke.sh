#!/usr/bin/env bash
# S1 DAgger — IsaacLab vertical-free lift, 240px camera, adaptive β.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_ROOT:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

MODE="${1:-B1}"  # B1, B3, or B7
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/s1_${MODE}_${RUN_TAG}}"
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
IMAGE_SIZE="${IMAGE_SIZE:-240}"
CAMERA_SIZE="${CAMERA_SIZE:-240}"
TOPDOWN_CLOSE="${TOPDOWN_CLOSE:-1}"
ROBOT_X_OFFSET="${ROBOT_X_OFFSET:--0.10}"
PANDA_JOINT6="${PANDA_JOINT6:-2.7}"
RANDOMIZE_CUBE_Z_ONLY="${RANDOMIZE_CUBE_Z_ONLY:-0}"
CUBE_Z_MIN="${CUBE_Z_MIN:-0.0}"
CUBE_Z_MAX="${CUBE_Z_MAX:-0.0}"
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
BETA_EVAL_METRIC="${BETA_EVAL_METRIC:-success_5cm}"
G_AUG_NOISE_STD="${G_AUG_NOISE_STD:-0.0}"
G_AUG_DROPOUT_PROB="${G_AUG_DROPOUT_PROB:-0.0}"
G_AUG_BLEND_GT_PROB="${G_AUG_BLEND_GT_PROB:-0.0}"
G_AUG_REACH_ONLY="${G_AUG_REACH_ONLY:-1}"
MAX_STORE_EPISODE_STEPS="${MAX_STORE_EPISODE_STEPS:-512}"
EVAL_PHASE_STALL_STEPS="${EVAL_PHASE_STALL_STEPS:-60}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-4}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RESUME_BUFFER="${RESUME_BUFFER:-}"
NO_RESUME_BUFFER="${NO_RESUME_BUFFER:-1}"

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
else
  common+=(--no-randomize_cube_z_only --randomize_cube)
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
if [[ -n "${AGGREGATE_STORAGE_DIR}" ]]; then
  common+=(--aggregate_storage_dir "${AGGREGATE_STORAGE_DIR}")
fi

if [[ "${BETA_ADAPTIVE}" == "1" ]]; then
  beta_flags=(--beta_adaptive)
else
  beta_flags=(--no-beta_adaptive)
fi

resume_flags=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  resume_flags+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
  if [[ -n "${RESUME_BUFFER}" ]]; then
    resume_flags+=(--resume_buffer "${RESUME_BUFFER}")
  elif [[ "${NO_RESUME_BUFFER}" == "1" ]]; then
    resume_flags+=(--no_resume_buffer)
  fi
fi

cube_dr_msg="xy[x=-0.1..0.1,y=-0.25..0.25,z=0] (G_θ clean240)"
[[ "${RANDOMIZE_CUBE_Z_ONLY}" == "1" ]] && cube_dr_msg="z_only[${CUBE_Z_MIN},${CUBE_Z_MAX}]"

if [[ "${MODE}" == "B1" ]]; then
  echo "=== S1 B1 vision-only (8ch) -> ${LOG_DIR} ==="
  echo "    iters=${MAX_ITERS} envs=${NUM_ENVS} collect=${COLLECT_STEPS} buf_cap=${MAX_AGGREGATE_EPISODES}"
  echo "    cube_dr=${cube_dr_msg} camera=${CAMERA_SIZE}px topdown_close=${TOPDOWN_CLOSE} image_size=${IMAGE_SIZE} robot_x_offset=${ROBOT_X_OFFSET} panda_joint6=${PANDA_JOINT6}"
  echo "    eval: every ${EVAL_INTERVAL} iters x ${EVAL_EPISODES} eps (${NUM_EVAL_ENVS} envs)"
  echo "    beta: warmup=${BETA_WARMUP_ITERS} adaptive=${BETA_ADAPTIVE} target=${BETA_SUCCESS_TARGET}"
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
  echo "    iters=${MAX_ITERS} envs=${NUM_ENVS} collect=${COLLECT_STEPS} buf_cap=${MAX_AGGREGATE_EPISODES}"
  echo "    cube_dr=${cube_dr_msg} camera=${CAMERA_SIZE}px topdown_close=${TOPDOWN_CLOSE} image_size=${IMAGE_SIZE} robot_x_offset=${ROBOT_X_OFFSET} panda_joint6=${PANDA_JOINT6}"
  echo "    eval: every ${EVAL_INTERVAL} iters x ${EVAL_EPISODES} eps (${NUM_EVAL_ENVS} envs)"
  echo "    beta: warmup=${BETA_WARMUP_ITERS} adaptive=${BETA_ADAPTIVE} target=${BETA_SUCCESS_TARGET}"
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
  echo "    iters=${MAX_ITERS} envs=${NUM_ENVS} collect=${COLLECT_STEPS} buf_cap=${MAX_AGGREGATE_EPISODES}"
  echo "    cube_dr=${cube_dr_msg} camera=${CAMERA_SIZE}px topdown_close=${TOPDOWN_CLOSE} image_size=${IMAGE_SIZE} robot_x_offset=${ROBOT_X_OFFSET} panda_joint6=${PANDA_JOINT6}"
  echo "    eval: every ${EVAL_INTERVAL} iters x ${EVAL_EPISODES} eps (${NUM_EVAL_ENVS} envs)"
  echo "    beta: warmup=${BETA_WARMUP_ITERS} adaptive=${BETA_ADAPTIVE} target=${BETA_SUCCESS_TARGET}"
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
