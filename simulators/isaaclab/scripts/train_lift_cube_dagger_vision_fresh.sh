#!/usr/bin/env bash
# Vision DAgger training for lift-cube dual-head binary-gripper student.
#
# Fresh run (mask + depth, 8 channels @ 64x64, same stack as offline CQL):
#   ./scripts/train_lift_cube_dagger_vision_fresh.sh
#
# Resume (policy only — skip 24GB+ buffer load):
#   RESUME_CHECKPOINT=./runs/lift_cube_dagger_vision/student_iter_300.pt \
#     ./scripts/train_lift_cube_dagger_vision_fresh.sh --no_resume_buffer
#
# Resume policy + buffer (needs ~24GB+ free RAM for dagger_aggregate.pt):
#   RESUME_CHECKPOINT=... RESUME_BUFFER=./runs/lift_cube_dagger_vision/dagger_aggregate.pt \
#     ./scripts/train_lift_cube_dagger_vision_fresh.sh
#
# Optional proprio (7D, no cube GT):
#   OBS_MODE=vision_proprio TASK=Isaac-Lift-Cube-Franka-Vision-Teacher-Student-Proprio-v0 \
#     ./scripts/train_lift_cube_dagger_vision_fresh.sh
#
# Note: ORACLE_DEMO_FILE (proprio HDF5) is NOT supported for vision DAgger.
# Use online aggregate only; maskdepth HDF5 preload is not wired yet.

set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

TASK="${TASK:-Isaac-Lift-Cube-Franka-Vision-Teacher-Student-v0}"
OBS_MODE="${OBS_MODE:-vision}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
VISUAL_CHANNELS="${VISUAL_CHANNELS:-8}"
NUM_ENVS="${NUM_ENVS:-32}"
RENDERING_MODE="${RENDERING_MODE:-performance}"
SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
COLLECT_STEPS="${COLLECT_STEPS:-64}"
HEADLESS="${HEADLESS:-1}"
LOG_DIR="${LOG_DIR:-./runs/lift_cube_dagger_vision}"

RANDOMIZE_CUBE="${RANDOMIZE_CUBE:-0}"
RANDOMIZE_LIFT_GOAL="${RANDOMIZE_LIFT_GOAL:-0}"
if [[ "${RANDOMIZE_CUBE}" != "0" || "${RANDOMIZE_LIFT_GOAL}" != "0" ]]; then
    MAX_AGGREGATE_EPISODES="${MAX_AGGREGATE_EPISODES:-2000}"
else
    MAX_AGGREGATE_EPISODES="${MAX_AGGREGATE_EPISODES:-500}"
fi
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-64}"
GRAD_INTERVAL="${GRAD_INTERVAL:-20}"
OPTIM_STEPS="${OPTIM_STEPS:-8}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-4}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}"
BUFFER_SAVE_INTERVAL="${BUFFER_SAVE_INTERVAL:-200}"

EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-30}"
PHASE_STALL_STEPS="${PHASE_STALL_STEPS:-180}"
PHASE_MIN_PROGRESS_M="${PHASE_MIN_PROGRESS_M:-0.0005}"
EARLY_DONE_MIN_SM_STATE="${EARLY_DONE_MIN_SM_STATE:-2}"

GRIPPER_BCE_WEIGHT="${GRIPPER_BCE_WEIGHT:-1.0}"
GRASP_SAMPLE_RATIO="${GRASP_SAMPLE_RATIO:-0.7}"
GRASP_SAMPLE_MIN_PHASE="${GRASP_SAMPLE_MIN_PHASE:-3}"
TEACHER_MIN_LIN_SPEED="${TEACHER_MIN_LIN_SPEED:-0.10}"
TEACHER_MIN_ANG_SPEED="${TEACHER_MIN_ANG_SPEED:-0.20}"
TEACHER_LIN_STD="${TEACHER_LIN_STD:-0.05}"
TEACHER_ANG_STD="${TEACHER_ANG_STD:-0.10}"

BETA_START="${BETA_START:-1.0}"
BETA_END="${BETA_END:-0.0}"
BETA_WARMUP_ITERS="${BETA_WARMUP_ITERS:-100}"
BETA_DECAY_ITERS="${BETA_DECAY_ITERS:-2000}"
BETA_ADAPTIVE="${BETA_ADAPTIVE:-1}"
BETA_SUCCESS_TARGET="${BETA_SUCCESS_TARGET:-0.5}"
BETA_SUCCESS_EMA="${BETA_SUCCESS_EMA:-0.7}"
SUCCESS_RESERVE_RATIO="${SUCCESS_RESERVE_RATIO:-0.35}"
FIXED_LIFT_GOAL="${FIXED_LIFT_GOAL:-1}"
LIFT_GOAL_X="${LIFT_GOAL_X:-0.5}"
LIFT_GOAL_Y="${LIFT_GOAL_Y:-0.0}"
LIFT_GOAL_Z="${LIFT_GOAL_Z:-0.375}"

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RESUME_BUFFER="${RESUME_BUFFER:-}"

MIN_BUFFER_EPISODES="${MIN_BUFFER_EPISODES:-32}"

if [[ ! -d "${ISAACLAB_DIR}" ]]; then
    echo "Isaac Lab directory not found: ${ISAACLAB_DIR}" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ISAACLAB_DIR}"

cmd=(
    ./isaaclab.sh -p scripts/reinforcement_learning/train_lift_cube_dagger_vision_binary_gripper.py
    --task "${TASK}"
    --obs_mode "${OBS_MODE}"
    --image_size "${IMAGE_SIZE}"
    --visual_channels "${VISUAL_CHANNELS}"
    --rendering_mode "${RENDERING_MODE}"
    --num_envs "${NUM_ENVS}"
    --seed "${SEED}"
    --max_iterations "${MAX_ITERATIONS}"
    --collect_steps "${COLLECT_STEPS}"
    --log_dir "${LOG_DIR}"
    --max_aggregate_episodes "${MAX_AGGREGATE_EPISODES}"
    --min_buffer_episodes "${MIN_BUFFER_EPISODES}"
    --batch_size "${BATCH_SIZE}"
    --seq_len "${SEQ_LEN}"
    --grad_interval "${GRAD_INTERVAL}"
    --optim_steps "${OPTIM_STEPS}"
    --episode_length_s "${EPISODE_LENGTH_S}"
    --phase_stall_steps "${PHASE_STALL_STEPS}"
    --phase_min_progress_m "${PHASE_MIN_PROGRESS_M}"
    --early_done_min_sm_state "${EARLY_DONE_MIN_SM_STATE}"
    --gripper_bce_weight "${GRIPPER_BCE_WEIGHT}"
    --grasp_sample_ratio "${GRASP_SAMPLE_RATIO}"
    --grasp_sample_min_phase "${GRASP_SAMPLE_MIN_PHASE}"
    --teacher_min_lin_speed "${TEACHER_MIN_LIN_SPEED}"
    --teacher_min_ang_speed "${TEACHER_MIN_ANG_SPEED}"
    --teacher_lin_std "${TEACHER_LIN_STD}"
    --teacher_ang_std "${TEACHER_ANG_STD}"
    --beta_start "${BETA_START}"
    --beta_end "${BETA_END}"
    --beta_warmup_iters "${BETA_WARMUP_ITERS}"
    --beta_decay_iters "${BETA_DECAY_ITERS}"
    --beta_success_target "${BETA_SUCCESS_TARGET}"
    --beta_success_ema "${BETA_SUCCESS_EMA}"
    --success_reserve_ratio "${SUCCESS_RESERVE_RATIO}"
    --lift_goal_x "${LIFT_GOAL_X}"
    --lift_goal_y "${LIFT_GOAL_Y}"
    --lift_goal_z "${LIFT_GOAL_Z}"
    --eval_interval "${EVAL_INTERVAL}"
    --eval_episodes "${EVAL_EPISODES}"
    --num_eval_envs "${NUM_EVAL_ENVS}"
    --buffer_save_interval "${BUFFER_SAVE_INTERVAL}"
)

if [[ "${BETA_ADAPTIVE}" != "0" ]]; then
    cmd+=(--beta_adaptive)
else
    cmd+=(--no-beta_adaptive)
fi
if [[ "${FIXED_LIFT_GOAL}" != "0" && "${RANDOMIZE_LIFT_GOAL}" == "0" ]]; then
    cmd+=(--fixed_lift_goal)
else
    cmd+=(--no-fixed_lift_goal)
fi
if [[ "${RANDOMIZE_CUBE}" != "0" ]]; then
    cmd+=(--randomize_cube)
else
    cmd+=(--no-randomize_cube)
fi
if [[ "${RANDOMIZE_LIFT_GOAL}" != "0" ]]; then
    cmd+=(--randomize_lift_goal)
else
    cmd+=(--no-randomize_lift_goal)
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    cmd+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi
if [[ -n "${RESUME_BUFFER}" ]]; then
    cmd+=(--resume_buffer "${RESUME_BUFFER}")
fi
if [[ "${HEADLESS}" != "0" ]]; then
    cmd+=(--headless)
fi

cmd+=("$@")

echo "Vision DAgger lift-cube training:"
printf ' %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
