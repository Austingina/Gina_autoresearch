#!/usr/bin/env bash
# Fresh dual-head binary-gripper training with lenient teacher (min-speed arm twist).
#
# Example:
#   ./scripts/train_lift_cube_dual_head_fresh.sh
#
# Resume from a checkpoint instead:
#   RESUME_CHECKPOINT=./runs/lift_cube_ts_dual_head_min_vel_fresh/student_iter_500.pt \
#   ./scripts/train_lift_cube_dual_head_fresh.sh
#
# Visualize teacher first:
#   ./scripts/view_lenient_teacher.sh
#
# Visualize student checkpoint mid-training:
#   CHECKPOINT=./runs/lift_cube_ts_dual_head_min_vel_fresh/student_iter_500.pt \
#   ./scripts/play_lift_cube_student_binary_gripper.sh

set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

TASK="${TASK:-Isaac-Lift-Cube-Franka-Teacher-Student-v0}"
NUM_ENVS="${NUM_ENVS:-256}"
SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
COLLECT_STEPS="${COLLECT_STEPS:-64}"
HEADLESS="${HEADLESS:-1}"
LOG_DIR="${LOG_DIR:-./runs/lift_cube_ts_dual_head_min_vel_fresh}"

BUFFER_SIZE="${BUFFER_SIZE:-200}"
INIT_STEPS="${INIT_STEPS:-2000}"
SEQ_LEN="${SEQ_LEN:-64}"
GRAD_INTERVAL="${GRAD_INTERVAL:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"

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

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
ORACLE_DEMO_FILE="${ORACLE_DEMO_FILE:-}"
ORACLE_DEMO_MAX_EPISODES="${ORACLE_DEMO_MAX_EPISODES:-100}"

if [[ ! -d "${ISAACLAB_DIR}" ]]; then
    echo "Isaac Lab directory not found: ${ISAACLAB_DIR}" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ISAACLAB_DIR}"

cmd=(
    ./isaaclab.sh -p scripts/reinforcement_learning/train_lift_cube_teacher_student_binary_gripper.py
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --seed "${SEED}"
    --max_iterations "${MAX_ITERATIONS}"
    --collect_steps "${COLLECT_STEPS}"
    --log_dir "${LOG_DIR}"
    --buffer_size "${BUFFER_SIZE}"
    --init_steps "${INIT_STEPS}"
    --seq_len "${SEQ_LEN}"
    --grad_interval "${GRAD_INTERVAL}"
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
    --eval_interval "${EVAL_INTERVAL}"
    --eval_episodes "${EVAL_EPISODES}"
    --num_eval_envs "${NUM_EVAL_ENVS}"
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    cmd+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

if [[ -n "${ORACLE_DEMO_FILE}" ]]; then
    cmd+=(--oracle_demo_file "${ORACLE_DEMO_FILE}")
    cmd+=(--oracle_demo_max_episodes "${ORACLE_DEMO_MAX_EPISODES}")
fi

if [[ "${HEADLESS}" != "0" ]]; then
    cmd+=(--headless)
fi

cmd+=("$@")

echo "Dual-head fresh training (lenient teacher + min-speed arm twist):"
printf ' %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
