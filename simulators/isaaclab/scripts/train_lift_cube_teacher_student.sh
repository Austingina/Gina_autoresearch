#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

TASK="${TASK:-Isaac-Lift-Cube-Franka-Teacher-Student-v0}"
NUM_ENVS="${NUM_ENVS:-512}"
SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
HEADLESS="${HEADLESS:-1}"
LOG_DIR="${LOG_DIR:-./runs/lift_cube_teacher_student_kl}"

BUFFER_SIZE="${BUFFER_SIZE:-200}"
INIT_STEPS="${INIT_STEPS:-2000}"
SEQ_LEN="${SEQ_LEN:-64}"
GRAD_INTERVAL="${GRAD_INTERVAL:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"

EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-30}"
PHASE_STALL_STEPS="${PHASE_STALL_STEPS:-20}"
PHASE_MIN_PROGRESS_M="${PHASE_MIN_PROGRESS_M:-0.002}"
EARLY_DONE_MIN_SM_STATE="${EARLY_DONE_MIN_SM_STATE:-2}"
KL_ARM_WEIGHT="${KL_ARM_WEIGHT:-1.0}"
KL_GRIPPER_WEIGHT="${KL_GRIPPER_WEIGHT:-1.0}"
TEACHER_GRIPPER_STD="${TEACHER_GRIPPER_STD:-0.02}"
TEACHER_MIN_LIN_SPEED="${TEACHER_MIN_LIN_SPEED:-0.10}"
TEACHER_MIN_ANG_SPEED="${TEACHER_MIN_ANG_SPEED:-0.20}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
ORACLE_DEMO_FILE="${ORACLE_DEMO_FILE:-}"
GRASP_SAMPLE_RATIO="${GRASP_SAMPLE_RATIO:-0.7}"
GRASP_SAMPLE_MIN_PHASE="${GRASP_SAMPLE_MIN_PHASE:-3}"

if [[ ! -d "${ISAACLAB_DIR}" ]]; then
    echo "Isaac Lab directory not found: ${ISAACLAB_DIR}" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ISAACLAB_DIR}"

cmd=(
    ./isaaclab.sh -p scripts/reinforcement_learning/train_lift_cube_teacher_student.py
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --seed "${SEED}"
    --max_iterations "${MAX_ITERATIONS}"
    --log_dir "${LOG_DIR}"
    --buffer_size "${BUFFER_SIZE}"
    --init_steps "${INIT_STEPS}"
    --seq_len "${SEQ_LEN}"
    --grad_interval "${GRAD_INTERVAL}"
    --episode_length_s "${EPISODE_LENGTH_S}"
    --phase_stall_steps "${PHASE_STALL_STEPS}"
    --phase_min_progress_m "${PHASE_MIN_PROGRESS_M}"
    --early_done_min_sm_state "${EARLY_DONE_MIN_SM_STATE}"
    --kl_arm_weight "${KL_ARM_WEIGHT}"
    --kl_gripper_weight "${KL_GRIPPER_WEIGHT}"
    --teacher_gripper_std "${TEACHER_GRIPPER_STD}"
    --teacher_min_lin_speed "${TEACHER_MIN_LIN_SPEED}"
    --teacher_min_ang_speed "${TEACHER_MIN_ANG_SPEED}"
    --grasp_sample_ratio "${GRASP_SAMPLE_RATIO}"
    --grasp_sample_min_phase "${GRASP_SAMPLE_MIN_PHASE}"
    --eval_interval "${EVAL_INTERVAL}"
    --eval_episodes "${EVAL_EPISODES}"
    --num_eval_envs "${NUM_EVAL_ENVS}"
)

if [[ -n "${ORACLE_DEMO_FILE}" ]]; then
    cmd+=(--oracle_demo_file "${ORACLE_DEMO_FILE}")
    cmd+=(--oracle_demo_max_episodes "${ORACLE_DEMO_MAX_EPISODES}")
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    cmd+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

if [[ "${HEADLESS}" != "0" ]]; then
    cmd+=(--headless)
fi

cmd+=("$@")

echo "Running teacher-student KL training:"
printf ' %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
