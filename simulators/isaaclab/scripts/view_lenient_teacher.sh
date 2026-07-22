#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

OUTPUT_DIR="${OUTPUT_DIR:-./runs/lenient_teacher_preview}"
HEADLESS="${HEADLESS:-0}"
MAX_EPISODES="${MAX_EPISODES:-20}"
BINARY_GRIPPER="${BINARY_GRIPPER:-1}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-30}"
TEACHER_MIN_LIN_SPEED="${TEACHER_MIN_LIN_SPEED:-0.10}"
TEACHER_MIN_ANG_SPEED="${TEACHER_MIN_ANG_SPEED:-0.20}"

if [[ ! -d "${ISAACLAB_DIR}" ]]; then
    echo "Isaac Lab directory not found: ${ISAACLAB_DIR}" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ISAACLAB_DIR}"

cmd=(
    ./isaaclab.sh -p scripts/tools/view_lenient_teacher.py
    --num_envs 1
    --output_dir "${OUTPUT_DIR}"
    --max_episodes "${MAX_EPISODES}"
    --episode_length_s "${EPISODE_LENGTH_S}"
    --teacher_min_lin_speed "${TEACHER_MIN_LIN_SPEED}"
    --teacher_min_ang_speed "${TEACHER_MIN_ANG_SPEED}"
)

if [[ "${BINARY_GRIPPER}" != "0" ]]; then
    cmd+=(--binary_gripper)
fi

if [[ "${HEADLESS}" != "0" ]]; then
    cmd+=(--headless)
fi

cmd+=("$@")

echo "Visualizing lenient teacher:"
printf ' %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
