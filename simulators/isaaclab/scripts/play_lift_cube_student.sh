#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

CHECKPOINT="${CHECKPOINT:-./runs/lift_cube_teacher_student_kl/student_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/student_best_preview}"
HEADLESS="${HEADLESS:-0}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-30}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ISAACLAB_DIR}"

cmd=(
    ./isaaclab.sh -p scripts/tools/play_lift_cube_student.py
    --checkpoint "${CHECKPOINT}"
    --num_envs 1
    --output_dir "${OUTPUT_DIR}"
    --episode_length_s "${EPISODE_LENGTH_S}"
    --loop
)

if [[ "${HEADLESS}" != "0" ]]; then
    cmd+=(--headless)
fi

cmd+=("$@")

echo "Playing student checkpoint:"
printf ' %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
