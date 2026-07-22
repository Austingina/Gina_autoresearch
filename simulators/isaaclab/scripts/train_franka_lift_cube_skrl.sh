#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
TASK="${TASK:-Motrix-Lift-Cube-Franka-v0}"
NUM_ENVS="${NUM_ENVS:-32768}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
ML_FRAMEWORK="${ML_FRAMEWORK:-torch}"
HEADLESS="${HEADLESS:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-7500}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found in PATH." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

cmd=(
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py
    --task "${TASK}"
    --ml_framework "${ML_FRAMEWORK}"
    --num_envs "${NUM_ENVS}"
    --seed "${SEED}"
    --device "${DEVICE}"
)

if [[ "${HEADLESS}" != "0" ]]; then
    cmd+=(--headless)
fi

if [[ -n "${MAX_ITERATIONS}" ]]; then
    cmd+=(--max_iterations "${MAX_ITERATIONS}")
fi

cmd+=("$@")

echo "Running Isaac Lab training:"
printf ' %q' "${cmd[@]}"
echo

exec env TERM="${TERM:-xterm}" "${cmd[@]}"
