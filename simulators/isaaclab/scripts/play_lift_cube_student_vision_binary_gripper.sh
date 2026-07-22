#!/usr/bin/env bash
# Play/visualize vision DAgger student checkpoint (mask+depth CNN+LSTM).
# Proprio DAgger (lift_cube_dagger_binary) -> play_lift_cube_student_binary_gripper.sh
# Offline CQL -> play_lift_cube_offline_cql_lstm.sh

set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

CHECKPOINT="${CHECKPOINT:-./runs/lift_cube_dagger_vision/student_best.pt}"
TASK="${TASK:-Isaac-Lift-Cube-Franka-Vision-Teacher-Student-v0}"
NUM_ENVS="${NUM_ENVS:-1}"
HEADLESS="${HEADLESS:-0}"
MAX_EPISODES="${MAX_EPISODES:-20}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ISAACLAB_DIR}"

cmd=(
    ./isaaclab.sh -p scripts/tools/play_lift_cube_student_vision_binary_gripper.py
    --checkpoint "${CHECKPOINT}"
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --max_episodes "${MAX_EPISODES}"
)

if [[ "${HEADLESS}" != "0" ]]; then
    cmd+=(--headless)
fi

cmd+=("$@")
exec "${cmd[@]}"
