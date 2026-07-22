#!/usr/bin/env bash
# Play/visualize proprio dual-head DAgger student checkpoint.
# Vision DAgger (lift_cube_dagger_vision) -> play_lift_cube_student_vision_binary_gripper.sh
# Offline CQL -> play_lift_cube_offline_cql_lstm.sh

set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

CHECKPOINT="${CHECKPOINT:-./runs/lift_cube_ts_binary_gripper_bce/student_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/student_binary_gripper_preview}"
HEADLESS="${HEADLESS:-0}"
MAX_EPISODES="${MAX_EPISODES:-20}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-30}"
LOOP="${LOOP:-1}"
RANDOMIZE_CUBE="${RANDOMIZE_CUBE:-0}"
RANDOMIZE_LIFT_GOAL="${RANDOMIZE_LIFT_GOAL:-0}"
FIXED_LIFT_GOAL="${FIXED_LIFT_GOAL:-1}"
LIFT_GOAL_X="${LIFT_GOAL_X:-0.5}"
LIFT_GOAL_Y="${LIFT_GOAL_Y:-0.0}"
LIFT_GOAL_Z="${LIFT_GOAL_Z:-0.375}"

if [[ ! -d "${ISAACLAB_DIR}" ]]; then
    echo "Isaac Lab directory not found: ${ISAACLAB_DIR}" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ISAACLAB_DIR}"

cmd=(
    ./isaaclab.sh -p scripts/tools/play_lift_cube_student_binary_gripper.py
    --checkpoint "${CHECKPOINT}"
    --num_envs 1
    --output_dir "${OUTPUT_DIR}"
    --max_episodes "${MAX_EPISODES}"
    --episode_length_s "${EPISODE_LENGTH_S}"
    --lift_goal_x "${LIFT_GOAL_X}"
    --lift_goal_y "${LIFT_GOAL_Y}"
    --lift_goal_z "${LIFT_GOAL_Z}"
)

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

if [[ "${LOOP}" != "0" ]]; then
    cmd+=(--loop)
fi

if [[ "${HEADLESS}" != "0" ]]; then
    cmd+=(--headless)
fi

cmd+=("$@")

echo "Playing dual-head binary-gripper student checkpoint:"
printf ' %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
