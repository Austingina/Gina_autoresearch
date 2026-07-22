#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

CHECKPOINT="${CHECKPOINT:-./runs/lift_cube_offline_cql_lstm/cql_sac_lstm/checkpoint_final.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/offline_cql_lstm_preview}"
HEADLESS="${HEADLESS:-0}"
MAX_EPISODES="${MAX_EPISODES:-20}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-30}"
LOOP="${LOOP:-0}"
METRICS_ONLY="${METRICS_ONLY:-0}"
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
    ./isaaclab.sh -p scripts/tools/play_lift_cube_offline_cql_lstm.py
    --checkpoint "${CHECKPOINT}"
    --num_envs 1
    --output_dir "${OUTPUT_DIR}"
    --max_episodes "${MAX_EPISODES}"
    --episode_length_s "${EPISODE_LENGTH_S}"
    --lift_goal_x "${LIFT_GOAL_X}"
    --lift_goal_y "${LIFT_GOAL_Y}"
    --lift_goal_z "${LIFT_GOAL_Z}"
)

if [[ "${FIXED_LIFT_GOAL}" != "0" ]]; then
    cmd+=(--fixed_lift_goal)
else
    cmd+=(--no-fixed_lift_goal)
fi
if [[ "${LOOP}" != "0" ]]; then
    cmd+=(--loop)
fi
if [[ "${METRICS_ONLY}" != "0" ]]; then
    cmd+=(--metrics_only)
fi
if [[ "${HEADLESS}" != "0" ]]; then
    cmd+=(--headless)
fi

cmd+=("$@")

echo "Playing offline CQL-SAC LSTM checkpoint:"
printf ' %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
