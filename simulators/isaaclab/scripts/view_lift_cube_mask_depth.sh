#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/lift_cube_mask_depth_preview}"
HEADLESS="${HEADLESS:-1}"

export TERM="${TERM:-xterm-256color}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ISAACLAB_DIR}"

cmd=(
    ./isaaclab.sh -p scripts/tools/view_lift_cube_mask_depth.py
    --output_dir "${OUTPUT_DIR}"
)
if [[ "${HEADLESS}" != "0" ]]; then
    cmd+=(--headless)
fi
cmd+=("$@")
exec "${cmd[@]}"
