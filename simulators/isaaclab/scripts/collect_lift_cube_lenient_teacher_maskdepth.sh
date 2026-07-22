#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
DATASET="${DATASET:-./datasets/lift_cube_lenient_teacher_maskdepth.hdf5}"
NUM_ENVS="${NUM_ENVS:-32}"
NUM_DEMOS="${NUM_DEMOS:-500}"
NOISE_TRIGGER_PROB="${NOISE_TRIGGER_PROB:-0.35}"
NOISE_STD="${NOISE_STD:-0.05}"
NOISE_HOLD_MAX="${NOISE_HOLD_MAX:-8}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ISAACLAB_DIR}"

export TERM="${TERM:-xterm-256color}"

./isaaclab.sh -p scripts/tools/lenient_teacher_datacollection_maskdepth.py \
    --headless \
    --num_envs "${NUM_ENVS}" \
    --num_demos "${NUM_DEMOS}" \
    --dataset_file "${DATASET}" \
    --noise_trigger_prob "${NOISE_TRIGGER_PROB}" \
    --noise_std "${NOISE_STD}" \
    --noise_hold_max "${NOISE_HOLD_MAX}" \
    --log_interval 200 \
    "$@"
