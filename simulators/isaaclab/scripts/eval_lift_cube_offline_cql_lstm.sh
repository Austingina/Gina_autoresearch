#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
DATASET="${DATASET:-./datasets/lift_cube_lenient_teacher_maskdepth}"
CHECKPOINT="${CHECKPOINT:-./runs/lift_cube_offline_cql_lstm/cql_sac_lstm/checkpoint_final.pt}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ISAACLAB_DIR}"

python scripts/reinforcement_learning/eval_lift_cube_offline_cql_lstm.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset_file "${DATASET}" \
    "$@"
