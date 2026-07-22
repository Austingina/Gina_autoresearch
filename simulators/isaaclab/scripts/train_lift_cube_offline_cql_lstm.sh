#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
DATASET="${DATASET:-./datasets/lift_cube_lenient_teacher_maskdepth}"
LOG_DIR="${LOG_DIR:-./runs/lift_cube_offline_cql_lstm}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ISAACLAB_DIR}"

python scripts/reinforcement_learning/train_lift_cube_offline_cql_lstm.py \
    --dataset_file "${DATASET}" \
    --log_dir "${LOG_DIR}" \
    --obs_mode vision \
    --lstm_seq_len 20 \
    --updates_per_epoch 20 \
    --batch_size 32 \
    --epochs 1000 \
    --eval_every 50 \
    --eval_episodes 20 \
    "$@"
