#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
CALVIN_G_ROOT="${CALVIN_G_ROOT:-/mnt/nvme33n1/wuk_code/calvin_g}"
PYTHON="${CALVIN_PYTHON:-${CALVIN_G_ROOT}/envs/calvin_h200/bin/python}"
CALVIN_DATA="${CALVIN_DATA:-${CALVIN_G_ROOT}/data}"
SOURCE_ZIP="${CALVIN_DATA}/task_ABC_D.zip"
COMPACT_ROOT="${CALVIN_DATA}/task_ABC_D_compact_blocks"
OUTPUT_ROOT="${CALVIN_DATA}/task_ABC_D_compact_blocks_dinov2_small"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=1

"${PYTHON}" framework/scripts/data/precompute_calvin_dinov2_features_from_zip.py \
  --source-zip "${SOURCE_ZIP}" \
  --compact-data-root "${COMPACT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --model-name facebook/dinov2-small \
  --batch-size 128 \
  --shard-size 2048 \
  --device cuda:0
