#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
PYTHON="/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python"
SOURCE_ZIP="/mnt/nvme33n1/wuk_code/calvin_g/data/task_ABC_D.zip"
COMPACT_ROOT="/mnt/nvme33n1/wuk_code/calvin_g/data/task_ABC_D_compact_blocks"
OUTPUT_ROOT="/mnt/nvme33n1/wuk_code/calvin_g/data/task_ABC_D_compact_blocks_dinov2_small"

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
