#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${REPO_ROOT}/.env.server" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.server"
fi
# Transformers v4 follows TRANSFORMERS_CACHE over HF_HOME; old per-host values
# can hide an already populated Hugging Face hub cache.
unset TRANSFORMERS_CACHE
ROOT="${AUTORESEARCH_ROOT:-${REPO_ROOT}}"
CALVIN_G_ROOT="${CALVIN_G_ROOT:-$(cd "${ROOT}/.." && pwd)/calvin_g}"
PYTHON="${CALVIN_PYTHON:-${CALVIN_G_ROOT}/envs/calvin_h200/bin/python}"
CALVIN_DATA="${CALVIN_DATA:-${CALVIN_G_ROOT}/data}"
SOURCE_ZIP="${CALVIN_SOURCE_ZIP:-${CALVIN_DATA}/task_ABC_D.zip}"
SOURCE_DIR="${CALVIN_SOURCE_DIR:-}"
COMPACT_ROOT="${CALVIN_COMPACT_DATA_ROOT:-${CALVIN_DATA}/task_ABC_D_compact_blocks}"
OUTPUT_ROOT="${CALVIN_DINOV2_FEATURE_ROOT:-${CALVIN_DATA}/task_ABC_D_compact_blocks_dinov2_small}"
SOURCE_ARGS=(--source-zip "${SOURCE_ZIP}")
if [[ -n "${SOURCE_DIR}" ]]; then
  SOURCE_ARGS=(--source-dir "${SOURCE_DIR}")
fi

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=1

"${PYTHON}" framework/scripts/data/precompute_calvin_dinov2_features_from_zip.py \
  "${SOURCE_ARGS[@]}" \
  --compact-data-root "${COMPACT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --model-name facebook/dinov2-small \
  --batch-size 128 \
  --shard-size 2048 \
  --device cuda:0
