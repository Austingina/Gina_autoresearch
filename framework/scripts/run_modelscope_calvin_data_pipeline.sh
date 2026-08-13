#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${REPO_ROOT}/.env.server" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.server"
fi

CALVIN_G_ROOT="${CALVIN_G_ROOT:?Set CALVIN_G_ROOT or provide .env.server}"
CALVIN_DATA="${CALVIN_DATA:-${CALVIN_G_ROOT}/data}"
CALVIN_PYTHON="${CALVIN_PYTHON:-${CALVIN_G_ROOT}/envs/calvin_h200/bin/python}"
SOURCE_REPO="${MODELSCOPE_CALVIN_REPO:-${CALVIN_DATA}/modelscope_calvin_abc_rlds}"
EXTRACT_ROOT="${MODELSCOPE_CALVIN_EXTRACT:-${CALVIN_DATA}/modelscope_calvin_abc_extracted}"
META_ROOT="${MODELSCOPE_CALVIN_META:-${CALVIN_DATA}/hf_calvin_abc_lerobot/meta}"
CAMERA_CONFIG_ROOT="${CALVIN_CAMERA_CONFIG_ROOT:-${CALVIN_DATA}/calvin_debug_dataset}"
COMPACT_ROOT="${MODELSCOPE_CALVIN_COMPACT:-${CALVIN_DATA}/lerobot_blocks_behavior_g_semantic}"
FEATURE_ROOT="${MODELSCOPE_CALVIN_DINO:-${CALVIN_DATA}/lerobot_blocks_behavior_g_semantic_dinov2_small}"
LOG_ROOT="${CALVIN_G_ROOT}/logs/modelscope_abc_download"

PART00="${SOURCE_REPO}/task-ABC-D.tar.gz.part00"
PART01="${SOURCE_REPO}/task-ABC-D.tar.gz.part01"
PART00_SIZE=32212254720
PART01_SIZE=25623781844

mkdir -p "${EXTRACT_ROOT}" "${LOG_ROOT}"
while [[ ! -f "${PART00}" || ! -f "${PART01}" ]] || \
      [[ "$(stat -c%s "${PART00}" 2>/dev/null || echo 0)" -ne "${PART00_SIZE}" ]] || \
      [[ "$(stat -c%s "${PART01}" 2>/dev/null || echo 0)" -ne "${PART01_SIZE}" ]]; do
  printf '%s waiting for complete LFS worktree files\n' "$(date -Is)"
  sleep 60
done

cd "${SOURCE_REPO}"
echo "91b1a2083024b1d6445db3d2bbc974e3a6d90beeef6855aa3b793ada77e6ea4f  task-ABC-D.tar.gz.part00" | sha256sum -c -
echo "b7f29eb2eac3f8f10370a309640b9540b5e3f9deb1ad58bede7b9fe0fc7abf24  task-ABC-D.tar.gz.part01" | sha256sum -c -

if [[ ! -f "${EXTRACT_ROOT}/.extract_complete" ]]; then
  cat "${PART00}" "${PART01}" | tar -xzf - -C "${EXTRACT_ROOT}"
  date -Is > "${EXTRACT_ROOT}/.extract_complete"
fi

CHUNK_DIR="$(find "${EXTRACT_ROOT}" -type d -path '*/data/chunk-*' -print -quit)"
if [[ -z "${CHUNK_DIR}" ]]; then
  echo "No LeRobot data/chunk-* directory found under ${EXTRACT_ROOT}" >&2
  exit 1
fi
LEROBOT_ROOT="$(dirname "$(dirname "${CHUNK_DIR}")")"

if [[ ! -f "${COMPACT_ROOT}/conversion_manifest.json" ]]; then
  "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/data/convert_calvin_lerobot_to_compact_npz.py" \
    --lerobot-root "${LEROBOT_ROOT}" \
    --meta-root "${META_ROOT}" \
    --camera-config-root "${CAMERA_CONFIG_ROOT}" \
    --output-dir "${COMPACT_ROOT}" \
    --include-tasks \
      push_red_block_left push_red_block_right \
      push_blue_block_left push_blue_block_right \
      push_pink_block_left push_pink_block_right \
    --flow-mode semantic_axis
fi

unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
if [[ ! -f "${FEATURE_ROOT}/manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="${CALVIN_DATA_GPU:-0}" "${CALVIN_PYTHON}" \
    "${REPO_ROOT}/framework/scripts/data/precompute_calvin_dinov2_features_from_zip.py" \
    --source-dir "${COMPACT_ROOT}" \
    --compact-data-root "${COMPACT_ROOT}" \
    --output-dir "${FEATURE_ROOT}" \
    --model-name facebook/dinov2-small \
    --batch-size "${CALVIN_DINO_BATCH_SIZE:-128}" \
    --shard-size 2048 \
    --device cuda:0
fi

date -Is > "${FEATURE_ROOT}/.pipeline_complete"
echo "ModelScope CALVIN data pipeline complete: ${FEATURE_ROOT}"
