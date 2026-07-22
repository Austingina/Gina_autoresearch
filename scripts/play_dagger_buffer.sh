#!/usr/bin/env bash
# OpenCV playback of all DAgger disk-buffer episodes (ep_*.pt).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/simulators/resolve_paths.sh"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

BUFFER_DIR="${BUFFER_DIR:-${ROOT}/runs/s1_B1_mosaic_20260701_141037/dagger_episodes}"
IMAGE_SIZE="${IMAGE_SIZE:-240}"
FPS="${FPS:-20}"
STRIDE="${STRIDE:-1}"
SCALE="${SCALE:-1}"
START_EPISODE="${START_EPISODE:-0}"
MAX_EPISODES="${MAX_EPISODES:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

extra=()
if [[ "${DEPTH_COLORMAP:-1}" != "0" ]]; then
  extra+=(--depth-colormap)
fi

echo "=== Play DAgger buffer (OpenCV) ==="
echo "  dir: ${BUFFER_DIR}"
echo "  keys: q quit | n next ep | p prev ep | space pause | +/- speed | g jump"

python3 "${ROOT}/scripts/visualize_train_maskdepth_video.py" \
  "${BUFFER_DIR}" \
  --image-size "${IMAGE_SIZE}" \
  --fps "${FPS}" \
  --stride "${STRIDE}" \
  --scale "${SCALE}" \
  --start-episode "${START_EPISODE}" \
  --max-episodes "${MAX_EPISODES}" \
  "${extra[@]}" \
  "$@"
