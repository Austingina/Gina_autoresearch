#!/usr/bin/env bash
# S1 clean teacher demo collection: no action noise, 240px, closer top-down camera.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

export STAGE=s1_random_cube
export RANDOMIZE_CUBE=1
export CLEAN_TEACHER=1
export RUN_TAG
export CAMERA_SIZE="${CAMERA_SIZE:-240}"
export TOPDOWN_CLOSE="${TOPDOWN_CLOSE:-1}"
export ROBOT_X_OFFSET="${ROBOT_X_OFFSET:--0.10}"
export PANDA_JOINT6="${PANDA_JOINT6:-2.7}"
export NUM_ENVS="${NUM_ENVS:-8}"
export NUM_DEMOS="${NUM_DEMOS:-100}"
export ENABLE_PREVIEW="${ENABLE_PREVIEW:-0}"
export SHOW_OPENCV="${SHOW_OPENCV:-0}"
export HEADLESS="${HEADLESS:-1}"
export PREVIEW_DIR="${PREVIEW_DIR:-${ROOT}/runs/collect_preview/s1_clean_${RUN_TAG}}"

echo "=== S1 clean teacher collect (headless) ==="
echo "  G_θ trains on mask+depth (8ch). camera=${CAMERA_SIZE}px num_envs=${NUM_ENVS} robot_x_offset=${ROBOT_X_OFFSET} panda_joint6=${PANDA_JOINT6}"
echo "  Preview off by default — set ENABLE_PREVIEW=1 SHOW_OPENCV=1 for debug (needs more VRAM)."

bash "${ROOT}/scripts/collect_lift_demos.sh" "$@"
