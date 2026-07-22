#!/usr/bin/env bash
# Collect lift-cube mask+depth demos into AutoResearch-owned data/ (never IsaacLab/datasets/).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/simulators/resolve_paths.sh"
ISAACLAB_DIR="${ISAACLAB_ROOT}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

TASK="${TASK:-lift}"
STAGE="${STAGE:-s0_fixed}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

DEST_DIR="${ROOT}/data/${TASK}/${STAGE}/demos"
mkdir -p "${DEST_DIR}"
DATASET="${DEST_DIR}/${STAGE}_${RUN_TAG}.hdf5"

NUM_ENVS="${NUM_ENVS:-32}"
NUM_DEMOS="${NUM_DEMOS:-100}"
NOISE_TRIGGER_PROB="${NOISE_TRIGGER_PROB:-0.35}"
NOISE_STD="${NOISE_STD:-0.05}"
NOISE_HOLD_MAX="${NOISE_HOLD_MAX:-8}"
CAMERA_SIZE="${CAMERA_SIZE:-84}"
TOPDOWN_CLOSE="${TOPDOWN_CLOSE:-0}"
CLEAN_TEACHER="${CLEAN_TEACHER:-0}"
ENABLE_PREVIEW="${ENABLE_PREVIEW:-0}"
SHOW_OPENCV="${SHOW_OPENCV:-0}"
PREVIEW_DIR="${PREVIEW_DIR:-${ROOT}/runs/collect_preview/${STAGE}_${RUN_TAG}}"
HEADLESS="${HEADLESS:-1}"
ROBOT_X_OFFSET="${ROBOT_X_OFFSET:-0.0}"
PANDA_JOINT6="${PANDA_JOINT6:-}"

# S0: fixed cube (default). S1: export RANDOMIZE_CUBE=1
EXTRA_ARGS=()
if [[ "${RANDOMIZE_CUBE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--randomize_cube)
fi
if [[ "${CLEAN_TEACHER}" == "1" ]]; then
  NOISE_TRIGGER_PROB=0
  NOISE_STD=0
  NOISE_HOLD_MAX=0
fi
if [[ "${TOPDOWN_CLOSE}" == "1" ]]; then
  EXTRA_ARGS+=(--topdown_close)
fi
if [[ "${LIFT_GOAL_ABOVE_CUBE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--lift_goal_above_cube --lift_height_above_cube "${LIFT_HEIGHT_ABOVE_CUBE:-0.20}")
fi
if [[ "${RANDOMIZE_CUBE:-0}" == "1" && "${ROBOT_X_OFFSET}" == "0.0" ]]; then
  ROBOT_X_OFFSET="-0.10"
fi
if [[ "${RANDOMIZE_CUBE:-0}" == "1" && -z "${PANDA_JOINT6}" ]]; then
  PANDA_JOINT6="2.7"
fi
EXTRA_ARGS+=(--camera_height "${CAMERA_SIZE}" --camera_width "${CAMERA_SIZE}")
EXTRA_ARGS+=(--robot_x_offset "${ROBOT_X_OFFSET}")
if [[ -n "${PANDA_JOINT6}" ]]; then
  EXTRA_ARGS+=(--panda_joint6 "${PANDA_JOINT6}")
fi

if [[ "${ENABLE_PREVIEW}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_preview --preview_dir "${PREVIEW_DIR}")
  if [[ "${SHOW_OPENCV}" == "1" ]]; then
    EXTRA_ARGS+=(--show_opencv)
    HEADLESS=0
  fi
fi

isaac_args=(--num_envs "${NUM_ENVS}")
if [[ "${HEADLESS}" != "0" ]]; then
  isaac_args+=(--headless)
fi

echo "=== AutoResearch collect (isolated) ==="
echo "  output: ${DATASET}"
echo "  stage:  ${STAGE}  randomize_cube=${RANDOMIZE_CUBE:-0}"
echo "  clean_teacher=${CLEAN_TEACHER}  noise_prob=${NOISE_TRIGGER_PROB}  camera=${CAMERA_SIZE}x${CAMERA_SIZE}  topdown_close=${TOPDOWN_CLOSE}  robot_x_offset=${ROBOT_X_OFFSET}  panda_joint6=${PANDA_JOINT6:-default}"
if [[ "${ENABLE_PREVIEW}" == "1" ]]; then
  echo "  preview: ${PREVIEW_DIR}/preview_latest.png  opencv=${SHOW_OPENCV}"
fi
if [[ "${CAMERA_SIZE}" -gt 128 ]]; then
  px_ratio=$(( CAMERA_SIZE * CAMERA_SIZE / 84 / 84 ))
  echo "  [WARN] Large camera (${CAMERA_SIZE}px) ~${px_ratio}x HDF5 size vs 84px — consider fewer NUM_DEMOS"
fi
echo "  IsaacLab datasets/ will NOT be written."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${ISAACLAB_DIR}"
export TERM="${TERM:-xterm-256color}"
if [[ "${TERM:-}" == dumb ]]; then
  export TERM=xterm-256color
fi

./isaaclab.sh -p scripts/tools/lenient_teacher_datacollection_maskdepth.py \
    "${isaac_args[@]}" \
    --num_demos "${NUM_DEMOS}" \
    --dataset_file "${DATASET}" \
    --noise_trigger_prob "${NOISE_TRIGGER_PROB}" \
    --noise_std "${NOISE_STD}" \
    --noise_hold_max "${NOISE_HOLD_MAX}" \
    --log_interval 200 \
    "${EXTRA_ARGS[@]}" \
    "$@"

MANIFEST_DIR="${ROOT}/data/manifests"
mkdir -p "${MANIFEST_DIR}"
python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

manifest = {
    "type": "isaac_collect",
    "collected_at": datetime.now(timezone.utc).isoformat(),
    "task": "${TASK}",
    "stage": "${STAGE}",
    "dataset": "${DATASET}",
    "num_demos": int("${NUM_DEMOS}"),
    "randomize_cube": "${RANDOMIZE_CUBE:-0}" == "1",
    "clean_teacher": "${CLEAN_TEACHER}" == "1",
    "noise_trigger_prob": float("${NOISE_TRIGGER_PROB}"),
    "camera_size": int("${CAMERA_SIZE}"),
    "topdown_close": "${TOPDOWN_CLOSE}" == "1",
    "isaaclab_root": "${ISAACLAB_DIR}",
}
path = Path("${MANIFEST_DIR}") / f"collect_${TASK}_${STAGE}_${RUN_TAG}.json"
path.write_text(json.dumps(manifest, indent=2))
print(f"[OK] Manifest: {path}")
PY

echo "[OK] Collected to ${DATASET}"
