#!/usr/bin/env bash
# Live oracle teacher + G_θ overlay (verify Step 1 before DAgger).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_ROOT:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

FIELD_CKPT="${FIELD_CKPT:-${ROOT}/checkpoints/lift/guidance_field_s1_random_cube.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/play_teacher_g/s1}"
CAMERA_SIZE="${CAMERA_SIZE:-240}"
TOPDOWN_CLOSE="${TOPDOWN_CLOSE:-1}"
MAX_EPISODES="${MAX_EPISODES:-20}"
HEADLESS="${HEADLESS:-0}"
SHOW_OPENCV="${SHOW_OPENCV:-1}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export AUTORESEARCH_ROOT="${ROOT}"
export TERM="${TERM:-xterm-256color}"
cd "${ISAACLAB_DIR}"

extra=()
[[ "${SHOW_OPENCV}" != "0" ]] && extra+=(--show_opencv)
[[ "${HEADLESS}" != "0" ]] && extra+=(--headless)
[[ "${TOPDOWN_CLOSE}" == "1" ]] && extra+=(--topdown_close)

./isaaclab.sh -p scripts/tools/view_teacher_guidance_field.py \
  --field_checkpoint "${FIELD_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_episodes "${MAX_EPISODES}" \
  --randomize_cube \
  --fixed_lift_goal \
  --no-randomize_lift_goal \
  --camera_height "${CAMERA_SIZE}" \
  --camera_width "${CAMERA_SIZE}" \
  --autoresearch_root "${ROOT}" \
  "${extra[@]}"

echo "[OK] Preview: ${OUTPUT_DIR}"
