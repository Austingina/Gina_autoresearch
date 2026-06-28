#!/usr/bin/env bash
# Play/visualize B1, B3, or B7 vision DAgger student (Isaac GUI + optional OpenCV overlay).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_ROOT:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

MODE="${1:-B1}"  # B1, B3, or B7
STAGE="${STAGE:-s0}"  # s0 = fixed cube; s1 = random cube + fixed lift goal
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/play_preview/${STAGE}_${MODE}}"
MAX_EPISODES="${MAX_EPISODES:-20}"
HEADLESS="${HEADLESS:-0}"
SHOW_OPENCV="${SHOW_OPENCV:-1}"

if [[ "${MODE}" == "B1" ]]; then
  GUIDANCE_MODE="anchor_only"
  if [[ "${STAGE}" == "s1" ]]; then
    CHECKPOINT="${CHECKPOINT:-${ROOT}/runs/s1_B1_20260622_173518/student_best.pt}"
  else
    CHECKPOINT="${CHECKPOINT:-${ROOT}/runs/s0_B1_restart/student_best.pt}"
  fi
elif [[ "${MODE}" == "B3" ]]; then
  GUIDANCE_MODE="anchor_field_rule"
  CHECKPOINT="${CHECKPOINT:-}"
elif [[ "${MODE}" == "B7" ]]; then
  GUIDANCE_MODE="anchor_field_demo"
  if [[ "${STAGE}" == "s1" ]]; then
    CHECKPOINT="${CHECKPOINT:-${ROOT}/runs/s1_B7_20260623_001554/student_best.pt}"
  else
    CHECKPOINT="${CHECKPOINT:-${ROOT}/runs/s0_B7_20260622/student_best.pt}"
  fi
else
  echo "Usage: STAGE=s0|s1 $0 B1|B3|B7"
  exit 1
fi

field_args=()
if [[ "${GUIDANCE_MODE}" == "anchor_field_demo" ]]; then
  if [[ "${STAGE}" == "s1" ]]; then
    FIELD_CKPT="${FIELD_CKPT:-${ROOT}/checkpoints/lift/guidance_field_s1_random_cube.pt}"
  else
    FIELD_CKPT="${FIELD_CKPT:-${ROOT}/checkpoints/lift/guidance_field_s0_fixed.pt}"
  fi
  field_args=(--field_checkpoint "${FIELD_CKPT}")
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export AUTORESEARCH_ROOT="${ROOT}"
export TERM="${TERM:-xterm-256color}"
if [[ "${TERM:-}" == dumb ]]; then
  export TERM=xterm-256color
fi
cd "${ISAACLAB_DIR}"

extra=()
if [[ "${SHOW_OPENCV}" != "0" ]]; then
  extra+=(--show_opencv)
fi
if [[ "${HEADLESS}" != "0" ]]; then
  extra+=(--headless)
fi

IMAGE_SIZE="${IMAGE_SIZE:-240}"
CAMERA_SIZE="${CAMERA_SIZE:-240}"
TOPDOWN_CLOSE="${TOPDOWN_CLOSE:-1}"
CUBE_Z_MIN="${CUBE_Z_MIN:-0.0}"
CUBE_Z_MAX="${CUBE_Z_MAX:-0.0}"

domain_args=(--fixed_lift_goal --no-randomize_lift_goal --randomize_cube --no-randomize_cube_z_only)
if [[ "${STAGE}" == "s1" ]]; then
  domain_args+=(--image_size "${IMAGE_SIZE}" --camera_height "${CAMERA_SIZE}" --camera_width "${CAMERA_SIZE}")
  if [[ "${TOPDOWN_CLOSE}" == "1" ]]; then
    domain_args+=(--topdown_close)
  else
    domain_args+=(--no-topdown_close)
  fi
else
  domain_args=(--fixed_lift_goal --no-randomize_lift_goal --no-randomize_cube --no-randomize_cube_z_only)
fi

echo "=== Play ${STAGE} ${MODE} | ${CHECKPOINT} ==="
./isaaclab.sh -p scripts/tools/play_lift_cube_student_vision_binary_gripper.py \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_episodes "${MAX_EPISODES}" \
  --guidance_mode "${GUIDANCE_MODE}" \
  "${field_args[@]}" \
  --autoresearch_root "${ROOT}" \
  "${domain_args[@]}" \
  "${extra[@]}"

echo "[OK] Preview video: ${OUTPUT_DIR}"
