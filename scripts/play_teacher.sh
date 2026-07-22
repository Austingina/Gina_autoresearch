#!/usr/bin/env bash
# Live OpenCV preview of lenient teacher + mask/depth (S1 lift-cube settings).
#
# Tune teacher speed via env vars, then watch ee/top mask+depth in OpenCV.
#
# Examples:
#   bash scripts/play_teacher.sh
#   TEACHER_MIN_LIN_SPEED=0.35 TEACHER_MAX_LIN_VEL=0.85 bash scripts/play_teacher.sh
#   TEACHER_MIN_LIN_SPEED=0.10 bash scripts/play_teacher.sh   # default slow teacher
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/simulators/resolve_paths.sh"
ISAACLAB_DIR="${ISAACLAB_ROOT}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/play_preview/s1_teacher}"
HEADLESS="${HEADLESS:-0}"
SHOW_OPENCV="${SHOW_OPENCV:-1}"
MAX_EPISODES="${MAX_EPISODES:-30}"
DISPLAY_SCALE="${DISPLAY_SCALE:-2}"
CAMERA_SIZE="${CAMERA_SIZE:-240}"
TOPDOWN_CLOSE="${TOPDOWN_CLOSE:-1}"
ROBOT_X_OFFSET="${ROBOT_X_OFFSET:--0.10}"
PANDA_JOINT6="${PANDA_JOINT6:-2.7}"

# Teacher speed (raise for faster EE motion / larger frame-to-frame changes).
TEACHER_MIN_LIN_SPEED="${TEACHER_MIN_LIN_SPEED:-2.5}"
TEACHER_MIN_ANG_SPEED="${TEACHER_MIN_ANG_SPEED:-2.5}"
TEACHER_MAX_LIN_VEL="${TEACHER_MAX_LIN_VEL:-4.5}"
TEACHER_MAX_ANG_VEL="${TEACHER_MAX_ANG_VEL:-4.0}"
TEACHER_GRASP_HOLD_STEPS="${TEACHER_GRASP_HOLD_STEPS:-2}"
TEACHER_ABOVE_OFFSET_Z="${TEACHER_ABOVE_OFFSET_Z:-0.04}"
ARM_SCALE="${ARM_SCALE:-1.0}"
GOAL_THRESHOLD="${GOAL_THRESHOLD:-0.025}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export TERM=xterm-256color
cd "${ISAACLAB_DIR}"

extra=()
if [[ "${SHOW_OPENCV}" != "0" ]]; then
  extra+=(--show_opencv)
fi
if [[ "${HEADLESS}" != "0" ]]; then
  extra+=(--headless)
fi
if [[ "${DEPTH_COLORMAP:-1}" != "0" ]]; then
  extra+=(--depth_colormap)
else
  extra+=(--no-depth_colormap)
fi
if [[ "${TOPDOWN_CLOSE}" == "1" ]]; then
  extra+=(--topdown_close)
else
  extra+=(--no-topdown_close)
fi

echo "=== Teacher preview (mask+depth) ==="
echo "  min_lin=${TEACHER_MIN_LIN_SPEED} max_lin=${TEACHER_MAX_LIN_VEL} camera=${CAMERA_SIZE}px"
echo "  output: ${OUTPUT_DIR}/preview.mp4"
echo "  press q in OpenCV window to quit"

./isaaclab.sh -p scripts/tools/view_lift_cube_mask_depth.py \
  --task Isaac-Lift-Cube-Franka-Vision-MaskDepth-Play-v0 \
  --num_envs 1 \
  --output_dir "${OUTPUT_DIR}" \
  --max_episodes "${MAX_EPISODES}" \
  --display_scale "${DISPLAY_SCALE}" \
  --camera_height "${CAMERA_SIZE}" \
  --camera_width "${CAMERA_SIZE}" \
  --robot_x_offset "${ROBOT_X_OFFSET}" \
  --panda_joint6 "${PANDA_JOINT6}" \
  --teacher_min_lin_speed "${TEACHER_MIN_LIN_SPEED}" \
  --teacher_min_ang_speed "${TEACHER_MIN_ANG_SPEED}" \
  --teacher_max_lin_vel "${TEACHER_MAX_LIN_VEL}" \
  --teacher_max_ang_vel "${TEACHER_MAX_ANG_VEL}" \
  --teacher_grasp_hold_steps "${TEACHER_GRASP_HOLD_STEPS}" \
  --teacher_above_offset_z "${TEACHER_ABOVE_OFFSET_Z}" \
  --goal_threshold "${GOAL_THRESHOLD}" \
  --arm_scale "${ARM_SCALE}" \
  --randomize_cube \
  --fixed_lift_goal \
  --lift_goal_above_cube \
  "${extra[@]}" \
  "$@"
