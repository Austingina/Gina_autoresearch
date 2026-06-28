#!/usr/bin/env bash
# Live visual play entrypoint for Franka lift / drawer / door / button tasks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_ROOT:-/home/l/code/IsaacLab}"
ARENA_DIR="${ARENA_ROOT:-/home/l/code/IsaacLab-Arena}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

TASK="${1:-}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_STEPS="${NUM_STEPS:-100000}"
CAMERA_SIZE="${CAMERA_SIZE:-240}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/play_visual_tasks}"
HEADLESS="${HEADLESS:-0}"
SHOW_OPENCV="${SHOW_OPENCV:-1}"
IG_VISUALIZE_INTERVAL="${IG_VISUALIZE_INTERVAL:-30}"
IG_VISUALIZE_OUTPUT="${IG_VISUALIZE_OUTPUT:-window}"
IG_VIDEO_DIR="${IG_VIDEO_DIR:-}"
IG_VIDEO_FPS="${IG_VIDEO_FPS:-20}"
IG_REACH_GAIN="${IG_REACH_GAIN:-6.0}"
IG_ACTUATE_GAIN="${IG_ACTUATE_GAIN:-4.0}"
IG_MAX_DELTA_POS="${IG_MAX_DELTA_POS:-0.12}"
IG_POSITION_TOLERANCE="${IG_POSITION_TOLERANCE:-0.12}"
LIFT_ARM_SCALE="${LIFT_ARM_SCALE:-1.0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/play_visual_tasks.sh lift
  scripts/play_visual_tasks.sh drawer
  scripts/play_visual_tasks.sh door
  scripts/play_visual_tasks.sh button
  scripts/play_visual_tasks.sh inspect_button
  scripts/play_visual_tasks.sh inspect_door
  scripts/play_visual_tasks.sh inspect_microwave
  scripts/play_visual_tasks.sh teleop_button
  scripts/play_visual_tasks.sh teleop_door
  scripts/play_visual_tasks.sh teleop_microwave

Environment overrides:
  HEADLESS=1          Run without Isaac GUI (no viewport; use IG_VISUALIZE_OUTPUT=save).
  SHOW_OPENCV=0      Disable OpenCV overlay for lift.
  NUM_STEPS=2000     Limit Arena button/door rollout steps (default for button/door visual).
  NUM_EPISODES=5     Optional: run fixed episode count instead of NUM_STEPS (button only).
  CAMERA_SIZE=240    Lift camera resolution.
  IG_VISUALIZE_INTERVAL=30  Wrist RGB/depth/mask panel refresh interval (0=off).
  IG_VISUALIZE_OUTPUT=window  OpenCV panel: window | save | text.
  IG_VIDEO_DIR=...       Save per-episode wrist rgb/depth/mask MP4s (success only by default).
  IG_VIDEO_FPS=20        FPS for IG_VIDEO_DIR exports.
  IG_VIDEO_SUCCESS_ONLY=1  Set to 0 to also save failed episodes.
  FRANKA_X/Y/Z/YAW_DEG      Override Franka initial placement for teleop.
  PLATFORM_HEIGHT/SIZE_X/SIZE_Y Override teleop platform.
  TELEOP_EPISODE_LENGTH_S=3600  Long teleop episode duration.
  TELEOP_ALLOW_TASK_END=1       Re-enable normal task termination during teleop.
EOF
}

if [[ -z "${TASK}" || "${TASK}" == "-h" || "${TASK}" == "--help" ]]; then
  usage
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export AUTORESEARCH_ROOT="${ROOT}"
export TERM="${TERM:-xterm-256color}"
export LW_API_ENDPOINT="${LW_API_ENDPOINT:-https://api-dev.lightwheel.net}"
if [[ "${TERM:-}" == "dumb" ]]; then
  export TERM=xterm-256color
fi

headless_args=()
if [[ "${HEADLESS}" != "0" ]]; then
  headless_args+=(--headless)
fi

mkdir -p "${OUTPUT_ROOT}"

ig_video_success_args=(--ig_video_success_only)
if [[ "${IG_VIDEO_SUCCESS_ONLY:-1}" == "0" ]]; then
  ig_video_success_args=(--no-ig_video_success_only)
fi

case "${TASK}" in
  lift)
    cd "${ISAACLAB_DIR}"
    opencv_args=()
    if [[ "${SHOW_OPENCV}" != "0" ]]; then
      opencv_args+=(--show_opencv)
    fi
    ./isaaclab.sh -p scripts/tools/view_lift_cube_mask_depth.py \
      "${headless_args[@]}" \
      "${opencv_args[@]}" \
      --num_envs "${NUM_ENVS}" \
      --camera_height "${CAMERA_SIZE}" \
      --camera_width "${CAMERA_SIZE}" \
      --max_preview_steps "${NUM_STEPS}" \
      --arm_scale "${LIFT_ARM_SCALE}" \
      --output_dir "${OUTPUT_ROOT}/lift"
    ;;

  drawer)
    cd "${ISAACLAB_DIR}"
    ./isaaclab.sh -p scripts/environments/state_machine/open_cabinet_sm.py \
      "${headless_args[@]}" \
      --num_envs "${NUM_ENVS}"
    ;;

  inspect_button|inspect_door|inspect_microwave)
    cd "${ISAACLAB_DIR}"
    inspect_task="${TASK#inspect_}"
    ./isaaclab.sh -p "${ROOT}/scripts/inspect_arena_interaction_names.py" \
      "${headless_args[@]}" \
      --arena_task "${inspect_task}" \
      --num_envs "${NUM_ENVS}" \
      --output_dir "${OUTPUT_ROOT}/${inspect_task}_name_inspect"
    ;;

  teleop_button|teleop_door|teleop_microwave)
    cd "${ISAACLAB_DIR}"
    teleop_task="${TASK#teleop_}"
    teleop_pose_args=()
    if [[ -n "${FRANKA_X:-}" ]]; then teleop_pose_args+=(--franka_x "${FRANKA_X}"); fi
    if [[ -n "${FRANKA_Y:-}" ]]; then teleop_pose_args+=(--franka_y "${FRANKA_Y}"); fi
    if [[ -n "${FRANKA_Z:-}" ]]; then teleop_pose_args+=(--franka_z "${FRANKA_Z}"); fi
    if [[ -n "${FRANKA_YAW_DEG:-}" ]]; then teleop_pose_args+=(--franka_yaw_deg "${FRANKA_YAW_DEG}"); fi
    if [[ "${TELEOP_ALLOW_TASK_END:-0}" == "1" ]]; then teleop_pose_args+=(--allow_task_end); fi
    ./isaaclab.sh -p "${ROOT}/scripts/teleop_ee_record.py" \
      "${headless_args[@]}" \
      --arena_task "${teleop_task}" \
      --num_envs "${NUM_ENVS}" \
      --pos_step "${TELEOP_POS_STEP:-0.030}" \
      --rot_step "${TELEOP_ROT_STEP:-0.080}" \
      --episode_length_s "${TELEOP_EPISODE_LENGTH_S:-3600}" \
      "${teleop_pose_args[@]}" \
      --platform_height "${PLATFORM_HEIGHT:-0.45}" \
      --platform_size_x "${PLATFORM_SIZE_X:-0.80}" \
      --platform_size_y "${PLATFORM_SIZE_Y:-0.80}" \
      --output_dir "${OUTPUT_ROOT}/${teleop_task}_ee_teleop"
    ;;

  button)
    cd "${ISAACLAB_DIR}"
    episode_args=(--num_steps "${NUM_STEPS}")
    if [[ -n "${NUM_EPISODES:-}" ]]; then
      episode_args=(--num_episodes "${NUM_EPISODES}")
    fi
    ./isaaclab.sh -p "${ARENA_DIR}/isaaclab_arena/evaluation/policy_runner.py" \
      "${headless_args[@]}" \
      --enable_cameras \
      --num_envs "${NUM_ENVS}" \
      --policy_type isaaclab_arena.policy.interaction_geometry_teacher_policy.InteractionGeometryTeacherPolicy \
      "${episode_args[@]}" \
      --ig_debug_interval 30 \
      --ig_visualize_interval "${IG_VISUALIZE_INTERVAL}" \
      --ig_visualize_output "${IG_VISUALIZE_OUTPUT}" \
      --ig_visualize_dir "${OUTPUT_ROOT}/button_ig" \
      --ig_video_dir "${IG_VIDEO_DIR:-${OUTPUT_ROOT}/button_ig_videos}" \
      --ig_video_fps "${IG_VIDEO_FPS}" \
      "${ig_video_success_args[@]}" \
      --ig_teacher_mode press_button \
      --ig_press_axis "${IG_PRESS_AXIS:-y}" \
      --ig_press_surface_offset_x "${IG_PRESS_SURFACE_OFFSET_X:-0.0}" \
      --ig_press_surface_offset_y "${IG_PRESS_SURFACE_OFFSET_Y:-0.0}" \
      --ig_press_surface_offset_z "${IG_PRESS_SURFACE_OFFSET_Z:-0.0}" \
      --ig_reach_gain "${IG_REACH_GAIN}" \
      --ig_actuate_gain "${IG_ACTUATE_GAIN}" \
      --ig_max_delta_pos "${IG_MAX_DELTA_POS}" \
      --ig_position_tolerance "${IG_POSITION_TOLERANCE}" \
      --ig_oracle_press_distance "${IG_ORACLE_PRESS_DISTANCE:-0.50}" \
      --ig_oracle_press_after_actuate_steps "${IG_ORACLE_PRESS_AFTER_ACTUATE_STEPS:-0}" \
      --ig_hold_steps "${IG_HOLD_STEPS:-24}" \
      --ig_retreat_steps "${IG_RETREAT_STEPS:-18}" \
      press_button \
      --embodiment franka \
      --kitchen_layout 1 \
      --kitchen_style 2 \
      --coffee_button_joint "${COFFEE_BUTTON_JOINT:-Button002_joint}" \
      --reset_pressedness "${RESET_PRESSEDNESS:-0.0}" \
      --pressedness_threshold "${PRESSEDNESS_THRESHOLD:-0.5}" \
      --episode_length_s "${BUTTON_EPISODE_LENGTH_S:-20.0}" \
      --press_target_x "${PRESS_TARGET_X:-0.455}" \
      --press_target_y "${PRESS_TARGET_Y:--0.331}" \
      --press_target_z "${PRESS_TARGET_Z:-1.117}" \
      --franka_x "${BUTTON_FRANKA_X:-0.60}" \
      --franka_y "${BUTTON_FRANKA_Y:--0.72}" \
      --franka_z "${BUTTON_FRANKA_Z:-0.45}" \
      --franka_yaw_deg "${BUTTON_FRANKA_YAW_DEG:-90.0}" \
      --platform_height "${PLATFORM_HEIGHT:-0.45}" \
      --platform_size_x "${PLATFORM_SIZE_X:-0.80}" \
      --platform_size_y "${PLATFORM_SIZE_Y:-0.80}"
    ;;

  door|open_door|fridge)
    cd "${ISAACLAB_DIR}"
    ./isaaclab.sh -p "${ARENA_DIR}/isaaclab_arena/evaluation/policy_runner.py" \
      "${headless_args[@]}" \
      --enable_cameras \
      --num_envs "${NUM_ENVS}" \
      --policy_type isaaclab_arena.policy.interaction_geometry_teacher_policy.InteractionGeometryTeacherPolicy \
      --num_steps "${NUM_STEPS}" \
      --ig_debug_interval 30 \
      --ig_visualize_interval "${IG_VISUALIZE_INTERVAL}" \
      --ig_visualize_output "${IG_VISUALIZE_OUTPUT}" \
      --ig_visualize_dir "${OUTPUT_ROOT}/door_ig" \
      --ig_video_dir "${IG_VIDEO_DIR:-${OUTPUT_ROOT}/door_ig_videos}" \
      --ig_video_fps "${IG_VIDEO_FPS}" \
      "${ig_video_success_args[@]}" \
      --ig_teacher_mode open_door \
      --ig_reach_gain "${IG_REACH_GAIN}" \
      --ig_actuate_gain "${IG_ACTUATE_GAIN}" \
      --ig_max_delta_pos "${IG_MAX_DELTA_POS}" \
      --ig_position_tolerance "${IG_POSITION_TOLERANCE}" \
      franka_open_fridge \
      --embodiment franka \
      --kitchen_layout 1 \
      --kitchen_style 2 \
      --reset_openness 0.0 \
      --episode_length_s "${DOOR_EPISODE_LENGTH_S:-20.0}" \
      --open_target_x "${DOOR_TARGET_X:-4.25}" \
      --open_target_y "${DOOR_TARGET_Y:--0.62}" \
      --open_target_z "${DOOR_TARGET_Z:-1.05}" \
      --franka_x "${FRANKA_X:-4.45}" \
      --franka_y "${FRANKA_Y:--1.20}" \
      --franka_z "${FRANKA_Z:-0.45}" \
      --franka_yaw_deg "${FRANKA_YAW_DEG:-90.0}" \
      --platform_height "${PLATFORM_HEIGHT:-0.45}" \
      --platform_size_x "${PLATFORM_SIZE_X:-0.80}" \
      --platform_size_y "${PLATFORM_SIZE_Y:-0.80}"
    ;;

  microwave)
    cd "${ISAACLAB_DIR}"
    ./isaaclab.sh -p "${ARENA_DIR}/isaaclab_arena/evaluation/policy_runner.py" \
      "${headless_args[@]}" \
      --enable_cameras \
      --num_envs "${NUM_ENVS}" \
      --policy_type isaaclab_arena.policy.interaction_geometry_teacher_policy.InteractionGeometryTeacherPolicy \
      --num_steps "${NUM_STEPS}" \
      --ig_debug_interval 30 \
      --ig_visualize_interval "${IG_VISUALIZE_INTERVAL}" \
      --ig_visualize_output "${IG_VISUALIZE_OUTPUT}" \
      --ig_visualize_dir "${OUTPUT_ROOT}/microwave_ig" \
      --ig_video_dir "${IG_VIDEO_DIR:-${OUTPUT_ROOT}/microwave_ig_videos}" \
      --ig_video_fps "${IG_VIDEO_FPS}" \
      "${ig_video_success_args[@]}" \
      --ig_teacher_mode open_door \
      --ig_reach_gain "${IG_REACH_GAIN}" \
      --ig_actuate_gain "${IG_ACTUATE_GAIN}" \
      --ig_max_delta_pos "${IG_MAX_DELTA_POS}" \
      --ig_position_tolerance "${IG_POSITION_TOLERANCE}" \
      franka_open_microwave \
      --embodiment franka \
      --reset_openness 0.0 \
      --episode_length_s "${MICROWAVE_EPISODE_LENGTH_S:-20.0}" \
      --open_target_x "${MICROWAVE_TARGET_X:-0.42}" \
      --open_target_y "${MICROWAVE_TARGET_Y:--0.34}" \
      --open_target_z "${MICROWAVE_TARGET_Z:-0.72}"
    ;;

  *)
    echo "Unknown task: ${TASK}" >&2
    usage >&2
    exit 1
    ;;
esac
