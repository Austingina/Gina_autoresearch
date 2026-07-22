#!/usr/bin/env bash
# S0 smoke DAgger — tuned so episodes actually finish (avoid "buffer=0 forever").
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/simulators/resolve_paths.sh"
ISAACLAB_DIR="${ISAACLAB_ROOT}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

MODE="${1:-B1}"  # B1 or B7
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/runs/s0_${MODE}_${RUN_TAG}"
FIELD_CKPT="${FIELD_CKPT:-$ROOT/checkpoints/lift/guidance_field_s0_fixed.pt}"

# Lift episodes are ~1500 steps @ 30s. collect_steps must be large enough per iter.
NUM_ENVS="${NUM_ENVS:-16}"
COLLECT_STEPS="${COLLECT_STEPS:-512}"
MIN_BUFFER_EPISODES="${MIN_BUFFER_EPISODES:-8}"
MAX_ITERS="${MAX_ITERS:-100}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-25}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export TERM="${TERM:-xterm-256color}"
export PYTHONUNBUFFERED=1
export AUTORESEARCH_ROOT="${ROOT}"
cd "${ISAACLAB_DIR}"

# isaaclab.sh calls `tabs`; dumb TERM breaks non-interactive runs
if [[ "${TERM:-}" == dumb ]]; then
  export TERM=xterm-256color
fi

common=(
  --headless
  --num_envs "${NUM_ENVS}"
  --max_iterations "${MAX_ITERS}"
  --collect_steps "${COLLECT_STEPS}"
  --min_buffer_episodes "${MIN_BUFFER_EPISODES}"
  --episode_length_s "${EPISODE_LENGTH_S}"
  --eval_episodes 20
  --eval_interval 50
  --save_interval 100
  --seed 42
  --fixed_lift_goal
  --no-randomize_cube
  --no-randomize_lift_goal
  --image_size 64
  --autoresearch_root "${ROOT}"
  --log_dir "${LOG_DIR}"
)

if [[ "${MODE}" == "B1" ]]; then
  echo "=== B1 vision-only (8ch) -> ${LOG_DIR} ==="
  ./isaaclab.sh -p scripts/reinforcement_learning/train_lift_cube_dagger_vision_binary_gripper.py \
    "${common[@]}" \
    --guidance_mode anchor_only \
    --visual_channels 8
elif [[ "${MODE}" == "B7" ]]; then
  echo "=== B7 vision+G (10ch) -> ${LOG_DIR} ==="
  [[ -f "${FIELD_CKPT}" ]] || { echo "Missing ${FIELD_CKPT}"; exit 1; }
  ./isaaclab.sh -p scripts/reinforcement_learning/train_lift_cube_dagger_vision_binary_gripper.py \
    "${common[@]}" \
    --guidance_mode anchor_field_demo \
    --field_checkpoint "${FIELD_CKPT}"
else
  echo "Usage: $0 B1|B7"
  exit 1
fi
