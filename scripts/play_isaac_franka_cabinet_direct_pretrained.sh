#!/usr/bin/env bash
set -euo pipefail

# Try IsaacLab's official RSL-RL play entry for the Direct Franka Cabinet task.
# This uses the published pretrained checkpoint if IsaacLab provides one.

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/l/code/IsaacLab}"
NUM_ENVS="${NUM_ENVS:-16}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"

cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Franka-Cabinet-Direct-v0 \
  --num_envs "${NUM_ENVS}" \
  --use_pretrained_checkpoint \
  --video \
  --video_length "${VIDEO_LENGTH}" \
  "$@"
