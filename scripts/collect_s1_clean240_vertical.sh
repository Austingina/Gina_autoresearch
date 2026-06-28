#!/usr/bin/env bash
# Re-collect S1 clean240 demos with the vertical ACTUATE lift goal (Interaction Geometry
# κ=vertical_free). Teacher lifts the grasped cube straight up to cube_xy + height.
#
# Run ONLY when no other Isaac process is active (single-process OOM guardrail).
# Output: data/lift/s1_random_cube/demos/<...>.hdf5 — then symlink as demos_clean240v
# before retraining G_θ.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

STAGE=s1_random_cube \
RANDOMIZE_CUBE="${RANDOMIZE_CUBE:-1}" \
CLEAN_TEACHER="${CLEAN_TEACHER:-1}" \
TOPDOWN_CLOSE="${TOPDOWN_CLOSE:-1}" \
CAMERA_SIZE="${CAMERA_SIZE:-240}" \
NUM_ENVS="${NUM_ENVS:-8}" \
NUM_DEMOS="${NUM_DEMOS:-100}" \
LIFT_GOAL_ABOVE_CUBE=1 \
LIFT_HEIGHT_ABOVE_CUBE="${LIFT_HEIGHT_ABOVE_CUBE:-0.20}" \
RUN_TAG="${RUN_TAG:-clean240v_$(date +%Y%m%d_%H%M%S)}" \
bash "${ROOT}/scripts/collect_lift_demos.sh"
