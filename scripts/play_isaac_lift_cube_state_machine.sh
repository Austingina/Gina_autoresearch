#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/l/code/IsaacLab}"

cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p /home/l/code/AutoResearch/scripts/play_isaac_lift_cube_sm.py "$@"
