#!/usr/bin/env bash
set -euo pipefail

# IsaacLab official open-drawer state-machine demo.
# This visualizes the Franka opening the Sektion cabinet top drawer by
# targeting the official drawer_handle_top frame.
#
# Under the hood this runs:
#   /home/l/code/IsaacLab/scripts/environments/state_machine/open_cabinet_sm.py
#
# Note: the official state-machine demo uses the manager-based IK absolute env
# `Isaac-Open-Drawer-Franka-IK-Abs-v0`, not the DirectRL training env
# `Isaac-Franka-Cabinet-Direct-v0`.

# ISAACLAB_ROOT set by simulators/resolve_paths.sh
NUM_ENVS="${NUM_ENVS:-1}"

cd "${ISAACLAB_ROOT}"
./isaaclab.sh -p scripts/environments/state_machine/open_cabinet_sm.py \
  --num_envs "${NUM_ENVS}" \
  "$@"
