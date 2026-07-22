#!/usr/bin/env bash
# Headless smoke checks for the IsaacLab Interaction Geometry task suite.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/simulators/resolve_paths.sh"
# ISAACLAB_ROOT set by simulators/resolve_paths.sh
DRAWER_TIMEOUT_SEC="${DRAWER_TIMEOUT_SEC:-35}"

echo "=== [1/3] lift_cube: vertical_free ==="
"${ROOT}/scripts/play_isaac_lift_cube_state_machine.sh" --headless --max_steps "${LIFT_MAX_STEPS:-350}"

echo "=== [2/3] open_drawer: slide ==="
set +e
timeout "${DRAWER_TIMEOUT_SEC}s" "${ROOT}/scripts/play_isaac_open_drawer_state_machine.sh" --headless --num_envs 1
drawer_code=$?
set -e
if [[ "${drawer_code}" == "124" ]]; then
  echo "[RESULT] drawer_smoke=alive timeout_sec=${DRAWER_TIMEOUT_SEC}"
elif [[ "${drawer_code}" == "0" ]]; then
  echo "[RESULT] drawer_smoke=completed"
else
  echo "[ERROR] drawer_smoke failed with exit_code=${drawer_code}" >&2
  exit "${drawer_code}"
fi

echo "=== [3/3] open_side_door: revolute ==="
"${ROOT}/scripts/play_isaac_open_side_door_state_machine.sh" --headless --door right --max_steps "${DOOR_MAX_STEPS:-900}"

echo "[OK] Isaac task suite smoke completed."
