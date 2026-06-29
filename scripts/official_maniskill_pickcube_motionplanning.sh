#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/official_maniskill_PickCube-v1_motionplanning_${TS}}"

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

# Official ManiSkill Panda motion-planning demo.
# Source: mani_skill.examples.motionplanning.panda.run
conda run -n maniskill python -m mani_skill.examples.motionplanning.panda.run \
  -e PickCube-v1 \
  -n "${NUM_TRAJ:-1}" \
  --only-count-success \
  --save-video \
  --render-mode rgb_array \
  --sim-backend cpu \
  --record-dir "${OUT_DIR}" \
  --num-procs 1

echo "[official PickCube motionplanning] Output: ${OUT_DIR}"
