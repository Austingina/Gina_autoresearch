#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Official ManiSkill GPU simulation visual demo.
# This is random-action visualization, not a successful demonstration.
# Source: mani_skill.examples.benchmarking.gpu_sim
conda run -n maniskill python -m mani_skill.examples.benchmarking.gpu_sim \
  -e OpenCabinetDrawer-v1 \
  -n "${NUM_ENVS:-4}" \
  -o "${OBS_MODE:-rgb+depth}" \
  --render-mode "${RENDER_MODE:-sensors}" \
  --save-video

echo "[official Drawer visualization] Output dir: ${ROOT_DIR}/videos/ms3_benchmark"
