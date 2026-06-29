#!/usr/bin/env bash
# S1 Step 1 — learn phase-aware 3D G_θ on random-cube vertical-free lift demos.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEMO_DIR="${DEMO_DIR:-${ROOT}/data/lift/s1_random_cube/demos}"
IMAGE_SIZE="${IMAGE_SIZE:-240}"

if [[ ! -d "${DEMO_DIR}" ]] || [[ -z "$(ls -A "${DEMO_DIR}"/*.hdf5 2>/dev/null)" ]]; then
  echo "[ERROR] No S1 demos in ${DEMO_DIR}"
  echo "  Collect first:"
  echo "    bash scripts/run_s1_collect_clean.sh"
  exit 1
fi

/home/l/miniconda3/envs/env_isaaclab/bin/python3 "${ROOT}/scripts/learn_guidance_field.py" \
  --task lift \
  --stage s1_random_cube \
  --demo-dir "${DEMO_DIR}" \
  --supervision subgoal \
  --input-mode mask_depth \
  --out-dim 3 \
  --g-phase \
  --image-size "${IMAGE_SIZE}" \
  "$@"
