#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/maniskill_OpenCabinetDrawer-v1_success_${TS}}"

cd "${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}" conda run -n maniskill python scripts/play_maniskill_cabinet_success.py \
  --env_id OpenCabinetDrawer-v1 \
  --out_dir "${OUT_DIR}" \
  --steps "${STEPS:-80}" \
  --fps "${FPS:-20}" \
  --seed "${SEED:-0}"
