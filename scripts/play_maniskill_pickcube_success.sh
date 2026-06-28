#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/maniskill_PickCube-v1_success_${TS}}"

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

echo "[PickCube-v1] Running fast oracle success visualizer..."
PYTHONPATH="${ROOT_DIR}" conda run -n maniskill python scripts/play_maniskill_pickcube_oracle_success.py \
  --out_dir "${OUT_DIR}" \
  --steps "${STEPS:-80}" \
  --fps "${FPS:-20}" \
  --seed "${SEED:-0}"

echo "[PickCube-v1] Done. Output: ${OUT_DIR}"
