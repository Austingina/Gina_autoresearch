#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/maniskill_OpenCabinetDrawer-v1_ee_teacher_${TS}}"

cd "${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}" conda run -n maniskill python scripts/play_maniskill_cabinet_ee_teacher.py \
  --env_id OpenCabinetDrawer-v1 \
  --out_dir "${OUT_DIR}" \
  --max_steps "${MAX_STEPS:-240}" \
  --seed "${SEED:-0}" \
  --pull_distance "${PULL_DISTANCE:-0.55}" \
  --engage_offset "${ENGAGE_OFFSET:-0.01}"
