#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/runs/maniskill_OpenCabinetDoor-v1_ee_teacher_${TS}}"

cd "${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}" conda run -n maniskill python scripts/play_maniskill_cabinet_ee_teacher.py \
  --env_id OpenCabinetDoor-v1 \
  --out_dir "${OUT_DIR}" \
  --max_steps "${MAX_STEPS:-300}" \
  --seed "${SEED:-4}" \
  --pull_distance "${PULL_DISTANCE:-0.9}" \
  --engage_offset "${ENGAGE_OFFSET:--0.06}" \
  --direction_mode "${DIRECTION_MODE:-kappa}" \
  --direction_sign "${DIRECTION_SIGN:--1}" \
  --pose_control
