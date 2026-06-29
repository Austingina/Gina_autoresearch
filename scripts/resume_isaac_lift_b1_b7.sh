#!/usr/bin/env bash
# Resume the IsaacLab-only lift experiment after teacher task smoke validation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DRY_RUN="${DRY_RUN:-1}"
SKIP_TASK_SMOKE="${SKIP_TASK_SMOKE:-0}"
RUN_STEP1="${RUN_STEP1:-1}"
RUN_B1="${RUN_B1:-1}"
RUN_B7="${RUN_B7:-1}"
FIELD_CKPT="${FIELD_CKPT:-${ROOT}/checkpoints/lift/guidance_field_s1_random_cube.pt}"
DEMO_DIR="${DEMO_DIR:-${ROOT}/data/lift/s1_random_cube/demos}"

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

echo "=== IsaacLab-only resume: lift B1/B7 ==="
echo "  DRY_RUN=${DRY_RUN}"
echo "  DEMO_DIR=${DEMO_DIR}"
echo "  FIELD_CKPT=${FIELD_CKPT}"
echo "  semantics=vertical_free, no hidden xy relocation"

if [[ "${SKIP_TASK_SMOKE}" != "1" ]]; then
  run_cmd "${ROOT}/scripts/smoke_isaac_task_suite.sh"
fi

if [[ "${RUN_STEP1}" == "1" ]]; then
  if [[ ! -d "${DEMO_DIR}" ]] || ! compgen -G "${DEMO_DIR}/*.hdf5" >/dev/null; then
    echo "[ERROR] No demos found in ${DEMO_DIR}" >&2
    echo "Collect vertical-free Isaac demos first:" >&2
    echo "  NUM_DEMOS=100 bash scripts/collect_s1_clean240_vertical.sh" >&2
    if [[ "${DRY_RUN}" != "1" ]]; then
      exit 1
    fi
  fi
  run_cmd env DEMO_DIR="${DEMO_DIR}" IMAGE_SIZE="${IMAGE_SIZE:-240}" bash "${ROOT}/scripts/run_s1_step1.sh"
fi

if [[ "${RUN_B1}" == "1" ]]; then
  run_cmd env \
    MAX_ITERS="${B1_MAX_ITERS:-300}" \
    LIFT_GOAL_ABOVE_CUBE=1 \
    LIFT_HEIGHT_ABOVE_CUBE="${LIFT_HEIGHT_ABOVE_CUBE:-0.20}" \
    bash "${ROOT}/scripts/run_s1_dagger_smoke.sh" B1
fi

if [[ "${RUN_B7}" == "1" ]]; then
  if [[ "${DRY_RUN}" != "1" && ! -f "${FIELD_CKPT}" ]]; then
    echo "[ERROR] Missing ${FIELD_CKPT}; Step 1 must finish before B7." >&2
    exit 1
  fi
  run_cmd env \
    FIELD_CKPT="${FIELD_CKPT}" \
    MAX_ITERS="${B7_MAX_ITERS:-300}" \
    LIFT_GOAL_ABOVE_CUBE=1 \
    LIFT_HEIGHT_ABOVE_CUBE="${LIFT_HEIGHT_ABOVE_CUBE:-0.20}" \
    bash "${ROOT}/scripts/run_s1_dagger_smoke.sh" B7
fi

echo "[OK] Resume pipeline command sequence finished."
