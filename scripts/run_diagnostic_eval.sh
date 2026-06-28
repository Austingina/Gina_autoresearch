#!/usr/bin/env bash
# Diagnostic eval with EE/cube trajectories + failure-mode JSON (AutoResearch-owned output).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_ROOT:-/home/l/code/IsaacLab}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

STAGE="${STAGE:-s0}"  # s0 = fixed cube; s1 = random cube + fixed lift goal
MODE="${MODE:-}"  # optional B1|B7 — sets defaults when CHECKPOINT unset
CHECKPOINT="${CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/diagnostic_eval}"
GUIDANCE_MODE="${GUIDANCE_MODE:-}"
FIELD_CKPT="${FIELD_CKPT:-}"
NUM_EPISODES="${NUM_EPISODES:-20}"
NUM_ENVS="${NUM_ENVS:-4}"
HEADLESS="${HEADLESS:-1}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export AUTORESEARCH_ROOT="${ROOT}"
export PYTHONUNBUFFERED=1
export TERM="${TERM:-xterm-256color}"
if [[ "${TERM:-}" == dumb ]]; then
  export TERM=xterm-256color
fi
cd "${ISAACLAB_DIR}"

if [[ -z "${CHECKPOINT}" ]]; then
  if [[ "${MODE}" == "B1" ]]; then
    GUIDANCE_MODE="${GUIDANCE_MODE:-anchor_only}"
    if [[ "${STAGE}" == "s1" ]]; then
      CHECKPOINT="${ROOT}/runs/s1_B1_20260622_173518/student_best.pt"
    else
      CHECKPOINT="${ROOT}/runs/s0_B1_restart/student_best.pt"
    fi
  elif [[ "${MODE}" == "B7" ]]; then
    GUIDANCE_MODE="${GUIDANCE_MODE:-anchor_field_demo}"
    if [[ "${STAGE}" == "s1" ]]; then
      CHECKPOINT="${ROOT}/runs/s1_B7_20260623_001554/student_best.pt"
    else
      CHECKPOINT="${ROOT}/runs/s0_B7_20260622/student_best.pt"
    fi
  else
    echo "Set CHECKPOINT=... or MODE=B1|B7 (with optional STAGE=s0|s1)"
    exit 1
  fi
fi
GUIDANCE_MODE="${GUIDANCE_MODE:-anchor_only}"
if [[ "${STAGE}" == "s1" ]]; then
  FIELD_CKPT="${FIELD_CKPT:-${ROOT}/checkpoints/lift/guidance_field_s1_random_cube.pt}"
else
  FIELD_CKPT="${FIELD_CKPT:-${ROOT}/checkpoints/lift/guidance_field_s0_fixed.pt}"
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

domain_args=(--fixed_lift_goal --no-randomize_lift_goal --episode_length_s 25)
if [[ "${STAGE}" == "s1" ]]; then
  domain_args=(--fixed_lift_goal --randomize_cube --no-randomize_lift_goal --episode_length_s 25)
else
  domain_args+=(--no-randomize_cube)
fi

cmd=(
  ./isaaclab.sh -p scripts/tools/diagnostic_eval_lift_cube_vision.py
  --checkpoint "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --num_envs "${NUM_ENVS}"
  --num_episodes "${NUM_EPISODES}"
  --guidance_mode "${GUIDANCE_MODE}"
  --autoresearch_root "${ROOT}"
  "${domain_args[@]}"
)

if [[ "${GUIDANCE_MODE}" == "anchor_field_demo" ]]; then
  cmd+=(--field_checkpoint "${FIELD_CKPT}")
fi
if [[ "${HEADLESS}" != "0" ]]; then
  cmd+=(--headless)
fi

echo "=== Diagnostic eval ${STAGE} ${MODE:-custom} -> ${OUTPUT_DIR} ==="
echo "    checkpoint: ${CHECKPOINT}"
"${cmd[@]}"

/home/l/miniconda3/envs/env_isaaclab/bin/python3 "${ROOT}/scripts/summarize_eval_report.py" \
  "${OUTPUT_DIR}/eval_diagnostics.json" \
  --markdown "${OUTPUT_DIR}/eval_analysis.md"

echo "[OK] Analysis: ${OUTPUT_DIR}/eval_analysis.md"
