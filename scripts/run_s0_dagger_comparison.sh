#!/usr/bin/env bash
# S0 comparison: B1 (vision-only) vs B7 (vision + learned guidance field).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/simulators/resolve_paths.sh"
ISAACLAB_DIR="${ISAACLAB_ROOT}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
FIELD_CKPT="${FIELD_CKPT:-$ROOT/checkpoints/lift/guidance_field_s0_fixed.pt}"

NUM_ENVS="${NUM_ENVS:-32}"
MAX_ITERS="${MAX_ITERS:-500}"
COLLECT_STEPS="${COLLECT_STEPS:-512}"
MIN_BUFFER_EPISODES="${MIN_BUFFER_EPISODES:-16}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-25}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
SEED="${SEED:-42}"

RUNS_ROOT="${ROOT}/runs/s0_dagger_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUNS_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export AUTORESEARCH_ROOT="${ROOT}"
export TERM="${TERM:-xterm-256color}"
export PYTHONUNBUFFERED=1
cd "${ISAACLAB_DIR}"

if [[ ! -f "${FIELD_CKPT}" ]]; then
  echo "[ERROR] Field checkpoint missing: ${FIELD_CKPT}"
  echo "  Run: python scripts/learn_guidance_field.py --stage s0_fixed"
  exit 1
fi

common_args=(
  --headless
  --num_envs "${NUM_ENVS}"
  --max_iterations "${MAX_ITERS}"
  --collect_steps "${COLLECT_STEPS}"
  --min_buffer_episodes "${MIN_BUFFER_EPISODES}"
  --episode_length_s "${EPISODE_LENGTH_S}"
  --eval_episodes "${EVAL_EPISODES}"
  --eval_interval 50
  --save_interval 100
  --seed "${SEED}"
  --fixed_lift_goal
  --no-randomize_cube
  --no-randomize_lift_goal
  --image_size 64
  --autoresearch_root "${ROOT}"
)

echo "=== B1 anchor_only (vision 8ch) ==="
./isaaclab.sh -p scripts/reinforcement_learning/train_lift_cube_dagger_vision_binary_gripper.py \
  "${common_args[@]}" \
  --guidance_mode anchor_only \
  --visual_channels 8 \
  --log_dir "${RUNS_ROOT}/B1_anchor_only" \
  2>&1 | tee "${RUNS_ROOT}/B1_train.log"

echo "=== B7 anchor_field_demo (vision 8ch + G 2ch) ==="
./isaaclab.sh -p scripts/reinforcement_learning/train_lift_cube_dagger_vision_binary_gripper.py \
  "${common_args[@]}" \
  --guidance_mode anchor_field_demo \
  --field_checkpoint "${FIELD_CKPT}" \
  --log_dir "${RUNS_ROOT}/B7_field_demo" \
  2>&1 | tee "${RUNS_ROOT}/B7_train.log"

python3 - <<PY
import json, re
from pathlib import Path
root = Path("${RUNS_ROOT}")

def best_success(log_path):
    if not log_path.exists():
        return None
    text = log_path.read_text()
    vals = [float(m.group(1)) for m in re.finditer(r"\[Eval \d+\] success=([0-9.]+)", text)]
    return max(vals) if vals else None

summary = {
    "runs_root": str(root),
    "B1_best_eval_success": best_success(root / "B1_train.log"),
    "B7_best_eval_success": best_success(root / "B7_train.log"),
    "field_checkpoint": "${FIELD_CKPT}",
}
out = root / "comparison_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "[OK] Comparison logs: ${RUNS_ROOT}"
