#!/usr/bin/env bash
# Remote B7-setting smoke: collect G demos -> train G_θ -> train B7 policy.
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-$HOME/AutoResearch}"
source "${ROOT}/simulators/resolve_paths.sh"
# ISAACLAB_ROOT set by simulators/resolve_paths.sh
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
# Isaac Sim RTX headless: mask physical GPU via CUDA_VISIBLE_DEVICES so activeGpu=0
# (direct --device cuda:N with N>0 can crash glinterop on this host).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
LOG_DIR="${LOG_DIR:-${ROOT}/runs/remote_b7_smoke_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

export PATH="${HOME}/miniconda3/bin:${PATH}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export TERM="${TERM:-xterm-256color}"
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
unset LD_PRELOAD LD_LIBRARY_PATH
export ISAACLAB_ROOT="${ISAACLAB_ROOT}"
export AUTORESEARCH_ROOT="${ROOT}"
export CUDA_VISIBLE_DEVICES

# Reduced smoke scale (full B7 defaults in run_s1_b7_fresh.sh).
NUM_DEMOS="${NUM_DEMOS:-12}"
G_EPOCHS="${G_EPOCHS:-10}"
MAX_ITERS="${MAX_ITERS:-50}"
FIELD_CKPT="${FIELD_CKPT:-${ROOT}/checkpoints/lift/guidance_field_s1_remote_smoke.pt}"

exec > >(tee -a "${LOG_DIR}/smoke.log") 2>&1
echo "=== remote B7 smoke $(date -Is) ==="
echo "  ROOT=${ROOT}"
echo "  ISAACLAB_ROOT=${ISAACLAB_ROOT}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  NUM_DEMOS=${NUM_DEMOS} G_EPOCHS=${G_EPOCHS} MAX_ITERS=${MAX_ITERS}"

python - <<'PY'
import torch, isaacsim
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())
print("isaacsim", isaacsim.__file__)
PY

echo "[1/3] Collect clean teacher demos (vertical ACTUATE G)..."
cd "${ROOT}"
COLLECT_CMD=(env NUM_DEMOS="${NUM_DEMOS}" NUM_ENVS=4 CAMERA_SIZE=240 TOPDOWN_CLOSE=1
  ROBOT_X_OFFSET=-0.10 PANDA_JOINT6=2.7 LIFT_GOAL_ABOVE_CUBE=1 LIFT_HEIGHT_ABOVE_CUBE=0.20
  RUN_TAG="remote_smoke_$(date +%Y%m%d_%H%M%S)" bash scripts/collect_s1_clean240_vertical.sh)
if command -v xvfb-run >/dev/null 2>&1; then
  xvfb-run -a "${COLLECT_CMD[@]}"
else
  echo "[WARN] xvfb-run not found — install: sudo apt-get install -y xvfb"
  "${COLLECT_CMD[@]}"
fi

DEMO_DIR="${ROOT}/data/lift/s1_random_cube/demos"
echo "[2/3] Train G_θ (mask+depth, subgoal, 3D, g-phase)..."
python "${ROOT}/scripts/learn_guidance_field.py" \
  --task lift \
  --stage s1_random_cube \
  --demo-dir "${DEMO_DIR}" \
  --supervision subgoal \
  --input-mode mask_depth \
  --out-dim 3 \
  --g-phase \
  --image-size 240 \
  --epochs "${G_EPOCHS}" \
  --output "${FIELD_CKPT}"

echo "[3/3] Train B7 policy (disk aggregate, anti-OOM profile)..."
cd "${ROOT}"
RUN_TAG="remote_smoke_$(date +%Y%m%d_%H%M%S)" \
LOG_DIR="${LOG_DIR}/b7" \
MAX_ITERS="${MAX_ITERS}" \
NUM_ENVS=4 \
COLLECT_STEPS=128 \
NUM_EVAL_ENVS=2 \
EVAL_EPISODES=3 \
EVAL_INTERVAL=10 \
AGGREGATE_BACKEND=disk \
AGGREGATE_CACHE_SIZE=4 \
FIELD_CKPT="${FIELD_CKPT}" \
ISAACLAB_ROOT="${ISAACLAB_ROOT}" \
CONDA_ENV="${CONDA_ENV}" \
AUTORESEARCH_ROOT="${ROOT}" \
bash scripts/run_s1_b7_fresh.sh

echo "[OK] Remote B7 smoke finished. Logs: ${LOG_DIR}"
