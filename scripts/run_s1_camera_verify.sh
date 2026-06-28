#!/usr/bin/env bash
# S1 camera-alignment ablation: B1 then B7 @ clean240 settings (240px + topdown_close).
# Validates whether matching G_θ demo camera fixes approach / B7 vs B1 gap.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export RUN_TAG="${RUN_TAG:-cam240_verify_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${ROOT}/runs/s1_camera_verify_${RUN_TAG}"
mkdir -p "${LOG_ROOT}"

unset RESUME_CHECKPOINT RESUME_BUFFER
export RESUME_CHECKPOINT=""
export RESUME_BUFFER=""

export IMAGE_SIZE=240
export CAMERA_SIZE=240
export TOPDOWN_CLOSE=1
export MAX_ITERS=400
export NUM_ENVS=4
export COLLECT_STEPS=256
export MAX_AGGREGATE_EPISODES=200
export EVAL_INTERVAL=25
export EVAL_EPISODES=10
export NUM_EVAL_ENVS=4
export BETA_ADAPTIVE=1
export NO_RESUME_BUFFER=1
export BUFFER_SAVE_INTERVAL=0

echo "=== S1 camera verify: B1 -> B7 ==="
echo "  tag=${RUN_TAG}"
echo "  camera=${CAMERA_SIZE}px topdown_close=${TOPDOWN_CLOSE} image_size=${IMAGE_SIZE}"
echo "  iters=${MAX_ITERS} envs=${NUM_ENVS} eval=${EVAL_INTERVAL}x${EVAL_EPISODES}"
echo "  logs=${LOG_ROOT}"

{
  echo "[$(date -Is)] Starting B1..."
  LOG_DIR="${LOG_ROOT}/B1" \
    bash "${ROOT}/scripts/run_s1_dagger_smoke.sh" B1
  echo "[$(date -Is)] B1 done."

  echo "[$(date -Is)] Starting B7..."
  LOG_DIR="${LOG_ROOT}/B7" \
    bash "${ROOT}/scripts/run_s1_dagger_smoke.sh" B7
  echo "[$(date -Is)] B7 done."
} 2>&1 | tee "${LOG_ROOT}/verify.log"

echo "[OK] TensorBoard:"
echo "  tensorboard --logdir ${LOG_ROOT}/B1/tb,${LOG_ROOT}/B7/tb"
echo "  dense eval tags: eval/pre_grasp_min_ee_cube, eval/post_grasp_dist_goal"
