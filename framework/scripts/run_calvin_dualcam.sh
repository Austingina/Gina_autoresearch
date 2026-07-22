#!/usr/bin/env bash
set -euo pipefail

PRESET="${1:-smoke}"
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/../../simulators/resolve_paths.sh"
ROOT="${AUTORESEARCH_ROOT}"
CALVIN_MODELS="$CALVIN_CODE/calvin_models"
GPU="${GPU:-2}"
RUN_NAME="${RUN_NAME:-calvin_dualcam_${PRESET}}"
STAMP="$(date +%F_%H%M%S)"
RUN_DIR="${RUN_DIR:-$ROOT/experiments/phase1_calvin/${STAMP}_${RUN_NAME}}"

mkdir -p "$RUN_DIR"/{logs,metrics,artifacts,configs}

test -d "$CALVIN_MODELS"
test -d "$CALVIN_ENV"
test -f "$CALVIN_DATASET/training/.hydra/merged_config.yaml"

COMMON_OVERRIDES=(
  "logger=tb_logger"
  "datamodule/datasets=vision_lang"
  "datamodule/observation_space=${CALVIN_OBS_SPACE:-lang_rgb_static_gripper_rel_act}"
  "model/perceptual_encoder=${CALVIN_ENCODER:-gripper_cam}"
  "datamodule.root_data_dir=$CALVIN_DATASET"
  "trainer.devices=1"
  "log_dir=$RUN_DIR/calvin_logs"
)

case "$PRESET" in
  smoke)
    OVERRIDES=(
      "datamodule.datasets.vision_dataset.batch_size=1"
      "datamodule.datasets.lang_dataset.batch_size=1"
      "datamodule.datasets.vision_dataset.num_workers=0"
      "datamodule.datasets.lang_dataset.num_workers=0"
      "datamodule.datasets.vision_dataset.min_window_size=2"
      "datamodule.datasets.vision_dataset.max_window_size=2"
      "datamodule.datasets.lang_dataset.min_window_size=2"
      "datamodule.datasets.lang_dataset.max_window_size=2"
      "+trainer.fast_dev_run=true"
    )
    ;;
  debug_train)
    OVERRIDES=(
      "datamodule.datasets.vision_dataset.batch_size=${BATCH_SIZE:-2}"
      "datamodule.datasets.lang_dataset.batch_size=${BATCH_SIZE:-2}"
      "datamodule.datasets.vision_dataset.num_workers=${NUM_WORKERS:-0}"
      "datamodule.datasets.lang_dataset.num_workers=${NUM_WORKERS:-0}"
      "datamodule.datasets.vision_dataset.min_window_size=${WINDOW_SIZE:-8}"
      "datamodule.datasets.vision_dataset.max_window_size=${WINDOW_SIZE:-8}"
      "datamodule.datasets.lang_dataset.min_window_size=${WINDOW_SIZE:-8}"
      "datamodule.datasets.lang_dataset.max_window_size=${WINDOW_SIZE:-8}"
      "trainer.max_epochs=${MAX_EPOCHS:-1}"
      "+trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES:-20}"
      "+trainer.limit_val_batches=${LIMIT_VAL_BATCHES:-0}"
      "+trainer.num_sanity_val_steps=0"
      "+trainer.log_every_n_steps=1"
    )
    ;;
  debug_eval)
    OVERRIDES=(
      "datamodule.datasets.vision_dataset.batch_size=${BATCH_SIZE:-1}"
      "datamodule.datasets.lang_dataset.batch_size=${BATCH_SIZE:-1}"
      "datamodule.datasets.vision_dataset.num_workers=${NUM_WORKERS:-0}"
      "datamodule.datasets.lang_dataset.num_workers=${NUM_WORKERS:-0}"
      "datamodule.datasets.vision_dataset.min_window_size=${WINDOW_SIZE:-2}"
      "datamodule.datasets.vision_dataset.max_window_size=${WINDOW_SIZE:-2}"
      "datamodule.datasets.lang_dataset.min_window_size=${WINDOW_SIZE:-2}"
      "datamodule.datasets.lang_dataset.max_window_size=${WINDOW_SIZE:-2}"
      "trainer.max_epochs=${MAX_EPOCHS:-1}"
      "+trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES:-1}"
      "+trainer.log_every_n_steps=1"
    )
    ;;
  full_train)
    OVERRIDES=(
      "datamodule.datasets.vision_dataset.batch_size=${BATCH_SIZE:-32}"
      "datamodule.datasets.lang_dataset.batch_size=${BATCH_SIZE:-32}"
      "datamodule.datasets.vision_dataset.num_workers=${NUM_WORKERS:-8}"
      "datamodule.datasets.lang_dataset.num_workers=${NUM_WORKERS:-8}"
      "trainer.max_epochs=${MAX_EPOCHS:-100}"
      "+trainer.log_every_n_steps=${LOG_EVERY_N_STEPS:-50}"
    )
    ;;
  *)
    echo "Unknown preset: $PRESET" >&2
    echo "Available presets: smoke, debug_train, debug_eval, full_train" >&2
    exit 2
    ;;
esac

CMD=(conda run --no-capture-output -p "$CALVIN_ENV" python calvin_agent/training.py "${COMMON_OVERRIDES[@]}" "${OVERRIDES[@]}")

cat > "$RUN_DIR/README.md" <<RUNEOF
# $RUN_NAME

- created_at: $STAMP
- preset: $PRESET
- gpu: $GPU
- calvin_code: $CALVIN_CODE
- calvin_env: $CALVIN_ENV
- dataset: $CALVIN_DATASET

## Intent

Run CALVIN dual-camera policy baseline using rgb_static + rgb_gripper.

## Interpretation

Fill this after the run finishes.
RUNEOF

cat > "$RUN_DIR/notes.md" <<RUNEOF
# Notes: $RUN_NAME

## Question

What does this run test?

## Setup

- preset: $PRESET
- observation: rgb_static + rgb_gripper
- representation: no explicit G
- action decoder: CALVIN default MCIL/logistic policy unless overridden

## Result

Pending.

## Next

Pending.
RUNEOF

{
  echo "created_at=$STAMP"
  echo "host=$(hostname)"
  echo "root=$ROOT"
  echo "preset=$PRESET"
  echo "gpu=$GPU"
  echo "run_dir=$RUN_DIR"
  echo "calvin_code=$CALVIN_CODE"
  echo "calvin_env=$CALVIN_ENV"
  echo "calvin_dataset=$CALVIN_DATASET"
  echo
  df -h / /mnt/nvme33n1 /mnt/raid || true
  echo
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
  echo
  conda run --no-capture-output -p "$CALVIN_ENV" python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('arch', torch.cuda.get_arch_list()); print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)" || true
} > "$RUN_DIR/env.txt" 2>&1

{
  echo "# AutoResearch Phase 0"
  git -C "$ROOT/phase0_isaac/autoresearch" rev-parse HEAD || true
  git -C "$ROOT/phase0_isaac/autoresearch" status --short || true
  echo
  echo "# CALVIN"
  git -C "$CALVIN_CODE" rev-parse HEAD || true
  git -C "$CALVIN_CODE" status --short || true
  echo
  echo "# CALVIN env submodule"
  git -C "$CALVIN_CODE/calvin_env" rev-parse HEAD || true
  git -C "$CALVIN_CODE/calvin_env" status --short || true
  echo
  echo "# MacroRobotics reference"
  git -C "$ROOT/external/MacroRobotics" rev-parse HEAD || true
} > "$RUN_DIR/git_status.txt" 2>&1

printf '%s\n' "${COMMON_OVERRIDES[@]}" > "$RUN_DIR/configs/common_overrides.txt"
printf '%s\n' "${OVERRIDES[@]}" > "$RUN_DIR/configs/preset_overrides.txt"

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'cd %q\n' "$CALVIN_MODELS"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${CMD[@]}"
  printf '2>&1 | tee %q\n' "$RUN_DIR/logs/stdout.log"
} > "$RUN_DIR/command.sh"
chmod +x "$RUN_DIR/command.sh"

printf '[run] %s\n' "$RUN_DIR"
printf '[cmd] %s\n' "$RUN_DIR/command.sh"

set +e
bash "$RUN_DIR/command.sh"
STATUS=$?
set -e

echo "$STATUS" > "$RUN_DIR/status.txt"
if [ "$STATUS" -eq 0 ]; then
  echo "completed" > "$RUN_DIR/state.txt"
  echo "[done] run completed: $RUN_DIR"
else
  echo "failed" > "$RUN_DIR/state.txt"
  echo "[fail] run failed with status $STATUS: $RUN_DIR" >&2
fi
exit "$STATUS"
