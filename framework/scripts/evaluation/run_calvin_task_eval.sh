#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
CALVIN_CODE="${CALVIN_CODE:-$ROOT/phase1_calvin/calvin_code}"
CALVIN_ENV="${CALVIN_ENV:-$ROOT/phase1_calvin/envs/calvin_h200}"
CALVIN_DATASET="${CALVIN_DATASET:-$ROOT/phase1_calvin/data/task_D_D}"
GPU="${GPU:-2}"
POLICY="${POLICY:-zero}"
NUM_SEQUENCES="${NUM_SEQUENCES:-2}"
EP_LEN="${EP_LEN:-60}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-calvin_task_metric_${POLICY}_n${NUM_SEQUENCES}_h${EP_LEN}}"
STAMP="$(date +%F_%H%M%S)"
RUN_DIR="${RUN_DIR:-$ROOT/experiments/phase1_calvin/${STAMP}_${RUN_NAME}}"
TRAIN_FOLDER="${TRAIN_FOLDER:-}"
CHECKPOINT="${CHECKPOINT:-}"

mkdir -p "$RUN_DIR"/{logs,metrics,artifacts,configs}

test -d "$CALVIN_CODE/calvin_models"
test -d "$CALVIN_ENV"
test -f "$CALVIN_DATASET/validation/.hydra/merged_config.yaml"

ARGS=(
  "$ROOT/framework/scripts/evaluation/calvin_task_metric_eval.py"
  "--calvin-code" "$CALVIN_CODE"
  "--dataset-path" "$CALVIN_DATASET"
  "--eval-log-dir" "$RUN_DIR/metrics/task_eval"
  "--policy" "$POLICY"
  "--device" "0"
  "--num-sequences" "$NUM_SEQUENCES"
  "--ep-len" "$EP_LEN"
  "--seed" "$SEED"
)

if [ "$POLICY" = "checkpoint" ]; then
  test -n "$TRAIN_FOLDER"
  test -n "$CHECKPOINT"
  ARGS+=("--train-folder" "$TRAIN_FOLDER" "--checkpoint" "$CHECKPOINT")
fi

cat > "$RUN_DIR/README.md" <<RUNEOF
# $RUN_NAME

- created_at: $STAMP
- policy: $POLICY
- gpu: $GPU
- calvin_code: $CALVIN_CODE
- calvin_env: $CALVIN_ENV
- dataset: $CALVIN_DATASET
- num_sequences: $NUM_SEQUENCES
- ep_len: $EP_LEN

## Intent

Run CALVIN long-horizon task-level metrics using the official evaluator.

## Primary Metrics

- average successful sequence length
- chain success rate for 1..5 instructions
- per-subtask success/total

## Interpretation

This is the primary evaluation layer for policy claims. Offline MSE/proxy
metrics are only representation diagnostics.
RUNEOF

{
  echo "created_at=$STAMP"
  echo "host=$(hostname)"
  echo "root=$ROOT"
  echo "policy=$POLICY"
  echo "gpu=$GPU"
  echo "run_dir=$RUN_DIR"
  echo "calvin_code=$CALVIN_CODE"
  echo "calvin_env=$CALVIN_ENV"
  echo "calvin_dataset=$CALVIN_DATASET"
  echo "train_folder=$TRAIN_FOLDER"
  echo "checkpoint=$CHECKPOINT"
  echo "num_sequences=$NUM_SEQUENCES"
  echo "ep_len=$EP_LEN"
  echo
  df -h / /mnt/nvme33n1 /mnt/raid || true
  echo
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo
  conda run --no-capture-output -p "$CALVIN_ENV" python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available())" || true
} > "$RUN_DIR/env.txt" 2>&1

{
  echo "# AutoResearch"
  git -C "$ROOT" rev-parse HEAD || true
  git -C "$ROOT" status --short || true
  echo
  echo "# CALVIN"
  git -C "$CALVIN_CODE" rev-parse HEAD || true
  git -C "$CALVIN_CODE" status --short || true
  echo
  echo "# CALVIN env"
  git -C "$CALVIN_CODE/calvin_env" rev-parse HEAD || true
  git -C "$CALVIN_CODE/calvin_env" status --short || true
} > "$RUN_DIR/git_status.txt" 2>&1

{
  echo "POLICY=$POLICY"
  echo "NUM_SEQUENCES=$NUM_SEQUENCES"
  echo "EP_LEN=$EP_LEN"
  echo "CALVIN_DATASET=$CALVIN_DATASET"
  echo "TRAIN_FOLDER=$TRAIN_FOLDER"
  echo "CHECKPOINT=$CHECKPOINT"
} > "$RUN_DIR/configs/eval_overrides.env"

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'cd %q\n' "$ROOT"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf 'PYTHONPATH=%q:%q:${PYTHONPATH:-} ' "$CALVIN_CODE/calvin_models" "$CALVIN_CODE/calvin_env"
  printf 'conda run --no-capture-output -p %q python ' "$CALVIN_ENV"
  printf '%q ' "${ARGS[@]}"
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
  echo "[done] task eval completed: $RUN_DIR"
else
  echo "failed" > "$RUN_DIR/state.txt"
  echo "[fail] task eval failed with status $STATUS: $RUN_DIR" >&2
fi
exit "$STATUS"
