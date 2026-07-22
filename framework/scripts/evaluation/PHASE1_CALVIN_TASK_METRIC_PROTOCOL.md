# Phase 1 CALVIN Task-Metric Protocol

**Date:** 2026-07-03

## Position

Offline G MSE and action-proxy MSE are diagnostic gates only. The paper-level
claim must be evaluated by CALVIN rollout metrics:

- average successful sequence length
- chain success rate for 1..5 instructions
- per-subtask success/total

## Evaluation Ladder

1. `zero` or `random` policy smoke: verify simulator, task oracle, rendering, and
   JSON output.
2. no-G checkpoint: establish the actual task baseline.
3. oracle-G checkpoint: upper-bound whether reusable interaction geometry helps
   downstream control when the representation is correct.
4. predicted-G checkpoint: only run after predicted G passes diagnostic gates.
5. shuffled/constant-G controls: separate useful geometry from nuisance capacity.

## Minimal Smoke Command

```bash
GPU=2 POLICY=zero NUM_SEQUENCES=2 EP_LEN=60 \
  /mnt/nvme33n1/wuk_code/AutoResearch/framework/scripts/run_calvin_task_eval.sh
```

This produces:

```text
metrics/task_eval/results.json
metrics/task_eval/task_metric_summary.json
logs/stdout.log
env.txt
git_status.txt
command.sh
```

## Checkpoint Eval Command

```bash
GPU=2 POLICY=checkpoint NUM_SEQUENCES=1000 EP_LEN=360 \
TRAIN_FOLDER=/path/to/calvin_logs/runs/YYYY-MM-DD/HH-MM-SS \
CHECKPOINT=/path/to/calvin_logs/runs/YYYY-MM-DD/HH-MM-SS/saved_models/epoch=N.ckpt \
  /mnt/nvme33n1/wuk_code/AutoResearch/framework/scripts/run_calvin_task_eval.sh
```

## Interpretation Rule

Do not claim reusable-G improves policy learning from MSE alone. The minimum
credible evidence is:

- no-G vs oracle-G improves at least one task-level metric under the same
  training/eval budget, and
- predicted-G does not underperform no-G once used as policy input, or is framed
  only as an auxiliary/regularization signal.
