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

## Smoke Result

Zero-policy smoke completed on the server:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_053325_calvin_task_metric_zero_smoke_n1_h5_final
```

Command:

```bash
GPU=3 POLICY=zero NUM_SEQUENCES=1 EP_LEN=5 \
RUN_NAME=calvin_task_metric_zero_smoke_n1_h5_final \
  /mnt/nvme33n1/wuk_code/AutoResearch/framework/scripts/run_calvin_task_eval.sh
```

Result:

```text
state: completed
status: 0
avg_seq_len: 0.0
chain_sr@1..5: 0.0, 0.0, 0.0, 0.0, 0.0
task_info: open_drawer 0 / 1
```

This confirms that the CALVIN simulator, task oracle, EGL rendering path, and
official task-metric JSON writer are usable in the current server environment.

Checkpoint-policy smoke also completed:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_061300_calvin_task_metric_checkpoint_debug_smoke_n1_h5_mixedpath
```

Command:

```bash
GPU=5 POLICY=checkpoint NUM_SEQUENCES=1 EP_LEN=5 \
CALVIN_DATASET=/mnt/nvme33n1/wuk_code/AutoResearch/phase1_calvin/data/calvin_debug_dataset \
TRAIN_FOLDER=/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-02_064810_calvin_dualcam_debug_train/calvin_logs/runs/2026-07-02/06-48-20 \
CHECKPOINT=/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-02_064810_calvin_dualcam_debug_train/calvin_logs/runs/2026-07-02/06-48-20/saved_models/epoch=0.ckpt \
RUN_NAME=calvin_task_metric_checkpoint_debug_smoke_n1_h5_mixedpath \
RUN_DIR=/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_061300_calvin_task_metric_checkpoint_debug_smoke_n1_h5_mixedpath \
  /mnt/nvme33n1/wuk_code/AutoResearch/framework/scripts/run_calvin_task_eval.sh
```

Result:

```text
state: completed
status: 0
checkpoint loaded: yes
avg_seq_len: 0.0
chain_sr@1..5: 0.0, 0.0, 0.0, 0.0, 0.0
task_info: open_drawer 0 / 1
```

This confirms that the official CALVIN checkpoint loading path is also usable.
The checkpoint was only a one-epoch debug model, so the metric has no method
significance.

## Wrapper Notes

The wrapper applies two compatibility fixes without editing CALVIN source:

- restores removed NumPy aliases such as `np.int` for older tacto/pyrender code
- resolves paths before CALVIN changes cwd, while passing `dataset_path` as a
  string and `checkpoint` as a Path to match CALVIN's mixed API expectations

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
