# CALVIN Task-Metric Protocol

Offline reconstruction/action MSE is not the final objective. It is a diagnostic
for whether the G representation is structured enough to justify policy-level
experiments.

Primary task metrics:

- average successful sequence length
- chain success rate for 1..5 instructions
- per-subtask success/total

Experiment ladder:

1. evaluator smoke with zero/random policy
2. no-G CALVIN policy baseline
3. oracle-G upper bound
4. predicted-G policy only after the predictor passes diagnostics
5. shuffled/constant-G controls

Canonical wrapper:

```bash
GPU=2 POLICY=zero NUM_SEQUENCES=2 EP_LEN=60 \
  /mnt/nvme33n1/wuk_code/AutoResearch/framework/scripts/run_calvin_task_eval.sh
```

Checkpoint evaluation:

```bash
GPU=2 POLICY=checkpoint NUM_SEQUENCES=1000 EP_LEN=360 \
TRAIN_FOLDER=/path/to/train_folder \
CHECKPOINT=/path/to/epoch.ckpt \
  /mnt/nvme33n1/wuk_code/AutoResearch/framework/scripts/run_calvin_task_eval.sh
```

Decision rule: paper claims should be based on rollout success metrics, not G
MSE alone.

Smoke verified on 2026-07-03:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_053325_calvin_task_metric_zero_smoke_n1_h5_final
state=completed, status=0
avg_seq_len=0.0, chain_sr@1..5=0.0
task_info: open_drawer 0/1
```

Checkpoint smoke also verified:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_061300_calvin_task_metric_checkpoint_debug_smoke_n1_h5_mixedpath
state=completed, status=0
checkpoint loaded successfully
avg_seq_len=0.0, chain_sr@1..5=0.0
task_info: open_drawer 0/1
```

The checkpoint was a one-epoch debug checkpoint, so this only validates the
evaluation path, not policy quality.
