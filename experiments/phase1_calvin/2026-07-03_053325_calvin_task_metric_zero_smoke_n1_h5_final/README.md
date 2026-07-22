# calvin_task_metric_zero_smoke_n1_h5_final

- created_at: 2026-07-03_053325
- policy: zero
- gpu: 3
- calvin_code: /mnt/nvme33n1/wuk_code/AutoResearch/phase1_calvin/calvin_code
- calvin_env: /mnt/nvme33n1/wuk_code/AutoResearch/phase1_calvin/envs/calvin_h200
- dataset: /mnt/nvme33n1/wuk_code/AutoResearch/phase1_calvin/data/task_D_D
- num_sequences: 1
- ep_len: 5

## Intent

Run CALVIN long-horizon task-level metrics using the official evaluator.

## Primary Metrics

- average successful sequence length
- chain success rate for 1..5 instructions
- per-subtask success/total

## Interpretation

This is the primary evaluation layer for policy claims. Offline MSE/proxy
metrics are only representation diagnostics.
