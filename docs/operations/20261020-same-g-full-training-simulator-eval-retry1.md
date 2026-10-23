# Operation 20261020-SAME-G-FULL-TRAINING-SIM-EVAL-RETRY1

- **Stage:** retry after verified CALVIN runtime recovery
- **Target:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Status:** `IN_PROGRESS`
- **Result:** `PENDING`
- **Created:** 2026-10-20T10:12:32+08:00

## Provenance

Retry of blocked operation `20261020-SAME-G-FULL-TRAINING-SIM-EVAL`; the blocked evidence remains unchanged. Prerequisites now verified: official validation config CRC/SHA, 13 exact dataset views, experiment-local hub `0.26.5` overlay, `tacto` path, camera-free state environment, and GREEN one-sequence simulator rollout after TDD recursion fix.

## Frozen protocol

- Conditions: old legacy source once plus all 12 new folds.
- Model: `chunk_bc_minimal_g_concat`, proprio-only, no direct language/task one-hot.
- Same articulation and primitive-G files and hashes across every run.
- Full source windows (`max_windows=0`), stride 2, chunk 16, batch 256, hidden 256, layers 3, 50 epochs.
- Seeds: `20260814` for initialization, loader, data ordering, split and rollout.
- Evaluation: balanced first task, manifest-declared held-out tasks only, 100 sequences per task, 160 steps, normal G; zero/sign-flip controls follow only after normal completion.
- Fail closed on config/hash/split/parity/CUDA/process mismatch; never modify raw data or the original repo.

## Outputs

- New evidence: `artifacts/evidence/20261020T101232-same-g-full-formal-retry1/`.
- Runs: `experiments/phase1_calvin/2026-10-20_same_g_full_formal/retry1/`.
- Existing quick checkpoints are not reused as formal checkpoints.

## Actual results

`PENDING`
