# Operation 20261020-SAME-G-FULL-TRAINING-SIM-EVAL

- **Stage:** CALVIN same-G full-data training and simulator evaluation
- **Target:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Status:** `BLOCKED`
- **Result:** `PENDING`
- **Operator:** Hermes Agent for user
- **Created:** 2026-10-20T09:56:47+08:00

## Authoritative requirements

Run a true old-coverage versus new-coverage comparison using the original repository's formal style: all available windows, 50 epochs, CALVIN task execution, fixed model/G/optimizer/seeds, and held-out-only evaluation. Preserve `/home/ai-sz-a26394-u1/AutoResearch` read-only at `master@3d4aeba1badaa3d6ce781033e0d9187729dfec54`.

## Baseline

- GPU: RTX 4090, CUDA tensor runtime available; 24,572,264,448 bytes free at 2026-10-20T09:56:01+08:00.
- Disk: 1,371,545,976,832 bytes free.
- No competing CALVIN training process was found.
- Old data: 324 segments, exact source 288 / held-out 36.
- New data: 432-segment superset, 12 exact folds, each source 324 / held-out 27.
- Existing quick experiment is offline-only and is not reused as formal evidence.
- Neither subset currently contains `validation/`; official sparse ZIP index contains `task_ABC_D/validation/.hydra/merged_config.yaml` (7,287 bytes; compressed 1,951 bytes).
- CALVIN `get_env()` reads `validation/.hydra/merged_config.yaml`; state-only rollout does not require validation episode NPZ files.

## Goal

1. Fetch and CRC-verify the official validation merged config without downloading the 555 GB archive.
2. Build immutable per-condition dataset views: `training` symlink to existing data plus a common official `validation/.hydra/merged_config.yaml`.
3. Train one old split and 12 new folds with all source windows, 50 epochs, identical `chunk_bc_minimal_g_concat` model, batch size 256, chunk 16, stride 2, seed `20260814`, proprio-only executor state, no direct language/task one-hot, fixed articulation and primitive-G geometry.
4. Run held-out CALVIN simulator rollouts with 100 balanced first-task sequences per held-out task where deterministic sequence supply permits.
5. Run `normal`, `zero_g`, and `sign_flip_f` audits without treating counterfactuals as expert training data.
6. Report per fold, mean, worst fold, population standard deviation, and source-to-held-out limitations.

## Non-goals

- No Go2/Isaac locomotion changes.
- No raw episode overwrite or deletion.
- No modification of `/home/ai-sz-a26394-u1/AutoResearch`.
- No claim of cross-simulator, cross-robot, vision-language, or universal G generalization.
- No full 555 GB CALVIN download.

## Change scope

Allowed:
- this operation record and a new timestamped evidence directory;
- experiment-specific validation metadata and symlinked dataset views;
- tests and launcher/report code in `Gina_autoresearch`;
- checkpoints, logs, rollout traces and metrics under `experiments/phase1_calvin/2026-10-20_same_g_full_formal/`.

Forbidden:
- raw `episode_*.npz`, existing manifests, old checkpoints, original repository, robot workspaces.

## Safety and fairness gates

- Fail if official validation config CRC/size differs from ZIP central-directory metadata.
- Fail if a dataset view resolves `training` anywhere other than the declared old/new roots.
- Fail on source/held-out `(task,start,end)` overlap.
- Fail if shared hyperparameters, G geometry hashes, model code hash, optimizer/update scheme, or seeds differ.
- `--max-windows 0`; use every source window.
- Evaluate only the fold's manifest-declared held-out tasks.
- No silent CUDA-to-CPU fallback.
- Preserve failed attempts and exact logs.

## Evidence

New directory: `artifacts/evidence/20261020T095647-same-g-full-formal/`; creation must refuse overwrite.

## Planned procedure

1. Create evidence directory and snapshot time/Git/GPU/disk/process state.
2. Fetch the one official validation config member by bounded HTTP Range using the local sparse ZIP index; verify size and CRC.
3. TDD a view/launcher preflight that enforces symlink target, 13 conditions, all-window mode, parity, and held-out isolation.
4. Instantiate a CALVIN environment from the minimal validation view and run reset + no-op close test.
5. Run one bounded full-loader training/one held-out simulator rollout smoke.
6. Launch sequential 1+12 full runs, each with 50 epochs and checkpoint preservation; then normal and G-counterfactual rollouts.
7. Aggregate results and update this same record with actual times, counts, hashes, incidents and conclusion.

## Rollback/recovery

- Restore modified pre-dirty source files only from timestamped backups; never use broad Git restore.
- Failed views/checkpoints/logs remain preserved and marked incomplete.
- Stop only process IDs launched by this operation.
- Symlink/view removal is allowed only in a separate explicitly authorized cleanup; raw data is never touched.

## Abort conditions

Unexpected concurrent GPU owner, changed source hashes, insufficient disk, validation config mismatch, source/held-out leakage, simulator reset failure, missing success oracle, parity mismatch, corrupted checkpoint, or unexplained process ownership.

## Completion criteria

- **PASS:** all 13 full trainings and manifest-held-out simulator evaluations complete with verified parity and aggregate report.
- **PARTIAL:** training completes but some simulator tasks lack deterministic sequence supply; report exact gaps without substituting offline metrics.
- **BLOCKED:** official validation config or simulator runtime cannot be made valid within this operation's declared scope.
- **FAILED:** leakage, raw mutation, parity violation, corrupted outputs, or unowned process conflict.

## Actual results

- **Execution status:** `BLOCKED` before formal training; no formal checkpoint was started.
- Official validation `merged_config.yaml` was fetched by bounded Range and verified: 7,287 bytes, ZIP CRC `0x38bb16dc`, SHA-256 `75c1c15067e4bd543d1b69a654769836f6cbbf55c4e44ae30da255c45609523f`.
- The 13-condition plan and dataset views passed exact source/held-out and symlink-target tests.
- State-only CALVIN environment reset passed after excluding unused cameras and adding the checked-out `tacto` package root to `PYTHONPATH`.
- The first rollout smoke then stopped during CALVIN evaluator import: Transformers `4.46.3` requires `huggingface-hub>=0.23.2,<1.0`, while the shared runtime contains `huggingface-hub==1.25.1`.
- Preserved evidence: `artifacts/evidence/20261020T095647-same-g-full-formal/`.
- Required recovery: create an experiment-local dependency overlay; do not mutate the shared AST runtime in place.

## Next allowed operation

Create and validate a separate experiment-local CALVIN runtime overlay, then open a new formal-training retry record and evidence directory.