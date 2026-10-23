# Operation 20261020-MATCHED-TASK-MULTISEED-RETRAIN

- **Operation ID:** `20261020-matched-task-multiseed-retrain`
- **Stage:** matched-task, fixed-reset, multi-seed CALVIN retraining and simulator comparison
- **Target:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Status:** `IN_PROGRESS`
- **Result:** `PENDING`
- **Operator:** Hermes Agent
- **Created:** `2026-10-20T15:24:33+08:00`

## Authoritative requirement

Retrain and retest the current CALVIN coarse-G conditions using a fair protocol that separates task difficulty from fold coverage and random training variation.

## Evidence-backed baseline

- Existing full-formal retry contains 13 completed 50-epoch checkpoints: one old legacy condition and 12 multifold conditions.
- Existing normal-G simulator outputs contain 4,000 task rollouts and show 13.50% old versus 15.56% pooled new success, but pooled fold rates range from 0% to 63.33% because each fold evaluates a different three-task held-out set.
- Existing per-task evidence shows strong task dependence: drawer tasks are usually 0%, while one `move_slider_right` fold is 100%.
- Existing runs use only seed `20260814`; seed variance is therefore unknown.
- At `2026-10-20T15:24:33+08:00`, `nvidia-smi` fails with kernel/userspace mismatch (`595.91.07` versus `595.99.02`); formal CUDA retraining must not start until both NVML and `torch.cuda.is_available()` pass.
- Independent process PID `188225` is extracting the exact original `task_D_D` dataset into a separate declared directory; this operation must not terminate, modify, or reuse its partial output.

## Scientific protocol

1. Compare folds only on the same held-out task identity; do not interpret pooled rates across different task sets as a causal fold comparison.
2. Reuse identical deterministic CALVIN sequence indices and reset/config provenance for every comparison of the same task.
3. Train each condition with seeds `20260814`, `20260815`, and `20260816`.
4. Freeze model, G geometry/schema, optimizer, epochs, batch size, stride, chunk size, source selection, evaluation horizon, and per-task rollout count; only the declared condition/fold and seed may vary.
5. Use complete source windows but report actual windows/examples per condition; also preserve the existing equal-budget 3,000-window comparison as a separate reference rather than mixing it with full-data results.
6. Report per-task/per-seed success, mean, standard deviation, minimum, and paired differences on matched task/reset cells.
7. Keep normal-G results separate from `zero_g` and `sign_flip_f` controls.
8. No claim of universal, cross-robot, or cross-simulator generalization.

## Scope

- Add a deterministic matched-task/multi-seed plan builder and tests if the current builder cannot express the protocol.
- Generate unique seed-specific output directories without overwriting existing checkpoints or evaluations.
- Run RED→GREEN tests, a bounded smoke, then formal CUDA training only after runtime gates pass.
- Run normal-G matched-task simulator evaluation first; run controls only after normal evaluation completes.
- Produce machine-readable and Markdown aggregate reports.

## Explicit non-goals

- Do not modify `/home/ai-sz-a26394-u1/AutoResearch`.
- Do not mutate raw CALVIN episodes, existing checkpoints, existing evaluation files, or the active `task_D_D` extraction.
- Do not reboot, replace NVIDIA packages, use `sudo`, or terminate unrelated processes without separate authorization.
- Do not compare unmatched tasks as though their success rates measured the same difficulty.

## Mutable paths

- `framework/scripts/policy_state_bc/` for new plan/aggregation code and tests only.
- `experiments/phase1_calvin/2026-10-20_matched_task_multiseed/` for plans, checkpoints, evaluations, and reports.
- `artifacts/evidence/20261020T152433-matched-task-multiseed/` for immutable evidence.
- This operation record and its mirror under `/home/ai-sz-a26394-u1/operation-records/`.

## Safety and concurrency gates

1. Create a new evidence directory and refuse overwrite.
2. Capture Git HEAD/status/diff hashes, target-file hashes, disk, process ownership, GPU/NVML, CUDA, and active extraction evidence before mutation.
3. Abort if any output path has an unexplained writer or already contains an undeclared run.
4. Abort formal training unless `nvidia-smi` and runtime `torch.cuda.is_available()` both pass; no silent CPU fallback.
5. Abort if any matched task uses different sequence indices, reset/config hash, horizon, or rollout count across compared conditions/seeds.
6. Abort on source/held-out segment overlap, G hash drift, non-seed hyperparameter drift, NaN/Inf, corrupt checkpoints, or raw-data mutation.

## Numbered execution plan

1. Create evidence directory; capture timestamp, Git/provenance, process, disk, GPU/CUDA, manifests, existing run hashes, and active-download ownership.
2. RED: add a test requiring three declared seeds, matched held-out task groups, identical evaluation sequence IDs per task, and fail-closed parity checks.
3. GREEN: minimally extend or add the plan builder; run target test, affected regression tests, and syntax checks.
4. Generate a no-overwrite formal plan and validate source/held-out separation and matched-task membership.
5. Run one seed/condition smoke through the production loader, checkpoint writer, and matched-task evaluator.
6. Recheck GPU/CUDA and process gates; if blocked, write `NOT_RUN` receipts and stop without claiming training completion.
7. Train all declared condition×seed runs with unique directories and immutable receipts.
8. Evaluate normal G on identical per-task sequence pools; then run zero/sign-flip controls.
9. Aggregate per-task/per-seed mean, standard deviation, worst seed, matched paired deltas, and limitations.
10. Verify artifacts and adoption, hash evidence, update this record, and mirror it to `/home/ai-sz-a26394-u1/operation-records/`.

## Expected commands

- Runtime: `/home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/envs/calvin_reuse/bin/python`
- Tests: run exact new test first, then existing same-G plan/cache/evaluator regressions.
- Training/evaluation commands will be emitted by the audited generated plan and saved verbatim before execution; no hand-edited per-run divergence is allowed.

## Failure and abort conditions

- NVML/CUDA unavailable at formal gate.
- Existing extraction or another process owns any planned output.
- Matched task/reset parity mismatch.
- Fold leakage, G/config/hash drift, checkpoint corruption, failed simulator initialization, or raw-data mutation.
- Insufficient disk for all unique runs.

## Rollback and recovery

- Existing data, checkpoints, and evaluations are immutable and need no rollback.
- New source files are restored from timestamped backups; pre-dirty files are never reverted with `git restore`.
- Failed generated runs remain preserved and marked incomplete; only processes started by this operation may be stopped.
- A blocked GPU gate requires a separate authorized system-recovery operation or a later retry with a new evidence directory.

## Completion criteria

- **PASS:** all declared multi-seed trainings and matched-reset normal/control evaluations complete; reports and hashes verify.
- **PARTIAL:** protocol/code/smoke pass but some formal runs or controls remain incomplete.
- **BLOCKED:** prerequisite gate prevents valid smoke or formal CUDA execution.
- **FAILED:** leakage, parity drift, raw-data mutation, unexplained ownership conflict, or corrupt artifacts.

## Actual results

`PENDING`

## Next allowed operation

Only after this record closes: interpret matched-task/multi-seed evidence, then decide whether to rebalance tasks, change G, or add new task families.
