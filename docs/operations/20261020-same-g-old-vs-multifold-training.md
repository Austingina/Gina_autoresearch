# Operation 20261020-SAME-G-COVERAGE-COMPARISON

- **Stage:** CALVIN coarse-G controlled training comparison
- **Target:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Status:** `COMPLETED_MIXED_RESULT`
- **Operator:** Hermes Agent for user
- **Created:** 2026-10-20T08:11:14+08:00

## Authoritative requirement

Modify and train the existing CALVIN pipeline to test whether the **same coarse guidance field G**, under the old single-fold coverage versus the new 12-fold combination coverage, produces more reliable compositional generalization.

## Scientific contract

The only intended causal variable is dataset coverage:

- old condition: 324 segments, single intent-factorial coverage;
- new condition: 432-segment superset, 12 deterministic semantic folds;
- fixed across conditions: G schema, geometry/calibration, preprocessing, policy mode, architecture, optimizer, learning-rate schedule, update/example budget, initialization seed, data-loader seed, validation rule, evaluation sequence pool, and metric aggregation;
- required reporting: source and held-out metrics, 12-fold mean, worst fold, cross-fold standard deviation, per-cell values, and G-control audits (`normal`, `zero_g`, `sign_flip_f`, `wrong_entity`) where labels permit;
- no claim of cross-simulator or cross-robot generalization from CALVIN-only evidence.

## Evidence-backed baseline

Captured at 2026-10-20T08:08:20+08:00:

- Git: `master@3d4aeba1badaa3d6ce781033e0d9187729dfec54`, dirty experimental copy; original `/home/ai-sz-a26394-u1/AutoResearch` remains the frozen source.
- Old data: `data/calvin/task_ABC_D_g_intent_factorial_s12_h3`, 324 selected segments.
- New data: `data/calvin/task_ABC_D_g_intent_multifold_s12`, manifest complete with 432 segments, 25,852 frames, 12 fold manifests.
- Neither target currently contains an interaction-volume/minimal-G packed cache for this comparison.
- New manifest explicitly reports:
  - `phase_label_status=not_generated_requires_state_grounded_phase_builder`;
  - `negative_trajectory_status=not_present_requires_simulator_rollout`;
  - `container_factorial_status=coverage_only_missing_crossed_primitive_container_tasks`.
- Existing cache builder writes only selected `--tasks` and marks `heldout_tasks_written=false`; fold-aware selection/adoption is not yet implemented.
- Existing audit venv `.venv-data-audit` has no PyTorch.
- CALVIN code/runtime paths used by historical runs are absent locally.
- GPU gate currently fails: kernel module `595.91.07`, userspace/NVML `595.99.02`; `nvidia-smi` reports `Driver/library version mismatch`.
- Disk has approximately 1.3 TiB free.
- No active target training process was found.

## Scope

1. Discover a valid local Python/PyTorch/CALVIN environment or record a prerequisite blocker.
2. Add deterministic, fold-aware data/cache selection with strict parity receipts using TDD.
3. Generate same-schema G caches for old and new data without overwriting raw episodes.
4. Run CPU or GPU smoke training that exercises the exact production path.
5. Run the formal controlled comparison only after GPU/runtime gates pass.
6. Aggregate metrics and update this record with actual evidence.

## Explicit non-goals

- No modification of `/home/ai-sz-a26394-u1/AutoResearch`.
- No deletion or replacement of episode files, prior manifests, or prior experiments.
- No fabricated failed expert trajectories; audit-only controls remain non-expert labels.
- No reboot, NVIDIA driver replacement, package-manager system mutation, or `sudo` without separate user authorization.
- No claim that offline MSE alone proves policy success; simulator rollout metrics are required for behavior claims.

## Expected changed paths

- `framework/scripts/data/` — fold-aware cache/selection code and tests.
- `framework/scripts/policy_state_bc/` — only if a tested fold-selection seam is absent.
- `experiments/phase1_calvin/2026-10-20_same_g_coverage_comparison/` — configs, commands, checkpoints, metrics, and logs.
- `artifacts/evidence/20261020T081114-same-g-coverage-comparison/` — immutable evidence receipts.
- This operation record and its mirrored copy under `/home/ai-sz-a26394-u1/operation-records/`.

## Evidence directory and no-overwrite policy

Planned: `artifacts/evidence/20261020T081114-same-g-coverage-comparison/`.

The directory must be newly created, must not overwrite prior evidence, and must contain initial provenance before any code/data mutation.

## Safety and concurrency gates

1. Abort writes if another process owns the same cache, output, or experiment directory.
2. Abort formal GPU training while `nvidia-smi` or `torch.cuda.is_available()` fails.
3. Refuse to overwrite an existing cache/checkpoint; use unique condition/fold paths.
4. Refuse comparison if G schema/calibration hashes differ between old and new conditions.
5. Refuse comparison if model/hyperparameter/update-budget receipts differ outside the declared coverage fields.
6. Keep held-out episode identities fixed per task/scene where the protocol requires it.
7. Keep held-out tasks/segments out of fitting and normalization statistics.

## Numbered execution plan

### 1. Capture provenance and prerequisites

Expected evidence:

- timestamp, branch/HEAD/status path list and hashes;
- dataset/manifest/fold hashes and counts;
- process ownership and disk state;
- Python, PyTorch, CALVIN and NVIDIA runtime gates.

Failure exit: mark `BLOCKED` and do not start formal training if no valid runtime exists.

### 2. TDD fold-aware selection/cache contract

RED expectations:

- old and new conditions cannot yet emit identical-G comparison receipts;
- fold held-out segments must be rejected from training/cache fitting;
- mismatch in G schema/calibration hash must fail closed;
- mismatch in non-coverage hyperparameters must fail closed.

GREEN expectations:

- deterministic source/held-out segment IDs for each fold;
- cache contains only declared source IDs, while evaluation references fixed held-out IDs;
- receipts expose dataset manifest hash, fold manifest hash, G schema hash, calibration hash, episode IDs, frame counts, and command/config hash.

### 3. Generate old/new same-G caches

Planned rule:

- same geometry/calibration file and hash;
- separate output paths, atomic temp-to-final replacement;
- no raw episode mutation;
- shape, dtype, frame-index, segment-ID, task/scene and finite-value checks.

### 4. Smoke training and evaluation

Run a bounded one-fold/one-seed smoke on the exact chosen policy mode and loader path, with small windows/epochs and explicit device. Verify checkpoint creation, metrics JSON, deterministic rerun compatibility, and held-out isolation.

### 5. Formal training

After all gates pass, run paired conditions with fixed seeds and equalized budget. Formal run should not silently fall back from CUDA to CPU.

### 6. Aggregate and interpret

Produce per-fold source/held-out metrics, mean, worst, standard deviation, paired deltas, G-control sensitivity, and limitations. Mark conclusions as offline-only if simulator rollouts cannot run.

## Rollback and recovery

- Source edits: restore exact pre-operation backups from the evidence directory, never `git restore` over pre-dirty files.
- Generated cache/experiment outputs: leave failed attempts preserved and marked incomplete; do not delete evidence.
- Processes: terminate only PIDs started by this operation and record their argv/start time.
- Raw data: immutable; any detected episode mutation is an immediate `FAILED` condition.

## PASS/PARTIAL/BLOCKED/FAILED criteria

- **PASS:** tested fold-aware path, hash-identical G contract across conditions, verified caches, completed paired formal training and held-out evaluation, metrics aggregated with no leakage.
- **PARTIAL:** code/caches/smoke pass but formal training or simulator rollout remains unavailable; no generalization claim.
- **BLOCKED:** runtime/GPU/CALVIN dependency gate prevents the first valid smoke or formal run.
- **FAILED:** leakage, raw-data mutation, parity mismatch, corrupted cache/checkpoint, or unexplained process/output ownership conflict.

## Actual results

### Implementation and gates

- Added manifest-exact `(task,start,end)` source/held-out selection with manifest SHA-256 receipts.
- Added immutable primitive-geometry loading and receipt; both conditions used `primitive_geometry.json` SHA-256 `5c950473476f103c7852ecb004ad25bd8f778eb50f9be8ea75a0925aded3e128`.
- Added verified generic packed state caches:
  - old: 19,354 frames, SHA-256 `faedceae7e37ff65b4f458bf82caeed07c00c6be67a09c33c62237e1afe95499`;
  - new: 25,852 frames, SHA-256 `d5185438a92eaad271a9ba4d7d27750b7c1f63ee8c5809d37e961b0c9cdcb0f8`.
- Added NumPy 2 pickle compatibility for the pinned NumPy 1 runtime and explicit trusted-checkpoint loading for PyTorch 2.6+.
- RED/GREEN tests and both old/new CUDA smoke runs passed; offline held-out evaluator passed.

### Formal training parity

Thirteen CUDA checkpoints completed: one old legacy split and all 12 new folds. Every run used exactly 3,000 training windows, 30 epochs, batch 256, chunk 16, stride 4, mode `chunk_bc_minimal_g_concat`, proprio-only executor state, no direct language/task one-hot, seed `20260814`, the same articulation geometry, and the same primitive-G hash. Old source selection contained 288 exact segments; each new fold contained 324 exact source segments; no held-out segment was used for fitting.

### Held-out offline result

- Old single legacy split: first-action MSE `0.0736221`, first-XYZ MSE `0.1063403`, gripper accuracy `0.983759`.
- New 12-fold aggregate:
  - first-action MSE mean `0.0617741`, population std `0.0268451`, best `0.0374333`, worst `0.1329268`;
  - first-XYZ MSE mean `0.0583089`, population std `0.0360360`, best `0.0270028`, worst `0.1403841`;
  - gripper accuracy mean `0.962738`, population std `0.0250601`, worst `0.881410`.
- Mean new-minus-old delta: first-action MSE `-0.0118479`, first-XYZ MSE `-0.0480313`, gripper accuracy `-0.0210208`.
- New-fold G-control sensitivity: mean prediction delta L2 was `0.278254` for `zero_g` and `0.432761` for `sign_flip_f`; however several folds improved rather than degraded under these counterfactuals, so consistent causal use of G is not established.

### Interpretation

The expanded coverage improves average offline translational imitation error, but it is **not yet more reliable in the strict worst-fold sense**: the worst new fold is worse than the old single split and fold variance remains material. The old condition is one legacy held-out split while the new condition is a 12-fold aggregate, so this is not a paired significance test. CALVIN simulator success-rate evaluation was not run because neither subset contains the required `validation/` environment dataset; conclusions are limited to offline held-out first-action imitation and counterfactual sensitivity.

### Outputs

- Aggregate report: `experiments/phase1_calvin/2026-10-20_same_g_coverage_comparison/comparison_report.json`
- Report SHA-256: `8fc282a25a4a53521022a11d9b42b89c0e07cb3f65f3eeb264e6032250809284`
- Formal checkpoints/evaluations: `experiments/phase1_calvin/2026-10-20_same_g_coverage_comparison/formal/`
- Evidence: `artifacts/evidence/20261020T081114-same-g-coverage-comparison/`

## Next allowed operation

Add the missing CALVIN validation environment data or construct a simulator-backed deterministic held-out reset set, then run success-rate rollouts and multi-seed paired models; do not claim simulator-level or universal G generalization from the present offline result.