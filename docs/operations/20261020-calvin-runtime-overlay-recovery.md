# Operation 20261020-CALVIN-RUNTIME-OVERLAY-RECOVERY

- **Stage:** prerequisite recovery for CALVIN simulator evaluation
- **Target:** experiment-local overlay under `experiments/phase1_calvin/2026-10-20_same_g_full_formal/common/runtime_overlay/`
- **Status:** `PARTIAL`
- **Operator:** Hermes Agent for user
- **Created:** 2026-10-20T10:07:56+08:00

## Baseline and blocker

The shared AST `calvin_reuse` Python has Transformers `4.46.3` and `huggingface-hub 1.25.1`; importing CALVIN evaluation fails because Transformers requires hub `<1.0`. The shared environment must remain unchanged.

## Goal

Install only `huggingface-hub==0.26.5` into an experiment-local `--target` overlay, prepend that overlay and the checked-out `tacto` package root to `PYTHONPATH`, then prove imports and a one-sequence CALVIN rollout smoke work.

## Scope and safety

- Allowed: new overlay files and timestamped evidence only.
- Forbidden: shared AST environment, raw datasets, original repo, robot workspaces.
- Verify the shared environment still reports hub `1.25.1` after recovery.
- Fail if overlay import is not exactly `0.26.5`, CALVIN evaluator import fails, or rollout cannot create a result JSON.

## Commands

1. `pip install --no-deps --target <overlay> huggingface-hub==0.26.5`.
2. Import gate with overlay first in `PYTHONPATH`.
3. Repeat state-only one-sequence rollout smoke.
4. Record package files, hashes, shared-runtime non-mutation and outputs.

## Rollback

Preserve failed overlay evidence; removal requires a later cleanup operation. No shared environment rollback should be needed because it is not modified.

## Actual results

- Overlay installation passed: `huggingface-hub==0.26.5` imports with Transformers `4.46.3` and CALVIN evaluator.
- The shared AST environment remains unchanged at `huggingface-hub==1.25.1`.
- `tacto.Sensor` resolves when the checked-out `calvin_env/tacto` root precedes `calvin_env` in `PYTHONPATH`.
- The rollout advanced through evaluator import, environment creation, deterministic sequence selection and policy invocation, proving the dependency recovery; it then exposed an unrelated source bug: `profile_eval_timing` and per-rollout tracing reuse the same closure variable name, causing `timed_model_step` to call itself recursively.
- Evidence: `artifacts/evidence/20261020T100756-calvin-runtime-overlay/`.
- **Result:** runtime dependency overlay recovered; rollout completion remains blocked on a separately scoped profiling-wrapper bug fix.
