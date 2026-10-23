# Operation 20261020-CALVIN-PROFILING-RECURSION-FIX

- **Stage:** TDD prerequisite repair for simulator rollout
- **Target:** `framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py`
- **Status:** `PASS`
- **Created:** 2026-10-20T10:10:14+08:00

## Reproduced defect

With `profile_eval_timing=True`, `timed_model_step` closes over local `model_step`; the rollout loop later reassigns that same local name to the installed wrapper, so the wrapper recursively calls itself and raises `RecursionError` before the first environment action.

## Scope

Rename the profiling closure's original-callable variables and the rollout trace callable variable so they cannot alias. No model, data, metric, environment or rollout semantics may change.

## TDD

1. Preserve pre-edit source hash and file.
2. Add an integration regression invoking one state-only CALVIN sequence with timing enabled; verify RED is `RecursionError`.
3. Apply the minimum variable-separation patch.
4. Verify GREEN, existing policy tests, compile, and operation-local diff.

## Rollback

Restore only the timestamped backup of the target file.

## Actual results

- RED reproduced exact `RecursionError` before any environment action.
- GREEN completed one deterministic state-only CALVIN rollout with timing enabled: 2 policy calls and 2 environment steps; the untrained-for-simulator quick checkpoint scored `0/1`, which is expected test evidence rather than a performance conclusion.
- Minimal change separated profiling closure callables from per-rollout trace callables; no model/data/metric logic changed.
- Python compile passed; operation-local diff and pre/post hashes are under `artifacts/evidence/20261020T101014-calvin-profiling-recursion/`.
