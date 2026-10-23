# Operation 20261021 — Gina LIBERO middle-drawer expert pure-logic TDD

- **Operation ID:** `20261021-gina-libero-middle-drawer-expert-tdd`
- **Stage:** implementation / unit verification
- **Target:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Status:** `SUPERSEDED BY MAIN-THREAD CONTRACT CORRECTION`
- **Operator:** Hermes Agent
- **Recorded:** `2026-10-21T23:07:51+08:00`
- **Git branch / HEAD:** `master` / `3d4aeba1badaa3d6ce781033e0d9187729dfec54`

## Authoritative requirements

User request in the current session: implement a minimal pure-logic online expert for pinned LIBERO `libero_goal` task 0, `open_the_middle_drawer_of_the_cabinet`, using strict RED→GREEN TDD. No LIBERO runtime, rollout, or installation is permitted. `.venv-calvin-unit/bin/python` is the accepted Python because it provides NumPy 1.26.4.

## Baseline

The requested implementation and test paths do not exist. The worktree is already dirty with unrelated user/project work; this operation owns only the declared paths below. Existing protected files:

- `framework/sim_data/gates.py`: SHA-256 `ab7f7a0aeb4005838fef5de1a1238d0cbbb1f79444542a10617df848d4767625`
- `framework/scripts/data/test_native_collection_contract.py`: SHA-256 `261663867736459b31ab02a628775a86201087f813c08a0efd30137d0b3d5cf3`

## Goal

Create and verify a dependency-light expert slice covering:

1. finite, clipped, exactly 7-dimensional actions;
2. fail-closed discovery of exactly one slide joint, finite ordered joint range, and associated body from fake MuJoCo-like model/data;
3. `MIDDLE_HANDLE_LOCAL` body-frame-to-world transformation;
4. pull direction selected so commanded motion decreases drawer joint position;
5. FSM `PREGRASP → APPROACH → CLOSE_GRIPPER → PULL → HOLD`, with consecutive error confirmations;
6. fail-safe behavior for NaN and missing geometry;
7. no success-label dependence on drawer qpos, reward, done, or G.

## Explicit non-goals

- No LIBERO installation, import, rollout, recorder, HDF5, or adapter.
- No modification to upstream LIBERO, `framework/sim_data/gates.py`, or any G code.
- No simulator success-label implementation; this module emits actions/state diagnostics only.

## Scope / mutable paths

- `framework/sim_data/libero_middle_drawer_expert.py` (new)
- `framework/scripts/data/test_libero_middle_drawer_expert.py` (new)
- `docs/operations/20261021-gina-libero-middle-drawer-expert-tdd.md` (this record)
- `docs/operations/evidence/20261021-gina-libero-middle-drawer-expert-tdd-230751/` (new, no overwrite)

No process, service, external system, dataset, upstream LIBERO file, gate, or G artifact may change.

## Prerequisites and safety gates

- Confirm evidence directory does not exist before creation.
- Confirm both requested code paths are absent before RED.
- Use only `.venv-calvin-unit/bin/python`; do not install dependencies.
- Record every command exit code in evidence.
- Preserve RED failure (expected exit 1 caused by missing implementation), then GREEN output.
- Re-hash protected files at close and require exact equality.
- Check scoped Git diff/status; do not stage, commit, restore, or modify unrelated dirty files.

## Planned steps and exact commands

1. Create the unique evidence directory and capture timestamp, Git baseline, Python/NumPy version, and path absence.
2. Write tests first at `framework/scripts/data/test_libero_middle_drawer_expert.py`.
3. RED:
   ```bash
   .venv-calvin-unit/bin/python -m unittest -v framework/scripts/data/test_libero_middle_drawer_expert.py
   ```
   Expected: exit `1`, import/file failure specifically because the implementation is absent.
4. Write the minimum implementation at `framework/sim_data/libero_middle_drawer_expert.py`.
5. GREEN: rerun the exact unittest command. Expected: exit `0`, all target tests pass.
6. Syntax gate:
   ```bash
   .venv-calvin-unit/bin/python -m py_compile framework/sim_data/libero_middle_drawer_expert.py framework/scripts/data/test_libero_middle_drawer_expert.py
   ```
   Expected: exit `0`.
7. Produce SHA-256 manifests, operation-local `/dev/null` diffs for both new code files, scoped Git diff/status, and protected-file hash verification.
8. Update this record with actual results and final status.

## Failure / abort conditions

Abort and mark `BLOCKED` or `FAILED` if the required interpreter/NumPy is unavailable, requested paths already exist, evidence path exists, RED does not fail for missing implementation, protected hashes change, tests cannot be made green without LIBERO runtime/install, or scope must expand into gates/G/upstream LIBERO.

## Rollback / recovery

All implementation artifacts are new. Rollback is deletion of only the two new code files and operation-owned `__pycache__` entries, while retaining this record and all evidence. Never use `git restore` in this dirty worktree. The operation record/evidence are retained even on failure.

## Completion criteria

- **PASS:** expected RED preserved; target GREEN and `py_compile` exit 0; contracts above tested; hashes/diffs recorded; protected files unchanged; scope respected.
- **PARTIAL:** implementation evidence exists but one non-safety verification remains incomplete.
- **BLOCKED:** prerequisite prevents execution without scope expansion/setup.
- **FAILED:** implementation or safety contract fails after execution.

## Actual results

- **Execution window:** `2026-10-21T23:07:51+08:00` through `2026-10-21T23:14:46+08:00`.
- Created the unique no-overwrite evidence directory
  `docs/operations/evidence/20261021-gina-libero-middle-drawer-expert-tdd-230751/`.
- Preflight used CPython 3.11.15 and NumPy 1.26.4 from `.venv-calvin-unit`; both requested code paths were confirmed absent.
- **RED:** the exact target unittest command exited `1`. Discovery failed at the intended boundary with
  `ImportError: cannot import name 'libero_middle_drawer_expert' from 'framework.sim_data'` because the production module did not exist. Full output and exit receipt are `10-red-unittest.txt` and `10-red-unittest.exit`.
- Implemented the minimum pure-logic module and fake-model tests. No LIBERO runtime/import/install or rollout was used.
- **GREEN:** the same target unittest command exited `0`; 9 tests ran in 0.014 seconds and reported `OK`. Full output and exit receipt are `20-green-unittest.txt` and `20-green-unittest.exit`.
- **Syntax:** the planned `py_compile` command exited `0` (`30-py-compile.*`).
- New-file SHA-256 values (`31-code.sha256`):
  - implementation: `c8fd102937d8d91882b60f0c49e30ed2d6767becc35856c1e78dbf83c08bef16`
  - test: `cf193eb1c969b833f8629693d113e83a5932c711b0989f7ef2409e7aa283cb05`
- New-file `/dev/null` diffs are preserved as `32-implementation.diff` (10,171 bytes) and `33-test.diff` (8,164 bytes).
- Protected-file before/after manifests compare byte-for-byte (`cmp` exit `0`); `gates.py` and the existing native collection contract remained unchanged. No upstream LIBERO or G file was touched.
- The repository had extensive pre-existing dirty/untracked work. Final status is preserved in `34-final-git-status.txt`; this operation neither staged nor restored any path.
- **Deviations/incidents/rollback:** none. Rollback was not invoked.
- **Result:** `SUPPORTED` — all requested pure-logic contracts are covered and green; runtime behavior remains intentionally untested because a LIBERO runtime was explicitly out of scope.

## Next allowed operation / unresolved items

After PASS, runtime integration and rollout remain separate future operations and require a real LIBERO environment. No such operation is authorized here.

## Main-thread review disposition

The original tests passed but encoded incorrect or unsafe assumptions: the handle local coordinate did not match the pinned XML, all model slide joints were required to be globally unique, non-finite action values were silently replaced, gripper signs were reversed relative to the provisional robosuite convention, and the pull target chased the moving handle. This record is retained as historical RED/GREEN evidence but is superseded by `20261021-gina-libero-middle-drawer-expert-correction.md`.
