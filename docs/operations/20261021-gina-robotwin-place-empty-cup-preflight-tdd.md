# Operation 20261021-GINA-ROBOTWIN-PLACE-EMPTY-CUP-PREFLIGHT-TDD

- **Operation ID:** `20261021-gina-robotwin-place-empty-cup-preflight-tdd`
- **Stage:** fail-closed preflight implementation and pure unit tests
- **Target:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **External read-only source:** `/home/ai-sz-a26394-u1/RoboTwin`
- **Status:** `PASS`
- **Result:** `SUPPORTED`
- **Operator:** Hermes Agent
- **Created:** `2026-10-21T23:07:48+08:00`

## Authoritative requirements

Implement a read-only, no-install RoboTwin `place_empty_cup` preflight in Gina using strict RED→GREEN TDD. It must verify exact project root and pinned commit, `PYTHONNOUSERSITE` semantics, an explicit required-package list, four required non-empty asset classes, task source definitions for `play_once` and `check_success`, collector admission requiring `plan_success`, `check_success`, and `planned_joints_legal`, and a bounded fixed pilot (`episode_num == 1`, positive `max_attempts`).

## Baseline

- Gina realpath/Git root: `/home/ai-sz-a26394-u1/Gina_autoresearch`.
- Gina branch/HEAD: `master` / `3d4aeba1badaa3d6ce781033e0d9187729dfec54`; worktree already dirty with unrelated changes.
- External RoboTwin HEAD: `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`, matching the requested pin.
- External required assets are currently absent: `assets/objects/021_cup`, `assets/objects/019_coaster`, `assets/embodiments/aloha-agilex`, and `assets/background_texture`.
- RoboTwin `scripts/collect_data.py` has the three native gates, but its unbounded `while suc_num < episode_num` has no `max_attempts`; the Gina preflight therefore must require an independently bounded pilot configuration.
- Approved test interpreter: `.venv-calvin-unit/bin/python` (Python 3.11.15), stdlib only.

## Goal

1. Add `framework/sim_data/robotwin_preflight.py` with no write/install behavior.
2. Add `framework/scripts/data/test_robotwin_preflight.py` with pure temporary-fixture unit tests.
3. Preserve expected RED, GREEN, compile, live fail-closed, hashes, and operation-local diff evidence.
4. Do not change `/home/ai-sz-a26394-u1/RoboTwin`.

## Non-goals

- No dependency or asset installation/download.
- No simulator import, render, seed search, collection, or rollout.
- No edits to external RoboTwin.
- No cleanup of unrelated Gina changes.

## Declared mutable paths

- `framework/sim_data/robotwin_preflight.py`
- `framework/scripts/data/test_robotwin_preflight.py`
- `docs/operations/20261021-gina-robotwin-place-empty-cup-preflight-tdd.md`
- `artifacts/evidence/20261021T230748-gina-robotwin-place-empty-cup-preflight-tdd/`

## Evidence policy

The evidence directory is unique and must not overwrite prior operation evidence. Logs preserve exit codes and exact outputs. Final SHA-256 and an operation-local `/dev/null` diff for each new source file are required.

## Planned steps and expected outputs

1. Capture baseline Git identity/status and external repository HEAD/status/hash manifest.
2. Write tests first and run them before production code; expected RED is import failure for missing `framework.sim_data.robotwin_preflight`.
3. Add the minimum stdlib-only checker and CLI; expected GREEN is all target tests passing.
4. Run `py_compile`, target tests, and a live preflight against the external checkout with `PYTHONNOUSERSITE=1`; expected live result is nonzero/fail-closed due to missing packages/assets, without install or writes.
5. Verify external RoboTwin HEAD/status and tracked diff hash are unchanged.
6. Generate SHA-256 manifest, Git diff/status receipt, operation-local diffs, and close this record.

## Safety gates / abort conditions

- Abort on any attempted package installation, asset download, rollout, simulator import, or external repository mutation.
- Checker may use only metadata discovery, Git read commands, filesystem reads/stat, AST parsing, and JSON output.
- Every missing, malformed, unreadable, or unverifiable prerequisite fails closed.
- Pilot configuration passes only for integer (not bool) `episode_num == 1` and positive integer (not bool) `max_attempts`.

## Rollback

All implementation files are new. Rollback is removal of only the declared new Gina files/evidence. Never use `git restore` against the pre-dirty worktree. External RoboTwin requires no rollback because it must remain untouched.

## Completion criteria

- **PASS:** expected RED preserved; target GREEN and `py_compile` pass; live check fails closed for truthful current blockers; hashes/diffs captured; external repo identity unchanged.
- **PARTIAL:** code/tests pass but one requested receipt cannot be captured.
- **BLOCKED:** test interpreter or repository identity cannot be verified.
- **FAILED:** external mutation, install/rollout, scope escape, or false-positive preflight.

## Actual results

### TDD and verification

- RED command: `.venv-calvin-unit/bin/python framework/scripts/data/test_robotwin_preflight.py`.
  - Exit `1`, expected `ModuleNotFoundError: No module named 'framework.sim_data.robotwin_preflight'`.
  - Evidence: `red.log`, `red.exitcode`.
- Initial GREEN: 10 tests passed in 0.374 seconds, exit `0`.
  - Evidence: `green-initial.log`, `green-initial.exitcode`.
- Final affected regression: all 10 new tests passed in 0.186 seconds and existing `test_native_collection_contract.py` printed `cross-simulator native collection contract: OK`; aggregate exit `0`.
  - Evidence: `regression.log`, `regression.exitcode`.
- `py_compile` for both new Python files exited `0`.
  - Evidence: `py_compile.log`, `py_compile.exitcode`.
- `git diff --check` for the declared source/document paths exited `0`.

### Live fail-closed result

The exact live command used `PYTHONNOUSERSITE=1`, fixed `--episode-num 1`, and bounded `--max-attempts 25`. It exited `1` with top-level `"ok": false`, as required for the current incomplete runtime:

- PASS: exact root `/home/ai-sz-a26394-u1/RoboTwin`;
- PASS: pinned HEAD `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`;
- PASS: non-empty `PYTHONNOUSERSITE` CPython semantics;
- PASS: task class defines `play_once` and `check_success`;
- PASS: collector admission structurally combines `plan_success`, `check_success()`, and `planned_joints_legal()`;
- PASS: pilot is exactly one episode and has positive `max_attempts=25`;
- BLOCKED: 15 of 16 checked import names are absent in the approved unit venv (`numpy` is present);
- BLOCKED: all four required asset directories are absent.

An additional CLI run with `PYTHONNOUSERSITE` present but empty exited `1` and marked `python_no_user_site` false, proving the non-empty-string semantics rather than generic key-presence.

Evidence: `live-preflight.json`, `live-preflight.exitcode`, `empty-python-nousersite.json`, and `empty-python-nousersite.exitcode`.

### Read-only and provenance verification

- No package installation, asset download, simulator import, seed search, render, collection, or rollout was run.
- External RoboTwin remained clean before and after final verification; HEAD stayed `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f` and tracked diff SHA-256 stayed the empty-diff hash `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Submodule identity at closure: `XPolicyLab` `fa431ecd893ee706883e64fe5fe1464ec8cd928d`.
- Final source/document/evidence hashes and operation-local new-file diffs are recorded in `sha256-manifest.txt` and `operation-local.diff` in the independent evidence directory.

### Deviations and incidents

- A provenance helper attempted to hash every `git ls-files` path and encountered the `XPolicyLab` gitlink as a directory. It did not alter the repository and returned shell exit `0` because the final pipeline hash succeeded. The authoritative read-only proof therefore uses the independently captured HEAD, empty `git status`, empty tracked-diff hash, and submodule SHA instead of that unsuitable aggregate. The failed approach remains visible in `robotwin-readonly-before-final.txt` and was not erased.

Final operation status is `PASS`: the implementation protocol completed and the live preflight truthfully refused readiness. Runtime/assets remain prerequisites for a later, separately authorized operation.

## Next allowed operation

A separate operation may install missing dependencies/assets or build a bounded collector only after current competing downloads permit it; this operation does neither.
