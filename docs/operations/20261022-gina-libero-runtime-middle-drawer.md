# OP-20261022-GINA-LIBERO-RUNTIME-MIDDLE-DRAWER-01

- **Created:** 2026-10-22T07:39:44+08:00
- **Status:** `PASS — RUNTIME/STATIC PREFLIGHT ONLY; ONLINE ROLLOUT NOT RUN`
- **Project:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Pinned upstream:** LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`

## Goal

Create a Gina-owned isolated LIBERO runtime from the pinned upstream commit, connect the existing middle-drawer pure-logic expert through a runtime adapter, validate the real gripper sign and joint/body/site semantics, then run three bounded init-state pilots using native `_check_success()` and independent artifact readback.

## Mutable scope

- `/home/ai-sz-a26394-u1/Gina_libero_runtime/`
- new Gina adapter/tests/evidence/operation files

Historical AST LIBERO directories are read-only provenance and must not receive changes or count as Gina completion.

## Method and gates

1. Clone/copy only the pinned clean upstream source into the dedicated runtime root and create an isolated environment.
2. Install per upstream declarations; record package freeze and import paths.
3. TDD the runtime adapter before implementation.
4. One-step gripper qpos sign smoke and live joint/body/handle transform readback.
5. Three fixed init states, bounded step count, fresh environment each.
6. Native `_check_success()` is the only semantic success gate; qpos/reward/done/G cannot create success labels.
7. Persist `T+1` observations/states versus `T` actions with hashes and independent readback.

## Abort conditions

Stop on commit mismatch, missing assets/BDDL, API mismatch, ambiguous joint/body resolution, non-finite action, GPU contention, or failed native Oracle. Do not weaken success thresholds.

## Evidence

`artifacts/evidence/20261022T073944-gina-libero-runtime/`

## Scope amendment — no-rollout runtime preflight

- **Authorized:** 2026-10-22 (current user request)
- This operation stops after installation and read-only / static interface preflight.
- GPU initialization, environment reset/step, one-step gripper smoke, and the three init-state pilots are explicitly `NOT_RUN` in this operation.
- The runtime source may be cloned at the pinned commit and an isolated virtual environment may be created under `/home/ai-sz-a26394-u1/Gina_libero_runtime/`; upstream tracked source must remain clean.
- Preflight must capture source URL/HEAD/status, Python identity, `pip freeze`, import paths, configured BDDL/assets visibility, and parse the target BDDL/init-state/model XML without creating a simulator environment.
- Run the existing 10 pure-logic Gina tests. Only if a real interface mismatch requires Gina adapter code: preserve target-file hashes/backups and use strict per-contract RED → minimal GREEN → full affected regression. No adapter change is required merely to produce an interface inventory.
- Abort on pinned commit mismatch, an upstream tracked-source change, attempted GPU/rollout, missing target resources, or an interface fact that cannot be grounded in the pinned runtime.

## Planned exact execution

1. Clone `https://github.com/Lifelong-Robot-Learning/LIBERO.git` into the dedicated root, checkout `8f1084e3132a39270c3a13ebe37270a43ece2a01`, and verify a clean detached worktree.
2. Create `.venv` with the host Python and install using the pinned upstream declaration without modifying tracked source.
3. Record source/runtime provenance and dependency inventory in the existing unique evidence directory.
4. Use import-safe probes plus Python/AST/XML/pickle parsing only; do not instantiate a LIBERO environment or call reset/step.
5. Run only `framework/scripts/data/test_libero_middle_drawer_expert.py`, expecting 10 pure-logic tests.
6. Recheck both Git worktrees and close this record with evidence hashes and a `PASS`, `PARTIAL`, `BLOCKED`, or `FAILED` status.

## Rollback

- Runtime rollback: remove only `/home/ai-sz-a26394-u1/Gina_libero_runtime/` after preserving evidence.
- Gina rollback: restore only operation-owned edits from evidence backups; never use Git restore on pre-existing dirty/untracked files.
- Evidence and failure logs are retained and never overwritten.

## Actual results

### 2026-10-22 parent-thread verification — `PARTIAL`

- Pinned source was materialized cleanly at `8f1084e3132a39270c3a13ebe37270a43ece2a01`; source receipt SHA-256 is retained in the evidence directory.
- The real target BDDL exists at `source/libero/libero/bddl_files/libero_goal/open_the_middle_drawer_of_the_cabinet.bddl` (not the earlier assumed `libero_90` path).
- The dedicated environment is `/home/ai-sz-a26394-u1/Gina_libero_runtime/.venv`. Live import failed because the upstream requirements install stopped while downloading the `torch-2.14.1` 554.6 MB wheel; consequently `libero`, MuJoCo interface inventory, and static runtime preflight are not yet accepted.
- The corrected 10 pure-logic Gina expert tests passed in the separate `.venv-calvin-unit` environment, which is logic evidence only and is not the LIBERO runtime.
- Requirements installation was restarted with 120-second read timeout and 20 retries as process session `proc_be2d5312e9de`. After completion, editable install, exact imports, target BDDL/init-state/XML parsing, package freeze and clean-worktree readback remain mandatory.
- No environment reset/step, gripper-sign smoke, native `_check_success()` call, or online trajectory has run; formal LIBERO additions remain `0`.

### 2026-10-22 final no-rollout runtime acceptance — `PASS`

- The resumed upstream requirements install exited `0`; final environment inventory contains 122 frozen distributions.
- The pinned source was installed without changing tracked files. Because this pinned commit uses a PEP 420 namespace layout (`libero.libero`) while its root editable build produces an empty mapping, the dedicated runtime adds only `/home/ai-sz-a26394-u1/Gina_libero_runtime/source` through `gina_libero_source.pth`; the upstream checkout remains clean at `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Runtime imports passed with Python `3.10.12`, robosuite `1.4.0`, MuJoCo `3.16.0`, NumPy `1.22.4`, and Torch `2.14.1+cu130`; Torch reported CUDA available. This was import/readback only, not simulator initialization.
- A dedicated `LIBERO_CONFIG_PATH=/home/ai-sz-a26394-u1/Gina_libero_runtime/config` prevents first-import prompts and leaves the user's `~/.libero` untouched.
- Sparse checkout initially contained only XML. The exact seven asset families referenced by this BDDL were materialized from ordinary blobs already present in the pinned Git object database: wooden cabinet, flat stove, wine rack, Akita black bowl, cream cheese, wine bottle, and plate. No LFS substitution or source edit was used.
- `scripts/preflight_libero_middle_drawer_runtime.py` passed fail-closed static validation. Receipt: `static-runtime-preflight-v2.json` (`status=PASS`, SHA-256 `35759722f325a9d97a2344acc9bd14e7c1788b76ee9122e8811ec8d12f5db2a3`).
- Verified target semantics: goal `open wooden_cabinet_1_middle_region`; object of interest identical; init states shape `(50, 79)`, dtype `float64`, all finite; exactly one `middle_level` slide joint with axis `0 1 0`, range `-0.16 0.01`, and matching `cabinet_middle` body / `middle_region` site.
- All `file=` references from the seven task asset XMLs resolve to non-empty files. The final source HEAD/status readback is pinned and clean.
- The pure-logic expert regression passed `10/10` in the accepted LIBERO environment. An earlier invocation with the host Python failed only because that interpreter lacks NumPy; both failure and corrected evidence are retained.
- Scope remains strict: no environment construction, reset, step, gripper-sign smoke, native `_check_success()`, HDF5, or trajectory acceptance occurred. Formal LIBERO online additions remain `0`; those require a separately authorized GPU rollout stage.
