# Operation 20261020-ORIGINAL-TASK-D-D-SPARSE-DATA-PREP

- **Operation ID:** `20261020-ORIGINAL-TASK-D-D-SPARSE-DATA-PREP`
- **Stage:** prerequisite data preparation for the original-repository full-scale comparison
- **Target:** `/home/ai-sz-a26394-u1/Gina_autoresearch/data/calvin/task_D_D_original_full_state`
- **Status:** `BLOCKED`
- **Operator:** Hermes Agent for user
- **Created:** 2026-10-20T13:45:59+08:00

## Authoritative requirements

- Preserve `/home/ai-sz-a26394-u1/AutoResearch` as read-only at `master@3d4aeba1badaa3d6ce781033e0d9187729dfec54`.
- Reproduce the original formal run's 14 included tasks and exact annotation-derived segment counts: 2,107 segments total, as recorded in `2026-07-13_minimal_g_projected_firstw16_e50_n100_h160_seed0/run_meta.json`.
- Use the existing 555-GB remote CALVIN ZIP only through its verified sparse central-directory index and bounded HTTP Range requests; never download the whole archive.
- Reuse already verified episode members by hard link where exact uncompressed size matches; separately report reused and newly downloaded bytes/files.
- Store only experiment data and receipts under `Gina_autoresearch`; never change the original repository or existing old/new datasets.

## Baseline

- Current 1+12-fold formal run completed successfully with exit code 0.
- Original `task_D_D` data paths recorded by the old run are absent on this host.
- Existing reusable multifold data: 432 segments, 25,852 frames, 7.0 GiB logical size.
- Sparse ZIP index: `artifacts/calvin_zip_index/task_ABC_D.sparse-index.zip` (219 MiB).
- Disk free before operation: approximately 1.3 TiB.
- Full official language annotation is already cached locally at `artifacts/data_distribution/intent_factorial/full_auto_lang_ann.npy`.

## Goal

Create and validate an exact, resumable, state/action-capable local view of every episode frame referenced by the original run's 14 tasks, with official metadata, task/interval manifest, CRC/size checks, and a packed state cache suitable for current same-G training.

## Explicit non-goals

- No training or simulator rollout in this operation.
- No extraction of RGB/tactile members not already embedded in the required episode NPZ members.
- No use of `close_drawer` as source training data: it was held out by the original run.
- No claim that the exact-original task set gives a four-task OOD comparison; the later training operation must separately label exact-original (`close_drawer` OOD) and matched-four-task source exclusion.

## Mutable paths

- `framework/scripts/data/extract_calvin_original_task_dd_subset.py`
- `framework/scripts/data/test_extract_calvin_original_task_dd_subset.py`
- `data/calvin/task_D_D_original_full_state/`
- `artifacts/evidence/20261020T134559-original-task-dd-data-prep/`

## Safety gates

1. Capture branch/HEAD/status, hashes, disk and process ownership in a new no-overwrite evidence directory.
2. Contract test must prove exact original task list/counts against the cached official annotation before any network transfer.
3. Dry-run must report 2,107 segments and exact per-task counts matching original `run_meta.json`.
4. Refuse archive members outside `task_ABC_D/training/` and refuse size/CRC mismatches.
5. Final gate requires complete manifest, zero missing files, exact interval isolation, packed-cache readability and source-input hashes.

## Planned steps

1. Add a generic exact-task sparse extractor by reusing the already-tested Range, retry and member-extraction helpers.
2. Run RED→GREEN contract tests and dry-run selection.
3. Start the resumable extraction with four workers, bounded retries, timeout and existing-data hard-link reuse.
4. On successful extraction, write official metadata, full selection manifest and download/reuse receipt.
5. Build packed state/action cache and verify all selected intervals.

## Failure and recovery

- Abort on changed source/index hashes, count mismatch, HTTP non-206 response after retries, CRC/size mismatch, disk pressure, concurrent writer, or unplanned path mutation.
- Preserve partial files and logs for resumable retry; do not delete prior verified data.
- Restore code only from timestamped pre-edit backups; data output may be quarantined but not silently overwritten.

## PASS criteria

- Exact 2,107 original segments and per-task counts match old `run_meta.json`.
- Every required frame is present with expected size and readable state/action fields.
- Manifest says complete with zero missing/extra selected intervals.
- Packed cache builds and verifies.

## Actual results

`BLOCKED` at the dry-run selection gate before any dataset network transfer or output data creation.

- Contract RED correctly failed before implementation.
- The first implementation then exposed a source-identity mismatch: the cached annotation and sparse index are for `task_ABC_D.zip`, whereas the original repository's `/data/task_D_D` intervals come from the distinct official `task_D_D.zip` archive (about 166 GB compressed).
- Evidence: original split manifest interval `push_red_block_left:[296701,296765]` is absent from the `task_ABC_D` annotation; using the wrong archive would not reproduce the original 2,107 segments.
- No remote episode bytes were downloaded and no training was started.
- Recovery requires building a sparse central-directory index for `http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip`, then rerunning exact selection against that archive's own annotation.
