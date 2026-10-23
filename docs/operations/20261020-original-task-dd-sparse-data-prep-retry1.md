# Operation 20261020-ORIGINAL-TASK-D-D-SPARSE-DATA-PREP-RETRY1

- **Operation ID:** `20261020-ORIGINAL-TASK-D-D-SPARSE-DATA-PREP-RETRY1`
- **Stage:** recover source identity, prepare exact original data view
- **Target:** `/home/ai-sz-a26394-u1/Gina_autoresearch/data/calvin/task_D_D_original_full_state`
- **Status:** `IN_PROGRESS`
- **Created:** 2026-10-20T13:53:49+08:00

## Why this retry exists

The first attempt stopped before transfer because it correctly detected that `task_ABC_D.zip` cannot reproduce the original repository's `task_D_D` frame intervals. The authoritative source is the distinct official archive `http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip` recorded by the original repository's `bootstrap.md` and download pipeline.

## Immutable source identity

- URL: `http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip`
- HTTP Last-Modified: `Thu, 23 Feb 2023 17:04:44 GMT`
- HTTP ETag: `"294ca3666e-5f5610261d19b"`
- Logical archive size: `177379436142` bytes
- Server advertises byte ranges.
- Original segment source: read-only `object_transfer_red_blue_to_pink.json` plus the original representative run's exact task counts.

## Scope and safety

1. Fetch only the end-of-central-directory/central-directory ranges required to construct a sparse local ZIP index.
2. Fetch `task_D_D`'s own official annotation/config/scene metadata by bounded Range.
3. Select exactly the original run's 2,107 segments using task identities and intervals from the original split manifest, then dry-run count-check before episode transfer.
4. Extract only episode members needed by those segments, with per-member ZIP CRC and size validation; resumable atomic writes, four workers, bounded 32-MiB ranges.
5. Build and verify packed state/action cache; preserve all previous datasets and the original repository.
6. On success, open a separate operation record before training or simulator evaluation.

## Mutable paths

- `artifacts/calvin_zip_index/task_D_D.*`
- `framework/scripts/data/extract_calvin_original_task_dd_subset.py`
- `framework/scripts/data/test_extract_calvin_original_task_dd_subset.py`
- `data/calvin/task_D_D_original_full_state/`
- `artifacts/evidence/20261020T135349-original-task-dd-data-prep-retry1/`

## Stop conditions

Stop without training on source-header drift, unsupported/non-206 ranges, sparse-index parse failure, segment/count mismatch, CRC/size mismatch, insufficient disk, concurrent writer, or mutation outside the declared paths.

## PASS criteria

- Sparse index opens with Python `ZipFile` and contains the exact official annotation plus all selected episodes.
- Exact original counts: 2,107 segments, 48,069 windows under the representative training configuration.
- Every selected member passes archive CRC/size validation and contains robot state, scene state and action.
- Packed cache and manifest both verify complete.

## Actual results

`IN_PROGRESS`

- First extraction process downloaded 5.661 GiB through group 325/1619, then received SIGTERM exactly at the 600-second process timeout; no CRC error or HTTP failure occurred.
- The extractor uses atomic per-member writes and validates existing members on startup, so the operation was resumed without deleting partial data as `proc_af2d927b35f4`, with a 10,800-second bound.
