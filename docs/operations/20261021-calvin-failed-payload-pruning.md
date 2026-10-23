# Operation 20261021 CALVIN Failed-Payload Pruning

- **Operation ID:** `20261021-calvin-failed-payload-pruning`
- **Created:** `2026-10-21T15:36:05+08:00`
- **Status:** `PASS`
- **Operator:** Hermes Agent
- **Target:** `/home/ai-sz-a26394-u1/Gina_autoresearch/data/calvin/generated_*/attempts/`
- **Authoritative instruction:** User explicitly requested deletion of approximately 8.53 GiB of failed data before decoupling data expansion from candidate G research.

## Baseline

- Git HEAD: `3d4aeba1badaa3d6ce781033e0d9187729dfec54`
- Worktree is already dirty and contains unrelated research changes; this operation will not edit tracked source files.
- No pre-existing CALVIN/PyBullet collector process was found; the only `pgrep` match was the probe command itself.
- Filesystem before operation: 1.8 TiB total, 531 GiB used, 1.2 TiB available.
- Failed payload roots: 27 `attempts/` directories.
- Allocated size: 8,943,340 KiB = 8.5290 GiB.
- Files: 3,302 total = 1,651 `.npz` payloads + 1,651 `.json` receipts.

## Goal

Delete the failed per-frame payload directories while retaining auditable evidence and leaving every accepted trajectory and collection-level manifest untouched.

## Non-goals

- Do not delete any `accepted/` file.
- Do not modify collection manifests or accepted receipts.
- Do not modify controllers, reset pools, source code, official CALVIN data, or training/evaluation artifacts.
- Do not begin LIBERO/RoboTwin collection in this operation.

## Safety gates

1. Refuse to run if a collector/PyBullet process exists.
2. Enumerate the exact 27 target directories and all files before deletion.
3. Write a SHA-256 inventory of every deleted file.
4. Archive all failed `.json` receipts before deleting their parent directories.
5. Record accepted-tree and collection-manifest hashes before and after; they must remain identical.
6. Delete only directories whose resolved path matches `data/calvin/generated_*/attempts`.

## Evidence directory

`artifacts/evidence/20261021T153605-calvin-failed-payload-pruning/`

No previous evidence directory may be overwritten.

## Planned steps

1. Capture before-state inventory, hashes, allocated bytes, and process check.
2. Create `failed_receipts.tar.gz` containing all failed JSON receipts with relative paths.
3. Validate the archive member count and SHA-256.
4. Remove the 27 validated `attempts/` directories.
5. Verify no `generated_*/attempts/` directory remains.
6. Recompute disk use and accepted/manifest hashes.
7. Update this record with actual results.

## Abort conditions

- Any active collector or PyBullet process.
- A target resolves outside the repository's `data/calvin` tree.
- Inventory count differs from 3,302 files before deletion.
- JSON archive does not contain exactly 1,651 receipts.
- Any accepted or collection-manifest hash changes.

## Recovery

Deleted NPZ payloads are intentionally not recoverable from the working tree. Their exact path, size, and SHA-256 will remain in the evidence inventory; JSON receipts will remain in the compressed evidence archive; collection manifests and all accepted trajectories remain in place. Reconstructing deleted failures would require rerunning the recorded controller/reset configuration.

## PASS criteria

- Exactly 27 validated `attempts/` directories removed.
- Exactly 3,302 failed files removed after evidence capture.
- Approximately 8.529 GiB released.
- All accepted files and collection manifests unchanged by hash.
- Receipt archive contains 1,651 JSON entries and passes archive read-back.

## Actual results

- **Executed:** `2026-10-21T15:36:05+08:00` to `2026-10-21T15:44:21+08:00`
- **Status:** `PASS`
- Preflight found no active collector or PyBullet process.
- Removed exactly 27 validated `generated_*/attempts/` directories.
- Removed exactly 3,302 failed files:
  - 1,651 frame/state/action NPZ payloads;
  - 1,651 JSON receipts.
- Released allocated space: 8,943,340 KiB = 8.5290 GiB.
- Archived all 1,651 failed JSON receipts before deletion:
  - archive: `artifacts/evidence/20261021T153605-calvin-failed-payload-pruning/failed_receipts.tar.gz`;
  - archive size: 728 KiB;
  - archive SHA-256: `944e4629fba1081cd4a055cc460eff1c4b4ee149ac4937cee52316dea226f24a`.
- Full deleted-file inventory includes path, byte size, and SHA-256 for all 3,302 files:
  - `artifacts/evidence/20261021T153605-calvin-failed-payload-pruning/before_inventory.json`;
  - inventory SHA-256: `5af7230b1499c34cd6f913e7f6abb857b8489650a8bf434117c35e6416da265c`.
- All 460 protected collection manifests and accepted files remained byte-identical by SHA-256.
- Remaining `generated_*/attempts/` directories: 0.
- Filesystem use changed from 531 GiB to 523 GiB.
- Accepted-only read-back after deletion passed:
  - strategy-1 Q1: 6/6 accepted hash/receipt/modalities verified;
  - Scene-D left/right Q25: 150/150 accepted hash/receipt verified, 25 per cell.

### Deviation and follow-up

The legacy `validate_collection()` routine assumes every failed payload referenced by a manifest is still present. Its full-collection audit now raises `FileNotFoundError` on intentionally pruned `attempts/*.npz`, although all accepted artifacts remain valid. This does not invalidate the deletion or accepted data, but the next decoupling operation must add an explicit pruned-failure sidecar/contract and teach validators to distinguish `failure payload intentionally pruned` from `unexpected missing accepted artifact`.

### Evidence

- `artifacts/evidence/20261021T153605-calvin-failed-payload-pruning/result.json`
- `artifacts/evidence/20261021T153605-calvin-failed-payload-pruning/post_delete_verification.json`
- `artifacts/evidence/20261021T153605-calvin-failed-payload-pruning/q1_accepted_readback.json`
- `artifacts/evidence/20261021T153605-calvin-failed-payload-pruning/q25_accepted_only_readback.json`
- `artifacts/evidence/20261021T153605-calvin-failed-payload-pruning/process_gate.txt`
