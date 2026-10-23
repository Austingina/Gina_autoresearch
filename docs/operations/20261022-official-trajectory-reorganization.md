# OP-20261022-GINA-OFFICIAL-TRAJECTORY-REORGANIZATION-01

- **Created:** 2026-10-22
- **Status:** `COMPLETE` — CALVIN + LIBERO + RoboTwin official payloads and canonical unified index independently verified
- **Authoritative project root:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Git HEAD at start:** `3d4aeba1badaa3d6ce781033e0d9187729dfec54`

## Goal

Build the short-term data-expansion line from upstream official successful demonstrations. Preserve simulator-native files and emit a common semantic index with task intent, phase boundaries, source hashes/provenance, and distribution reports. This operation is **official-trajectory reorganization + label enrichment**, not new simulator trajectory collection.

## Scope and boundaries

1. Keep CALVIN, LIBERO, and RoboTwin native state/action schemas separate.
2. Do not use candidate G as an admission gate, distribution axis, or default input.
3. Do not create held-out folds or run training/evaluation in this collection-only operation.
4. Keep full parent trajectories authoritative; phase slices are derived views and never independently relabeled as native success.
5. Contact-like boundaries inferred from state/action are explicitly named proxies, not force/contact ground truth.
6. Accept only official demonstrations already designated successful by the upstream dataset.
7. Historical AST repositories/data may be inspected read-only, but AST experiment products are not Gina completion evidence. Any reused official payload must be imported byte-for-byte into a Gina-owned root with source and SHA-256 provenance.
8. RoboTwin is included only if an official trajectory payload is actually present and readable; assets or plans do not count as data.

## Starting inventory

- CALVIN authoritative local source: `data/calvin/task_ABC_D_g_intent_multifold_s12/`
- CALVIN verified inventory: 432 segments, 25,852 frames, 12 tasks, 3 scenes; manifest `complete=true`; official language annotations define successful task segments.
- LIBERO and RoboTwin: inventory pending read-only audit.

## TDD and verification plan

1. RED→GREEN canonical intent parsing and strict phase partition contracts.
2. RED→GREEN CALVIN state-progress extraction and phase-boundary labeling.
3. RED→GREEN LIBERO HDF5 demo inventory and action/gripper-based phase proxy, if official files are admitted.
4. Build immutable JSONL/JSON/CSV/Markdown outputs in a new project-owned directory.
5. Independently read back every index row against native files, verify ranges, fields, hashes, counts, percentages, and zero/missing simulator cells.
6. Report parent trajectories, unique transitions/frames, and derived phase views separately to prevent count inflation.

## Abort conditions

- Missing or unreadable native payload.
- Source hash mismatch against authoritative manifest.
- Phase ranges overlap, leave gaps, or fall outside the parent episode.
- Any attempt to treat a derived slice as an independently Oracle-verified success.
- Any write outside the project-owned output/import roots without explicit authorization.

## Evidence root

`artifacts/evidence/20261022-official-trajectory-reorganization/`

## Actual results

### Implemented

- Added `framework/scripts/data/build_official_trajectory_semantic_index.py` and 8 focused unit tests.
- Imported the official LIBERO cache into the Gina-owned root `data/libero/official_f13aa24a3da8c43c7225569f28c562979fa0e35a/` without adopting AST experiment outputs.
- Official LIBERO source: `huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets`, revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`.
- Verified all 40 HDF5 payloads against the source inventory: 33,784,856,577 bytes, 2,000 demos, and 338,575 transitions.
- Built `data/semantic_indexes/official_reorg_20261022_v1/` with:
  - 2,432 complete official parent trajectories (CALVIN 432, LIBERO 2,000);
  - 364,427 native frames/transitions (CALVIN 25,852, LIBERO 338,575);
  - 9,728 derived phase views, all explicitly marked `independent_success=false`;
  - 52 native task IDs across 7 simulator-scene/suite domains;
  - 9 simulator-qualified intent-family cells.
- Against the previous CALVIN-only indexed baseline, parent count increased by 2,000 to 5.629630×; native task-ID coverage increased from 12 to 52 (+40, 4.333333×); domain coverage increased from 3 to 7.
- Candidate G keys are absent from the index and were not used for admission, distribution, or default model input.
- RoboTwin is a declared zero/missing cell: no official trajectory payload was present; runtime/assets were not counted as demonstrations.

### Failure found and fixed before release

The first full build stopped before output because 95 CALVIN block-push segments had zero progress under a global color→state-slot assumption. Dataset evidence showed that block ordering is scene-specific: scene A is pink/blue/red, B is red/blue/pink, and C is blue/red/pink. A regression test was added first, the extractor was changed to require scene-aware mapping, and the previously failing scene-C segment then measured 0.256436 state progress with contiguous phase ranges.

### Offline verification

- Unit tests: `8/8 PASS`.
- Python compile checks: `PASS`.
- Independent readback: `PASS` for all 2,432 index rows, 25,852 CALVIN frame references, 2,000 LIBERO HDF5 groups, 338,575 LIBERO transitions, 40 payload hashes, parent-ID uniqueness, phase continuity, terminal success fields, and absence of candidate-G keys.
- Index record hash: `d205592a24c61f293e043c30cb9ccac6e08ccd7391c907bd18ad08684fe594e2`.
- Summary hash: `1ccce858ebff84d077428825cd7ac431a6ab708b365ab8d03184cd78bc4e9f96`.
- Independent receipt: `artifacts/evidence/20261022-official-trajectory-reorganization/independent_readback.json`.

### Provenance-line separation decision

- The official-data index remains `official_trajectory_reorganization_and_label_enrichment` only.
- The five accepted RoboTwin `place_empty_cup` simulator executions are sealed in `data/semantic_indexes/online_native_expert_20261022_v1/` as `online_native_expert_collection`; they are not official RoboTwin data.
- The partially constructed `data/semantic_indexes/multisource_20261022_v2/` is preserved for audit but marked `PAUSED_NON_AUTHORITATIVE`; it must not be used for training or counts.
- RoboTwin official source is `huggingface.co/datasets/TianxingChen/RoboTwin2.0`, pinned revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`.
- Remote inventory confirms 50 `dataset/<task>/demo_clean.zip` official payload archives totaling 23,375,659,439 bytes. The inventory at `data/robotwin/official_3dc3b798668feb99ac61cc9086d84cbcc3d79186/remote_inventory.json` is metadata only until payload download and hash verification complete.

### Current RoboTwin gate

RoboTwin may enter the official index only after all pinned payload archives are present, each SHA-256 matches its HF LFS OID, all HDF5 parent episodes are inventoried, native image/state/action streams pass readback, and phase sidecars are generated without altering native payloads.

### Final RoboTwin and unified-index acceptance

- Pinned RoboTwin source revision: `3dc3b798668feb99ac61cc9086d84cbcc3d79186`.
- Download receipt: 50/50 archives verified, no failures, 23,375,659,439 declared payload bytes.
- Canonical RoboTwin index: `data/semantic_indexes/robotwin_official_20261022_v3/`.
  - 2,500 complete official parent trajectories across 50 tasks;
  - 548,893 native transitions;
  - 2,195,572 decoded RGB frames across four native camera streams;
  - 12,026 derived phase views;
  - records SHA-256 `8b62ecdeaa9b4b9d5c853724c07ec599fe5ac8bdaaf244ec50627922547eebd9`.
- Canonical three-simulator semantic view: `data/semantic_indexes/official_reorg_20261022_v4/`.
  - 4,932 unique official parent trajectories: CALVIN 432, LIBERO 2,000, RoboTwin 2,500;
  - 913,320 native frames/transitions;
  - 21,754 derived phase views, never counted as independent successes;
  - records SHA-256 `57e5ef3b1e07c0c4c3fc4a5dfd19d5ddd94958ee549ba8f19d7481dc53d38050`.
- Independent receipt: `artifacts/evidence/20261022-official-trajectory-reorganization/unified_v4_independent_readback.json` reports zero empty, gapped, overlapping, or out-of-range phase intervals, unique parent IDs, official-only provenance, and no candidate-G admission keys.
- Native CALVIN, LIBERO, and RoboTwin action/state spaces remain separate; the combined product is a semantic view, not a vector merge.

### Superseded phase-index artifacts

Independent readback found 21 empty `transport_or_manipulation` intervals in `robotwin_official_20261022_v1/v2`, inherited by `official_reorg_20261022_v2/v3`. Root cause was an immediate/late gripper-release fallback that allowed `start > end`. Two regression cases and a fail-fast contiguous-complete-nonempty partition validator were added. These four old directories are audit-only and must not be used for training or statistics.

### Not online-verified / limitations

- No simulator was launched and no new physical trajectory was executed by this official-data operation; this line is strictly official-trajectory reorganization + label enrichment.
- CALVIN phase boundaries use task-native state-motion proxies. LIBERO uses gripper-transition proxies where available and action-motion proxies otherwise. These are not force/contact ground truth.
- No training, held-out evaluation, cross-simulator vector mixing, or G validation was performed.
- RoboTwin official integration remains incomplete until the pinned payload passes download, hash, schema, image/state/action readback, and indexing gates.
