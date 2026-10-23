# OP-20261023-GINA-VIMA-OFFICIAL-TRAJECTORY-EXPANSION-01

- **Created:** 2026-10-23T08:49:11+08:00
- **Status:** `COMPLETE — pinned official VIMA sparse payload, success/alignment-gated semantic index, and five-simulator view independently verified`
- **Authoritative project root:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Git HEAD at start:** `3d4aeba1badaa3d6ce781033e0d9187729dfec54`
- **Pre-existing worktree:** dirty; protect all unrelated files and artifacts.

## Decision

Use **VIMA-Bench**, not CLIPort, for this increment.

Reasons:

1. VIMA publishes a pinned official 650K-oracle-trajectory payload with RGB, segmentation, oracle actions, multimodal prompts, task/object metadata, and per-trajectory success evidence.
2. The official CLIPort repository publishes the oracle generator and model checkpoints, but its documented full demonstration corpus is generated locally rather than distributed as an immutable official payload. Using CLIPort now would belong to the separate online-native collection line, not the rapid official-trajectory reorganization line.
3. VIMA directly adds compositional prompt grounding, rotation, constraint satisfaction, ordered imitation, and restore/memory tasks missing from the current inventory.

## Frozen upstream

- Dataset: `VIMA/VIMA-Data`
- Hugging Face revision: `f35cee89382278b8d63d6d6f9c27cfeda3deb3f5`
- File: `vima.zip`
- Archive bytes: `21,475,254,280`
- Published LFS SHA-256 OID: `65d3f1bec7d95ca80214eb1ce1c5deb1a79f0ceb9517eb516b8a5e88dae52510`
- VIMA-Bench source commit: `97e0af11e126fd477c9d81eeabfb6c739021c680`
- Source snapshot SHA-256: `3cefd19d994392b90cb67e261411ae9856a6bc4c6b23906a49a863aea3fd6a4b`
- Dataset license: CC BY 4.0
- Source license: MIT

## Remote ZIP audit

- ZIP64 entries: `7,588,631`
- Central-directory offset: `20,551,603,971`
- Central-directory size: `923,650,211`
- Required tail bytes: `923,650,309`
- HTTP byte ranges: supported
- Full archive will not be duplicated locally. The central directory is cached as a sparse index; selected members are range-fetched and CRC-verified before persistence.

## Prospective task selection

Select exactly 1,000 official parent **candidates** from each of six training tasks, uniformly across each task's sorted native trajectory IDs. Admission remains conditional on the native success and alignment gates below; malformed candidates are excluded rather than forcing the quota:

| Native task | Canonical family | Gap addressed |
|---|---|---|
| `rotate` | orientation manipulation | explicit rotation |
| `twist` | orientation manipulation | novel-concept rotation/twist |
| `sweep_without_exceeding` | constraint-aware tool interaction | bounded sweep |
| `follow_order` | ordered imitation | temporal ordering |
| `pick_in_order_then_restore` | long-horizon memory manipulation | ordered act + restore |
| `rearrange_then_restore` | long-horizon rearrangement | multi-object restore |

This is a fixed official-data subset, not a train/test split and not an evaluation protocol. No candidate G is used for selection, admission, or default model input.

## Acceptance gates

1. Every selected parent has `action.pkl`, `obs.pkl`, `trajectory.pkl`, and aligned `rgb_front`/`rgb_top` JPEG sequences.
2. `trajectory.pkl` records native `success=true` and `failure=false`.
3. `steps == action length`, and segmentation/RGB/end-effector sequences are `steps+1`.
4. Every extracted member matches central-directory uncompressed size and CRC; persist per-member SHA-256.
5. Parent IDs are unique and preserve native task/trajectory identity.
6. Derived phase views are contiguous, non-empty, cover the parent exactly, and remain `independent_success=false`.
7. Independent readback verifies every local member and every index record before merge.
8. Keep VIMA native observations/actions separate from other simulators; normalize semantic labels only.

## Claim boundary

- Provenance class: official-demonstration reorganization + label enrichment.
- New physical trajectories: 0.
- Online VIMA/PyBullet replay in this operation: not required and not to be claimed unless separately executed.
- Training, held-out evaluation, and generalization claims: out of scope.

## Actual results

Completed at `2026-10-23T10:04:15+08:00` on the unchanged start HEAD `3d4aeba1badaa3d6ce781033e0d9187729dfec54`; the worktree remained dirty and unrelated changes were not modified.

### Frozen source and sparse extraction

- Verified the pinned Hugging Face metadata at revision `f35cee89382278b8d63d6d6f9c27cfeda3deb3f5` and persisted `data/vima/official_f35cee89382278b8d63d6d6f9c27cfeda3deb3f5/source_revision.json` (`SHA-256 14933911d69fe7dbb5f4d82fb68f11db9e8b20f5e862189488bb7f18e59d89aa`).
- Preserved the pinned VIMA-Bench source tarball (`SHA-256 3cefd19d994392b90cb67e261411ae9856a6bc4c6b23906a49a863aea3fd6a4b`).
- Downloaded and parsed the complete 923,650,309-byte ZIP64 tail. Central-directory-tail SHA-256: `c82fc16942a43a51bf6570ffeba1e156cb433e497bed10bd615621883081d180`.
- Created a 21,475,254,280-byte logical sparse ZIP index backed by approximately 923.7 MB of physical blocks. Sparse-index SHA-256: `9067e98f1c62aa83ff422fcce3cc5544cedc271cf9a0d455b72f5e3facfb47ad`.
- Froze 6,000 deterministic candidates in `selection_manifest.json` (`SHA-256 96a19df470f7d4aaa70a44b2b7bfdcafc79a0c2aca4e56921301f4e9156056f0`). Selected compressed member payload was 212,395,872 bytes; all 6,000 extracted candidates occupy 5,867,382,477 native bytes across 55,344 member files. The accepted 5,999 parents account for 5,866,259,008 bytes across 55,331 files.
- Every persisted member passed ZIP central-directory size/CRC validation and received a local SHA-256 receipt. Final extraction receipt SHA-256: `ecf54e56843a87660aa3aaeb8d4699d02c38f823604450d836907c80c815ee9b`.

### Admission result

- Candidates: **6,000**.
- Accepted official successful parents: **5,999**.
- Excluded: **1** — `pick_in_order_then_restore/105017`. Native metadata reports success, but it has five RGB frames while its three actions require four T+1 observation frames. The alignment gate was preserved; no task-specific exception or threshold relaxation was added.
- Accepted distribution:
  - `rotate`: 1,000
  - `twist`: 1,000
  - `sweep_without_exceeding`: 1,000
  - `follow_order`: 1,000
  - `pick_in_order_then_restore`: 999
  - `rearrange_then_restore`: 1,000
- Native action transitions: **12,668**.
- Native aligned observation frames: **18,667**.
- Derived phase views: **12,231**, all non-independent semantic views.
- Every accepted parent retains official `trajectory.pkl`, `action.pkl`, `obs.pkl`, front RGB, top RGB, segmentation, prompt payload, native trajectory ID, and per-member provenance.

Canonical VIMA index:

```text
data/semantic_indexes/vima_official_20261023_v2/
```

- `records.jsonl` SHA-256: `d28f4e9bdd782640f78fd0e44f1921829cce1efde64034c02ac5adac5b78c945`
- `summary.json` SHA-256: `7158b36b84871ae593944ddb87482c0510a75a49372f6e42a8e5ff9bbcce9a32`
- `vima_official_20261023_v1` is superseded because it predates the common unified-index field contract.

### Independent readback

The independent verifier did not import the index builder. It re-opened all accepted pickle/RGB payloads, re-hashed all **55,331 files / 5,866,259,008 bytes**, rechecked native success/failure fields, action/observation/RGB alignment, task counts, parent-ID uniqueness, phase coverage, provenance, candidate-G absence, and summary hashes. Result: `PASS`.

Evidence:

```text
artifacts/evidence/20261023-vima-official-expansion/independent-readback-v2.log
```

Log SHA-256: `249017f01d34bfb16f6d228c3002d86c50e16f139a3e7253dd00353aa1e9ecda`.

### Unified official semantic view

Canonical five-simulator view:

```text
data/semantic_indexes/official_reorg_20261023_v6/
```

- Total official successful parents: **26,193** (`+5,999`, or `1.297068×` the four-simulator v5 total).
- Total native frames/transitions: **2,507,756**.
- Total derived phase views: **95,033**.
- By simulator: CALVIN 432; LIBERO 2,000; RoboTwin 2,500; ManiSkill 15,262; VIMA 5,999.
- Existing `pick_place` count remains 2,050; its share falls from 10.15% in v5 to **7.83%** in v6.
- Unified `records.jsonl` SHA-256: `81023ae40c4a5e7680e026e782cf3ccaab64ef08836b0087dc00170817b26a2f`.
- Independent unified readback: `PASS`; evidence log SHA-256 `65c9b22c09457691917d259b59342ea0214110ab74edb66cc87b8a041909ad93`.

### Tests and claim boundary

- TDD RED was recorded before each new extractor/index behavior.
- Final focused suites: VIMA extractor 3/3, VIMA builder 4/4, ManiSkill regression 6/6, RoboTwin regression 3/3; all passed. Python static compilation also passed.
- **Implemented and offline-verified:** yes.
- **Online VIMA-Bench/PyBullet environment replay:** **not run**.
- **Newly executed simulator trajectories:** **0**.
- This operation therefore supports only official-data ingestion, integrity, native success/alignment, RGB/modality, and semantic-index claims—not online replay, policy training, or generalization claims.
