# OP-20261023-GINA-MANISKILL-OFFICIAL-TRAJECTORY-EXPANSION-01

- **Created:** 2026-10-23T08:22:56+08:00
- **Status:** `COMPLETE` — pinned ManiSkill official payloads, success-only semantic index, and four-simulator view independently verified
- **Authoritative project root:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Git HEAD at start:** `3d4aeba1badaa3d6ce781033e0d9187729dfec54`
- **Pre-existing worktree:** dirty; all existing modifications are protected and will not be reverted.

## Goal

Add pinned official ManiSkill demonstrations to the short-term `official_trajectory_reorganization_and_label_enrichment` line, emphasizing task families that are underrepresented in the current 4,932-parent CALVIN/LIBERO/RoboTwin index.

## Fixed source

- Dataset: `https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations`
- Pinned revision: `d674485bbffdd533914e52d272fdda34c0515608`
- Upstream code/docs observed at ManiSkill `main@62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` only as schema documentation; payload identity is governed by the pinned dataset revision above.

## Selected official task payloads

1. `LiftPegUpright-v1` — orientation/upright manipulation
2. `PegInsertionSide-v1` — insertion/assembly
3. `PlugCharger-v1` — insertion/assembly
4. `PokeCube-v1` — tool/contact operation
5. `PullCubeTool-v1` — tool-mediated manipulation
6. `PushT-v1` — planar contact manipulation
7. `RollBall-v1` — rolling/contact manipulation
8. `DrawTriangle-v1` — tool trajectory / longer sequential contact
9. `StackPyramid-v1` — multi-object arrangement

Ordinary `PickCube-v1` is intentionally excluded because `pick_place` is already the dominant family. Cabinet demos are not listed by the current official ManiSkill downloader and therefore are not invented or substituted.

## Scope and boundaries

- This is acquisition and reindexing of official demonstrations, **not new simulator execution**.
- Preserve ZIP/HDF5/JSON payload bytes; write semantic records as sidecars only.
- Parent success must be demonstrated by official per-episode metadata and/or native `success` streams; file existence alone is insufficient.
- Preserve action/control mode, reset kwargs, env kwargs, `T` actions, and `T+1` environment-state alignment.
- Raw official demos are compressed state/action datasets and may contain no RGB/RGB-D observations. Missing image modalities remain explicit; they are not synthesized in this operation.
- Candidate G is absent from admission, distribution, and default inputs.
- Phase labels are non-authoritative action/gripper/motion proxies and every phase has `independent_success=false`.
- No held-out split, training, or cross-simulator raw-vector merge is performed.

## Planned artifacts

- Native payload root: `data/maniskill/official_d674485bbffdd533914e52d272fdda34c0515608/`
- Simulator index: `data/semantic_indexes/maniskill_official_20261023_v1/`
- Four-simulator semantic view: `data/semantic_indexes/official_reorg_20261023_v5/`
- Evidence root: `artifacts/evidence/20261023-maniskill-official-expansion/`

## TDD and verification gates

1. RED→GREEN synthetic native fixture for success gating, T/T+1 alignment, semantic mapping, and nonempty complete phase coverage.
2. Verify archive byte size and SHA-256 against pinned HF LFS metadata before indexing.
3. Full readback of every selected HDF5 parent and associated JSON episode metadata.
4. Independently verify record uniqueness, native references, hashes, success evidence, modality alignment, phase coverage, no candidate-G keys, and official-only provenance.
5. Combine only already verified official indexes; verify the combined records hash and counts independently.

## Abort conditions

- Source revision/hash mismatch, unsafe ZIP member path, missing JSON/HDF5 pair, missing native success evidence, non-finite actions, T/T+1 mismatch, invalid phase partition, output directory collision, or any write outside the declared Gina-owned roots.

## Actual results

### Acquisition and provenance

- Downloaded and SHA-256 verified all **9/9** selected official ZIP archives from pinned revision `d674485bbffdd533914e52d272fdda34c0515608`.
- Verified payload size: **265,543,076 bytes**.
- Native archives remain unchanged under `data/maniskill/official_d674485bbffdd533914e52d272fdda34c0515608/archives/`.
- Remote inventory SHA-256: `5da3c3d8426a93a1135303d93715ccc31b021c223537298899a6b2321af35172`.

### TDD implementation

- Added `framework/scripts/data/build_maniskill_official_trajectory_index.py`.
- Added `framework/scripts/data/test_build_maniskill_official_trajectory_index.py`.
- RED→GREEN corrections made from actual payload behavior:
  1. mixed-success archives are filtered to official successful parents while exclusions remain audited;
  2. parent identity includes the full native HDF5 member path so same-basename controller datasets cannot collide;
  3. native `obs_mode` and full environment kwargs are retained for modality auditing.
- Focused tests: **6/6 PASS**.

### Canonical ManiSkill index

Canonical output: `data/semantic_indexes/maniskill_official_20261023_v2/`.

- **15,262** official successful parent trajectories;
- **1,581,768** native transitions;
- **61,048** derived phase views, all `independent_success=false`;
- **9** native tasks across six underrepresented semantic families;
- **25** unsuccessful `PegInsertionSide-v1` candidates excluded from the semantic index without altering the native archive;
- records SHA-256: `0d2a7c4bb0a7c0fae6ba61b42a50355530921cc64c1db5aebf5524791c4a2870`;
- summary SHA-256: `654186c2a15b7c6d2b742f22d6a3afc7bef79df550e08ef00646b1b716d64267`.

Family counts:

- insertion/assembly: 2,975;
- orientation manipulation: 2,008;
- tool/contact operation: 4,300;
- planar push: 2,606;
- rolling contact: 2,373;
- multi-object arrangement: 1,000.

### Native modality boundary

The pinned official compressed payloads use only `none`, `state`, or unspecified observation modes. A native `obs` group is present for 5,975 indexed parents, but these are state observations, not RGB/RGB-D trajectory streams. Therefore this operation adds state/action/reset/success demonstrations and semantic phase sidecars; it does **not** claim image expansion. Image replay, if later requested, must be a separate persisted replay operation with simulator/runtime validation.

### Four-simulator canonical view

Canonical output: `data/semantic_indexes/official_reorg_20261023_v5/`.

- **20,194** official successful parents: CALVIN 432, LIBERO 2,000, RoboTwin 2,500, ManiSkill 15,262;
- **2,495,088** native frames/transitions;
- **82,802** derived phase views;
- increase over v4: **+15,262 parents**, from 4,932 to 20,194 (**4.094485×** total scale);
- `pick_place` remains 2,050 parents but falls from 41.57% to **10.151530%** because the added ManiSkill selection intentionally targets other families;
- records SHA-256: `4a232497f99d8abf9571f970b198c000bd167f6aaa8056d997ee29e78724895c`;
- summary SHA-256: `b6c5b63e76d5f19a27d8ff57dda15a80c6031f4eb1b463def29c5da7815b5026`.

### Independent verification

- ManiSkill full readback: `artifacts/evidence/20261023-maniskill-official-expansion/independent_readback_v2.json` — **PASS** for all 9 archive hashes, 15,262 HDF5 parent groups, 1,581,768 transitions, T/T+1 alignment, success evidence, phase coverage, unique IDs, and candidate-G absence.
- Unified v5 readback: `artifacts/evidence/20261023-maniskill-official-expansion/unified_v5_independent_readback.json` — **PASS** for 20,194 unique official parents, source-index hashes, phase partitions, official-only provenance, and candidate-G absence.

### Claim boundary

- Implemented and offline/full-payload verified: **yes**.
- Simulator replay or online native-Oracle execution in this operation: **no**.
- New physical trajectories generated: **0**.
- Training or held-out evaluation performed: **no**.
