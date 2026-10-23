# AutoResearch data (project-owned)

**Rule:** All demos, checkpoints, and run logs for this project live under `AutoResearch/`.
Do **not** read/write `$ISAACLAB_ROOT/datasets/` from training scripts — those files may be used for CQL / other experiments.

## Layout

```text
data/
├── manifests/              # JSON provenance (small, safe to commit)
└── lift/
    ├── s0_fixed/demos/     # S0: fixed cube + fixed goal
    ├── s1_random_cube/demos/
    ├── s2_random_goal/demos/
    └── s3_ood_object/demos/

checkpoints/lift/           # guidance_field_*.pt
runs/                       # eval logs
```

## Official-trajectory reorganization + label enrichment

The short-term multi-simulator expansion line keeps native payloads separate and writes only a common semantic index:

```text
data/
├── calvin/task_ABC_D_g_intent_multifold_s12/       # admitted CALVIN parent segments
├── libero/official_f13aa24a3da8c43c7225569f28c562979fa0e35a/
│   ├── libero_10|libero_goal|libero_object|libero_spatial/  # native HDF5
│   └── official_inventory_manifest.json
├── robotwin/official_<hf_revision>/                 # official payload/inventory only
├── maniskill/official_d674485bbffdd533914e52d272fdda34c0515608/
│   ├── archives/                                    # 9 pinned native official ZIPs
│   └── remote_inventory.json
├── vima/official_f35cee89382278b8d63d6d6f9c27cfeda3deb3f5/
│   ├── upstream_source/                             # pinned VIMABench source tarball
│   ├── source_revision.json                         # HF revision, size, and LFS SHA-256
│   └── selected_semantic_gap_v1/                    # frozen sparse official subset
│       ├── payload/                                 # untouched selected native members
│       ├── selection_manifest.json
│       └── extraction_receipt.json
└── semantic_indexes/
    ├── official_reorg_20261022_v1/                  # CALVIN + LIBERO verified source index
    │   ├── records.jsonl
    │   ├── summary.json
    │   ├── distribution.csv
    │   └── REPORT.md
    ├── robotwin_official_20261022_v3/               # canonical verified RoboTwin official index
    ├── maniskill_official_20261023_v2/               # canonical verified ManiSkill official index
    ├── vima_official_20261023_v2/                    # canonical verified VIMA official subset index
    ├── official_reorg_20261022_v4/                   # canonical 3-simulator predecessor
    ├── official_reorg_20261023_v5/                   # canonical 4-simulator predecessor
    ├── official_reorg_20261023_v6/                   # canonical 5-simulator official semantic view
    ├── online_native_expert_20261022_v1/            # newly executed expert pilots only
    │   ├── records.jsonl
    │   ├── summary.json
    │   └── REPORT.md
    └── multisource_20261022_v2/STATUS.json          # paused, non-authoritative
```

The two provenance lines are authoritative **only as separate indexes**:

- `official_trajectory_reorganization_and_label_enrichment`: CALVIN, LIBERO, RoboTwin, ManiSkill, and VIMA fixed-revision official successful parent trajectories. Full local archives are hash-verified where downloaded; VIMA uses a fixed 21.5 GB archive revision/LFS SHA-256 plus CRC- and SHA-verified sparse native-member extraction. The canonical combined semantic view is `official_reorg_20261023_v6/`.
- `online_native_expert_collection`: newly executed Isaac/CALVIN/RoboTwin expert trajectories. The five accepted RoboTwin `place_empty_cup` pilots belong here and are excluded from official-data counts.

The former direct multisource merge is paused and must not be used for training or distribution statistics. A unified semantic view may join the indexes for search, but it must preserve provenance and simulator-native action/state boundaries.

`robotwin_official_20261022_v1`, `robotwin_official_20261022_v2`, `official_reorg_20261022_v2`, and `official_reorg_20261022_v3` are superseded audit artifacts: an independent readback found 21 empty phase intervals in each. `maniskill_official_20261023_v1` is also audit-only because v2 adds the explicit native `obs_mode`/environment schema fields required for downstream modality auditing. `vima_official_20261023_v1` is audit-only because its records predate the common unified-index field contract; v2 contains the same accepted parents with canonical `native_task_id`, `canonical_task`, `phase_views`, and provenance fields. They must not be used for training or statistics.

The canonical ManiSkill v2 index contains 15,262 official successful parents from 9 selected non-ordinary-pick tasks and 1,581,768 transitions. The source archives contain only `none`/`state` observation modes (5,975 parents have a native state observation group); no RGB/RGB-D trajectory stream is claimed. Twenty-five unsuccessful PegInsertion candidates remain in the untouched archive but are explicitly excluded from the success-only semantic index.

The canonical VIMA v2 index contains 5,999 official successful parents selected uniformly from six semantic-gap tasks: `rotate`, `twist`, `sweep_without_exceeding`, `follow_order`, `pick_in_order_then_restore`, and `rearrange_then_restore`. It preserves 12,668 native oracle actions, 18,667 aligned observations, and both official front/top RGB streams. One selected `pick_in_order_then_restore` candidate (`105017`) is excluded because its five RGB files disagree with its native T+1 observation length of four; thresholds were not relaxed. This is a sparse immutable-member extraction from the pinned official archive, not a simulator rollout.

The phase rows in `records.jsonl` are derived views over complete parent demonstrations. They are **not** new independent success evidence. CALVIN, LIBERO, RoboTwin, ManiSkill, and VIMA native image/state/action spaces remain simulator-specific; candidate G is not an admission field or default input. The current consolidated inventory report is `docs/official-trajectory-inventory-20261023.md`. See also `docs/operations/20261022-official-trajectory-reorganization.md`, `docs/operations/20261023-maniskill-official-trajectory-expansion.md`, and `docs/operations/20261023-vima-official-trajectory-expansion.md`.

## Import (copy from Isaac, read-only upstream)

```bash
python scripts/import_isaac_demos.py \
  --source $ISAACLAB_ROOT/datasets/lift_cube_lenient_teacher_maskdepth_smoke.hdf5 \
  --task lift --stage s0_fixed
```

## Collect (new demos, write only here)

```bash
NUM_DEMOS=50 bash scripts/collect_lift_demos.sh
# S1: RANDOMIZE_CUBE=1 STAGE=s1_random_cube bash scripts/collect_lift_demos.sh
```

## Compare without polluting

You may diff metrics or visualizations against Isaac originals, but always train from copies under `data/`.
