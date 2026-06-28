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
