# Data isolation policy

## Principle

| Location | Role | AutoResearch may |
|----------|------|------------------|
| `$ISAACLAB_ROOT/datasets/` | Shared upstream (CQL, DAgger, etc.) | **Read + copy only** |
| `AutoResearch/data/` | This project's demos | **Read + write** |
| `AutoResearch/checkpoints/` | Field / policy weights | **Read + write** |
| `AutoResearch/runs/` | Eval logs | **Read + write** |
| `results.tsv`, `findings.md` | Autoresearch ratchet | **Read + write** (tracked in git) |

## Workflows

### 1. Bootstrap from existing Isaac data

```bash
./scripts/init_workspace.sh

python scripts/import_isaac_demos.py \
  --source $ISAACLAB_ROOT/datasets/lift_cube_lenient_teacher_maskdepth_smoke.hdf5 \
  --stage s0_fixed

# Optional: import a full shard (large — copies, does not move)
python scripts/import_isaac_demos.py \
  --source $ISAACLAB_ROOT/datasets/lift_cube_lenient_teacher_maskdepth \
  --stage s0_fixed --glob 'lift_cube_lenient_teacher_maskdepth_v3.hdf5'
```

Each import writes `data/manifests/import_*.json` with `source`, `dest`, `sha256`.

### 2. Collect new demos

```bash
STAGE=s0_fixed NUM_DEMOS=100 bash scripts/collect_lift_demos.sh
```

Output: `data/lift/s0_fixed/demos/s0_fixed_<timestamp>.hdf5`  
Never passes `--dataset_file` under IsaacLab.

### 3. Train Step 1

```bash
python scripts/learn_guidance_field.py \
  --task lift --stage s0_fixed
# default demo-dir: data/lift/s0_fixed/demos/
```

`learn_guidance_field.py` refuses `--demo-dir` under `IsaacLab/datasets/` unless `--allow-external-data`.

### 4. Compare with upstream

- Run `analyze_offline_cql_dataset.py` on **both** paths separately.
- Do not symlink Isaac datasets into `data/` (breaks isolation); use `import_isaac_demos.py`.

## Curriculum mapping

| Stage | Collect | Import source (example) |
|-------|---------|-------------------------|
| S0 | default `collect_lift_demos.sh` | smoke / v3 success-heavy shards |
| S1 | `RANDOMIZE_CUBE=1 STAGE=s1_random_cube` | — |
| S2 | + future `RANDOMIZE_LIFT_GOAL` flag | — |
