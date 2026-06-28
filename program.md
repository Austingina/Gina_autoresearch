# AnchorMask + GuidanceField — Autoresearch

## One-sentence goal

Learn **guidance field G_θ** from demo mask trajectories, then prove **π(o, M, G_θ)** beats **π(o, M)** — starting on **fixed cube**, then harder curricula.

## Read first

1. [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) — **curriculum S0→S1→S2**, exit criteria
2. [docs/REFERENCES.md](docs/REFERENCES.md) — papers + `$ISAACLAB_ROOT` / `$HIEVT_ROOT` files
3. [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) — Step 1 / Step 2

## Policy contract

```python
a_t = pi(o_t, M_t, G_t)
```

## Two-step pipeline (every curriculum stage)

```text
Step 1  Learn G_θ from demos     →  field_alignment (val)
Step 2  Train π with (M, G_θ)    →  success_rate, post_grasp_wander_m
        Compare B7 vs B1; B3 = oracle upper bound only
```

## Curriculum (do not skip)

| Stage | split | Cube | Goal | Open when |
|-------|-------|------|------|-----------|
| **S0** | `s0_fixed` | fixed | fixed | **START** |
| **S1** | `s1_random_cube` | random xy | fixed | S0 pass |
| **S2** | `s2_random_goal` | random | random | S1 pass |
| **S3** | `s3_ood_object` | random | random + OOD appearance | Month 2 |

## Data isolation (mandatory)

- **Never** train or collect into `$ISAACLAB_ROOT/datasets/`.
- Import copies: `scripts/import_isaac_demos.py`
- New collection: `scripts/collect_lift_demos.sh` → `data/lift/<stage>/demos/`
- See [docs/DATA_ISOLATION.md](docs/DATA_ISOLATION.md)

## Setup

```bash
export ISAACLAB_ROOT=/home/l/code/IsaacLab
export HIEVT_ROOT=/home/l/code/Hierarchical-Instruction-aware-Embodied-Visual-Tracking-master
export EVT_ROOT=/home/l/code/Offline_RL_Active_Tracking
export DUALGF_ROOT=/home/l/code/DualGF
export TARGF_ROOT=/home/l/code/TarGF   # https://github.com/AaronAnima/TarGF
python scripts/check_flywheel.py
git checkout -b autoresearch/<tag>
```

## Agent rules

- Edit **`experiment.py` only** (field predictor + hyperparams).
- **Current stage:** `s0_fixed` until exit criteria met (see EXPERIMENT_PLAN.md).
- Step 1 ratchet: `field_alignment` ↑ ≥ 0.01.
- Step 2 ratchet: `success_rate` ↑ ≥ 0.005 OR `post_grasp_wander_m` ↓ ≥ 0.02.
- Tag commits: `step1-s0-...` or `step2-s0-...`.

## Commands

```bash
# Step 1 (S0) — isolated data
./scripts/init_workspace.sh
python scripts/import_isaac_demos.py \
  --source $ISAACLAB_ROOT/datasets/lift_cube_lenient_teacher_maskdepth_smoke.hdf5 \
  --stage s0_fixed

python scripts/learn_guidance_field.py --task lift --stage s0_fixed

# Step 2 eval
python run_experiment.py --task lift --split s0_fixed --backend dry_run > run.log 2>&1
grep "^success_rate:\|^post_grasp_wander\|^field_alignment" run.log
```

## Baselines

| ID | guidance_mode | Role |
|----|---------------|------|
| B1 | anchor_only | negative |
| B7 | anchor_field_demo | **main** (learned G_θ) |
| B3 | anchor_field_rule | oracle ceiling |

## results.tsv

```
commit	success_rate	post_grasp_wander_m	field_alignment	guidance_mode	backend	task	split	status	description
```

## References (code)

| Repo | Key script |
|------|------------|
| IsaacLab | `train_lift_cube_dagger_vision_binary_gripper.py`, `record_oracle_goal_conditioned_demos.py` |
| HIEVT | `train_offline_goal_conditioned_bbox.py` |
| EVT | mask pipeline in `Offline_RL_Active_Tracking` |
| [TarGF](https://github.com/AaronAnima/TarGF) | `main.py --mode train_gf`, `score_matching/` |
| DualGF | `sde_run.sh` → `Runners/*/BallSDE.py`, `NaviSDEImg.py` |

WM (MWM) = optional B8 ablation only — [docs/REFERENCES.md](docs/REFERENCES.md).

**NEVER STOP** the loop once started.
