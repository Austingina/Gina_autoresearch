# AutoResearch — AnchorMask + GuidanceField

Unified **visual task interface** for policy learning:

```python
a_t = pi(o_t, M_t, G_t)   # M = anchor mask, G = guidance field
```

## Goals (clear)

1. **Learn** G_θ from demonstration mask trajectories (Step 1)
2. **Prove** G_θ helps policy vs mask-only B1 (Step 2)
3. **Generalize** via curriculum: fixed cube → random cube → random goal (S0→S1→S2)

Representation is valid only when **both** field learning and policy metrics improve.

## Data isolation

**All project demos and checkpoints live under `AutoResearch/data/`** — never write to `$ISAACLAB_ROOT/datasets/`.

| Doc | Content |
|-----|---------|
| [docs/DATA_ISOLATION.md](docs/DATA_ISOLATION.md) | **Import / collect policy** |
| [data/README.md](data/README.md) | Directory layout |

```bash
./scripts/init_workspace.sh
python scripts/import_isaac_demos.py \
  --source $ISAACLAB_ROOT/datasets/lift_cube_lenient_teacher_maskdepth_smoke.hdf5 \
  --stage s0_fixed
```

## Install

Core training environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -e ".[visual]"     # local plotting / video review
pip install -e ".[maniskill]"  # ManiSkill validation scripts
```

IsaacLab / Arena scripts require external local repositories; configure their paths with the
environment variables below.

## Docs (read in order)

| Doc | Content |
|-----|---------|
| [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) | **Start here** — curriculum S0–S3, exit criteria |
| [docs/REFERENCES.md](docs/REFERENCES.md) | Papers + local repos + files to reuse |
| [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) | Step 1 / Step 2 protocol |
| [docs/REMOTE_SETUP.md](docs/REMOTE_SETUP.md) | Remote H200 setup, artifact sync, sanity checks |
| [docs/RESEARCH_CONTEXT.md](docs/RESEARCH_CONTEXT.md) | Hypotheses |
| [program.md](program.md) | Autoresearch agent loop |

## Repos

```bash
export ISAACLAB_ROOT=/home/l/code/IsaacLab
export HIEVT_ROOT=/home/l/code/Hierarchical-Instruction-aware-Embodied-Visual-Tracking-master
export EVT_ROOT=/home/l/code/Offline_RL_Active_Tracking
export DUALGF_ROOT=/home/l/code/DualGF
export TARGF_ROOT=/home/l/code/TarGF   # git clone https://github.com/AaronAnima/TarGF
```

## Remote workflow

Use Git for code/config/docs and `rsync` for large artifacts:

```bash
git remote add origin https://github.com/wukui-muc/AutoResearch.git 2>/dev/null || \
  git remote set-url origin https://github.com/wukui-muc/AutoResearch.git
git push -u origin master

ssh wukui@34.52.175.145
git clone https://github.com/wukui-muc/AutoResearch.git
cd AutoResearch
pip install -r requirements.txt
./scripts/init_workspace.sh
```

See [docs/REMOTE_SETUP.md](docs/REMOTE_SETUP.md) for H200 checks and dataset/checkpoint sync commands.

## Quick start (S0)

```bash
cd /home/l/code/AutoResearch
pip install -e .
python scripts/check_flywheel.py

# Step 1 (S0) — project-owned data only
./scripts/init_workspace.sh
python scripts/import_isaac_demos.py \
  --source $ISAACLAB_ROOT/datasets/lift_cube_lenient_teacher_maskdepth_smoke.hdf5 \
  --stage s0_fixed

python scripts/learn_guidance_field.py --task lift --stage s0_fixed
# demo-dir default: data/lift/s0_fixed/demos/

# Step 2 — policy eval (after checkpoint)
python run_experiment.py --task lift --split s0_fixed --backend dry_run
```

## Three-file contract

| File | Role |
|------|------|
| `prepare.py` | Metrics + curriculum splits (immutable) |
| `experiment.py` | Field predictor + config (agent edits) |
| `program.md` | Agent instructions |
