# Bootstrap — rebuild AutoResearch from Git alone

**Read this first** after cloning when local or server disks are gone.

Git tracks **code, configs, progress journals, experiment evidence (metrics/notes), and env recipes**.
Large simulators, conda envs, demos, and checkpoints are **recreated** from the recipes below.

## 0. Clone

```bash
git clone https://github.com/wukui-muc/AutoResearch.git
cd AutoResearch
export AUTORESEARCH_ROOT="$PWD"
source simulators/resolve_paths.sh   # sets ISAACLAB_ROOT, CALVIN_*, hosts
```

Optional IsaacLab backend (vendored under this repo):

```bash
# After clone, Isaac Lab sources are already at:
#   simulators/isaaclab/   (README, isaaclab.sh, source/, research scripts)
# Install Isaac Sim + conda env (not stored in Git):
bash scripts/remote_setup_env_isaaclab.sh
```

Legacy: a separate fork at `https://github.com/wukui-muc/IsaacLab.git` is optional;
prefer the vendored tree in this repository.
## 1. What lives where

| Tracked in Git | Not in Git (rebuild / rsync) |
|----------------|------------------------------|
| `PROGRESS.md`, `findings.md`, `docs/`, `knowledge_base/` | `runs/`, `checkpoints/*.pt`, HDF5 demos |
| `scripts/`, `goals/`, `framework/`, `configs/` | `simulators/isaaclab/runs|datasets|_isaac_sim*` |
| `simulators/isaaclab/` source + install scripts | conda / Isaac Sim packages (~GBs) |
| `experiments/**/*.json|md|yaml|sh` (metrics/notes) | `experiments/**/*.pt` weights |
| `shared/envs/*.pip.freeze.txt` | conda/venv installs (~GBs) |
| `simulators/registry.yaml`, platform docs | CALVIN `task_ABC_D` (~500G+), `task_D_D` |

Journals:

| File | Role |
|------|------|
| [PROGRESS.md](../PROGRESS.md) | Index + current active goal |
| [docs/progress_phase1_calvin.md](progress_phase1_calvin.md) | **Active** CALVIN claim ladder (E3/E4/ABC→D) |
| [docs/progress_phase0_isaac.md](progress_phase0_isaac.md) | Legacy Isaac S0/S1 lift DAgger |
| [findings.md](../findings.md) | Isaac ratchet / failure-mode notes |
| [knowledge_base/](../knowledge_base/) | Paper hypotheses + experiment writeups |

Hosts / paths: [HOSTS.md](HOSTS.md) · Env recipes: [ENVIRONMENTS.md](ENVIRONMENTS.md) · Simulators: [SIMULATORS.md](SIMULATORS.md)

## 2. Environments (pick by track)

### Phase 0 — Isaac Lab (lift / DAgger)

```bash
# See docs/ENVIRONMENTS.md and scripts/remote_setup_env_isaaclab.sh
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
# Install Isaac Sim 5.1 + Isaac Lab into simulators/isaaclab (or ISAACLAB_ROOT)
bash scripts/remote_setup_env_isaaclab.sh   # on H200-class hosts
pip install -e ".[visual]"                  # AutoResearch extras
```

Verified stack (H200, 2026-07): Python 3.11, `torch 2.7.0+cu128`, isaacsim 5.1, isaaclab 0.54.x.  
Pin dump: `shared/envs/env_isaaclab.pip.freeze.txt`.

### Phase 1 — CALVIN (compact `G=[A,F]`)

```bash
# Official CALVIN + H200 torch build (sm_90)
export CALVIN_G_ROOT=/path/to/calvin_g     # code + data + envs live here
# Clone official repos into $CALVIN_G_ROOT/code/calvin (@fa03f01)
# Create env from shared/envs/calvin_h200.pip.freeze.txt
#   torch==2.3.1+cu121  (NOT the official 1.13 cu117 env for training on H200)

ln -sfn "$CALVIN_G_ROOT" phase1_calvin/calvin_g
ln -sfn "$CALVIN_G_ROOT/code/calvin" phase1_calvin/calvin_code
ln -sfn "$CALVIN_G_ROOT/envs/calvin_h200" phase1_calvin/envs/calvin_h200
ln -sfn "$CALVIN_G_ROOT/data" phase1_calvin/data
```

Readiness:

```bash
source simulators/resolve_paths.sh
framework/scripts/check_calvin_ready.sh
```

## 3. Data regeneration

### Isaac demos (project-owned under `data/`)

```bash
./scripts/init_workspace.sh
# Import from an existing Isaac HDF5 (never write into IsaacLab/datasets/)
python scripts/import_isaac_demos.py \
  --source "$ISAACLAB_ROOT/datasets/<shard>.hdf5" --stage s0_fixed
# Or collect fresh:
STAGE=s1_random_cube RANDOMIZE_CUBE=1 NUM_DEMOS=100 bash scripts/collect_lift_demos.sh
```

Policy: [DATA_ISOLATION.md](DATA_ISOLATION.md).

### CALVIN datasets

Download official ABC_D / D_D / debug from CALVIN project pages into `$CALVIN_G_ROOT/data/`.
Minimum for smoke: `calvin_debug_dataset`. Full ABC→D gates need `task_ABC_D` (+ optional DINOv2 cache).

## 4. Resume the research queue

1. Read **PROGRESS.md** → follow the active-goal link (currently Phase 1 CALVIN).
2. Skim latest gate in `docs/progress_phase1_calvin.md` (ABC→D visual, E3, E4).
3. Reproduce or extend via `framework/scripts/` and `experiments/phase1_calvin/<run>/`.
4. For Isaac bring-up only: `docs/progress_phase0_isaac.md` + `scripts/run_s1_*.sh`.

Canonical new runs:

```bash
bash shared/scripts/init_experiment.sh phase1_calvin short_name
# → experiments/phase1_calvin/<stamp>_<short_name>/{command.sh,config.yaml,notes.md,...}
```

## 5. Push / sync discipline

- **Git:** code, docs, journals, lightweight metrics.
- **rsync:** demos, `student_*.pt`, CALVIN zips, Isaac `runs/` — never force into Git.
- After disk recovery: clone → `resolve_paths.sh` → install env → recreate/link simulators → regenerate or rsync data → continue from journals.
