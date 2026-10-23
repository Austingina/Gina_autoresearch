# Gina AutoResearch — Foundation-G / Interaction Geometry

Unified research flywheel for **whether a compact geometric goal interface G improves visuomotor generalization**.

```text
a_t = π(o_t, M_t, G_t)     # research interface (platform-agnostic)
```

**Code:** <https://github.com/Austingina/Gina_autoresearch><br>
**Derived dataset:** <https://huggingface.co/datasets/AUSTINGINA/Gina_autoresearch><br>
**Recovery entry point:** [bootstrap.md](bootstrap.md)<br>
**Dataset ledger:** [dataset.md](dataset.md)<br>
(If disks expire: clone this GitHub repo, rebuild the public inputs, and rerun the recorded gates.)

## Active vs legacy tracks

| Phase | Backend | Status | Where |
|-------|---------|--------|-------|
| **1 (active)** | CALVIN dual-cam, compact `G=[A,F]` | E3 causal pass; ABC→D visual weak-pass (2026-07-21) | `framework/`, `docs/progress_phase1_calvin.md`, `knowledge_base/` |
| **0 (legacy)** | Isaac Lab lift / DAgger (AnchorMask + G_θ) | S1 bring-up; strict success@2cm still open | `scripts/`, `goals/`, `simulators/isaaclab`, `docs/progress_phase0_isaac.md`, `findings.md` |

## Layout

```text
AutoResearch/
  bootstrap.md               # rebuild from an empty machine
  dataset.md                 # data URLs, hashes, and derivation recipes
  docs/HOSTS.md              # local + H200 paths
  docs/ENVIRONMENTS.md       # conda / pip recipes
  PROGRESS.md                # index → phase journals
  framework/                 # Phase-1 CALVIN trainers / gates
  knowledge_base/            # paper + evidence writeups
  experiments/               # run records (metrics tracked; weights not)
  simulators/                # IsaacLab vendored sources + registry
  scripts/ goals/ configs/   # Phase-0 Isaac flywheel
  shared/envs/               # pip freezes
  phase0_isaac/ phase1_calvin/  # phase READMEs + CALVIN symlinks
```

## Quick start

```bash
git clone https://github.com/Austingina/Gina_autoresearch.git && cd Gina_autoresearch
export AUTORESEARCH_ROOT="$PWD"
source simulators/resolve_paths.sh

# Phase 0 — Isaac
# (install env_isaaclab; IsaacLab sources already under simulators/isaaclab)
./scripts/init_workspace.sh
python scripts/check_flywheel.py

# Phase 1 — CALVIN (after linking calvin_g — see bootstrap.md)
framework/scripts/check_calvin_ready.sh
GPU=0 framework/scripts/run_calvin_dualcam.sh smoke
```

## Docs

| Doc | Content |
|-----|---------|
| [bootstrap.md](bootstrap.md) | **Start here after a wipe** |
| [dataset.md](dataset.md) | Data sources, hashes, split semantics, and generated caches |
| [docs/HOSTS.md](docs/HOSTS.md) | Local / H200 path map |
| [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) | `env_isaaclab` + `calvin_h200` |
| [docs/SIMULATORS.md](docs/SIMULATORS.md) | Pluggable backends |
| [docs/DATA_ISOLATION.md](docs/DATA_ISOLATION.md) | Demos never write into Isaac `datasets/` |
| [docs/REMOTE_SETUP.md](docs/REMOTE_SETUP.md) | H200 setup + rsync |
| [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) | Isaac curriculum S0–S3 |
| [docs/progress_phase1_calvin.md](docs/progress_phase1_calvin.md) | Latest CALVIN gates |
| [docs/progress_phase0_isaac.md](docs/progress_phase0_isaac.md) | Isaac S0/S1 journal |

## Git vs artifacts

- **Git:** code, configs, journals, KB, lightweight `experiments/**` metrics.
- **rsync / regenerate:** HDF5, `*.pt`, CALVIN ABC_D/D_D, Isaac `runs/`/`datasets/` (not in Git).
