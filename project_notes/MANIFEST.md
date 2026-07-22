# Project Manifest

Updated: 2026-07-22 (unified Git recovery layout)

## Canonical sources

| Source | Path / URL |
|--------|------------|
| **Git (recovery)** | https://github.com/wukui-muc/AutoResearch.git |
| Local workstation | `/home/l/code/AutoResearch` |
| H200 NVMe working copy | `/mnt/nvme33n1/wuk_code/AutoResearch` |
| CALVIN external | `/mnt/nvme33n1/wuk_code/calvin_g` |

## Phase 0: IsaacLab

- Code: **repo root** (`scripts/`, `goals/`, `simulators/`)
- Nested IsaacLab: `simulators/isaaclab` (local) or `$HOME/code/IsaacLab` (H200)
- Journal: `docs/progress_phase0_isaac.md`, `findings.md`
- Env: `env_isaaclab` — freeze `shared/envs/env_isaaclab.pip.freeze.txt`

Historical H200 nested copy (pre-unification): `phase0_isaac/autoresearch` @ `08b824a` (superseded by GitHub master).

## Phase 1: CALVIN

- Framework code: `framework/` (now tracked in Git)
- Knowledge base: `knowledge_base/`
- Journal: `docs/progress_phase1_calvin.md`
- Evidence (metrics): `experiments/phase1_calvin/`
- Env: `calvin_h200` — freeze `shared/envs/calvin_h200.pip.freeze.txt`
- CALVIN commit: `fa03f01` · calvin_env: `1431a46` (EGL headless patch)

## Storage policy

- Git: code, docs, journals, KB, lightweight metrics
- Disk: demos, `*.pt`, CALVIN ABC_D/D_D, nested simulator trees
- Rebuild: [docs/BOOTSTRAP.md](../docs/BOOTSTRAP.md)

## Migration decisions (2026-07-22)

- Merged H200 untracked `framework/` + `knowledge_base/` + phase-1 PROGRESS into GitHub AutoResearch.
- Kept Phase-0 flat layout at repo root (did not nest under `phase0_isaac/`).
- IsaacLab remains nested/external git; CALVIN remains external `calvin_g` with symlinks.
