# Simulators in AutoResearch

## Why

Physics / rendering engines are **pluggable backends**. AutoResearch owns the
research flywheel; each platform keeps its own checkout or external tree.

## Default paths

| Variable | Default |
|----------|---------|
| `AUTORESEARCH_ROOT` | this repo |
| `ISAACLAB_ROOT` | `$AUTORESEARCH_ROOT/simulators/isaaclab` |
| `MANISKILL_ROOT` | `$AUTORESEARCH_ROOT/simulators/maniskill` |
| `CALVIN_G_ROOT` | `/mnt/nvme33n1/wuk_code/calvin_g` (H200) or `phase1_calvin/calvin_g` |
| `CALVIN_CODE` / `CALVIN_ENV` / `CALVIN_DATASET` | resolved by `simulators/resolve_paths.sh` |

```bash
source simulators/resolve_paths.sh
```

## Nested IsaacLab git

`simulators/isaaclab` is a **nested git repository** (not vendored into AutoResearch history).

- Push/pull IsaacLab with its own `origin` (recreate on GitHub if deleted).
- AutoResearch `.gitignore` ignores the nested tree except platform docs.
- Do not commit `simulators/isaaclab/runs/` or `datasets/`.

```bash
cd simulators/isaaclab
git remote add origin git@github.com:<you>/IsaacLab.git   # if missing
git push -u origin main
```

## CALVIN (Phase 1)

CALVIN is an **external** workspace (`calvin_g`), linked under `phase1_calvin/`.
Code entry: `framework/scripts/`. Env recipe: `docs/ENVIRONMENTS.md`.

## Registry

See `simulators/registry.yaml` for declared platforms.

## Launch pattern (Isaac)

```bash
source simulators/resolve_paths.sh
cd "$ISAACLAB_ROOT"
./isaaclab.sh -p scripts/…
```

Wrappers under `scripts/*.sh` source `resolve_paths.sh` automatically.

Full recovery: [BOOTSTRAP.md](BOOTSTRAP.md) · Hosts: [HOSTS.md](HOSTS.md).
