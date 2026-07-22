# Simulators in AutoResearch

## Why

Physics / rendering engines are **pluggable backends**. AutoResearch owns the
research flywheel; each platform keeps install docs and (where practical) source
under this repo or an external linked tree.

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

## Isaac Lab (vendored)

`simulators/isaaclab` is **checked into AutoResearch** (README, `isaaclab.sh`,
`source/`, research scripts, docs). Clone AutoResearch → you get the IsaacLab
tree used by Phase 0.

Still **not** in Git (local only):

- `simulators/isaaclab/runs/`, `datasets/`
- `_isaac_sim*`, `.venv/` (Isaac Sim / conda live in the env, not in the tree)

Install: [ENVIRONMENTS.md](ENVIRONMENTS.md) + `scripts/remote_setup_env_isaaclab.sh`.

A detached history backup may exist locally as `simulators/isaaclab.git.backup/`
(gitignored) if you previously used a nested clone.

## CALVIN (Phase 1)

CALVIN remains an **external** workspace (`calvin_g`), linked under `phase1_calvin/`.
Code entry: `framework/scripts/`. Env recipe: [ENVIRONMENTS.md](ENVIRONMENTS.md).

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
