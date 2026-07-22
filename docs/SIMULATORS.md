# Simulators in AutoResearch

## Why

Experiments (especially lift DAgger) run on **Isaac Lab**, but AutoResearch also
uses **ManiSkill** (and may add more). Platforms are nested under `simulators/`
so the research repo owns orchestration while each engine keeps its own checkout.

## Default paths

| Variable | Default |
|----------|---------|
| `AUTORESEARCH_ROOT` | this repo |
| `ISAACLAB_ROOT` | `$AUTORESEARCH_ROOT/simulators/isaaclab` |
| `MANISKILL_ROOT` | `$AUTORESEARCH_ROOT/simulators/maniskill` |

Override any time: `export ISAACLAB_ROOT=/elsewhere/IsaacLab`.

Legacy sibling path `/home/l/code/IsaacLab` is supported via symlink after migration.

## Nested IsaacLab git

`simulators/isaaclab` is a **nested git repository** (not vendored into AutoResearch history).

- Push/pull IsaacLab code with its own `origin` (recreate on GitHub if deleted).
- AutoResearch `.gitignore` ignores the nested tree except platform docs.
- Do not commit `simulators/isaaclab/runs/` or `datasets/` (local only).

```bash
# Recreate remote after deletion (example)
cd simulators/isaaclab
git remote remove origin 2>/dev/null || true
git remote add origin git@github.com:<you>/IsaacLab.git
git push -u origin main
```

## Launch pattern

```bash
source simulators/resolve_paths.sh
cd "$ISAACLAB_ROOT"
./isaaclab.sh -p scripts/reinforcement_learning/train_lift_cube_dagger_vision_binary_gripper.py ...
```

Wrappers under `scripts/*.sh` source `resolve_paths.sh` automatically.
