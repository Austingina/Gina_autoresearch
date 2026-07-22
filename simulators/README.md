# Simulators — pluggable backends for AutoResearch

AutoResearch owns the **research flywheel** (data, G_θ, policy ablations, metrics).
Physics / rendering engines live under `simulators/<name>/` as **platform backends**.

```text
a_t = π(o_t, M_t, G_t)     # research interface (platform-agnostic)
         │
         ▼
   simulators/<platform>/  # IsaacLab | ManiSkill | …
```

## Layout

| Path | Role |
|------|------|
| `simulators/registry.yaml` | Declared platforms + env vars |
| `simulators/resolve_paths.sh` | Shell helper: set `AUTORESEARCH_ROOT`, `ISAACLAB_ROOT`, … |
| `simulators/isaaclab/` | **Isaac Lab** checkout (nested git; Phase-0 lift/DAgger) |
| `simulators/maniskill/` | Adapter notes for ManiSkill (code in `goals/maniskill_bridge.py`) |
| `phase1_calvin/` + `framework/` | **CALVIN** (Phase-1; external `CALVIN_G_ROOT`, not under `simulators/`) |

## Contract for a platform

Each backend should provide (docs + scripts, not necessarily one Python ABC):

1. **Env / task ids** used by AutoResearch configs
2. **Demo I/O** that can be imported into `AutoResearch/data/` (never write platform `datasets/` in-place)
3. **Train / play entrypoints** invoked from `scripts/*.sh` via `./<launcher> -p …`
4. **Path env var** (e.g. `ISAACLAB_ROOT`) with default = `$AUTORESEARCH_ROOT/simulators/<name>`

## Adding a new simulator

1. Create `simulators/<name>/` (nested clone or vendor)
2. Add an entry to `registry.yaml`
3. Add `scripts/run_<name>_*.sh` wrappers that `source simulators/resolve_paths.sh`
4. Document data import into `data/` (see `docs/DATA_ISOLATION.md`)
5. Keep platform-specific weights/runs **inside** the nested checkout or under `AutoResearch/runs/` — do not pollute other platforms

## IsaacLab (primary)

```bash
source simulators/resolve_paths.sh
echo "$ISAACLAB_ROOT"   # → …/AutoResearch/simulators/isaaclab
cd "$ISAACLAB_ROOT" && ./isaaclab.sh -p scripts/…
```

See `docs/SIMULATORS.md` and `simulators/isaaclab` nested README.
