# Isaac Lab backend (vendored)

Location: `simulators/isaaclab/` — **tracked in AutoResearch** (source + install scripts).

Heavy local artifacts stay gitignored: `runs/`, `datasets/`, `_isaac_sim*`, `.venv/`.

```bash
source simulators/resolve_paths.sh
test -x "$ISAACLAB_ROOT/isaaclab.sh" && echo "OK: $ISAACLAB_ROOT"
# Install Isaac Sim into the conda env (not into this tree):
#   bash scripts/remote_setup_env_isaaclab.sh
```

- Path: `simulators/isaaclab`
- Legacy symlink (this workstation): `/home/l/code/IsaacLab` → that path
- Role: primary lift / vision DAgger / mask+depth demos
- Upstream reference: https://github.com/isaac-sim/IsaacLab (+ research patches for AutoResearch)
