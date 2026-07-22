# Isaac Lab backend (nested checkout)

Expected location: `simulators/isaaclab/` (gitignored nested repo).

```bash
# If missing, clone or move your IsaacLab tree here:
#   mv /path/to/IsaacLab simulators/isaaclab
#   # or: git clone <your-fork> simulators/isaaclab

source simulators/resolve_paths.sh
test -x "$ISAACLAB_ROOT/isaaclab.sh" && echo "OK: $ISAACLAB_ROOT"
```

Pinned locally for this workstation after 2026-07-22 migration:
- Path: `simulators/isaaclab`
- Legacy symlink: `/home/l/code/IsaacLab` → that path
- Role: primary lift / vision DAgger / mask+depth demos
