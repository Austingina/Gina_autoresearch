# Hosts and path map

AutoResearch runs on at least two machines. Always prefer env vars over hard-coded paths.

## Machines

| Role | SSH / host | Canonical AutoResearch root |
|------|------------|-----------------------------|
| **Local workstation** | `l-Z790-UD-AX` | `/home/l/code/AutoResearch` |
| **Remote H200** | `ssh unrealzoo-remote` → `wukui@34.52.175.145` | `/mnt/nvme33n1/wuk_code/AutoResearch` |

```bash
# ~/.ssh/config (local)
Host unrealzoo-remote
    HostName 34.52.175.145
    User wukui
    IdentityFile ~/.ssh/id_ed25519
```

## Path conventions

```bash
export AUTORESEARCH_ROOT=<clone root>          # required
source "$AUTORESEARCH_ROOT/simulators/resolve_paths.sh"
# → ISAACLAB_ROOT, MANISKILL_ROOT, CALVIN_G_ROOT, CALVIN_CODE, CALVIN_ENV, CALVIN_DATASET
```

| Variable | Local default | H200 default (if present) |
|----------|---------------|---------------------------|
| `AUTORESEARCH_ROOT` | `/home/l/code/AutoResearch` | `/mnt/nvme33n1/wuk_code/AutoResearch` |
| `ISAACLAB_ROOT` | `$AR/simulators/isaaclab` | `/home/wukui/code/IsaacLab` or nested |
| `CALVIN_G_ROOT` | *(set when CALVIN installed)* | `/mnt/nvme33n1/wuk_code/calvin_g` |
| `CALVIN_CODE` | `$AR/phase1_calvin/calvin_code` | symlink → `$CALVIN_G_ROOT/code/calvin` |
| `CALVIN_ENV` | `$AR/phase1_calvin/envs/calvin_h200` | `$CALVIN_G_ROOT/envs/calvin_h200` |
| `CALVIN_DATASET` | debug or ABC_D under `phase1_calvin/data/` | same via symlink |

Legacy compatibility (H200):

```text
/home/wukui/AutoResearch → …/AutoResearch/phase0_isaac/autoresearch   # old flat Isaac tree
/home/l/code/IsaacLab    → …/AutoResearch/simulators/isaaclab         # local only
```

See also [archive/old_paths.md](../archive/old_paths.md).

## Disk policy

| Content | Where |
|---------|--------|
| Git clone (code/docs) | anywhere; keep small |
| Isaac runs / HDF5 | local NVMe or `$AR/runs`, `$AR/data` (gitignored) |
| CALVIN ABC_D / D_D | `$CALVIN_G_ROOT/data` on large NVMe (H200: `/mnt/nvme33n1`) |
| Do not put heavy artifacts on root `/` or full RAID volumes |

## Git remotes

| Repo | URL | Notes |
|------|-----|-------|
| AutoResearch | `https://github.com/wukui-muc/AutoResearch.git` | **canonical** recovery source |
| IsaacLab fork | `https://github.com/wukui-muc/IsaacLab.git` | recreate if deleted; upstream `isaac-sim/IsaacLab` |
| CALVIN | `https://github.com/mees/calvin.git` @ `fa03f01` | + `calvin_env` @ `1431a46` |

The H200 unified tree historically lived **outside** Git; as of 2026-07-22 the recoverable framework is merged into this GitHub repo. Keep server working copies in sync with `git pull` + selective `rsync` for artifacts.
