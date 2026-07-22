# Phase 0: IsaacLab / AutoResearch

This phase is the **Isaac Lab** lift / DAgger track (AnchorMask + guidance field G_θ).

On the **canonical Git repo** (and the local workstation), Phase-0 code lives at the **repository root**
(`scripts/`, `goals/`, `simulators/`, `findings.md`), not under a nested copy.

On the **H200 NVMe** historical layout, an older checkout also exists at:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/phase0_isaac/autoresearch
/home/wukui/AutoResearch → that path   # compatibility symlink
```

Prefer the GitHub `AutoResearch` root going forward. Sync with:

```bash
git pull
source simulators/resolve_paths.sh
```

Journal: [docs/progress_phase0_isaac.md](../docs/progress_phase0_isaac.md) · Findings: [findings.md](../findings.md) · Bootstrap: [docs/BOOTSTRAP.md](../docs/BOOTSTRAP.md).
