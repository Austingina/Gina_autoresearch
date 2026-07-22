# Old Paths

Updated: 2026-07-02

## Preserved Compatibility Paths

- `/home/wukui/AutoResearch` now symlinks to `/mnt/nvme33n1/wuk_code/AutoResearch/phase0_isaac/autoresearch`.

## Existing External Paths

- `/home/wukui/code/IsaacLab` remains in place and is linked from `external/IsaacLab`.
- `/mnt/nvme33n1/wuk_code/calvin_g` remains in place and is linked from `phase1_calvin/calvin_g`.

## Reasoning

- The old AutoResearch repo was small and safe to move under the new NVMe root.
- CALVIN was already under NVMe and contains conda envs with absolute paths, so it was not moved.
- IsaacLab may contain path-sensitive local environment state, so it is kept as an external dependency for now.
