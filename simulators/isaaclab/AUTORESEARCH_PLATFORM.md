# AutoResearch integration note

This Isaac Lab tree is the **primary simulator backend** for
[AutoResearch](https://github.com/wukui-muc/AutoResearch), **vendored** at:

```text
AutoResearch/simulators/isaaclab
```

Clone AutoResearch to obtain these sources (README, `isaaclab.sh`, `source/`,
lift/DAgger scripts). Do not expect `runs/` or `datasets/` from Git.

Launch from AutoResearch:

```bash
source simulators/resolve_paths.sh
cd "$ISAACLAB_ROOT"
./isaaclab.sh -p …
```

Install Isaac Sim into `env_isaaclab` via `scripts/remote_setup_env_isaaclab.sh`.
