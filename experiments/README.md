# Experiment records

New runs go under:

```text
experiments/<phase>/<YYYY-MM-DD_HHMMSS>_<short_name>/
```

Scaffold:

```bash
bash shared/scripts/init_experiment.sh phase1_calvin short_name
```

Each run should keep at least: `command.sh`, `config.yaml`/`env.txt`, `notes.md`, `metrics/`, `logs/`.

**Git policy:** track metrics/notes/configs (`*.json`, `*.md`, `*.yaml`, `*.sh`).  
**Ignore:** `*.pt`, `*.pth`, `*.ckpt`, large videos, Hydra lightning dumps.

`experiments/phase1_calvin/` currently holds recovered lightweight evidence from the H200 NVMe workspace (2026-07 gates: E3/E4/ABC→D, etc.). Weights were not copied — retrain from `framework/scripts/` if needed.
