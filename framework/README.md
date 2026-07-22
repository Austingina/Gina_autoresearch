# AutoResearch Framework

This framework is the executable layer for the unified project.

It follows the same engineering pattern as `external/MacroRobotics/geo_vla`:

```text
configs/      # experiment presets
scripts/      # runnable launchers and checks
autoresearch/ # lightweight Python helpers and package namespace
docs/         # framework usage notes
tests/        # smoke checks
```

## CALVIN Quickstart

Check the server and CALVIN installation:

```bash
/mnt/nvme33n1/wuk_code/AutoResearch/framework/scripts/check_calvin_ready.sh
```

Run the verified dual-camera smoke experiment:

```bash
GPU=2 /mnt/nvme33n1/wuk_code/AutoResearch/framework/scripts/run_calvin_dualcam.sh smoke
```

Run a small train-only debug job:

```bash
GPU=2 MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=20 \
  /mnt/nvme33n1/wuk_code/AutoResearch/framework/scripts/run_calvin_dualcam.sh debug_train
```

Each run creates a canonical record under:

```text
experiments/phase1_calvin/<timestamp>_<preset>/
```

with `command.sh`, `env.txt`, `git_status.txt`, `logs/stdout.log`, `notes.md`, and framework metadata.
