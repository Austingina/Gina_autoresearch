# CALVIN Experiment Protocol

## Goal

Establish a strong no-G dual-camera baseline before adding G-like representations or flow-matching action decoders.

## Standard Observation

- `rgb_static`
- `rgb_gripper`
- robot proprioception from CALVIN datamodule
- no explicit G input for the first baseline

## Standard Presets

```bash
# readiness check
framework/scripts/check_calvin_ready.sh

# full train+val smoke, includes EGL rollout initialization
GPU=2 framework/scripts/run_calvin_dualcam.sh smoke

# small train-only debug run
GPU=2 MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=20 framework/scripts/run_calvin_dualcam.sh debug_train

# debug train with validation/rollout init
GPU=2 framework/scripts/run_calvin_dualcam.sh debug_eval
```

## Run Record Contract

Every canonical run must include:

- `command.sh`: exact executable command
- `env.txt`: machine, disk, CUDA, torch, dataset, GPU
- `git_status.txt`: relevant commits and dirty files
- `configs/common_overrides.txt`
- `configs/preset_overrides.txt`
- `logs/stdout.log`
- `notes.md`
- `state.txt` and `status.txt`

## Next Milestones

1. Debug dataset no-G train-only sanity.
2. Download official task D dataset.
3. Task D no-G baseline with CALVIN default action decoder.
4. Add flow-matching action decoder under the same run contract.
5. Reintroduce G-like representation only after the no-G baseline is reliable.
