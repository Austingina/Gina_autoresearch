# OP-20261021-GINA-ISAAC51-RUNTIME-SETUP-01

- **Created:** 2026-10-21T22:08:00+08:00
- **Stage:** Gina Isaac lift production recovery
- **Target:** isolated conda environment `env_gina_isaac51`
- **Status:** `PARTIAL_PASS_CONTROL_READY_RENDER_BLOCKED`
- **Operator:** Hermes Agent

## Authoritative project root

`/home/ai-sz-a26394-u1/Gina_autoresearch`

No source, operation, or artifact from `/home/ai-sz-a26394-u1/AST` is adopted by this operation.

## Gina evidence and baseline

- `scripts/remote_setup_env_isaaclab.sh:45-61` pins Torch 2.7/cu128, Isaac Sim `5.1.0.0`, and editable Gina-vendored IsaacLab extensions.
- `shared/envs/env_isaaclab.pip.freeze.txt:87-117` records the former working stack at IsaacLab commit `50c72036...` and Isaac Sim `5.1.0.0`, but its editable paths are from old host `/home/wukui` and are provenance only.
- Current `env_isaaclab`: Python 3.12.14, IsaacLab 6.1.11, no `isaacsim` or `isaacsim-kernel` package.
- Gina `simulators/isaaclab/_isaac_sim`: absent.
- GPU: RTX 4090, 130 MiB used, 0% utilization.
- No LIBERO or simulator collection process owns the GPU.
- Disk available: more than 1 TB.

## Goal

Create an isolated Python 3.11 environment matching Gina's documented Isaac Sim 5.1 stack, install Gina-vendored IsaacLab extensions editable, and pass import/render startup gates. Lift collection is a separate subsequent operation.

## Non-goals

- Do not alter or remove the existing `env_isaaclab` or standalone Isaac Sim 6.0.1.
- Do not modify lift/drawer expert code.
- Do not generate trajectories.
- Do not start bulk production.

## Allowed changes

- `/home/ai-sz-a26394-u1/miniforge3/envs/env_gina_isaac51/`
- pip/conda caches
- editable-install metadata for Gina's `simulators/isaaclab/source/*`
- `artifacts/evidence/20261021T220800-gina-isaac51-runtime-setup/`

## Installation sequence

1. Create `env_gina_isaac51` with Python 3.11 and pip.
2. Install `isaacsim[all,extscache]==5.1.0.0` from NVIDIA PyPI with `--resume-retries 100 --timeout 120 --retries 20`.
3. Install Torch 2.7.0 and torchvision 0.22.0 from cu128 index.
4. Install Gina-vendored `isaaclab`, `isaaclab_assets`, `isaaclab_contrib`, `isaaclab_tasks`, and `isaaclab_rl` editable.
5. Read back package versions and resolved source paths.
6. Run import and headless rendering-startup probes.

## Abort conditions

- Resolver selects a simulator version other than 5.1.0.0.
- Existing `env_isaaclab` or `/home/ai-sz-a26394-u1/isaacsim/6.0.1` changes.
- Disk falls below 150 GB.
- Import paths resolve outside Gina for IsaacLab packages.
- Fatal Kit/SyntheticData/Replicator startup error.

## Rollback

Remove only `env_gina_isaac51` after preserving receipts. Existing runtimes remain untouched.

## PASS criteria

- Python 3.11;
- Isaac Sim package set exactly 5.1.0.0;
- IsaacLab modules resolve from Gina's vendored source;
- import and render-stack startup gates pass.

## Evidence

`artifacts/evidence/20261021T220800-gina-isaac51-runtime-setup/`

## Actual results

### Installation and source recovery

- `isaacsim[all,extscache]==5.1.0.0` installed in `env_gina_isaac51`.
- Gina's editable IsaacLab extensions were installed.
- The first lean launch exposed a missing `isaaclab.utils.datasets` source package. Root cause was the repository-level `datasets/` ignore rule. The current `recorder_manager.py` exactly matched public IsaacLab tag `v2.3.2`, so only that tag's four `utils/datasets` files were restored and precisely unignored. See `20261021-gina-isaaclab-datasets-restoration.md`.
- `pip check` is not globally clean because Isaac Sim 5.1 metadata pins older `click`, `starlette`, and `typing-extensions` than the Gina IsaacLab v2.3.2 extension requirements. Gina's prior freeze uses the newer IsaacLab-side versions. Runtime gates, rather than the inconsistent metadata set, are authoritative here.

### Verified lean physics/control gate — PASS

One environment, seed 7, 650 bounded headless steps:

```text
[RESULT] success=True max_height_gain=0.1756 max_xy_drift=0.0160
ISAACLAB_SH_RC=0
```

Evidence: `artifacts/evidence/20261021T220800-gina-isaac51-runtime-setup/lift-control-smoke-unbuffered.log`.

### Rendering/Replicator gate — BLOCKED

The one-environment/one-demo RGB+depth+mask pilot crashed before scene creation with exit code 139 in `librtx.scenedb.plugin.so`. No target HDF5 exists and zero trajectories were created. A minimal full Kit probe and the actual rendering collector both reproduced the same RTX SceneDB failure.

The host driver is `595.99.02`. Isaac Sim 5.1's archived requirements specify Linux driver `580.65.06`, and public IsaacLab issue #6785 reports the same Isaac Sim 5.1 / 595-driver `librtx.scenedb.plugin.so` crash with a recommendation to use a production driver.

No system driver was changed. RGB+depth+mask HDF5 validation and production remain blocked pending an explicitly approved driver/runtime change.

Machine-readable receipt: `artifacts/evidence/20261021T220800-gina-isaac51-runtime-setup/runtime_validation_receipt.json`.
