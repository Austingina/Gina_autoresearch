# Operation 20261020-ROBOTWIN2-INSTALL

- **Operation ID:** `20261020-robotwin2-install`
- **Stage:** isolated RoboTwin 2.0 simulator installation
- **Target code:** `/home/ai-sz-a26394-u1/RoboTwin`
- **Target runtime:** `/home/ai-sz-a26394-u1/RoboTwin_runtime/env`
- **Status:** `IN_PROGRESS`
- **Result:** `PENDING`
- **Operator:** Hermes Agent
- **Created:** `2026-10-20T16:01:40+08:00`

## Authoritative requirement

Install RoboTwin so CALVIN, LIBERO, and RoboTwin can later participate in a simulator-neutral coarse-G dataset expansion and distribution comparison.

## Authoritative sources

- Canonical repository: `https://github.com/RoboTwin-Platform/RoboTwin`, default branch `main`, observed HEAD `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`.
- Official install documentation: `https://robotwin-platform.github.io/doc/usage/robotwin-install.html`.
- Official requirements: Linux, Python 3.10, CUDA 12.1 recommended, Vulkan, recursive XPolicyLab submodule, `scripts/_install.sh`, then `scripts/_download_assets.sh`.

## Evidence-backed baseline

- Target code directory does not exist.
- Host is Ubuntu 22.04.5 x86_64 with approximately 1.3 TiB free.
- `libvulkan1` and `mesa-vulkan-drivers` are installed; `ffmpeg` exists.
- System `nvcc` is CUDA 11.5, so the installation must not compile against it accidentally when a package expects a different toolkit.
- An isolated micromamba executable is available at `/home/ai-sz-a26394-u1/miniforge3/micromamba`.
- NVIDIA userspace is currently `595.99.02` while the loaded kernel module is `595.91.07`; code/environment installation may proceed, but GPU rendering/simulator PASS is blocked until reboot and post-reboot verification.
- Existing CALVIN extraction PID `188225` and LIBERO training PID `211323` are unrelated and must not be terminated or have their outputs modified.

## Goal

1. Clone the exact canonical RoboTwin repository recursively into a new directory.
2. Create an isolated Python 3.10 runtime at the declared prefix.
3. Install official Python requirements, XPolicyLab, CuRobo, and documented environment-local compatibility edits without modifying shared environments.
4. Download and verify required RoboTwin assets when their size and source gates pass.
5. Run import, repository, asset, and bounded simulator smoke checks; defer GPU-rendering PASS until the driver is coherent.
6. Produce hashes, inventory, and a reusable environment invocation.

## Non-goals

- Do not modify CALVIN, LIBERO, AutoResearch, Gina raw datasets, or shared Python environments.
- Do not use `sudo`, reboot, unload GPU modules, stop active jobs, or modify system CUDA/driver packages.
- Do not download the full 100,000-trajectory RoboTwin training dataset in this installation operation; dataset selection and cross-simulator expansion require a separate manifest-driven operation.
- Do not claim simulator usability from package installation alone.

## Mutable paths

- `/home/ai-sz-a26394-u1/RoboTwin/`
- `/home/ai-sz-a26394-u1/RoboTwin_runtime/`
- `/home/ai-sz-a26394-u1/Gina_autoresearch/artifacts/evidence/20261020T160140-robotwin2-install/`
- This record and its mirror under `/home/ai-sz-a26394-u1/operation-records/`.

## Safety and concurrency gates

1. Refuse if either target directory already exists.
2. Create a unique evidence directory before cloning.
3. Record repository HEAD/submodule SHAs, environment package inventory, disk, processes, CUDA/Vulkan state, and hashes.
4. Keep all package writes inside the declared runtime prefix or repository.
5. Review official scripts before execution; do not run hidden/system package-manager actions.
6. Abort asset extraction on download failure, insufficient disk, archive corruption, or unexpected paths.
7. Do not run GPU simulator smoke while NVML/kernel mismatch remains; preserve a `NOT_RUN_DRIVER_MISMATCH` receipt.

## Numbered execution plan

1. Create evidence and capture timestamp, disk, processes, GPU/Vulkan, target nonexistence, repository/API identity, and official command sources.
2. Clone `main` recursively and record exact main/submodule SHAs and Git status.
3. Create Python 3.10 environment at the isolated prefix and record interpreter/pip identity.
4. Install pinned official requirements in stages, preserving logs and stopping on the first real failure.
5. Install XPolicyLab and CuRobo in editable mode; apply only documented environment-local compatibility edits and record operation-local diffs/hashes.
6. Run import gates for NumPy, PyTorch, SAPIEN, mplib, XPolicyLab, CuRobo, HDF5, and the RoboTwin task modules.
7. Audit asset source names/sizes, then download/extract required assets with resumable logs and verify expected directories/files.
8. Run non-rendering configuration/task discovery smoke.
9. After driver coherence is available, run Vulkan and one bounded headless RoboTwin simulator smoke; until then mark that gate blocked rather than fabricating success.
10. Update this record with actual outputs, hashes, deviations, final status, and next required dataset operation.

## Failure/abort conditions

- Existing target, unexplained concurrent writer, repository/submodule failure, environment prefix escape, dependency conflict that breaks core imports, failed asset checksum/archive, insufficient disk, or any modification to protected projects.
- GPU simulator smoke is blocked, not failed, when the known driver mismatch remains.

## Rollback/recovery

- New isolated code/runtime paths may be preserved for diagnosis; deletion requires a later cleanup decision.
- No shared environment or system package mutation is planned, so no system rollback should be necessary.
- Partial downloads remain resumable and must not be reported complete.

## Completion criteria

- **PASS:** exact code/submodules, isolated environment, required assets, imports, task discovery, and bounded simulator smoke all verify.
- **PARTIAL:** code/environment/assets verify but rendering smoke is blocked by the pre-existing driver mismatch.
- **BLOCKED:** prerequisite prevents installation before usable imports exist.
- **FAILED:** scope escape, protected-data mutation, corrupt artifacts, or unexplained ownership conflict.

## Actual results

`PENDING`

## Next allowed operation

Create a separate manifest-driven RoboTwin task/data selection operation, align task/observation/action/G semantics with CALVIN and LIBERO, and generate the requested original-versus-expanded distribution tables.
