# OP-20261022-GINA-ROBOTWIN-RUNTIME-ASSETS-01

- **Created:** 2026-10-22T07:39:44+08:00
- **Status:** `IN_PROGRESS`
- **Project:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **External runtime source:** `/home/ai-sz-a26394-u1/RoboTwin@ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`
- **Isolated environment:** `/home/ai-sz-a26394-u1/RoboTwin_runtime/env`

## Goal

Install the pinned RoboTwin runtime dependencies and official assets required for `place_empty_cup`, pass the existing Gina fail-closed preflight, then run one bounded native-Oracle pilot only after all prerequisites pass.

## Mutable scope

- `/home/ai-sz-a26394-u1/RoboTwin_runtime/`
- `/home/ai-sz-a26394-u1/RoboTwin/assets/` official downloaded payloads only
- Gina evidence/operation files and a future Gina-owned adapter/manifest

The external RoboTwin tracked source must remain clean. Environment-level upstream patches, if required by the official installer, must be recorded by pre/post hashes.

## Gates

1. `PYTHONNOUSERSITE=1`; exact Python and package readback.
2. Required imports including SAPIEN, Torch, h5py, OpenCV, Open3D and MPLib.
3. Task-minimal official assets must carry source-revision, archive LFS identity, ZIP CRC and extracted-file hash receipts; `background_texture` is required only when `random_background=true`. Full-archive installation, if deferred, must not be claimed complete.
4. Gina preflight passes exactly for `episode_num=1`, finite `max_attempts`.
5. GPU ownership rechecked before pilot.
6. Acceptance requires `plan_success && check_success() && planned_joints_legal()[0]`, readable HDF5, aligned state/action/vision lengths, successful `decode_image_bit`, size and SHA-256.

## Abort conditions

Stop on dirty external tracked source, wrong commit, missing/corrupt assets, dependency ABI conflict, unbounded seed search, GPU contention, or any failed native gate. File existence alone is not success.

## Evidence

`artifacts/evidence/20261022T073944-gina-robotwin-runtime-assets/`

## Actual results

### 2026-10-22T09:09:35+08:00 baseline

- Authoritative project root confirmed: `/home/ai-sz-a26394-u1/Gina_autoresearch`, HEAD `3d4aeba1badaa3d6ce781033e0d9187729dfec54`.
- External RoboTwin root confirmed clean at pinned HEAD `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`; XPolicyLab gitlink `fa431ecd893ee706883e64fe5fe1464ec8cd928d`.
- Host GPU: NVIDIA GeForce RTX 4090, driver `595.99.02`, 23,945 MiB free; no compute process was present at the check time.
- Project filesystem free space: approximately 1.2 TiB.
- Isolated runtime Python exists (`3.10.22`) but metadata-only probing with `PYTHONNOUSERSITE=1` found all 16 required import modules absent.
- Required official assets `021_cup`, `019_coaster`, `aloha-agilex`, and `background_texture` are absent.
- Result: collector launch is `NOT_RUN`; prerequisite installation is now the active step under this operation. No trajectory or simulator-success claim exists yet.

### 2026-10-22T09:54:45+08:00 implementation/install checkpoint

- The upstream collector's success loop is unbounded and ignores `max_attempts`; it was therefore not launched directly. A Gina-owned bounded two-pass adapter was added with fresh simulator instances per seed, a hard 25-attempt limit, planning admission on all three native gates, and replay Oracle admission before HDF5 writing.
- RED→GREEN evidence is retained in the current attempt directory. Combined RoboTwin preflight/adapter suite passes `16/16`; no simulator was needed for these contract tests.
- Physical-quality thresholds were frozen before rollout: native Oracle true, `xy < 0.035 m`, `|dz| < 0.015 m`, cup tilt `<= 10 deg`, linear speed `<= 0.05 m/s`, angular speed `<= 0.5 rad/s`, and cup/coaster contact true. A failed quality gate may remain diagnostic but is not scale-up/training eligible.
- Official `embodiments.zip` was reused from a pre-existing local stage only after exact LFS verification: `219847741` bytes, SHA-256 `85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789`. Extracted `aloha-agilex` contains 82 non-empty files and `config.yml`.
- For the fixed `random_background=false` pilot, background textures are explicitly non-required; enabling random backgrounds remains fail-closed without them. This conditional contract is regression-tested.
- To avoid waiting for the full 3.74 GB object bundle, the official revision-pinned `objects.zip` was read by HTTP Range and only `objects/021_cup` plus `objects/019_coaster` were extracted. ZIP CRC validation plus per-file SHA-256 covers 44 regular files / 72,187,719 bytes. Manifest: `task-minimal-objects-manifest.json` under the current evidence attempt. This is task-minimal pilot readiness, **not** a claim that the complete objects archive is installed.
- Two ordinary pip attempts failed while streaming the 798.9 MB PyTorch wheel. The active recovery uses a persistent resumable file under `/home/ai-sz-a26394-u1/RoboTwin_runtime/wheels/`; partial bytes are not accepted as an installed package.
- Current state remains `NOT_RUN`: assets required by this task are present, runtime installation is in progress, and no RoboTwin episode has yet been executed or counted.

### 2026-10-22T10:02+08:00 parent-thread verification

- Task-minimal official assets remain present with receipts: 44 cup/coaster files totaling `72,187,719` bytes and 82 non-empty `aloha-agilex` files. This is not a full-object-bundle claim; background textures remain intentionally unnecessary only for `random_background=false`.
- Live import verification failed immediately at `numpy`; therefore the 16/16 result above is an offline contract-suite result, not a runtime preflight pass.
- The earlier single-stream curl recovery (`proc_70ea3c6710f5`) was terminated with exit `143` before checksum or installation and is retained as a superseded failed attempt. It did not promote or install the partial wheel.
- A single bounded parallel HTTP Range downloader now owns the same output, using eight 4 MiB chunks per batch and two mirrors. At `2026-10-22T10:39:11+08:00` it had completed `402,653,184 / 798,942,091` bytes (`50.40%`); the last two committed batches increased the file from `369,098,752` to `402,653,184` bytes. It remains an active partial download, not an installed package.
- Completion must still be followed by exact-size/package validation, SHA-256, pip installation, remaining requirements, live imports, and the correct preflight invocation.
- GPU pilot and trajectory/HDF5 acceptance remain `NOT_RUN`; formal RoboTwin additions remain `0`.
