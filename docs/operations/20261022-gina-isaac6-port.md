# OP-20261022-GINA-ISAAC-SIM6-PORT-01

- **Created:** 2026-10-22T07:39:44+08:00
- **Status:** `PARTIAL — PHYSICS PASS; VISUAL COLLECTION HOLD`
- **Project:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Existing runtime:** `/home/ai-sz-a26394-u1/isaacsim/6.0.1`
- **Protected baseline:** `env_gina_isaac51` and its evidence remain unchanged.

## Goal

Port Gina's vendored IsaacLab task/collector integration to Isaac Sim 6.0.1 in a separate compatibility path and restore the lift RGB+depth+mask persisted-pilot gate before attempting drawer integration.

## Method

1. Freeze Isaac 6 Python/Kit APIs, package paths and driver/render startup behavior.
2. Add source-contract tests first for every compatibility shim or launcher change; preserve RED and GREEN evidence.
3. Keep 5.1 behavior intact; no broad in-place replacement.
4. Validate in layers: import → lean 1-env physics lift → rendering/Replicator → one RGB+depth+mask HDF5 → independent field/shape/dtype/nonblank/length/SHA-256 readback.
5. Only after lift passes may drawer collector work proceed.

## Mutable scope

- Gina compatibility adapter/tests/docs/evidence
- a dedicated Isaac 6 environment or launcher metadata under Gina/runtime-specific paths

Do not modify the standalone Isaac Sim 6.0.1 installation unless a separately recorded runtime-local configuration change is necessary and reversible.

## Abort conditions

Stop on mixed 5.1/6.0 module resolution, incompatible vendored IsaacLab/task APIs, RTX crash, missing completion marker, absent HDF5, or failed readback. Do not use path overrides to conceal incompatible framework versions.

## Evidence

`artifacts/evidence/20261022T073944-gina-isaac6-port/`

## Actual results

### CPU compatibility stage — `PASS` (2026-10-22)

- No Kit, GPU physics, RTX, Replicator, or renderer was started in this stage.
- Frozen identities: standalone Isaac Sim `6.0.1-rc.7+release.42383.32955d8d.gl` (Python 3.12.13); Gina vendored IsaacLab `2.3.2` / core extension `0.54.4`; dependency Python 3.12.14.
- Static standalone-6 sources contain all three previously reported missing APIs: `omni.kit.commands.register_callback`, `omni.kit.commands.unregister_module_commands`, and `omni.usd.get_context`. The protected 5.1 sources contain them too. The earlier AttributeErrors therefore indicate mixed/partially initialized namespaces, not removal of those APIs in 6.0.1.
- Confirmed mixing hazard: the Python-3.12 dependency env's default `isaaclab` is external version `6.1.11` from `/home/ai-sz-a26394-u1/IsaacLab`, while Gina must resolve its vendored `0.54.4`. The standalone Python by itself also lacks Torch/Gymnasium.
- Added a fail-closed isolated launcher, `scripts/run_gina_isaac6.py`, which scrubs inherited Python/conda/venv paths, disables user site, uses the standalone `python.sh`, sources only Gina's vendored IsaacLab packages first, and uses the matching Python-3.12 dependency interpreter for third-party packages.
- Added version-scoped Isaac 6 experience wrappers under `simulators/isaaclab/apps/isaacsim_6_0/`; the protected 5.1 app files and runtime were not modified.
- TDD receipts preserve focused RED exit 1 and GREEN. Final affected regression: 9/9 passed; compile exit 0.
- CPU preflight and isolated import passed without Kit. Resolved modules: Gina vendored `isaaclab`, standalone-6 `isaacsim`, dependency-env `torch`.
- Important remaining compatibility fact: vendored IsaacLab v2.3.2 formally declares only Isaac Sim 4.5/5.0/5.1. This CPU stage is port implementation evidence, not a claim that Isaac 6 physics/rendering works.

Detailed findings, exact unexecuted lean/render commands, and evidence paths: `artifacts/evidence/20261022T073944-gina-isaac6-port/cpu-compat-audit.md`.

### Live compatibility stage — `PARTIAL` (2026-10-22)

- The isolated Isaac 6 launcher passed the real 1-env, 650-step, no-camera lift gate: `success=True`, maximum cube height gain `0.1757 m`, maximum XY drift `0.0162 m`.
- Runtime TDD then fixed the real Kit dependency solver, nested app-source path, PhysX tensor namespace split between 5.1/6.0, removed deformable schema handling, headless viewport `None`, Conda `libstdc++` ordering, and the 5.1/6.0 material-command keyword split. The protected 5.1 app/runtime files were not edited.
- The first persisted visual run created a structurally readable 650-frame HDF5, but the episode was a native failure (`success=false`) and independent readback found all six visual tensors entirely zero. Receipt: `lift-hdf5-pilot-v4-readback.json`; SHA-256 `cd7660e98676b17d019caf530c2ee52347578885486fdee440b85e7064afec3b`.
- A fail-closed startup camera preflight was added so future runs cannot silently persist all-zero RGB/depth/mask data.
- A follow-up Isaac-6 RTX Hydra attachment probe changed empty annotators into an RTX/Fabric initialization path but hung on the first explicit render. The timed-out residual Kit process was explicitly terminated and GPU process readback was empty afterward.
- Current gate: physics/control `PASS`; RTX/Replicator camera pixels `FAIL/HOLD`; successful visual HDF5 `NOT ACHIEVED`; formal Isaac online additions remain `0`.
- Drawer integration and any multi-task production remain blocked behind a non-zero lift-camera readback. Do not count the failure HDF5 as accepted training data.
