# Operation 20261021 Cross-Simulator Native Collection Contract

- **Operation ID:** `20261021-cross-simulator-native-collection-contract`
- **Created:** `2026-10-21T16:16:59+08:00`
- **Status:** `PASS — OFFLINE VERIFIED`
- **Operator:** Hermes Agent
- **Repository:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Baseline commit:** `3d4aeba1badaa3d6ce781033e0d9187729dfec54`

## Objective

Turn the proven Isaac Lab pattern into a simulator-neutral collection contract:

```text
Native expert → native simulator execution → native success checker → native artifact recorder
```

The common layer governs provenance, Oracle evidence, artifact hashing, acceptance, and metadata-only failures. It does **not** impose a shared action array, shared state vector, or candidate G representation.

## Upstream API evidence

- LIBERO repository commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
  - `OffScreenRenderEnv`/`ControlEnv` provides `reset`, `step`, `set_init_state`, `get_sim_state`, and public `check_success()` delegating to the task's `_check_success()`.
- RoboTwin repository commit: `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`.
  - `scripts/collect_data.py` runs `setup_demo()`, `play_once()`, accepts only when `plan_success and check_success()` plus `planned_joints_legal()`, then calls `save_traj_data()`.
- CALVIN currently uses same-rollout initial/final info with `get_task_info_for_set(..., {task})`.

## Scope

1. Add a simulator-neutral native episode contract with no G field.
2. Add fail-closed native Oracle adapters for CALVIN, LIBERO, and RoboTwin.
3. Add artifact hash/readback validation and a quota runner that stores failed attempts as metadata only.
4. Add contract tests using fakes so the common layer is testable without heavyweight simulator runtimes.
5. Register LIBERO and RoboTwin as external runtimes; do not pretend they are installed.
6. Document exact runtime integration and expansion order.

## Non-goals

- Do not fabricate LIBERO or RoboTwin trajectories.
- Do not install heavyweight runtimes in the current CALVIN environment.
- Do not normalize native actions or states into a lossy universal vector.
- Do not add G to acceptance, sampling, or the common schema.
- Do not claim simulator verification until each native runtime has produced and read back an accepted artifact.

## TDD plan

1. RED: accepted episodes require explicit native Oracle evidence and existing hashed artifacts.
2. RED: failed Oracle evidence cannot enter accepted output.
3. RED: adapter checkers fail closed when native APIs are absent or malformed.
4. RED: common manifests contain native/canonical task metadata but no G.
5. Implement contracts, gates, runner, and registry entries.
6. GREEN: run unit contracts and existing CALVIN collector regressions.

## Safety

- New code only; no existing dataset mutation in this operation.
- Existing uncommitted user/project changes are preserved.
- Runtime-specific imports must be lazy so missing LIBERO/RoboTwin installations do not break CALVIN.
- Evidence directory: `artifacts/evidence/20261021T161659-cross-simulator-contract/`.

## PASS criteria

- Common contract has no G dependency.
- CALVIN/LIBERO/RoboTwin native gates are separately tested and fail closed.
- Failed attempts are receipt-only by default.
- Accepted artifact hashes are independently read back.
- Registry and implementation plan distinguish `OFFLINE VERIFIED` from `SIMULATOR VERIFIED`.

## Actual results

- **Completed:** `2026-10-21T16:25:33+08:00`
- **Status:** `PASS — OFFLINE VERIFIED`
- Added `framework/sim_data/` with:
  - a G-free native episode/request/Oracle contract;
  - fail-closed CALVIN, LIBERO, and RoboTwin native success gates;
  - a quota runner that hashes accepted artifacts and stores failures as metadata-only receipts;
  - independent manifest readback that fails on missing or changed accepted artifacts.
- Added a TDD contract covering native gates, one failed receipt followed by one accepted artifact, SHA-256 readback, absence of G fields, and missing-accepted fail-closed behavior.
- Added LIBERO and RoboTwin external-runtime entries to `simulators/registry.yaml` with pinned upstream commits and native success checkers.
- Added `docs/CROSS_SIM_NATIVE_COLLECTION.md` with native fields, staged rollout order, and platform-specific limitations.
- Verified upstream APIs against:
  - LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`;
  - RoboTwin `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`.
- Regression output:
  - `cross-simulator native collection contract: OK`;
  - scripted, parametric, and geometry CALVIN collector contracts: OK;
  - simulator registry YAML: OK.
- Current machine check: `libero` and `sapien` are not importable, and `LIBERO_ROOT`/`ROBOTWIN_ROOT` are unset. Therefore no LIBERO/RoboTwin simulator rollout is claimed.
- Key SHA-256 values:
  - contracts: `c437f41c4d5f5907bdd59b3e416ac26e800e9f1c642ee36be9fc4e32536ee9e1`;
  - gates: `ab7f7a0aeb4005838fef5de1a1238d0cbbb1f79444542a10617df848d4767625`;
  - runner: `887c0da6f6ce68f6671156d7f5faa98da6cbbd6b30386b907b46c279e36d472e`;
  - contract test: `261663867736459b31ab02a628775a86201087f813c08a0efd30137d0b3d5cf3`.

## Verification boundary

The architecture and API bindings are **OFFLINE VERIFIED**. Existing CALVIN collections remain simulator-verified from their original runs. LIBERO and RoboTwin remain `NOT IMPLEMENTED` at the runtime/expert execution layer on this host until their isolated environments and assets are installed and a native pilot artifact passes Oracle and hash readback.
