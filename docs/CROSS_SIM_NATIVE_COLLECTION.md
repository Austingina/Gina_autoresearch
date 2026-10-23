# Cross-Simulator Native Data Expansion

## Authoritative project root

All implementation, environment checks, rollouts, operation records, and generated artifacts in this program are rooted at:

`/home/ai-sz-a26394-u1/Gina_autoresearch`

Other checkouts, including `/home/ai-sz-a26394-u1/AST`, may be used only as read-only historical references. Their runtime state, modifications, or artifacts do not establish progress for this program.

## Principle

Reuse the **method** proven in Isaac Lab, not Isaac Lab's action arrays or simulator classes:

```text
native expert
  → native simulator rollout
  → native success checker
  → full native artifact recording
  → hash/readback validation
```

The shared layer standardizes only episode identity, task semantics, provenance, native Oracle evidence, failure receipts, artifact hashes, and quotas. Native observations/actions remain native. Candidate `G` is outside this contract.

## Common contract

Implementation:

- `framework/sim_data/contracts.py`
- `framework/sim_data/gates.py`
- `framework/sim_data/runner.py`
- `framework/scripts/data/test_native_collection_contract.py`

An accepted episode must have:

1. a concrete reset/seed identity;
2. a native task name and optional canonical semantic description;
3. native Oracle evidence from the same rollout;
4. at least one artifact under `accepted/`;
5. byte size and SHA-256 read back from disk.

A failed attempt stores only a JSON receipt by default. Missing accepted artifacts fail closed.

## Platform bindings

| Platform | Native expert/executor | Authoritative success gate | Native recording | Current status |
|---|---|---|---|---|
| Isaac lift | `LiftCubeOraclePolicy` and pick/lift state machine | task success plus consecutive stable-success latch | HDF5 RGB/depth/mask, state, teacher/executed actions, phase, success/failure export | collector implemented in Gina; current-host persisted pilot must be revalidated before production |
| Isaac drawer | existing `OpenDrawerSm` | relative drawer progress plus predeclared velocity/consecutive-stability contract | drawer-specific recorder/collector still required | state-machine source exists; not yet a recorded or simulator-validated data task |
| CALVIN | scripted replay, parameterized teacher, geometry-contact teacher | `get_task_info_for_set(initial_info, final_info, {task})` | NPZ dual RGB + robot/scene/action/raw geometry | Q1/Q25 data simulator-verified; common gate offline-tested |
| LIBERO | task-specific expert to be implemented over `OffScreenRenderEnv` | public `ControlEnv.check_success()` → task `_check_success()` | native observation dict, MuJoCo state, action, BDDL/init state | runtime and expert not installed; gate offline-tested |
| RoboTwin 2.0 | native `setup_demo()` → `play_once()` | `plan_success && check_success() && planned_joints_legal()` | native `save_traj_data()` HDF5 plus instruction/seed | upstream expert path identified; runtime not installed; gate offline-tested |

Pinned upstream API baselines:

- LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- RoboTwin: `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`

## Native fields to preserve

### CALVIN

- dual RGB observations;
- `robot_obs`, `scene_obs`;
- absolute command and executed relative action;
- object position/orientation;
- contact-detected flag, live contact position, surface normal;
- task direction, reset state, controller version;
- initial/final info-derived Oracle receipt.

### LIBERO

- complete observation dictionary and camera names;
- raw 7-D action exactly as passed to `env.step`;
- MuJoCo flattened simulator state from `get_sim_state()`;
- BDDL file, task suite/id/language, exact init state;
- `done`, reward, info, and independent `check_success()` result;
- environment and asset commit/config hashes.

### RoboTwin

- native HDF5 from `save_traj_data()` without schema flattening;
- task config, embodiment, seed, instruction and domain-randomization config;
- head/wrist camera payloads, joint/end-pose actions and simulator state already emitted by RoboTwin;
- `plan_success`, task `check_success()`, joint-legality result/excess;
- task class/config and upstream commit hashes.

## Expansion order and gates

### Stage A — Isaac lift production recovery

1. Resolve and freeze the runtime from the Gina checkout itself; do not inherit claims from another working copy.
2. Run a bounded lift control smoke, then a 1–3 episode persisted HDF5 pilot.
3. Independently reopen the HDF5 and verify episode count, success labels, image/state/action alignment, teacher-versus-executed actions, file size, and SHA-256.
4. Start lift-only production only after this persisted pilot passes. Report it as single-intent production.

### Stage B — Isaac drawer as the second recorded task

1. Reuse the existing `OpenDrawerSm` only as the controller; do not call the state-machine demo a collector.
2. Add episode-start capture, drawer progress/velocity measurement, consecutive stable-success confirmation, recorder wiring, success/failure export, and artifact manifest.
3. Run a bounded online pilot and independently reopen every accepted artifact.
4. Do not claim Isaac multi-task production until both lift and drawer have independently passed current-host online collection and readback gates.

### Stage C — CALVIN continuation

1. Run geometry-contact push at six cells ×3 distinct native resets.
2. Audit Oracle rate, lateral drift, tilt, vertical displacement and table support.
3. Expand only stable cells; do not jump directly to Q25.
4. Add parameterized drawer and slider experts after push stability is established.

### Stage D — LIBERO middle-drawer pilot

1. Use the Gina-pinned runtime and verify one official init state.
2. Implement one task-specific online expert for `open_the_middle_drawer_of_the_cabinet`.
3. Collect three accepted varied init states using `check_success()` only.
4. Preserve BDDL/init-state provenance and full MuJoCo state.
5. Add experts task family by task family—never treat reward, motion progress, or a candidate G as success.

### Stage E — RoboTwin `place_empty_cup` pilot

RoboTwin already provides the closest equivalent to Isaac Lab's teacher collection. After installing the pinned runtime/assets:

1. Install the pinned SAPIEN runtime and required assets without altering the Isaac/CALVIN/LIBERO environments.
2. Run `place_empty_cup` for one bounded episode through native `play_once()`.
3. Require `plan_success`, task `check_success()`, legal planned joints, and native HDF5 persistence.
4. Independently reopen HDF5 and verify modalities and hashes before adding seeds or a second task.

## Bulk-production promotion rule

Isaac may be called a **multi-task bulk production line** only after at least `lift_cube` and `open_drawer` each have:

1. a registered environment;
2. an online state/geometry-driven expert;
3. an authoritative task-success gate with stable confirmation where needed;
4. a recorder with success/failure separation;
5. a current-host persisted pilot;
6. independent artifact readback and hashes.

Until then, lift collection is reported as lift-only production and drawer as adapter development/pilot. CALVIN, LIBERO, and RoboTwin trajectories remain separate platform datasets even when task semantics are aligned.

## Status vocabulary

- `NOT IMPLEMENTED`: no executable adapter/expert.
- `OFFLINE VERIFIED`: common contract/native gate tested with fakes or existing artifacts.
- `SIMULATOR VERIFIED`: native runtime executed and produced Oracle-accepted artifacts that were reopened and hash-verified.

As of 2026-10-21, the cross-simulator contract and native gate bindings are **OFFLINE VERIFIED**. Existing CALVIN Q1/Q25 datasets remain **SIMULATOR VERIFIED**. Isaac lift collection exists in source but still requires a fresh Gina-root current-host persisted pilot before production resumes; Isaac drawer is not yet a recorded data task. No new LIBERO or RoboTwin trajectory is claimed. The earlier LIBERO training process has exited and is no longer the current GPU blocker.
