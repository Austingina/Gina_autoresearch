# Operation 20261020-CALVIN-PARAMETRIC-COMPOSITIONAL-COLLECTION

- **Operation ID:** `20261020-calvin-parametric-compositional-collection`
- **Status:** `PASS — SIMULATOR VERIFIED FOR 6 CELLS × 2 RESETS`
- **Created:** `2026-10-20T18:43:55+08:00`
- **Repository:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Baseline commit:** `3d4aeba Record interaction volume parser closure`
- **Scope:** implement a parameterized closed-loop CALVIN block-push expert, oracle-gated batch collector, and combination-isolated multi-fold manifests for `object × direction` generalization.

## Requested deliverables

1. Parameterized expert for every cell in:
   - objects: `red`, `blue`, `pink`;
   - directions: `left`, `right`.
2. Closed-loop state machine driven by named live scene information, not fixed action replay.
3. Batch collection with varied deterministic reset states, bounded retries, exact task-oracle acceptance, fresh state/dual-camera/action/G/phase output, and failed-attempt isolation.
4. Multi-fold manifests that hold out unseen object-direction combinations while keeping each held-out object's and direction's atoms visible in source training cells.
5. Leakage/integrity checks and actual simulator read-back evidence.

## Pre-operation live evidence

- GPU: NVIDIA GeForce RTX 4090, driver `595.99.02`, 130 MiB used of 24,564 MiB.
- No active CALVIN collector/evaluator was found.
- Existing worktree contains unrelated modified and untracked research artifacts; none may be reset or overwritten.
- Existing fixed-script collector is present at `framework/scripts/data/collect_calvin_scripted_expert.py` and has already produced three fixed-reset simulator-verified trajectories.
- CALVIN runtime:
  - code: `/home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/code/calvin`;
  - Python: `/home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/envs/calvin_reuse/bin/python`;
  - EGL renderer previously validated on the RTX 4090.

## Safety, isolation, and scientific gates

- Strict TDD: observe RED before each production behavior, then GREEN and regression.
- Use `env.get_info()["scene_info"]["movable_objects"][name]` for object identity; do not depend on cross-scene `scene_obs` object ordering in the controller.
- Accept an expert rollout only when CALVIN `get_task_info_for_set(start_info, current_info, {task})` confirms the exact requested task.
- Failed attempts remain under `attempts/`; they must never enter training/source manifests.
- Use a dedicated generated directory and refuse overwrites.
- Record deterministic seed/reset provenance and simulator/task-config hashes.
- Every held-out pair must be absent from source cells while both its object and its direction occur in source cells.
- Rotate bridge objects across folds; report exact cell membership rather than calling random episodes compositional generalization.
- Do not claim robust multi-seed collection until all requested cells and reset variants are actually simulator-confirmed and read back.

## Planned artifacts

- `framework/scripts/data/collect_calvin_parametric_push_expert.py`
- `framework/scripts/data/test_collect_calvin_parametric_push_expert.py`
- generated batch under `data/calvin/`
- collection manifest, exact per-attempt receipts, and fold manifests
- RED/GREEN, simulator, read-back, hash, and leakage-audit evidence under `artifacts/evidence/`

## Actual results

- **Completed:** `2026-10-20T18:56:05+08:00`
- **Implementation state:** `SIMULATOR VERIFIED` for all six `color × direction` cells, each on two distinct reset states.
- **Collector:** `framework/scripts/data/collect_calvin_parametric_push_expert.py`
- **Unit contract:** `framework/scripts/data/test_collect_calvin_parametric_push_expert.py`
- **Primary generated batch:** `data/calvin/generated_parametric_push_20261020_s20261020_q2_v1/`
- **Collection manifest:** `data/calvin/generated_parametric_push_20261020_s20261020_q2_v1/collection_manifest.json`
- **Evidence:** `artifacts/evidence/20261020T184355-calvin-parametric-push/`

### Controller implemented

The expert is a parameterized named-object closed-loop state machine:

1. parse an exact task such as `push_blue_block_right` into object `block_blue` and flow `[+1,0,0]`;
2. read the live object position from `env.get_info()["scene_info"]["movable_objects"][object_name]`;
3. calculate the contact side from direction, then execute `hover → contact → push` absolute-pose waypoints;
4. require stable Cartesian/orientation convergence before advancing states;
5. evaluate the exact requested task through CALVIN `get_task_info_for_set(start_info, final_info, {task})` after every action;
6. retain only exact oracle successes in `accepted/` while preserving bounded failures in `attempts/`.

Each trajectory stores fresh `rgb_static`, fresh `rgb_gripper`, `robot_obs`, `scene_obs`, executed 7-D absolute pose actions, corresponding CALVIN-scaled 7-D relative-action labels, named object positions, controller state, state-grounded phase, `post_contact`, 6-D `G=[A,F]`, reset arrays, reset segment/scene, and exact task/factor identity.

### Simulator collection receipt

| Combination | Accepted | Failed before quota | Distinct accepted resets | Oracle final progress |
|---|---:|---:|---:|---|
| red × left | 2 | 4 | 2 | 0.1211, 0.1040 |
| red × right | 2 | 0 | 2 | 0.1092, 0.1038 |
| blue × left | 2 | 6 | 2 | 0.1265, 0.1509 |
| blue × right | 2 | 2 | 2 | 0.1284, 0.1069 |
| pink × left | 2 | 7 | 2 | 0.1094, 0.1046 |
| pink × right | 2 | 2 | 2 | 0.1065, 0.1059 |
| **Total** | **12** | **21** | **12** | exact CALVIN oracle success for every accepted rollout |

The nonzero failure count is retained intentionally: the controller is not yet robust on every available reset. Failed traces are auditable but excluded from every fold and training/source partition.

### Balanced connected-checkerboard folds

Six exhaustive factor folds were emitted under `folds/`:

- each of red, blue, and pink is the bridge object in two complementary polarities;
- each fold contains four source cells and two held-out combination cells;
- every held-out object and every held-out direction appears independently in source cells;
- source/held-out pair overlap is empty;
- every one of the six object-direction cells is held out exactly twice across the six folds;
- with quota two, each fold contains 8 source trajectories and 4 held-out trajectories;
- no failed attempt is referenced by a fold manifest.

This is a true unseen-combination split, not a random episode split. It does not claim unseen-object, unseen-direction, unseen-scene, or cross-simulator generalization.

### Read-back and regression evidence

- `accepted_verified=12`, two per cell, `complete=true`.
- Six fold manifests hash-verified and leakage-checked.
- Accepted trajectories have aligned modalities, 7-D actions and relative actions, 6-D G, both phase values, and dual-camera shapes `(T,200,200,3)` and `(T,84,84,3)`.
- Generated first frames differ from source demonstration frames by 53.4–72.2% of static-camera channel values and 52.2–89.8% of gripper-camera channel values; source RGB was not copied.
- New parameterized collector tests, Python compilation, previous fixed-script collector tests, simulator batch run, and read-back validator all exited 0.
- Primary batch size: 183 MiB.
- No CALVIN collector or PyBullet process was left running after verification.

### Final hashes

- collector: `22217b60b716c68289d6a88f86b1764e09c1f4b67f1e50e15f60db24931e3e5a`
- test: `3bf36bf83b25858c7b1876a03d0c8c76fdacb9c1b72ebe1ac1c3d113e9639b55`
- primary manifest: `6f72e10c83201ea35c7faa390d269f58156b7422e217c4759e0d1b2fd3cc25cc`

### Remaining boundary

This proves the implemented expert and collector on 12 accepted simulator rollouts. It does **not** yet prove benchmark-scale data sufficiency or learned-policy compositional generalization. The next stage is to raise per-cell reset diversity, separate scene/seed pools, train one model per fold under a fixed budget, and evaluate held-out combinations with matched reset caches and multiple training seeds.
