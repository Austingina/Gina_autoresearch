# CALVIN Q3 Stability Gate and Multitask Expansion

- **Operation ID:** `20261021-calvin-q3-stability-gate-and-multitask-expansion`
- **Created:** `2026-10-21T17:28:39+08:00`
- **Status:** `Q3 AUDIT COMPLETE — INTEGRITY/REPEATABILITY PASS; PHYSICAL-QUALITY HOLD; MULTITASK BULK COLLECTION NOT STARTED`
- **Repository:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Scope:** simulator data collection only; no model training, G discovery, held-out folds, or LIBERO/RoboTwin rollout collection.

## User-approved order

1. Complete CALVIN Q3 stability audit.
2. Only after Q3 passes, expand CALVIN to additional task families.
3. LIBERO official-data download/environment preparation may continue, but no formal LIBERO trajectory expansion is allowed before the CALVIN gate passes.

## Frozen baseline

- Target design: `red/blue/pink × forward/backward × 3 distinct native resets = 18 accepted trajectories`.
- Existing first collection: `data/calvin/generated_geometry_contact_y_q3_20261021_s20261027_v1/`
  - 12 accepted, 74 failed metadata-only.
- Existing recovery collection: `data/calvin/generated_geometry_contact_y_q3_forward_recovery_20261021_s20261028_v1/`
  - 5 accepted, 34 failed metadata-only.
- Current total: 17/18; missing one `push_blue_block_forward` trajectory.
- Remediated collector SHA-256: `2e59f96e9151d166c57031f2935ac9e1e44e019b773a1f562c83720fd27cc929`.
- Acceptance gate: same-rollout CALVIN `get_task_info_for_set(initial, final, {requested_task})` only.
- Candidate G: disabled and not part of sampling, control, validation, or acceptance.

## Why a clean recollection is required

The first 12 accepted trajectories predate the final fresh-environment isolation fix. Their Oracle receipts, hashes, fields, temporal alignment, and support preservation passed readback, but they were not all generated under the final isolation contract. Therefore they remain auditable historical artifacts but will not alone satisfy the final stability gate.

The final Q3 candidate must be a clean 18-trajectory collection generated with:

- a fresh CALVIN/PyBullet environment for every candidate preview;
- a separate fresh environment for every persisted final rollout;
- two-pass native pose reset plus movable-body linear/angular velocity clearing;
- three distinct reset IDs per object-direction cell;
- no color-specific or held-out-specific controller branch.

## Predeclared Q3 pass gates

### A. Completeness and provenance

- Exactly six object-direction cells, each with at least 3 accepted trajectories.
- Exactly 3 selected trajectories per cell in the final audit set, from 3 distinct reset IDs.
- Collector, reset pool, task configuration, and accepted artifacts have recorded SHA-256 values.

### B. Native success and data integrity

- Every accepted trajectory is confirmed by the same-rollout native CALVIN Oracle for the exact requested task.
- Every accepted NPZ hash matches its receipt/manifest.
- Required raw fields are present and temporally aligned: dual RGB, robot state, scene state, native/relative action where applicable, object/contact geometry, task/reset provenance.
- Candidate `G`, `phase`, and `post_contact` are absent by default.

### C. Simulator isolation and repeatability

- Collector unit contracts and syntax checks pass.
- Fresh-environment differential probe remains repeatable for the previously unstable blue-forward and pink-forward cases.
- No reused-environment result is used as evidence of deterministic execution.

### D. Physical-quality audit

For every selected accepted trajectory:

- table/support contact is preserved;
- no downward displacement greater than `0.005 m`;
- Oracle progress reaches its native strict success condition;
- lateral drift, tilt, execution length, contact counts, and vertical motion are reported per trajectory and per cell.

Lateral drift and tilt are audit metrics, not substitutes for the native Oracle. They will be used to decide whether the push family is clean enough for scale-up; thresholds will not be relaxed to rescue a failed Oracle rollout.

## Remediation plan

1. Generate a new namespaced Scene-D native reset pool using seed `20261029`.
2. Run a clean six-cell Q3 collection with the remediated fresh-environment collector into a new, non-overwriting output directory.
3. Independently read back all selected accepted files and produce a combined audit JSON.
4. If any pass gate fails, stop before multitask expansion and remediate/recollect only through simulator-wide rules.
5. If all gates pass, freeze the Q3 evidence and begin a small CALVIN multitask pilot.

## Multitask pilot after Q3 pass

Initial pilot scope is deliberately small:

- articulation: `open_drawer`, `close_drawer`, `move_slider_left`, `move_slider_right`;
- container transport: `push_into_drawer`, `place_in_slider`;
- start with one Oracle-accepted trajectory per task under fresh-environment isolation;
- preserve images, native state/action, contact/task/reset provenance;
- failures metadata-only;
- no G emitted;
- expand quotas only after each task family passes its own reproducibility and physical-quality audit.

## Pre-operation system evidence

At `2026-10-21T17:28:39+08:00`:

- root filesystem: 1.8 TiB total, 1.2 TiB available;
- memory: 31 GiB total, about 25 GiB available;
- GPU: RTX 4090, 130 MiB used, 0% utilization;
- no CALVIN collector/PyBullet collection process active;
- LIBERO official dataset downloader active separately; it is not a simulator collection process.

## Actual results

### 1. Tests and reset-pool generation

- At `2026-10-21T17:28:39+08:00`, both the fixed-scene reset-pool unit contract and geometry-contact planner contract passed.
- A new reset-pool offset option was added test-first so deterministic native samples can be continued without reusing the first selected candidates.
- Seed `20261029` produced only red=1, blue=0, pink=1 candidates after 10,000 native reset samples under the strict unobstructed-corridor contract.
- Continuing seed `20261022` to 30,000 samples produced only red=5, blue=6, pink=3 strict candidates, so the planned new `24 per color` pool was correctly rejected rather than created with relaxed geometry.
- A smaller strict continuation pool was generated with two new non-duplicate resets per color, then merged with the existing 72-reset pool into `data/calvin/fixed_scene_D_y_push_resets_20261021_q3_audit_union_v1.json` (77 physical states after exact scene-state deduplication).

### 2. Clean fresh-environment recollection

- Output: `data/calvin/generated_geometry_contact_y_q3_fresh_audit_20261021_s20261030_v1/`
- Process exit: code `2`, deliberately indicating incomplete quota.
- Accepted: 17/18.
- Per cell: red-forward 3, red-backward 3, blue-forward 2, blue-backward 3, pink-forward 3, pink-backward 3.
- This clean run confirms the historical 17/18 result under the final fresh-environment contract; the blue-forward deficit is reproducible and not a reused-environment artifact.

### 3. Blue-forward completion without relaxing Oracle

- The 24 failed blue-forward attempts showed that many trajectories moved more than 0.10 m but lost support, tilted, or drifted laterally; CALVIN correctly rejected them.
- An existing geometry-only filter over a native Scene-D `n=200` reset pool provided 18 forward-clear blue candidates. It did not use trajectory outcome or G.
- Output: `data/calvin/generated_geometry_contact_blue_forward_q1_20261021_s20261031_v1/`
- Process exit: code `0`.
- Result: 1/1 Oracle-accepted trajectory on distinct reset `scene_D_native_y_forward_n200v1_blue_0010`.
- Final combined Q3 set: exactly 18 trajectories, 3 per object-direction cell, and 3 distinct reset IDs per cell.

### 4. Joint readback and stability result

Evidence root: `artifacts/evidence/20261021T175600-calvin-q3-stability-final/`.

- final audit: `final_q3_stability_audit.json`
- final audit SHA-256: `6ff71bf377c3e93a34995ffc59c6a4ebd58474c56c35bfa941006b6fc426bc9c`
- all 18 accepted files passed SHA-256 readback;
- all required native fields are present and temporally aligned;
- all 18 preserve support contact;
- maximum downward displacement: `0.001222 m`, below the predeclared `0.005 m` gate;
- candidate G/phase/post-contact fields are absent;
- targeted fresh-environment probes remain repeatable;
- total accepted frames: 1,486.

Therefore Q3 completeness, native Oracle, artifact integrity, isolation, and targeted repeatability gates **pass**.

### 5. Physical-quality hold

The final quality envelope also reports:

- maximum lateral drift: `0.135191 m`;
- maximum quaternion orientation change: `2.886926 rad` (this includes yaw and must not be interpreted as pure tipping angle).

No lateral-drift threshold had been predeclared, so it would be scientifically invalid to silently call these trajectories clean enough for scale-up. The final status is therefore:

- **Q3 integrity/repeatability:** PASS;
- **Q3 physical-quality scale-up gate:** HOLD;
- **CALVIN multitask bulk collection:** NOT STARTED.

### 6. Controller-quality differential and rollback

A test-first hypothesis locked the live push target to its original lateral axis:

- on blue-forward reset `scene_D_native_y_forward_n200v1_blue_0010`, lateral drift improved from `0.135191 m` to `0.021230 m`, with native Oracle success and support preserved;
- full six-cell validation then produced no accepted red-forward trajectory before termination, so the behavior did not generalize;
- a second test-first hypothesis using 50% lateral tracking retained blue-forward success but failed native Oracle on representative red-forward reset `scene_D_native_y_red_0004`;
- both controller variants and their unit-test contracts were reverted to the last validated implementation (collector SHA-256 restored to `2e59f96e9151d166c57031f2935ac9e1e44e019b773a1f562c83720fd27cc929`);
- the incomplete axis-lock output is diagnostic only and is excluded from all final Q3 evidence.

### 7. Multitask expansion boundary

Repository inspection confirms that the existing articulation collector replays official demonstration action arrays from exact resets. It is useful for simulator smoke tests but is **not** a parameterized online expert and will not be reported as new behavioral data. Online articulation and container-expert designs are being investigated separately; no formal multitask rollout is permitted until the physical-quality gate is prospectively defined and the push controller passes all six cells.

Updated at `2026-10-21T18:22:31+08:00`.

### 8. Multitask online-expert design investigation

Completed at `2026-10-21T18:25:37+08:00`. This was source inspection plus live read-only Scene-D geometry probing; **no articulation/container rollout was executed and no data was added**.

Verified articulation facts:

- `scene_obs[0]` is slider joint `base__slide`; `scene_obs[1]` is drawer joint `base__drawer`.
- Native relative-motion thresholds are strict: slider left/right `±0.15`, drawer open/close `±0.12`, always relative to one frozen episode-start `start_info`.
- Runtime slider range is `[0, 0.28]` along world `-x`; drawer range is `[0, 0.22]` along world `-y`.
- CALVIN exposes live link state but no named handle pose; the first pilot must transform an asset-local calibrated contact point through the current link pose every control cycle.
- A proposed online controller uses fresh-environment resets and live states `hover → contact → grip/preload → transport → native-Oracle stop`; it must not replay official action arrays or use G.

Verified container-task constraints:

- `place_in_slider` requires the scored episode to begin with exactly one object already held and unsupported, then end released on `plank_link`; an online grasp must therefore be an unscored setup phase followed by a fresh `start_info` capture, or be encoded in a valid already-held reset.
- `push_into_drawer` checks a support transition from table `base_link` to `drawer_link`, but has no displacement, low-velocity, drawer-open, or exclusive-contact condition.
- Its intended robot non-contact clause compares an integer UID with `(uid, link_id)` tuples and is ineffective; any new collector must independently require endpoint robot-object separation, in-container COM, persistent low velocity, support contact, and bounded local-up-axis tilt.

Feasibility decision:

- online articulation expert: feasible and best next implementation target;
- `place_in_slider`: feasible after held-reset/grasp-setup infrastructure exists;
- `push_into_drawer`: feasible but higher risk because it needs table-edge transition and gravity-settle control;
- formal rollout remains blocked by the current physical-quality hold; these findings are design evidence only.
