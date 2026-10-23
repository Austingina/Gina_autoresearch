# Operation 20261021 CALVIN Geometry-Contact Push Planner

- **Operation ID:** `20261021-calvin-geometry-contact-push-planner`
- **Created:** `2026-10-21T07:39:47+08:00`
- **Status:** `PASS — SIMULATOR VERIFIED Q1 SIX-CELL GATE`
- **Repository:** `/home/ai-sz-a26394-u1/Gina_autoresearch`

## Goal

Replace hand-tuned Cartesian correction for world-Y block pushes with one geometry-driven teacher algorithm. The planner must derive contact poses from live object geometry and task flow, rank short-horizon simulator rollouts, execute the selected candidate freshly, and preserve Oracle-gated full trajectories.

## Scientific constraint

- One algorithm across colors and directions.
- Geometry-derived dimensions/contact frames are allowed.
- No color/task-label-specific hand-tuned controller constants.
- Accepted remains gated only by CALVIN Oracle.
- No training, held-out generation, or evaluation.

## Safety and data gates

1. Candidate search uses exact embedded Scene-D reset arrays.
2. Candidate preview outputs are diagnostics, never accepted demonstrations.
3. Selected candidate is rerun from the exact reset to collect the final trajectory.
4. Failures remain isolated and lightweight.
5. Do not scale beyond one accepted trajectory per Y cell until all six cells pass.
6. Existing Q25 left/right collection must revalidate unchanged.

## Planned evidence

- PyBullet AABB/contact/body-ID probe.
- RED/GREEN contracts for contact-frame generation and candidate scoring.
- Candidate-search manifest including every candidate, score, and selected geometry.
- Six-cell simulator smoke manifest and accepted artifact read-back.
- Regression logs and SHA-256 hashes.

## Actual results

### Implementation

The independent strategy-1 collector is implemented at:

- `framework/scripts/data/collect_calvin_geometry_contact_push_expert.py`
- SHA-256: `aac938d997a9f10915dbe5d47c14def1c5e7554583a8ebefeda55531aea809c8`

The frozen Q1 controller uses one label-independent algorithm for all six cells:

1. Resolve the target by name from `env.get_info()["scene_info"]["movable_objects"]`.
2. Read collision half-extents and live object quaternion from PyBullet.
3. Intersect the ray from object COM along `-F` with the oriented collision box.
4. Generate the same Cartesian product of four relative contact heights and four quarter-turn wrist orientations for every object and direction.
5. Preview candidates with short simulator rollouts and rank progress, lateral drift, tilt, contact, and support diagnostics.
6. Use PyBullet robot-object contact feedback for preload.
7. During push, recompute the OBB/COM-ray contact frame from the live pose every step and issue a short contact preload target along `F`.
8. Accept only a fresh full-rate rerun whose initial/final info passes the CALVIN task oracle.

No red/blue/pink-specific control constant or task-combination branch was introduced.

### Same-frozen-version six-cell simulator gate

Collection:

`data/calvin/generated_geometry_contact_y_q1_20261021_s20261025_v2/`

Manifest SHA-256:

`735d5a4e03f2ac7544e8ea9fc16c783ef956fd568c231b3fd497886243247a6d`

The manifest contains 57 task/reset result rows. Six accepted trajectories were produced by the same controller file and passed the same-rollout CALVIN oracle:

| Cell | Reset | Candidate | Frames | Progress (m) | Lateral drift (m) | Tilt (rad) | Support preserved |
|---|---|---:|---:|---:|---:|---:|---|
| red-forward | `scene_D_native_y_red_0002` | `h0.35_q1` | 55 | 0.10110 | 0.07431 | 0.59319 | yes |
| red-backward | `scene_D_native_y_red_0022` | `h0.65_q3` | 29 | 0.10200 | 0.05651 | 0.43918 | yes |
| blue-forward | `scene_D_native_y_blue_0018` | `h0.50_q3` | 107 | 0.10492 | 0.09557 | 0.23587 | yes |
| blue-backward | `scene_D_native_y_blue_0016` | `h0.35_q3` | 97 | 0.10003 | 0.02163 | 0.91108 | yes |
| pink-forward | `scene_D_native_y_pink_0019` | `h0.35_q1` | 54 | 0.10165 | 0.02645 | 0.27585 | yes |
| pink-backward | `scene_D_native_y_pink_0008` | `h0.20_q1` | 58 | 0.10200 | 0.00098 | 0.51853 | yes |

Result: `accepted_verified=6`, one per cell, `complete=true`, folds absent.

### Accepted artifact read-back

The CALVIN runtime Python independently reopened all six accepted NPZ files with `allow_pickle=False` and verified:

- every trajectory SHA-256 against both the manifest and its receipt;
- `oracle_success=true` and the requested task in `oracle_achieved_tasks`;
- aligned full-rate `rgb_static`, `rgb_gripper`, `robot_obs`, `scene_obs`, `actions`, `rel_actions`, `G`, `phase`, `post_contact`, `object_pos`, `interaction_anchor`, and `controller_state` arrays;
- static RGB shape `(T, 200, 200, 3)`;
- gripper RGB shape `(T, 84, 84, 3)`;
- action shape `(T, 7)` and `G` shape `(T, 6)`.

Evidence:

- `artifacts/evidence/20261021T073947-calvin-geometry-contact-planner/q1_readback.json`
- `artifacts/evidence/20261021T073947-calvin-geometry-contact-planner/final_hashes.sha256`

### Regression evidence

The following unit contracts pass under the CALVIN runtime:

- geometry-contact planner;
- legacy parametric push expert;
- Scene-D Y-push reset pool;
- Scene-D left/right reset pool;
- fixed scripted expert collector.

The frozen Scene-D left/right Q25 collection was read back again after the strategy-1 changes:

- 150 accepted and hash-verified;
- 479 failed attempts;
- 3,457 accepted frames;
- exactly 25 accepted in each of six left/right cells;
- `complete=true`;
- folds absent;
- manifest SHA-256 before and after audit remained identical:
  `9539e1346953227420ded8542a19ada66b3b74717f06b75ec3c30f77d8ee4e1c`.

Evidence:

- `artifacts/evidence/20261021T073947-calvin-geometry-contact-planner/q25_regression_audit.json`
- `artifacts/evidence/20261021T073947-calvin-geometry-contact-planner/q25_regression_summary.json`
- `artifacts/evidence/20261021T073947-calvin-geometry-contact-planner/q25_manifest_before.sha256`
- `artifacts/evidence/20261021T073947-calvin-geometry-contact-planner/q25_manifest_after.sha256`

### Scope and remaining gate

This operation passes only the required **six cells × one accepted trajectory** simulator gate. It does not establish collection-scale reliability or model generalization. In particular, the single successful trajectories still show lateral drift up to `0.09557 m` and tilt up to `0.91108 rad`; these are diagnostics, not a claim of stable pushing quality.

Do not scale directly to Q25. The next permitted stage is six cells × three distinct resets under this same frozen controller, followed by distribution-level audits of success rate, lateral drift, tilt, vertical displacement, and support identity. No model training, held-out split, fold generation, or generalization claim was made.
