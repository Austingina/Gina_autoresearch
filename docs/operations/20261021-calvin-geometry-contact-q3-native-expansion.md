# Operation 20261021 CALVIN Geometry Contact Q3 Native Expansion

- **Operation ID:** `20261021-calvin-geometry-contact-q3-native-expansion`
- **Created:** `2026-10-21T16:26:36+08:00`
- **Status:** `PASS — Q3 QUOTA AND ARTIFACT INTEGRITY; STABILITY GATE NOT PASSED`
- **Operator:** Hermes Agent
- **Repository:** `/home/ai-sz-a26394-u1/Gina_autoresearch`

## Objective

Execute the next gated CALVIN expansion stage: red/blue/pink × forward/backward, quota three Oracle-accepted trajectories per cell from distinct native Scene D resets. This is data expansion only; candidate G remains disabled.

## Frozen inputs

- Geometry collector SHA-256: `171c2343f98a1af287bbb55a2415a046a8ac21b6448e6e229448ae0aff710f28`
- Base collector SHA-256: `3bb48a1aa8f67512a5f915a63030c4b7808fae5b03458b5d89036b3089fa2a1a`
- Reset pool: `data/calvin/fixed_scene_D_y_push_resets_20261020_s20261022_n24_v1.json`
- Reset pool SHA-256: `23c98a4d2a2d6bc7d51561c538d465a7bfe3afe938e9a0a76402ed6983f75c41`
- CALVIN code: `/home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/code/calvin`
- Runtime Python: `/home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/envs/calvin_reuse/bin/python`
- Source split: `/home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/data/calvin_debug_dataset/training`
- Output: `data/calvin/generated_geometry_contact_y_q3_20261021_s20261027_v1/`
- Seed: `20261027`

## Collection parameters

- `quota_per_cell=3`
- `max_attempts_per_cell=24`
- `max_candidates=16`
- `preview_steps=260`
- `final_steps=300`
- `motion_threshold=0.003`
- `failure_storage=metadata_only`
- `emit_provisional_g=false` (default; flag intentionally absent)

## Acceptance and quality gates

1. Accepted only by same-rollout CALVIN `get_task_info_for_set(initial, final, {task})`.
2. Six cells must each have three accepted trajectories from three distinct reset IDs.
3. Accepted NPZ/receipt hashes and modality alignment must pass independent readback.
4. Audit lateral drift, tilt, vertical displacement, and table support before any Q25 decision.
5. Failure attempts remain metadata-only.
6. No folds or model training are generated.

## Preflight

- Output path absent.
- No active geometry collector process found using exact Python argv inspection.
- Existing accepted datasets are not modified.

## Command

```bash
CALVIN_PYTHON=/home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/envs/calvin_reuse/bin/python
$CALVIN_PYTHON framework/scripts/data/collect_calvin_geometry_contact_push_expert.py \
  --calvin-code /home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/code/calvin \
  --source-split /home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/data/calvin_debug_dataset/training \
  --reset-manifest data/calvin/fixed_scene_D_y_push_resets_20261020_s20261022_n24_v1.json \
  --out-dir data/calvin/generated_geometry_contact_y_q3_20261021_s20261027_v1 \
  --quota-per-cell 3 --max-attempts-per-cell 24 \
  --max-candidates 16 --preview-steps 260 --final-steps 300 \
  --motion-threshold 0.003 --failure-storage metadata_only --seed 20261027
```

## Actual results

### First Q3 attempt

- Process `proc_f9aa2a763a00` exited with code `2` because the collector intentionally returns `2` when `complete=false`; this was not an argparse/startup failure.
- Manifest result: 12 accepted, 74 failed, incomplete.
- Cell counts:
  - red-forward: 3;
  - red-backward: 3;
  - blue-forward: 0;
  - blue-backward: 3;
  - pink-forward: 0;
  - pink-backward: 3.
- All 12 accepted trajectories:
  - have matching manifest/file SHA-256;
  - contain all required native state/action/dual-camera/contact geometry fields;
  - have aligned modalities;
  - contain no `g`, `phase`, `post_contact`, or `interaction_anchor` arrays;
  - use three distinct resets in each completed cell;
  - preserve support contact.
- Stability envelope of the accepted subset:
  - max lateral drift: `0.09486938267946243 m`;
  - max tilt: `0.9901319022385179 rad`;
  - max Z drop: `0.0004138648509979248 m`.
- Evidence: `artifacts/evidence/20261021T162636-calvin-geometry-q3/partial_readback.json`.

### Root cause evidence

The process log showed preview-success/final-failure divergence. Two exact reset/candidate pairs that succeeded in Q1 failed in Q3:

- blue-forward, `scene_D_native_y_blue_0018`, `h0.50_q3`: Q1 Oracle success at `0.10492 m`; Q3 final failure despite `0.11783 m` progress and the same candidate;
- pink-forward, `scene_D_native_y_pink_0019`, `h0.35_q1`: Q1 Oracle success at `0.10165 m`; Q3 final failure at `0.06319 m`.

CALVIN's `MovableObject.reset()` calls `resetBasePositionAndOrientation()` but does not call `resetBaseVelocity()`. `PlayTableSimEnv.reset()` then immediately performs one physics step. Consequently linear/angular velocity left by a prior candidate can survive a pose reset and contaminate the next preview/final rollout. This is a simulator-reset hygiene defect, not evidence for a color-specific controller rule.

### Remediation implementation and verification

- Added a simulator-wide two-pass reset that clears linear/angular velocities of every movable body.
- A reused-environment probe still diverged on the third repetition, proving that velocity clearing alone does not eliminate all PyBullet/contact-solver history.
- A fresh-environment probe then ran blue-forward and pink-forward three times each:
  - all 6 outcomes and metrics were exactly repeatable;
  - pink-forward succeeded 3/3 at the same reset/candidate;
  - blue-forward failed 3/3 consistently, showing the former Q1 success was history-dependent rather than a stable clean rollout.
- Collector now creates a fresh CALVIN/PyBullet environment for every candidate preview and every final recording rollout.
- Unit contracts and `py_compile` pass.
- Remediated collector SHA-256: `2e59f96e9151d166c57031f2935ac9e1e44e019b773a1f562c83720fd27cc929`.
- Evidence:
  - `reset_hygiene_red.log` / `reset_hygiene_green.log`;
  - `reset_reproducibility.log` (reused environment, expected failure);
  - `fresh_env_reproducibility.json` and `.log` (fresh environment, pass);
  - `fresh_env_collector_green.log`.

### Recovery collection

Only the missing blue-forward and pink-forward cells will be collected in a new directory. The original 12 accepted artifacts remain unchanged and separately auditable.

- Output: `data/calvin/generated_geometry_contact_y_q3_forward_recovery_20261021_s20261028_v1/`
- Tasks: `push_blue_block_forward`, `push_pink_block_forward`
- Quota: 3 per cell
- Failures: metadata-only
- Candidate G: not emitted
- Candidate and final isolation: one fresh CALVIN environment per rollout

### Recovery result and reset-pool extension

The first recovery collection produced five independently read-back accepted trajectories:

- blue-forward: 2/3;
- pink-forward: 3/3;
- all five hashes and modality lengths match;
- candidate G remains absent;
- all five preserve table support.

The original 24 blue resets are exhausted, so the final blue-forward sample will use a newly sampled Scene-D-native reset pool. The reset generator now supports an explicit ID namespace to prevent IDs in different pools from referring to different states. No success threshold or controller rule is changed.

- New pool seed: `20261029`
- New ID namespace: `s20261029`
- New pool target: 24 resets per color
- Output: `data/calvin/fixed_scene_D_y_push_resets_20261021_s20261029_n24_v2.json`

The attempted 24-per-color pool was correctly rejected: only red 1, blue 1, and pink 2 eligible resets were found in 12,000 native samples, so no incomplete manifest was written. Instead, 18 blue-forward-safe resets were deterministically selected from the existing 200-reset native Scene-D pool using a direction-aware free-corridor test. This changes only reset eligibility, not rollout acceptance.

- Derived pool: `data/calvin/fixed_scene_D_blue_forward_resets_20261021_from_n200_v1.json`
- Derived pool SHA-256: `3a504ef165c4595d517bb813db7e746a8325a5c0493782d625b68d4e89b7611d`
- Count: 18 unique reset IDs, zero ID overlap with the original Y pool
- Source: existing Scene-D-native 200-reset manifest; original state arrays are retained verbatim
- Final recovery target: one additional Oracle-accepted blue-forward trajectory in a new collection

## Final aggregate result

Completed at `2026-10-21T17:38:29+08:00`.

The final recovery rollout succeeded on `scene_D_native_y_forward_n200v1_blue_0001` with candidate `h0.65_q3`, Oracle progress `0.1002`, 53 stored frames, lateral drift `0.01114 m`, tilt `0.35843 rad`, and preserved table support.

Across the original, first recovery, and final recovery collections:

- accepted: **18**;
- all six object×direction cells: **3/3**;
- three distinct reset IDs in every cell;
- all artifact SHA-256 values match;
- all required fields and per-frame modalities align;
- all 18 trajectories omit provisional candidate G;
- all 18 preserve table support.

Evidence:

- aggregate audit: `artifacts/evidence/20261021T162636-calvin-geometry-q3/q3_aggregate_audit.json`;
- aggregate audit SHA-256: `80bc373b0a4e0c7e113d23df027168fc8821b3a24e03585561323b9c10584813`;
- final recovery manifest SHA-256: `ec05d0f998731cb6dcaa19bfcc3eb9f8e2bfbae9a63b56a0ad2911aad3f2baf7`.

### Gate interpretation

- Q3 quota gate: **PASS**.
- Artifact integrity gate: **PASS**.
- Stability gate: **NOT PASSED**.

The accepted data remain valid Oracle-success trajectories, but the aggregate maximum endpoint lateral drift is `0.10152 m`. The legacy manifest `tilt` value (`1.00060 rad` maximum) is a total quaternion angle and therefore mixes planar yaw with actual toppling. A corrected framewise physical audit was subsequently run:

- definition: angle between the block local +Z axis and world ±Z, invariant to planar yaw;
- maximum trajectory lateral drift: `0.10383 m`;
- maximum physical tilt: `0.79255 rad`;
- all 18 trajectories retain support contact;
- evidence: `artifacts/evidence/20261021T162636-calvin-geometry-q3/q3_physical_stability_v2.json`;
- evidence SHA-256: `7191bc88fad5eb98b1204e79dbb406e30aff687a995bf16edbdf6459269ef1e5`.

Thus the stability gate remains not passed even after correcting the metric. These quality measurements do not retroactively replace the native Oracle or invalidate the artifacts; they block immediate Q25 scale-up until a predeclared physical-quality policy and unified controller improvement are verified.
