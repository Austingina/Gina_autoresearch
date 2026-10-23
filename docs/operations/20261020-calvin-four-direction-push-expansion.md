# Operation 20261020 CALVIN Four-Direction Push Expansion

- **Operation ID:** `20261020-calvin-four-direction-push-expansion`
- **Created:** `2026-10-20T22:24:11+08:00`
- **Status:** `PARTIAL — OFFLINE VERIFIED 4-DIRECTION SUPPORT; SIMULATOR VERIFIED 2/6 Y CELLS`
- **Repository:** `/home/ai-sz-a26394-u1/Gina_autoresearch`

## Scope

Data expansion only:

1. audit successful Q25 left/right initial-position bins;
2. extend the named-object push expert from world-X left/right to world-Y forward/backward;
3. define exact Y-direction CALVIN oracle tasks at runtime without modifying the external CALVIN installation;
4. generate Scene-D-native resets with safe Y travel margin;
5. collect and read back a small simulator-verified forward/backward batch before any scale-up.

No training, held-out split, fold generation, evaluation, or generalization claim is in scope.

## Direction convention

- `left`: world `[-1, 0, 0]`
- `right`: world `[+1, 0, 0]`
- `forward`: world `[0, +1, 0]`, away from the robot base
- `backward`: world `[0, -1, 0]`, toward the robot base

The manifest and every trajectory must preserve the explicit world-frame flow so the language label is not ambiguous.

## Preconditions and gates

- Runtime and reset scene must remain `calvin_scene_D`.
- Y-direction tasks must use CALVIN's existing `push_object` predicate with strict `>0.1m` displacement and table/support contact.
- Planned push travel keeps margin above the oracle threshold.
- New direction logic and reset schema require red/green TDD evidence.
- Small simulator batch must contain fresh RGB, state, action, G, phase, reset provenance, and exact oracle receipts.
- Failures stay outside accepted data; no evaluation folds are emitted.

## Pre-operation distribution finding

Q25 left/right accepted data has broad successful coverage but sparse bins remain: red-right has two empty 3×3 XY bins, blue-left one, and pink-right one. This is a descriptive collection audit only.

## Actual results

- **Completed audit/implementation pass:** `2026-10-20T23:00:22+08:00`
- **Overall status:** partial. The generic 4-direction schema, parser, controller geometry, runtime Y-task injection, task-specific native reset loading, Y-reset builder, dynamic validator, and dynamic distribution auditor are implemented and offline verified. The Y-direction expert is not yet reliable enough for a balanced dataset.
- **Scope remained data expansion only:** no folds, training, held-out split, or evaluation.

### Q25 position-bin audit

Using X edges `[0.000, 0.047, 0.094, 0.141]` and Y edges `[-0.150, -0.110, -0.070, -0.030]`:

- red-left: 0 empty bins;
- red-right: 2 empty bins;
- blue-left: 1 empty bin;
- blue-right: 0 empty bins;
- pink-left: 0 empty bins;
- pink-right: 1 empty bin.

Evidence: `artifacts/evidence/20261020T222411-calvin-four-direction/q25_position_bin_audit.json`.

### Implemented

- `parse_push_task` now supports `left/right/forward/backward` with explicit world-frame flows.
- Collector CLI now accepts `--directions`; default remains `left,right`, preserving prior behavior.
- Runtime Oracle config injects `push_*_block_forward=[push_object, block_*, 0, +0.1]` and backward with `-0.1`; the external CALVIN installation is unchanged.
- Fixed-scene reset banks can declare task-specific eligible subsets while old six-cell reset manifests remain valid.
- Validation and distribution audit now derive task cells from each collection manifest instead of hard-coding left/right.
- New Scene-D-native Y-reset builder and unit contract were added.
- Y-direction controller includes a geometric stop/settle gate at progress `>=0.102m` to avoid continuing past a valid threshold toward the table edge.

### Native Y-reset pools

- `data/calvin/fixed_scene_D_y_push_resets_20261020_s20261022_n24_v1.json`
  - 72 target-specific resets: 24 each for red/blue/pink;
  - selected from 8,497 native Scene-D samples;
  - each reset is shared by that object's forward/backward tasks.
- A stricter unobstructed-corridor experiment produced only 3 resets per color after 19,793 samples:
  `data/calvin/fixed_scene_D_y_push_resets_20261020_s20261023_n3_v2.json`.
- This confirms that simultaneous Y-margin, reachable X, pair separation, and clear-corridor states are rare under the native Scene-D sampler.

### Final simulator smoke result

Collection: `data/calvin/generated_fixed_scene_D_y_push_20261020_s20261022_q1_v8/`

| Cell | Attempted | Oracle accepted | Failed | Max verified status |
|---|---:|---:|---:|---|
| red-forward | 24 | 0 | 24 | failed smoke |
| red-backward | 6 | 1 | 5 | simulator verified |
| blue-forward | 24 | 0 | 24 | failed smoke |
| blue-backward | 24 | 0 | 24 | failed smoke |
| pink-forward | 24 | 0 | 24 | failed smoke |
| pink-backward | 20 | 1 | 19 | simulator verified |
| **Total** | **122** | **2** | **120** | **incomplete** |

- The two accepted trajectories contain 78 frames (42 red-backward, 36 pink-backward).
- Both accepted files were hash/read-back verified with Scene D native reset provenance, static/gripper RGB, state, 7-D action, 6-D G, phase, post-contact, and Oracle receipts.
- No Y collection is marked complete, balanced, or suitable for merging into the formal Q25 dataset.
- Experimental Y collections occupy about 3.7G in total and remain isolated from formal accepted data.

### Root-cause evidence

- Original Y pushes frequently generated a stable +2.4–2.8cm end-effector X offset at contact, causing lateral deflection instead of pure Y motion.
- Fixed X compensation did not generalize across resets.
- Open-gripper and alternate-yaw variants did not produce balanced success.
- Several backward attempts exceeded 0.1m but lost required support contact or fell beyond the table edge, so CALVIN correctly rejected them.
- The progress-based settle gate converted one additional pink-backward case into an Oracle-accepted trajectory but did not solve forward or blue cells.

### Regression and compatibility

- `CALVIN fixed-scene Y-push reset pool unit contract: OK`
- `CALVIN fixed-scene reset pool unit contract: OK`
- `CALVIN parametric push expert unit contract: OK`
- `CALVIN scripted expert collector unit contract: OK`
- Existing formal Q25 collection revalidated unchanged: 150 accepted, exactly 25 per left/right cell, complete, zero folds.
- No collector/PyBullet process remained active.

### Final hashes

- collector: `d59df4e2ed96d1e8bfc0d7b676a568431dfb482467e9cb450a5b8ec6234fa460`
- Y-reset builder: `217c065ffd40c40d487dfd4e2c32ef9632f830640ad6f54fd620e5dae1047cc4`
- 72-reset manifest: `23c98a4d2a2d6bc7d51561c538d465a7bfe3afe938e9a0a76402ed6983f75c41`
- final Y smoke manifest: `13bdc15c4d6af2401ef9f58745df44333d2905e783058df5989d237ebda39c1e`
- Q25 position-bin audit: `ed9b64f218ee4c7914f8ae0a7a73f38355a9276c4c65eeed589efe4ff4c033d0`

### Required next engineering step

Do not scale Y-direction collection yet. The remaining blocker is not task parsing or persistence but physically stable `+Y` contact for red/blue blocks. The next controller should use contact-force or collision-link feedback, or a short-horizon simulator planner over several candidate contact poses, instead of additional hand-tuned Cartesian thresholds.

## Closed-loop continuation — 2026-10-20T23:12:59+08:00 to 23:45:47+08:00

### Implemented and offline verified

- Lower Y contact height (`0.004 m` above object center by default).
- Live end-effector/object lateral feedback and bounded X correction.
- Short-standoff incremental Y push targets rather than one distant static push waypoint.
- Direction/object-specific braking thresholds and a settle phase.
- Positive-velocity progress projection for anticipating overshoot.
- Bounded initial-lane/current-object X blending to avoid unconstrained lateral chasing.
- Lightweight failed-attempt storage: every eighth frame plus the terminal frame; accepted trajectories remain full-rate.
- Failure receipts now distinguish original executed steps from stored diagnostic steps.

### Real simulator evidence

`data/calvin/generated_fixed_scene_D_y_push_20261020_s20261024_q1_v9/`:

- red-backward: 1 accepted;
- blue-backward: 1 accepted;
- pink-backward: 1 accepted;
- all three forward cells: 0 accepted;
- 3 accepted files were hash/read-back verified; collection remains incomplete.

`data/calvin/generated_fixed_scene_D_y_forward_20261020_s20261024_q1_v12/`:

- pink-forward: 1 accepted;
- red-forward and blue-forward: 0 accepted;
- accepted file was hash/read-back verified; collection remains incomplete.

Across isolated experimental versions, actual Oracle-confirmed coverage reached 4/6 Y cells, but this is **not** a balanced collection and not evidence that one final controller revision covers all four cells simultaneously. No Y data was merged into Q25.

### Remaining failure mechanism

- Red/blue forward attempts can achieve `+Y > 0.1 m`, but their X position can diverge catastrophically and they lose support contact before Oracle acceptance.
- Example observed blue failure: initial X `0.0875 m`, terminal X `-0.6422 m`, terminal Z `0.0200 m`; CALVIN correctly rejected it.
- Full lane lock prevents lateral runaway but often loses contact; unconstrained live-object tracking preserves contact but can amplify drift. A 75/25 bounded lane blend was offline verified but did not complete a forward cell in the final smoke run.
- Further threshold tuning is no longer justified without collision/contact feedback or planning.

### Storage effect

- New incomplete diagnostic collections are approximately `84–97 MB` each rather than roughly `0.9–1.1 GB` for earlier full-rate failure batches.
- This is roughly an order-of-magnitude reduction while preserving aligned keyframes and the final state.

### Final verification

- Four collector/reset/script regression contracts passed.
- Formal Q25 collection still validates as 150 accepted, 25 per left/right cell, complete, zero folds.
- q1-v9 validates as 3 accepted and incomplete.
- q1-v12 validates as 1 accepted and incomplete.
- No collector/PyBullet process remained active.
- Disk free remained approximately 1.3 TB.

### Final hashes for this continuation

- collector: `c82b9914408bb47d8f0d5bd32b37c7ba9e91d652b58c2248aa2f9395414386f9`
- collector contract: `16077e14b6522388982933ae22f8979949ee274d0f7eda43fb3879380af7a9a5`
- q1-v9 manifest: `cd22d3b246b9f2d857a778675703771a94ce1792d146f2ca968d1b1402b4d56d`
- q1-v12 manifest: `fee2bbfb53f5f71acf2525d9f190ac770a0c704a6723be614d2491e574d997aa`
