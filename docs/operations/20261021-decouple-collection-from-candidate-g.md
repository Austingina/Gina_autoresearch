# Operation 20261021 Decouple Data Expansion from Candidate G

- **Operation ID:** `20261021-decouple-collection-from-candidate-g`
- **Created:** `2026-10-21T15:45:07+08:00`
- **Status:** `PASS — OFFLINE VERIFIED`
- **Operator:** Hermes Agent
- **Repository:** `/home/ai-sz-a26394-u1/Gina_autoresearch`

## Authoritative requirement

The current objective is broader, more diverse, native simulator data. Candidate guidance feature `G` is not yet validated and must not be required for collection, distribution design, Oracle acceptance, or default training input. Rich native observations, states, actions, contacts, task intent, reset provenance, and success receipts must remain sufficient to derive and compare future candidate representations.

## Baseline

- Current CALVIN collectors require `g`, `phase`, and `post_contact` in their trajectory validators.
- Geometry and parametric controllers internally use task direction and geometric contact targets; those are native task/control requirements, not evidence that `G=[A,F]` is a validated generalization feature.
- Failed payload pruning intentionally removed all `attempts/` payload directories while retaining collection manifests and archived receipts. Legacy validation currently treats intentionally pruned failed paths as corruption.
- Pre-operation hashes:
  - geometry collector: `aac938d997a9f10915dbe5d47c14def1c5e7554583a8ebefeda55531aea809c8`;
  - geometry test: `9c019ab18dfbef017bcbd48cc515f390b34a7455f9b9fa0e394de25811fa451d`;
  - parametric collector: `c82b9914408bb47d8f0d5bd32b37c7ba9e91d652b58c2248aa2f9395414386f9`;
  - parametric test: `16077e14b6522388982933ae22f8979949ee274d0f7eda43fb3879380af7a9a5`;
  - scripted collector: `6a664ae3fad17da0217d8c89198e24fc39b6dcd9837c39f1ed408310a24ce9c8`;
  - scripted test: `32108646c648c250b38c7d6eb2b80754d7d6edb69985310f51430c6e78b49414`.

## Goal

1. Make native data the required trajectory contract.
2. Make candidate G/phase labels optional, versioned, and disabled by default for new collection.
3. Preserve the teacher's native task direction and geometry internally without describing them as validated G.
4. Store raw contact/orientation/task-intent fields needed to derive future representations.
5. Default failed attempts to metadata-only persistence.
6. Add explicit support for intentionally pruned failed payloads while continuing to fail on missing accepted data.

## Non-goals

- Do not change the authoritative native Oracle acceptance rule.
- Do not retrain or evaluate a policy.
- Do not define held-out folds.
- Do not claim the current G is accurate or useful.
- Do not start LIBERO/RoboTwin runtime collection in this operation.

## TDD plan

1. Add RED tests proving native-only trajectories validate without `g`, `phase`, `post_contact`, or `interaction_anchor`.
2. Add RED tests proving the optional provisional-label bundle is all-or-none and explicitly versioned.
3. Add RED tests proving missing pruned failures are accepted only with an explicit prune sidecar/record, while missing accepted files still fail.
4. Add RED tests proving metadata-only failure persistence writes no failure NPZ.
5. Implement the minimum schema/persistence changes.
6. Run target GREEN and all affected collector tests.
7. Re-read the existing Q1/Q25 accepted datasets and the pruned-failure collections.

## Safety and rollback

- Capture backups of every modified pre-dirty file in the evidence directory before editing.
- No existing accepted dataset or collection manifest will be rewritten.
- Rollback source changes from operation-local backups, never with `git restore`.
- Stop if any accepted-artifact validation regresses.

## Evidence directory

`artifacts/evidence/20261021T154507-decouple-collection-from-g/`

## PASS criteria

- New collection defaults to native-only labels and metadata-only failures.
- `--emit-provisional-g` is explicit opt-in.
- Existing G-bearing trajectories remain readable.
- Explicitly pruned failures no longer make accepted-only validation fail.
- Missing accepted payloads still fail closed.
- All affected unit contracts and accepted Q1/Q25 read-backs pass.

## Actual results

- **Completed:** `2026-10-21T16:08:52+08:00`
- **Status:** `PASS — OFFLINE VERIFIED`
- New CALVIN scripted, parametric, and geometry collections now default to native-only storage. `G`, `phase`, `post_contact`, and `interaction_anchor` are no longer required by the native trajectory contract.
- Experimental labels are emitted only with `--emit-provisional-g`; manifests mark them as `candidate_g_af_v1` and explicitly state that they are not an acceptance or distribution gate.
- The geometry collector now stores raw `object_orn`, `contact_detected`, `task_flow`, live contact position, and surface normal alongside native observations/actions. These support future recomputation without depending on the current G.
- All three collectors default failed rollout persistence to metadata-only receipts. Accepted trajectories remain full-rate.
- Added explicit `failure_payload_prune.json` sidecars to all 27 affected historical collections. Each sidecar contains paths, byte counts, and pre-deletion SHA-256 values linked to the authoritative inventory and archived receipts.
- The validators accept only failures explicitly recorded as metadata-only or explicitly listed in a prune sidecar. Missing accepted payloads still fail closed (covered by the RED/GREEN unit contract).
- Historical compatibility verified:
  - scripted collection: 3/3 accepted artifacts and hashes;
  - Scene D left/right Q25: 150 accepted, six cells ×25, 479 pruned failures, `complete=true`;
  - geometry forward/backward Q1: 6 accepted, six cells ×1, 4 pruned failures, `complete=true`;
  - all 27 pruned collections passed validator readback;
  - independent geometry Q1 NPZ readback verified six accepted trajectories.
- Tests passed:
  - scripted expert unit contract;
  - parametric push expert unit contract;
  - geometry-contact planner unit contract;
  - fixed-scene reset and Y-reset contracts;
  - Python compilation for all three collectors.
- Post-operation hashes:
  - geometry collector: `171c2343f98a1af287bbb55a2415a046a8ac21b6448e6e229448ae0aff710f28`;
  - geometry test: `9c019ab18dfbef017bcbd48cc515f390b34a7455f9b9fa0e394de25811fa451d`;
  - parametric collector: `3bb48a1aa8f67512a5f915a63030c4b7808fae5b03458b5d89036b3089fa2a1a`;
  - parametric test: `172e3026f03c05572a8d8f759d5d2dbc2069bd85c6da97ef8e58142031232847`;
  - scripted collector: `c3f508267b92462707772c34715f26d0ec3d614d398299a5526273ac2e0cc195`;
  - scripted test: `3b3080381cc75151e4ad37e3c5241c352ca0557e2e8912c3cdd9e706e32283bf`.

## Verification level

- Code/schema/persistence behavior: **OFFLINE VERIFIED**.
- Historical accepted data integrity: **READBACK VERIFIED**.
- No new simulator rollout was claimed in this operation; the next expansion run must be reported separately as **SIMULATOR VERIFIED** only after Oracle-gated rollout and artifact readback.
