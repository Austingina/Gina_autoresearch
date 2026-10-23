# Operation 20261020 CALVIN Scene-D Q25 Data Expansion

- **Operation ID:** `20261020-calvin-sceneD-q25-data-expansion`
- **Created:** `2026-10-20T21:55:38+08:00`
- **Status:** `PASS — SIMULATOR VERIFIED Q25 DATA EXPANSION`
- **Repository:** `/home/ai-sz-a26394-u1/Gina_autoresearch`

## Scope

Data expansion only. Expand the six fixed-Scene-D planar push cells to 25 oracle-confirmed trajectories per cell (150 accepted total), using a larger native reset pool and fresh simulator observations.

Out of scope: model training, held-out split creation, fold generation, policy evaluation, or generalization claims.

## Gates

- Actual runtime and reset schema must both be `calvin_scene_D`.
- Reset pool states must be native Scene-D states and reusable across all six cells.
- Accepted trajectories require the exact CALVIN task oracle on the same rollout.
- Failures remain outside `accepted/`.
- Images, states, actions, G, phase, post-contact, task factors, reset provenance, and receipts must be aligned and readable.
- Collector overwrite protection remains enabled.
- No evaluation folds are emitted.
- Final reporting must include attempted/accepted/failed counts, frames, reset diversity, initial-position ranges, disk usage, hashes, and limitations.

## Pre-operation state

- Previous fixed-scene Q3 collection: 18 accepted (3 per cell), 33 failed, 484 accepted frames.
- Root filesystem free space: approximately `1.3T`.
- No collector or PyBullet process was active.

## Actual results

- **Completed:** `2026-10-20T22:08:11+08:00`
- **Scope executed:** data expansion and descriptive distribution audit only; no folds, training, held-out split, or evaluation.
- **Reset pool:** 200 unique native `calvin_scene_D` states selected from 3,756 simulator-native reset samples.
- **Reset pool path:** `data/calvin/fixed_scene_D_push_resets_20261020_s20261021_n200_v1.json`
- **Collection path:** `data/calvin/generated_fixed_scene_D_push_20261020_s20261021_q25_v1/`
- **Accepted:** 150 oracle-confirmed trajectories, exactly 25 per cell.
- **Accepted frames:** 3,457.
- **Failed attempts:** 479, isolated under `attempts/`; none entered `accepted/`.
- **Successful reset diversity:** 25 distinct reset IDs per cell; 96 distinct reset IDs across the 150 accepted trajectories.
- **Collection size:** 3.9G total: 183M accepted data and 3.7G failed-attempt diagnostics.
- **Evaluation folds:** 0.

| Cell | Attempted | Accepted | Failed | Success rate | Frames | Length range | Initial x range | Initial y range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| red-left | 109 | 25 | 84 | 22.9% | 546 | 19–25 | 0.027–0.137 | -0.150–-0.035 |
| red-right | 83 | 25 | 58 | 30.1% | 494 | 17–25 | 0.010–0.138 | -0.143–-0.040 |
| blue-left | 138 | 25 | 113 | 18.1% | 575 | 18–39 | 0.024–0.122 | -0.149–0.060 |
| blue-right | 97 | 25 | 72 | 25.8% | 500 | 17–24 | 0.004–0.136 | -0.149–-0.037 |
| pink-left | 133 | 25 | 108 | 18.8% | 686 | 18–159 | 0.002–0.134 | -0.146–-0.048 |
| pink-right | 69 | 25 | 44 | 36.2% | 656 | 16–96 | 0.003–0.124 | -0.149–-0.037 |
| **Total** | **629** | **150** | **479** | **23.8%** | **3,457** | — | — | — |

## Read-back evidence

- All 150 accepted NPZ files, receipts, and hashes were read back successfully.
- Accepted counts are balanced at 25 for all six cells.
- Runtime scene read from accepted files is exclusively `calvin_scene_D`.
- Reset source kind is exclusively `native_runtime_reset`.
- Images are fresh simulator frames: static `200×200×3`, gripper `84×84×3`.
- State, 7-D actions, 6-D G, phase, post-contact, task factors, and reset metadata are temporally aligned.
- No evaluation fold files were emitted.
- Final unit and regression contracts passed:
  - `CALVIN fixed-scene reset pool unit contract: OK`
  - `CALVIN parametric push expert unit contract: OK`
  - `CALVIN scripted expert collector unit contract: OK`
- No collector or PyBullet process remained active after completion.

## Distribution artifacts

- `data/calvin/generated_fixed_scene_D_push_20261020_s20261021_q25_v1/data_distribution_summary.json`
- `artifacts/evidence/20261020T215538-calvin-sceneD-q25/`

## Final hashes

- reset pool: `696482856fe06e814a9f810f057c81b076114e113c89143215f6fe42fd5a3864`
- collection manifest: `9539e1346953227420ded8542a19ada66b3b74717f06b75ec3c30f77d8ee4e1c`
- distribution summary: `a3a5cc92b959defb56da03933aaf3b7a700f8fb05211550d1f56145e5d71eb49`
- reset builder: `caa39024aa6589170c448168e0b0d1a16f50ff2e6ad61e7b48847b6301ac8ce2`
- collector: `4318034c900499d5fb485b7184e4763d8aa42d50e4ba435541235ca35e54ef92`
- auditor: `01a590ec92c118e9130567508433503a77101a3cf3f1d3934bab99cb5f934da0`

## Boundaries and limitations

- This is still one task family, one runtime scene, and two directions; no grasp/container/articulation/spatial-relation data was added.
- Every cell uses the same 200-reset candidate pool, but oracle gating yields different accepted subsets: there are 96 unique successful resets globally and no single reset succeeded for all six cells. This is acceptable for distribution expansion, but it is not a matched-reset evaluation set.
- Accepted x positions are concentrated on the positive side of the reachable workspace even though the reset pool includes negative x values; subsequent expansion should explicitly stratify successful samples by position bin.
- The overall expert success rate is 23.8%, and failed diagnostics consume 3.7G. Future large batches should pre-screen resets or store receipt-only failures to improve collection efficiency without weakening accepted-data integrity.
- No generalization or model-performance claim is made.
