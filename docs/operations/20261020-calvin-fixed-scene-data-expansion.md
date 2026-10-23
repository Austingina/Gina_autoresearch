# Operation 20261020 CALVIN Fixed-Scene Data Expansion

- **Operation ID:** `20261020-calvin-fixed-scene-data-expansion`
- **Created:** `2026-10-20T20:30:37+08:00`
- **Status:** `PASS — SIMULATOR VERIFIED DATA EXPANSION`
- **Repository:** `/home/ai-sz-a26394-u1/Gina_autoresearch`

## Scope

This operation is **data expansion only**. It will:

1. generate a deterministic native reset pool for the actual CALVIN `calvin_scene_D` runtime;
2. use identical scene/camera/robot semantics across all `target object × direction` cells;
3. collect fresh simulator images, states, G labels, phase labels, executed expert actions, and oracle receipts;
4. balance accepted trajectory counts across red/blue/pink and left/right;
5. preserve failed attempts separately and report collection distributions.

It will **not** train a model, create an evaluation protocol, run held-out tests, or claim compositional generalization. Existing experimental fold files are outside this operation's deliverable.

## Pre-operation evidence

- Live dataset config names the runtime scene `calvin_scene_D`.
- Scene D movable-object order is `block_red`, `block_blue`, `block_pink`.
- Scene D contains official push tasks for all three objects in both left and right directions.
- Previous smoke data mixed reset vectors labelled A/B/C into a Scene D runtime; those trajectories remain smoke evidence and will not be treated as the formal fixed-scene expansion.
- No collector or PyBullet process was active at operation start.
- Existing unrelated worktree changes must not be reset or overwritten.

## Gates

- Strict TDD for reset schema and collector changes.
- Reset rows must declare and encode `calvin_scene_D`; no cross-scene vectors.
- Every reset must be reusable across all six object-direction cells to support controlled crossing.
- Exact task-oracle success is required for accepted data.
- Output paths must be new and overwrite-protected.
- New collection manifests must default to data statistics only; evaluation folds are opt-in and disabled here.
- Final status must distinguish implemented, offline-tested, and simulator-verified.

## Actual results

- **Completed:** `2026-10-20T20:42:25+08:00`
- **Scope actually executed:** data expansion and distribution audit only; no model training, held-out split, fold generation, or evaluation.
- **Runtime scene:** `calvin_scene_D` for every reset and rollout.
- **Reset pool:** 24 unique native Scene-D states selected from 408 native samples. Every reset is eligible for all six object-direction cells.
- **Reset pool:** `data/calvin/fixed_scene_D_push_resets_20261020_s20261020_v1.json`
- **Collection:** `data/calvin/generated_fixed_scene_D_push_20261020_s20261020_q3_v1/`
- **Collection size:** `282M`
- **Accepted:** 18 oracle-confirmed trajectories, exactly 3 per cell.
- **Rejected and isolated:** 33 failed attempts; none entered `accepted/`.
- **Accepted frames:** 484.
- **Evaluation folds emitted:** 0.
- **Observations:** fresh simulator `rgb_static` (`200×200×3`) and `rgb_gripper` (`84×84×3`), plus aligned state, 7-D actions, 6-D G, phase, post-contact, reset provenance, and oracle receipts.

| Cell | Attempted | Accepted | Failed | Accepted frames | Successful reset IDs |
|---|---:|---:|---:|---:|---:|
| `push_red_block_left` | 6 | 3 | 3 | 117 | 3 |
| `push_red_block_right` | 5 | 3 | 2 | 60 | 3 |
| `push_blue_block_left` | 9 | 3 | 6 | 70 | 3 |
| `push_blue_block_right` | 9 | 3 | 6 | 63 | 3 |
| `push_pink_block_left` | 14 | 3 | 11 | 69 | 3 |
| `push_pink_block_right` | 8 | 3 | 5 | 105 | 3 |
| **Total** | **51** | **18** | **33** | **484** | — |

### Read-back verification

- `validate_collection`: 18 accepted files verified, all six cells meet quota 3, collection complete.
- Runtime scenes read from accepted NPZ files: only `calvin_scene_D`.
- Reset source kinds read from accepted NPZ files: only `native_runtime_reset`.
- Accepted-cell counts are balanced.
- Every accepted cell uses three distinct successful reset IDs.
- G/action dimensions are 6/7 and all modalities are temporally aligned.
- No `folds/` evaluation output was generated.
- Final regressions passed:
  - `CALVIN fixed-scene reset pool unit contract: OK`
  - `CALVIN parametric push expert unit contract: OK`
  - `CALVIN scripted expert collector unit contract: OK`
- No collector or PyBullet Python process remained active after completion.

### Distribution artifacts

- `data/calvin/generated_fixed_scene_D_push_20261020_s20261020_q3_v1/data_distribution_summary.json`
- `artifacts/evidence/20261020T203037-calvin-fixed-scene-expansion/`

### Final hashes

- reset builder: `caa39024aa6589170c448168e0b0d1a16f50ff2e6ad61e7b48847b6301ac8ce2`
- collector: `4318034c900499d5fb485b7184e4763d8aa42d50e4ba435541235ca35e54ef92`
- distribution auditor: `01a590ec92c118e9130567508433503a77101a3cf3f1d3934bab99cb5f934da0`
- reset pool: `07a839dadfe5cc641008a789cd8c76c4d49e3832e8886f81b1c50484e3f80bba`
- collection manifest: `40f7fed77f2ac57674151927fdaf453d5d0c149f6e1f86944308d6b86eab6dda`
- distribution summary: `3a1b8b4f50c7997d50d3966e3dc17a621e3e0f3314070fe1b26ae01d110d22a5`

## Boundary and limitations

This operation expands only the first task family: planar left/right pushing for red, blue, and pink blocks. Three accepted resets per cell establish real diversity but are still a small dataset. Pink-left required 14 attempts for 3 successes, so controller robustness is not uniform across cells. No grasping, container, articulation, spatial-relation, LIBERO, or RoboTwin data were added in this operation, and no generalization claim is made.
