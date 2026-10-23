# Operation 20261020-CALVIN-SCRIPTED-EXPERT-COLLECTION

- **Operation ID:** `20261020-calvin-scripted-expert-collection`
- **Status:** `PASS — SIMULATOR VERIFIED (3 fixed scripted replays)`
- **Created:** `2026-10-20T18:16:57+08:00`
- **Repository:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Baseline commit:** `3d4aeba Record interaction volume parser closure`
- **Scope:** implement and exercise a success-gated CALVIN scripted-expert collector for one block-push, one drawer, and one slider task.

## Requested deliverable

1. Deterministic/reproducible task scripts or state-machine controllers for:
   - `push_blue_block_left`
   - `open_drawer`
   - `move_slider_left`
2. Fixed initial-state/seed manifest.
3. Bounded environment step loop and exact CALVIN task-oracle success receipt.
4. Persist accepted trajectories only after oracle success; retain failed-attempt diagnostics separately.
5. Per-step accepted data must include `robot_obs`, `scene_obs`, `rgb_static`, `rgb_gripper`, executed action, contact/interaction phase, compact `G`, task, seed/reset identity, and simulator/config provenance.
6. Read-back validation for shapes, lengths, phase/G alignment, manifest provenance, and success-gating.
7. Actual simulator evidence for every task family claimed complete.

## Pre-operation live evidence

- Git worktree already contains unrelated modified/untracked research artifacts; these must be preserved and not reset.
- GPU: NVIDIA GeForce RTX 4090, driver `595.99.02`, 24,564 MiB total, 130 MiB used at inspection.
- Disk: 1.3 TiB available on `/`.
- No active CALVIN collection/evaluation process was found.
- Last validated CALVIN runtime:
  - code: `/home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/code/calvin`
  - Python: `/home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/envs/calvin_reuse/bin/python`
  - dataset/config source: `/home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/data/calvin_debug_dataset`
  - previous EGL smoke recorded NVIDIA renderer and successful create/reset/step/close.
- Official filtered CALVIN files exist under `data/calvin/`; these remain source/calibration inputs and must not be relabeled as newly generated expert rollouts.

## Safety and isolation gates

- New outputs go to a dedicated generated-rollout directory; never overwrite official/filtered datasets.
- Do not modify the detached CALVIN runtime checkout unless a repository-local compatibility shim is unavoidable and separately recorded.
- Run strict TDD: failing test first, then minimal implementation, then targeted and regression tests.
- Fixed reset states must be recorded explicitly; no acceptance based only on filenames or policy intent.
- A rollout enters `accepted/` only when `task_oracle.get_task_info_for_set(start_info, final_info, {task})` confirms the exact requested task.
- Failed attempts remain under `attempts/` with diagnostics and are never labeled expert.
- No claim above `OFFLINE VERIFIED` until an actual simulator-generated trajectory is oracle-confirmed, saved, and read back.
- No claim of all three families until block, drawer, and slider each have at least one actual accepted trajectory.

## Planned evidence artifacts

- production collector/controller module and CLI;
- focused unit/integration tests with preserved RED/GREEN output;
- generated `collection_manifest.json` and per-attempt receipts;
- accepted `.npz` trajectories with dual-camera images and aligned metadata;
- read-back validator output and SHA-256 inventory;
- final Git diff restricted to files created/changed by this operation.

## Actual results

- **Completed:** `2026-10-20T18:32:56+08:00`
- **Evidence state:** `SIMULATOR VERIFIED` for exactly one fixed-reset rollout in each of the three requested families.
- **Implementation:**
  - `framework/scripts/data/collect_calvin_scripted_expert.py`
  - `framework/scripts/data/test_collect_calvin_scripted_expert.py`
- **Generated collection:** `data/calvin/generated_scripted_expert_20261020_s0/`
- **Manifest:** `data/calvin/generated_scripted_expert_20261020_s0/collection_manifest.json`
- **Method:** replay official CALVIN relative-action sequences from their exact recorded reset states as deterministic expert scripts; collect fresh simulator state and dual-camera observations; gate `accepted/` exclusively with CALVIN `get_task_info_for_set` for the exact task.

### Oracle-confirmed accepted rollouts

| Task | Family | Executed steps | Oracle success step | Final oracle progress | Phase counts | Artifact SHA-256 |
|---|---|---:|---:|---:|---|---|
| `push_blue_block_left` | block push | 21 | 20 | 0.100174 m | approach 6; contact/transport 15 | `58b7d82905a5c196ceea5f03ebc1cb84a1d6a4f7c0141a1f249b4feae19e4f0a` |
| `open_drawer` | drawer | 37 | 36 | 0.121255 joint displacement | approach 26; contact/transport 11 | `1abaa8db66c9902f0f29d001361f5fa69735d955218ec61893c2ee283e67ceeb` |
| `move_slider_left` | slider | 63 | 62 | 0.151569 joint displacement | approach 36; contact/transport 27 | `0560c03b4ecf9b3bd65a282f446b16e97fbed4246978021c879f4ef987dcf78e` |

Manifest totals: **3 accepted, 0 failed**. Each task receipt contains the exact requested task in `oracle_achieved_tasks`; no failed artifact was placed under `accepted/`.

### Stored schema and read-back

Each accepted `.npz` was read back with SHA-256 verification and contains aligned:

- `robot_obs` `(T,15)` and `scene_obs` `(T,24)`;
- `rgb_static` `(T,200,200,3)` and `rgb_gripper` `(T,84,84,3)`;
- relative `actions` `(T,7)`;
- compact `g` `(T,6)`;
- `phase`, `post_contact`, and state-grounded progress;
- task, fixed reset state, source frame IDs, script provenance, and success flag.

`phase/post_contact` is a state-grounded interaction-phase label: it switches when target object/joint motion first exceeds `0.002`, not from a force/torque contact sensor. Phase transitions were monotonic in all three accepted trajectories.

Fresh-render comparison against the source files found mean static-image differing-pixel fractions of `0.6268`, `0.7075`, and `0.6529`; therefore images were generated by the replay run rather than copied from the official episodes.

### Test evidence

- RED was observed for the missing collector and separately for the missing read-back validator.
- Unit contract: `CALVIN scripted expert collector unit contract: OK`.
- NumPy annotation compatibility regression: `OK`.
- State-only CALVIN environment regression: `OK` with NVIDIA EGL renderer.
- Collection process exit: `0`; read-back validator exit: `0`; compile/regression/hash step exit: `0`.
- Evidence directory: `artifacts/evidence/20261020T181657-calvin-scripted-expert/`.

### Scope boundary

This closes the requested **small-scale scripted collection vertical slice** for block, drawer, and slider. It does **not** yet provide a closed-loop controller robust to unseen reset states, multiple seeds, disturbances, or task variants: the current expert is a deterministic replay script calibrated from one official successful segment per task, while all accepted observations and oracle receipts are newly generated in the simulator.
