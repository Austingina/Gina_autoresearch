# OP-20261021-GINA-CALVIN-CONTACT-REACQUIRE-TDD-01

- **Created:** 2026-10-21T23:02:00+08:00
- **Stage:** CALVIN push physical-quality remediation
- **Target:** geometry-contact controller contact hysteresis and reacquisition
- **Status:** `ROLLED_BACK AFTER ONLINE QUALITY-GATE FAILURE`
- **Operator:** Hermes Agent

## Project root

`/home/ai-sz-a26394-u1/Gina_autoresearch`

## Baseline and motivation

- Q3 accepted 18/18 under the native CALVIN Oracle, but scale-up physical quality remains `HOLD`.
- 13/18 accepted trajectories lose robot-object contact after entering `push`.
- `scene_D_native_y_blue_0003` contains a 40-frame contact-loss run while the controller remains in `push`.
- Existing code enters `push` after one contact frame and never leaves `push` after contact loss.
- Existing target files are untracked; exact byte backups and hashes are mandatory before editing.

## Frozen change scope

Only:

- `framework/scripts/data/test_collect_calvin_geometry_contact_push_expert.py`
- `framework/scripts/data/collect_calvin_geometry_contact_push_expert.py`
- `.venv-calvin-unit/` as a Gina-local disposable Python/NumPy unit-test environment, because system Python has no NumPy and AST-named runtimes are out of scope

Do not change candidate geometry, action scales, native Oracle, G labels, candidate ranking, reset eligibility, existing Q3 artifacts, or manifests.

## Contract

1. `contact → push` requires two consecutive contact frames.
2. One contact dropout during `push` is ignored.
3. Two consecutive contact-loss frames enter `reacquire`.
4. `reacquire` targets the current live contact frame plus existing bounded preload depth.
5. Two consecutive contact frames are required before `reacquire → push`.
6. The command and receipt expose confirmation/loss counts, `reacquire_events`, and longest consecutive contact-loss run for audit.

Thresholds are prospectively frozen at two frames by reusing the controller's existing `stable_steps=2`; no lateral-drift acceptance threshold is introduced here.

## TDD sequence

1. Capture hashes and exact backups.
2. Run existing baseline unit contract.
3. Add two RED tests:
   - `test_push_requires_two_consecutive_contact_frames`
   - `test_push_ignores_single_dropout_but_reacquires_after_two_lost_frames`
4. Observe expected RED against current implementation.
5. Add the minimum state/counters and receipt fields.
6. Re-run target tests and complete unit contract.
7. Compile both files and compare operation-local diff.

## Online pilot boundary

No simulator rollout is part of this TDD operation. A later fresh-environment operation will run three fixed reset/candidate probes ×3 repeats. Oracle success remains trajectory acceptance; physical-quality metrics only gate scale-up.

## Evidence

`artifacts/evidence/20261021T230200-gina-calvin-contact-reacquire-tdd/`

## Actual results

- Existing baseline contract passed in the Gina-local `.venv-calvin-unit` with NumPy 1.26.4.
- RED was observed with exit code 1 at the expected old behavior: the first contact frame transitioned directly to `push`.
- Minimum implementation added two-frame contact confirmation, one-frame dropout tolerance, two-frame loss transition to `reacquire`, live reacquisition waypoints, and auditable counters.
- GREEN contract passed with exit code 0; both files compiled successfully.
- Native Oracle logic, G emission/dependency, candidate generation/scoring, reset eligibility, action scales, and existing Q3 artifacts were not changed.
- No simulator rollout was run in this operation; physical-quality scale-up remains `HOLD`.

Post-change hashes:

- controller: `d4e106637639b295a2eec0ad849bba55d34fcce1a342e5ef08bf1898a3fabec8`
- tests: `be651aba363f26529959241bdc8caf92003254b24db9b2b9a35e41c2d240fc43`

Receipts:

- `artifacts/evidence/20261021T230200-gina-calvin-contact-reacquire-tdd/tdd_receipt.json`
- `artifacts/evidence/20261021T230200-gina-calvin-contact-reacquire-tdd/operation.diff`
- `artifacts/evidence/20261021T230200-gina-calvin-contact-reacquire-tdd/red.log`
- `artifacts/evidence/20261021T230200-gina-calvin-contact-reacquire-tdd/green.log`

## Subsequent online disposition

The first frozen online probe (`blue0010 + h0.35_q1`, three fresh-environment repeats) produced native Oracle success 3/3, support preservation 3/3 and zero Z drop, but lateral drift was `0.1460630596 m` in every repeat. This exceeded the frozen baseline `0.1351911873 m` by about `0.010872 m`.

The state-machine change therefore failed its first online physical-quality gate. The remaining six probe rollouts were not started, the generated trajectories were marked diagnostic-only and excluded from formal data, and the contact-confirmation/reacquisition implementation and its tests were rolled back. The separately tested exact-probe CLI remains available.
