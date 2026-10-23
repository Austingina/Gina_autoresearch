# OP-20261021-GINA-CALVIN-REACQUIRE-PILOT-01

- **Created:** 2026-10-21T23:18:00+08:00
- **Stage:** CALVIN contact-reacquisition online pilot
- **Target:** three fixed reset/candidate probes × three fresh-environment repeats
- **Status:** `FAILED AT FIRST FROZEN PROBE — STOPPED; CONTROLLER ROLLED BACK`
- **Operator:** Hermes Agent

## Project and runtime boundary

- Authoritative project root, scripts, records, output and evidence: `/home/ai-sz-a26394-u1/Gina_autoresearch`
- External simulator dependency only: `/home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g`
- The external runtime is not a project root and will not receive source or artifact changes.

## Frozen probes

1. `push_blue_block_forward`, reset `scene_D_native_y_forward_n200v1_blue_0010`, candidate `h0.35_q1`
2. `push_red_block_forward`, reset `scene_D_native_y_red_0004`, candidate `h0.65_q1`
3. `push_blue_block_forward`, reset `scene_D_native_y_blue_0003`, candidate `h0.35_q0`

Each probe must run three times, each attempt in a newly created CALVIN/PyBullet environment. Candidate search is disabled in probe mode.

## Predeclared gates

- Same-rollout native Oracle success: 9/9.
- Support preserved: 9/9.
- `z_drop <= 0.005 m`: 9/9.
- No one-frame dropout may trigger `reacquire`.
- `scene_D_native_y_blue_0003` must exhibit a real `reacquire → push` transition and must not continue a 40-frame blind push.
- Each high-drift probe must not exceed its frozen baseline lateral drift.
- Repeats must agree in Oracle result, state-transition sequence and quality metrics.
- Passing these gates only permits six-cell validation; physical-quality scale-up remains `HOLD` until a prospective lateral threshold is separately frozen.

## Required CLI remediation

Existing CLI searches candidates and attempts each reset once, so it cannot execute the frozen experiment. Add test-first, fail-closed options:

- exact reset-id filter;
- exact fixed candidate ID;
- repeat count per selected reset;
- unique attempt IDs for repetitions;
- manifest fields that prove probe mode and frozen selectors.

## Allowed source changes

- `framework/scripts/data/test_collect_calvin_geometry_contact_push_expert.py`
- `framework/scripts/data/collect_calvin_geometry_contact_push_expert.py`

No Oracle, G, geometry generation, controller gains, action scales or existing artifacts may change.

## Evidence

`artifacts/evidence/20261021T231800-gina-calvin-reacquire-pilot/`

## Actual results

### Probe CLI TDD

- Exact reset filtering, fixed candidate selection and deterministic repeats were added test-first.
- RED exit code: 1 (selectors absent).
- GREEN exit code: 0; compile passed.
- Exact-probe CLI was retained after the controller rollback.

### First online probe

- Probe: `scene_D_native_y_forward_n200v1_blue_0010 + h0.35_q1`.
- Three repeats, each using fresh CALVIN/PyBullet environments.
- Native Oracle: 3/3 success.
- Support preservation: 3/3.
- Maximum Z drop: `0.0 m`.
- Lateral drift: `0.14606305956840515 m` in all three repeats.
- Frozen baseline: `0.1351911872625351 m`.
- Regression: approximately `0.010872 m`.
- Reacquire events: 0 in all three repeats.

The prospective physical-quality gate failed. The red-0004 and blue-0003 groups were not run. The three generated trajectories remain diagnostic artifacts and are explicitly excluded from any formal manifest by:

`data/calvin/pilot_reacquire_blue0010_h035q1_r3_20261021_v1/EXCLUDED_FROM_FORMAL_DATA.json`

### Rollback verification

- Contact-confirmation/reacquisition state-machine changes and their tests were removed.
- Existing controller behavior plus exact-probe CLI unit contract passed.
- Compile passed.
- Final controller SHA-256: `d64d1345dd8d753bf5bf4003d64bf442c18f4564a3be331233821b38fa613133`.
- Final test SHA-256: `4d8a8350e31c763dca5762120614b9b9a49ac4d2f889ac51f332e4f11da4796a`.
- Scale-up status remains `HOLD`.
