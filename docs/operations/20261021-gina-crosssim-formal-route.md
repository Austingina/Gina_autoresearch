# OP-20261021-GINA-CROSSSIM-FORMAL-ROUTE-01

- **Created:** 2026-10-21T22:02:58+08:00
- **Stage:** formal cross-simulator data-expansion route correction
- **Target:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Status:** `PASS`
- **Operator:** Hermes Agent

## Authoritative user direction

All implementation, runtime validation, operation records, and generated artifacts belong to `Gina_autoresearch`. The formal route is:

1. Isaac: restore and validate lift production; wire the existing drawer state machine as a second recorded task.
2. CALVIN: continue small-scale push physical-quality remediation.
3. LIBERO: implement one online expert for the middle-drawer task.
4. RoboTwin: install SAPIEN/assets and run a native-Oracle `place_empty_cup` pilot.
5. Do not claim Isaac multi-task bulk production until at least lift and drawer both pass online collection/readback gates.

## Baseline

- Gina HEAD: `3d4aeba1badaa3d6ce781033e0d9187729dfec54`.
- Worktree is pre-dirty; no unrelated file may be reverted or reformatted.
- GPU at 2026-10-21T22:02:58+08:00: RTX 4090, 130 MiB used, 0% utilization.
- No LIBERO training or simulator collector process is running.
- Gina contains `simulators/isaaclab/scripts/tools/lenient_teacher_datacollection_maskdepth.py`.
- Gina contains `simulators/isaaclab/scripts/environments/state_machine/open_cabinet_sm.py`.
- Drawer currently has a state machine but is not yet a recorded/validated second production task.
- CALVIN Q3 integrity/Oracle/repeatability passed earlier, while physical-quality scale-up remains on hold.

## Goal

Amend `docs/CROSS_SIM_NATIVE_COLLECTION.md` so it is the current formal route and explicitly binds every claim to the Gina project root.

## Non-goals

- No simulator launch.
- No environment installation.
- No trajectory generation.
- No claim that Isaac drawer, LIBERO, or RoboTwin is already simulator-verified.

## Allowed change

- `docs/CROSS_SIM_NATIVE_COLLECTION.md`
- this operation record and a new evidence receipt

## PASS criteria

- The formal route appears exactly in the shared document.
- Isaac lift and drawer maturity are represented separately.
- The stale statement that LIBERO training blocks the GPU is not retained as current fact.
- Multi-task bulk collection remains gated on two independently validated Isaac tasks.

## Evidence

`artifacts/evidence/20261021T220258-gina-crosssim-formal-route/`

## Actual results

The formal route was written into `docs/CROSS_SIM_NATIVE_COLLECTION.md` under an explicit Gina-only project-root invariant.

- post-change SHA-256: `89d3c8449a6236782d3d59318baa3194d954820e2ab5480bad1b8506c6fea107`
- five ordered stages present: yes
- Isaac lift and drawer represented as separate maturity rows: yes
- two-task multi-task promotion gate present: yes
- stale LIBERO GPU-blocker statement corrected: yes
- simulator launches or generated trajectories: none

Evidence:

- `artifacts/evidence/20261021T220258-gina-crosssim-formal-route/formal_route.diff`
- `artifacts/evidence/20261021T220258-gina-crosssim-formal-route/receipt.json`

Next allowed operation: a Gina-root-only Isaac runtime and persisted-lift-pilot preflight. No bulk collection is authorized yet.
