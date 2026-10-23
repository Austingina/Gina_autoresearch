# Operation 20261021 — Gina LIBERO middle-drawer expert contract correction

- **Operation ID:** `20261021-gina-libero-middle-drawer-expert-correction`
- **Target:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Status:** `PURE-LOGIC TDD PASS — RUNTIME/ROLLOUT NOT VERIFIED`
- **Reason:** main-thread review found that the initial pure-logic tests encoded geometry/control assumptions inconsistent with the pinned LIBERO source audit.

## Frozen corrections

1. Use pinned XML handle center `[-0.00278, -0.10161, 0.11041]`.
2. Resolve one explicit target joint supplied by the future `object_sites_dict` adapter; do not require the whole MuJoCo model to contain only one slide joint.
3. Require slide type, range approximately `[-0.16, 0.01]`, and owning body name containing `cabinet_middle`.
4. Reject non-finite actions instead of silently replacing them.
5. Use provisional robosuite convention `-1=open, +1=close`; the runtime pilot must still perform a one-step gripper-qpos sign smoke before rollout.
6. Latch the PULL target when grip closure is confirmed so the target does not chase the moving live handle indefinitely.
7. Continue to forbid controller-generated success labels and all dependence on reward/done/G.

## Method

RED → minimum implementation → GREEN → compile → hash/diff receipt. No LIBERO installation or rollout in this operation.

## Actual result

- RED: exit `1`; failures covered the missing explicit joint selector, wrong XML handle constant, silent non-finite action replacement, reversed gripper convention and unlatchable pull target.
- GREEN: exit `0`; 10 focused tests ran and reported `OK`.
- `py_compile`: exit `0`.
- Corrected implementation SHA-256: `d998123a2d42d91d4a52ac007c92dd1d01f5cdd1bbfc84826aff214f459dd77a`.
- Corrected test SHA-256: `bda3a27d9c5fe8813b25c40c780568dbcd3af951d98a74fbc0eeef996c0b3bc1`.
- Evidence: `artifacts/evidence/20261021T232000-gina-libero-middle-drawer-correction/`.
- No runtime adapter, simulator rollout, native `_check_success()` call, recorder or HDF5 was claimed or executed.
