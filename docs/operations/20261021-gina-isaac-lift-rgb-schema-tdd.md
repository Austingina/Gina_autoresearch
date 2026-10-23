# Operation 20261021 — Gina Isaac lift RGB schema TDD

- **Target:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Status:** `SOURCE TDD PASS — ISAAC RUNTIME/HDF5 VERIFICATION PENDING`
- **Scope:** add RGB observation terms to the existing lift collector schema while Isaac Sim 5.1 installs.
- **No simulator claim:** this operation is source-contract TDD only; real camera tensors and HDF5 remain pending runtime pilot/readback.

## Frozen contract

- Existing camera `data_types` must continue to contain `rgb`, segmentation and depth.
- Add normalized finite three-channel `camera_rgb` observation term.
- Add `obs/ee_rgb` and `obs/third_person_rgb` to the policy observation dictionary recorded by the existing `ActionStateRecorderManagerCfg` observation recorder.
- Preserve Depth, Mask, student state, teacher action and executed action fields.
- G remains absent from collection and admission.

## Method

RED → minimum source implementation → GREEN → compile → hash/diff receipt → later Isaac runtime/HDF5 verification.

## Actual result

- RED: exit `1`; four expected RGB-schema assertions failed while the existing camera annotator and recorder-observation checks passed.
- GREEN: exit `0`; 6 source-contract tests ran and reported `OK`.
- `py_compile`: exit `0` for all changed Python files.
- Added finite normalized float32 `camera_rgb`, exported it through lift `mdp`, and added `ee_rgb` plus `third_person_rgb` only to the formal collection observation group. Existing DAgger input groups were not changed.
- Confirmed from source that `PreStepFlatPolicyObservationsRecorder.record_pre_step()` returns `obs_buf["policy"]`; with `concatenate_terms=False`, the new RGB keys follow the existing nested observation export path.
- Evidence: `artifacts/evidence/20261021T232500-gina-isaac-lift-rgb-schema-tdd/`.
- This result does **not** claim that Isaac cameras initialized, RGB pixels are non-degenerate, or HDF5 contains the fields. Those are runtime pilot gates after installation completes.
