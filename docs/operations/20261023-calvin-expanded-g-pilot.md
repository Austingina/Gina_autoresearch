# Operation 20261023 — CALVIN expanded-data G closed-loop pilot

- **Operation ID:** `20261023T155803-calvin-expanded-g-pilot`
- **Project root:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Git HEAD:** `3d4aeba1badaa3d6ce781033e0d9187729dfec54`
- **Status:** COMPLETE — diagnostic gate blocked formal scale-up

## Purpose

Begin measured guidance-field evaluation on the expanded official CALVIN arm before attempting cross-simulator claims.

## Frozen model and held-out scope

- Fold/model: `new_block_bridge_pink__art_hold_slider_left`
- Checkpoint: `experiments/phase1_calvin/2026-10-20_same_g_full_formal/retry1/new_block_bridge_pink__art_hold_slider_left/chunk_bc_minimal_g_concat/policy.pt`
- Training source: expanded CALVIN multifold official demonstrations.
- Held-out tasks used in this pilot:
  - `push_blue_block_left`
  - `push_red_block_right`
- These tasks are absent from this fold's source set according to `formal_plan.json`.

## Matched protocol

- One immutable cache of 20 standard CALVIN initial states, reused byte-for-byte across conditions.
- Episode horizon: 160.
- Policy sample seed: 20260814.
- Action execution horizon: 1.
- Online guidance geometry: `behavior_block_volume`.
- Tool clearance: 0.018012 m.
- Directed object-motion switch threshold: 0.002 m.
- Native CALVIN task Oracle supplies success.
- Full rollout traces retained for all 20 episodes per condition.

## Conditions

1. `normal G`
2. `zero G`
3. `sign_flip G`

Only `query_g_control` changes; model, reset order, task list, simulator, horizon, policy seed, and evaluator stay fixed.

## Gates

- Environment/CUDA/EGL smoke already passed after reboot.
- Do not claim formal generalization from this one-fold 20-reset pilot.
- Report task-level success numerators/denominators and contact-switch counts.
- Continue to larger CALVIN evaluation only if all three conditions finish and artifacts read back.
- LIBERO/RoboTwin/ManiSkill/VIMA remain blocked on per-platform G/action/Oracle adapters; official successful recordings alone are not closed-loop evidence.

## Evidence root

`artifacts/evidence/20261023T155803-calvin-expanded-g-pilot/`

## Diagnostic continuation

The initially selected `new_block_bridge_pink__art_hold_slider_left` fold completed all matched conditions but scored 0/40 in each. Historical `generated_query` results show that the same checkpoint was already 0/100 on each selected held-out block task, so this fold cannot diagnose whether repaired online G improves a capable policy. A second matched pilot therefore uses `new_block_bridge_pink__art_hold_slider_right`, whose historical generated-query held-out results were 48/100 on `push_blue_block_left` and 42/100 on `push_red_block_right`; all other controls remain matched and outputs are stored under `diagnostic_known_capable_fold/`.

## Result and gate decision

- Initial incapable-fold pilot: normal/zero/sign-flip were each 0/40; normal produced 10 traced switches but no success.
- Known-capable-fold matched repaired-G pilot: normal 2/20, zero 0/20, sign-flip 0/20; normal produced 6/20 phase switches.
- Exact same-reset legacy control: generated-query 5/20 (blue-left 3/10, red-right 2/10).
- Paired outcomes: both succeed 2, legacy-only succeed 3, repaired-only succeed 0, both fail 15.
- All evaluator exits were zero and fixed-reset/summary hashes passed readback.
- Decision: do not scale the old checkpoint under repaired G. Rebuild training labels/conditions with the same repaired online definition, retrain, then repeat this immutable-reset protocol before formal multi-fold evaluation.
