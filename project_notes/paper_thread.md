# Paper Thread

The project should be treated as one paper thread, not as separate IsaacLab and CALVIN projects.

## Central Question

Does an interaction/geometric representation, called G in the current notes, improve visuomotor policy generalization, or can a strong visual/action model match it without explicit G?

## Phase 0: IsaacLab Evidence

The IsaacLab / AutoResearch phase provides early evidence and failure modes:

- Learned G can mislead the policy when G has systematic error.
- Existing comparisons include no-G, learned-G, and GT-G variants.
- S1/B1 visible buffer, mask+depth, and random-cube generalization are boundary tests.
- These experiments motivate why naive G is risky and why no-G baselines must be strong.

## Phase 1: CALVIN Rebuild

CALVIN is not a new project direction. It is a cleaner benchmark stage for the same research question:

- Simulator shift: IsaacLab -> CALVIN
- Camera structure: reuse CALVIN `rgb_static` + `rgb_gripper`
- Action model shift: LSTM-style decoder -> stronger flow-matching/action model candidates
- Goal: test whether the Phase 0 conclusions survive with a standardized dual-view manipulation pipeline.

## Working Hypothesis

The paper should not claim G is always beneficial. A stronger and safer framing is:

- Explicit or learned G is useful only when reliable and properly coupled to the policy.
- Learned G can hurt under representation error.
- Dual-view visual policies may already capture part of the needed interaction geometry.
- The contribution is to characterize this boundary and propose a better way to inject geometry if needed.
