# Story Line Update After ReinFlow And Mask World Model

## Main Update

The project should no longer be framed as "can we build a Foundation G that beats raw manipulation policies?" That framing is too close to world models, world-action models, and latent representation learning.

The stronger framing is:

> Modern visual and flow-based action policies can learn much of the needed geometry implicitly. We study the boundary where explicit interaction geometry helps, hurts, or becomes redundant.

This keeps the original IsaacLab evidence useful while making the CALVIN direction coherent.

## Two Orthogonal Axes

Axis 1: representation.

- no-G dual camera
- mask/depth/proprioception baselines
- GT or explicit geometry
- learned G
- soft/gated interaction-geometry conditioning

Axis 2: action model and adaptation.

- old LSTM baseline
- CALVIN imitation baseline
- flow matching policy
- optional online RL fine-tuning

The paper should not mix these axes. LSTM-to-flowmatching is an action-decoder upgrade. IsaacLab-to-CALVIN is a simulator/benchmark upgrade. The paper question remains the G/no-G interaction-geometry boundary.

## Revised Claim

We should avoid claiming:

- "A new general world model."
- "A new flow-policy RL algorithm."
- "G always improves manipulation."

We can more credibly claim:

- learned G can mislead policies when injected naively;
- strong no-G visual policies may already cover simple manipulation geometry;
- explicit interaction geometry becomes useful only under certain reliability and task-complexity regimes;
- flow matching and online RL provide a stronger testbed for this question than the earlier LSTM setting.

## Why This Is More Defensible

ReinFlow, FPO, FlowPRO, and ForesightFlow make flow-policy RL a competitive but crowded line. Mask/world-model work makes generic latent predictive representation a crowded line. Our project can sit between them by asking a more surgical question:

> What geometry should be explicit when the action model is already strong?

This is a cleaner top-conference story because it turns our negative results into evidence, not failure.

## Recommended Experiment Order

1. Finish the CALVIN no-G dual-camera baseline.
2. Implement or reuse a flow-matching action head under the same CALVIN observations.
3. Establish the no-G boundary under object/task randomization before adding learned G.
4. Add GT/explicit geometry as an upper-bound or diagnostic condition.
5. Reintroduce learned G only with soft/gated injection or reliability controls.
6. Treat online RL fine-tuning as a later stress test, inspired by ReinFlow/FPO/FlowPRO, rather than the first milestone.

## Updated Top-Conference Bet

The easier path to a credible top-conference submission is not "Foundation G as a broad model." It is a careful empirical and methodological paper:

> Interaction geometry under strong generative robot policies: when explicit geometry helps, when learned geometry hurts, and when no-G visual policies are enough.

This story is narrower, but it is harder to dismiss as a duplicate of world-action models or flow-policy RL.
