# Phase 1 CALVIN Next Experiments

## Current Priority

Establish the no-G boundary before reintroducing G.

The immediate question is whether dual-camera CALVIN observations, proprioception, and a stronger action decoder can generalize without explicit G. This directly answers whether the IsaacLab learned-G failure was caused by G prediction error, by the LSTM action head, or by the task/simulator setup.

## Minimal Experiment Ladder

1. CALVIN no-G dual-camera smoke and debug training.
2. CALVIN no-G longer training run on the same observation interface.
3. CALVIN no-G with stronger object/task randomization if supported cleanly.
4. Flow-matching action head with the same no-G input.
5. GT/explicit geometry upper-bound.
6. learned-G with soft/gated injection.
7. optional online RL fine-tuning after the imitation setup is stable.

## Decision Rule

If no-G flow policy already generalizes strongly, then G should not be pitched as necessary for simple manipulation. The paper should instead focus on the regimes where no-G fails: occlusion, contact ambiguity, camera shift, object randomization, precise geometric constraints, or multi-stage loco-manipulation.

If no-G fails while GT/explicit geometry helps, then the project has a stronger case for interaction geometry as an inductive bias.

If learned-G hurts while GT-G helps, then the central contribution becomes reliability-aware geometry injection.
