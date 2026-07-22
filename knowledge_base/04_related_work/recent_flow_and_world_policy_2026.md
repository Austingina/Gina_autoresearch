# Recent Flow-Policy And World-Model Papers

This note tracks recent papers that affect the AutoResearch story line after the move from IsaacLab/LSTM toward CALVIN/flow matching.

## Flow Policy Optimization

Source: https://arxiv.org/abs/2510.09976

Flow Policy Optimization formulates RL fine-tuning for pre-trained vision-language-action flow policies. It is especially relevant because it targets large flow-based VLA policies rather than only small task-specific policies.

Project implication: online RL for flow policies is becoming crowded. We should use it as a strong baseline/adaptation layer, not as the core novelty.

## FlowPRO

Source: https://arxiv.org/abs/2606.05468

FlowPRO focuses on exploration during online RL for flow matching policies. It identifies a practical issue in flow-policy RL: after supervised pretraining, stochastic flow-policy samples can remain too close to the demonstration distribution, limiting exploration.

Project implication: if we later fine-tune online in CALVIN or IsaacLab, exploration behavior will matter. Geometry-conditioned policies may look better or worse depending on whether they help exploration, not only whether they encode state.

## ForesightFlow

Source: https://arxiv.org/abs/2606.04968

ForesightFlow extends flow matching toward embodied decision making by reasoning over future observations and actions, with both supervised and reinforcement learning stages.

Project implication: predictive action/observation modeling is moving into the same territory as "world action model" and "Foundation G." Our story should not claim generic predictive modeling as the new contribution.

## Diffusion/Flow Policy Optimization Family

Source: https://arxiv.org/abs/2601.20701

Recent work on diffusion model policy optimization studies efficient RL fine-tuning for denoising generative policies. Together with ReinFlow and FPO, this suggests that generative action-policy RL is now a standard competitive direction.

Project implication: our method comparisons should include at least one strong generative action decoder. LSTM-only evidence is useful historical motivation, but not sufficient for a top-conference claim.

## World Action Model / Latent World Model Direction

Source examples:

- OA-WAM: https://arxiv.org/abs/2605.06481
- User-provided Mask World Model robot-policy paper.

These works learn latent or action-conditioned predictive structure for embodied decision making. They are close enough to "Interaction Geometry / Foundation G" that our framing must be specific.

Project implication: define G as an interaction-geometry bias or diagnostic variable, not as a generic latent world model. The core claim should be about when explicit geometry should be injected into a policy, not that latent predictive modeling is useful in general.
