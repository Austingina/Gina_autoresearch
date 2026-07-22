# Core Hypotheses

## H1: Learned G Risk

A learned G representation can hurt policy learning when its prediction error is structured and the policy over-trusts it.

Evidence source: Phase 0 IsaacLab learned-G runs and notes.

## H2: Strong No-G Boundary

Dual-view visual policies may absorb enough interaction geometry for simple manipulation tasks, especially when the camera setup is stable and proprioception is available.

Evidence source: Phase 1 CALVIN no-G dual-camera experiments.

## H3: G Is Useful Under Reliability Controls

Explicit G should help most when it is either ground truth, high-confidence, uncertainty-aware, or injected as a soft bias rather than a hard policy input.

Evidence source: future CALVIN and IsaacLab ablations.

## H4: Action Decoder Matters

The LSTM-era results should not be overinterpreted until a stronger action decoder, such as flow matching, is evaluated under the same data and observation conditions.

Evidence source: planned Phase 1 action-decoder ablations.

## H5: Flow Matching Is A Strong Testbed, Not The Main Claim

ReinFlow, Flow Policy Optimization, FlowPRO, and related papers make online RL for flow policies a competitive but crowded direction. Flow matching should be used to strengthen the evaluation of the G/no-G question, not as the primary novelty claim.

Evidence source: `04_related_work/reinflow.md` and `04_related_work/recent_flow_and_world_policy_2026.md`.

## H6: Foundation G Must Avoid Generic World-Model Framing

Mask/world-model and world-action-model papers overlap with broad claims about predictive latent representations. To stay differentiated, G should be framed as interaction geometry: an explicit, reliability-aware bias for manipulation contact and spatial relations.

Evidence source: `04_related_work/mask_world_model_robot_policy.md` and `05_paper/story_line_2026_update.md`.
