# ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning

Source: https://arxiv.org/abs/2505.22094

Status: NeurIPS 2025; arXiv version updated in 2026.

## Core Idea

ReinFlow studies how to fine-tune a flow matching policy with online reinforcement learning. Its key contribution is not a new visual representation, but an RL fine-tuning recipe for an action-generation model that was originally trained with a flow matching objective.

The paper proposes Flow-GRPO, which combines Group Relative Policy Optimization with flow matching policy training. The claimed result is that flow matching policies can be improved online after imitation pretraining, with stronger or comparable performance to diffusion-policy RL baselines across simulated tasks, and a real-world few-shot result from a small number of demonstrations.

## Why It Matters Here

This paper validates the direction of replacing the old LSTM action head with a stronger generative action decoder. Flow matching is now a mainstream action-model choice, and online RL fine-tuning is a plausible next stage after CALVIN imitation training.

It also creates a novelty risk. If our paper only says "we fine-tune a flow matching robot policy with RL", it will collide directly with ReinFlow and related work. The action model should therefore be treated as the strong baseline or training substrate, not the main conceptual claim.

## Implications For AutoResearch

1. Keep the representation question separate from the action-decoder question.
2. Use flow matching to reduce the chance that LSTM-specific failure modes dominate conclusions.
3. Do not claim novelty on online RL for flow policies unless we implement a meaningfully different objective.
4. A stronger story is: "Under modern flow-policy and optional online-RL training, when does explicit interaction geometry still help or hurt?"

## Experiment Hooks

- CALVIN no-G dual-camera imitation baseline.
- CALVIN flow-matching policy baseline under the same observation and action interface.
- Optional online RL fine-tuning stage after the imitation baseline is stable.
- Compare no-G, explicit/GT-G, and learned-G variants under the same flow-policy training recipe.
