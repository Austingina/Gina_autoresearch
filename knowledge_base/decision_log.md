# Decision Log

## 2026-07-02: Unified Project Root

Decision: use `/mnt/nvme33n1/wuk_code/AutoResearch` as the canonical active project root.

Reason: root disk is limited; NVMe has enough space for datasets, logs, checkpoints, and CALVIN outputs.

## 2026-07-02: Treat CALVIN as Phase 1

Decision: CALVIN is not a separate project. It is Phase 1 of the same G/no-G research question.

Reason: IsaacLab Phase 0 provides motivation and failure evidence; CALVIN provides a cleaner dual-camera benchmark.

## 2026-07-02: Start CALVIN With No-G Dual Camera

Decision: begin with `rgb_static + rgb_gripper` no-G baseline.

Reason: learned-G failure means no-G boundary must be established first.

## 2026-07-02: Add ReinFlow And Mask World Model To Related Work

Decision: track ReinFlow as evidence that flow matching plus online RL is a strong modern action-policy direction, and track Mask World Model as evidence that broad predictive/latent world-model framing is crowded.

Reason: this sharpens the paper story. We should not claim novelty as a generic world model or generic flow-policy RL method. The defensible claim is interaction geometry under strong policies.

## 2026-07-02: Separate Representation Axis From Action-Model Axis

Decision: maintain two independent experiment axes:

- representation: no-G, mask/depth, GT/explicit G, learned G, soft/gated G;
- action model: LSTM, CALVIN imitation baseline, flow matching, optional online RL fine-tuning.

Reason: the earlier IsaacLab/LSTM results are still useful, but top-conference evidence should show whether the G/no-G conclusion survives stronger action decoders.
