# Project Overview

This is one paper project with two experimental phases.

## Phase 0: IsaacLab / AutoResearch

Purpose: early validation of the G hypothesis and its failure modes.

Evidence to preserve:

- Learned G can mislead a policy when G has systematic error.
- GT-G, learned-G, and no-G controls exist in the historical records.
- S1/B1 visible-buffer, mask+depth, and random-cube generalization define the current no-G boundary question.

## Phase 1: CALVIN Dual-Camera Rebuild

Purpose: rebuild the same question in a standardized dual-view manipulation setup.

Key choices:

- Simulator/backend: CALVIN.
- Cameras: `rgb_static` and `rgb_gripper` from official CALVIN configuration.
- Initial policy path: strong no-G baseline with CALVIN dual-view data.
- Next policy path: flow matching / stronger temporal action decoder.
- Optional later path: reintroduce G-like information only after the no-G boundary is established.

## Paper Framing

The paper should not assume explicit G is always useful. A stronger claim is that geometry helps only when its reliability and coupling to the policy are controlled. The contribution is to characterize this boundary and design a safer representation pathway.
