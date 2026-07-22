# Reusable Interaction Geometry Experiment Plan

Created: 2026-07-02

## Central Question

Can a task-agnostic interaction geometry representation improve cross-task and low-data generalization for modern robot policies?

The project should not only ask whether G helps a single task. The stronger question is whether G captures reusable structure that transfers across manipulation tasks:

- end-effector/object spatial relations;
- contact and near-contact;
- containment/support relations;
- articulated-object progress;
- object-centric future mask/depth dynamics;
- action-conditioned changes in these relations.

## Working Thesis

Naively learned geometry can hurt when injected as a hard policy input. But a reusable interaction-geometry module, trained across source tasks and injected with reliability control, can improve held-out task generalization under strong flow/diffusion policies.

## Why This Scope Is Competitive

Recent flow-policy RL papers make modern generative action decoders a strong default. Recent mask/world/action-model papers show that predictive visual abstractions are important. We can integrate both trends, but focus on a more specific unsolved issue:

> What task-agnostic geometry should be shared across tasks, and how should it condition a policy without amplifying prediction errors?

## CALVIN Fit

The current server CALVIN setup exposes useful observations in the debug dataset:

- `rgb_static`, `rgb_gripper`;
- `depth_static`, `depth_gripper`;
- `robot_obs`, `scene_obs`;
- `actions`, `rel_actions`;
- language annotations and task labels.

CALVIN task families in `new_playtable_tasks.yaml` include:

- rotate block;
- push block;
- open/close drawer and move slider;
- lift block from table/slider/drawer;
- place object in slider/drawer;
- stack/unstack;
- turn on/off lightbulb or LED;
- push into drawer.

This makes CALVIN suitable for source/target splits by primitive, object, and interaction type.

## Data Sources

### Tier 1: Immediate

1. CALVIN debug dataset
   - Use for script validation, data inspection, G-label prototype, and smoke training.
   - Already available at `/mnt/nvme33n1/wuk_code/calvin_g/data/calvin_debug_dataset`.

2. Full CALVIN dataset
   - Main Phase 1 benchmark.
   - Store under `/mnt/nvme33n1/wuk_code/calvin_g/data` or symlink into `/mnt/nvme33n1/wuk_code/AutoResearch/data/calvin`.

3. Phase 0 IsaacLab AutoResearch logs/data
   - Use as historical evidence and controlled stress tests for random-cube/visible-buffer questions.

### Tier 2: Near-Term Transfer

1. LIBERO
   - Useful for multi-task and lifelong manipulation transfer after the CALVIN pipeline is stable.
   - Best role: external validation that reusable G is not CALVIN-specific.

2. MacroRobotics / geo_vla
   - Useful as a code-organization and policy-training reference.
   - Best role: reuse structure, not necessarily inherit every model choice.

### Tier 3: Later Real-Data Pretraining

1. BridgeData V2
   - Real-world manipulation data.
   - Best role: visual encoder or mask/geometry pretraining, not first-stage policy evaluation.

2. Open X-Embodiment / RT-X
   - Large multi-robot data source.
   - Best role: pretrained encoders or VLA baselines after the CALVIN story is stable.

## Reusable Pretrained Nodes

### Visual Backbones

- DINOv2 or similar self-supervised visual encoder for object-centric features.
- CLIP/SigLIP-style language-image features for language-conditioned variants.
- CALVIN's existing visual encoder path as the first local baseline.

Recommendation: start with the CALVIN baseline encoder, then add frozen DINOv2/SigLIP only after the no-G and G-label scripts are reliable.

### Mask And Object Nodes

- Simulator labels first when available.
- SAM 2 or GroundingDINO + SAM-style pipelines for external datasets and manual inspection.
- Color/object-name heuristics for early CALVIN blocks, only as a debugging bridge.

Recommendation: do not depend on SAM masks for the first scientific result. Use GT/sim/scene information where possible, then add learned masks as a robustness test.

### Depth And Geometry Nodes

- CALVIN `depth_static` and `depth_gripper` are immediate.
- Depth Anything V2 or similar monocular depth can be used later for real-data transfer.
- Object center, end-effector position, relative vector, distance, and near-contact can be derived from depth/scene state for prototypes.

### Action Decoders

- CALVIN imitation baseline first.
- Flow matching or diffusion action head second.
- ReinFlow/FPO-style online RL fine-tuning only after imitation baselines and G injection are stable.

### Proposed G Modules

1. `GEncoder`
   - Inputs: multi-view RGBD, robot state, optional language.
   - Outputs: object-centric slots or relational geometry tokens.

2. `GDynamics`
   - Inputs: G tokens and action chunk.
   - Outputs: future G, future masks, or future geometry deltas.

3. `GConfidence`
   - Inputs: G tokens and prediction residual history.
   - Outputs: confidence/reliability for gated policy conditioning.

4. `GPolicyAdapter`
   - Injects G into the action policy through FiLM, cross-attention, or gated residual conditioning.

## Core Experimental Axes

### Axis A: Representation

1. no-G RGB
2. no-G RGBD
3. RGBD + auxiliary mask/depth/geometry prediction
4. GT-G hard conditioning
5. learned-G hard conditioning
6. learned-G auxiliary-only
7. learned-G gated conditioning
8. shared reusable G frozen on target tasks

### Axis B: Action Model

1. CALVIN baseline action head
2. diffusion/flow action head
3. optional online RL fine-tuning

### Axis C: Generalization

1. in-task validation
2. held-out object/color
3. held-out primitive
4. held-out task family
5. low-data target adaptation
6. camera/occlusion/distractor perturbation

## Proposed CALVIN Splits

### Split 1: Object Transfer

Train G on red/blue block variants and evaluate policy transfer on pink block variants.

Purpose: test whether G represents interaction structure rather than color-specific policy shortcuts.

### Split 2: Primitive Transfer

Train G on push/lift/rotate, evaluate on place/stack/push-into-drawer.

Purpose: test whether geometry helps compose contact and object-motion primitives.

### Split 3: Articulation Transfer

Train G on slider/drawer interaction subsets, evaluate on held-out open/close directions or held-out articulated object progress.

Purpose: test whether G captures progress along constrained geometry.

### Split 4: Low-Data Target Tasks

Pretrain shared G on many source tasks. For each target task, train policy with 1%, 5%, 10%, and 100% demonstrations.

Purpose: strongest test for reusable interaction geometry.

## Minimal Experiment Ladder

### E0: Data And Visualization Validation

Goal: prove that we can inspect CALVIN episodes, task labels, RGBD, robot state, scene state, and candidate G labels.

Deliverables:

- episode visualization grid;
- task split manifest;
- candidate G-label dump;
- 20 manually checked videos/images.

### E1: Strong No-G Baseline

Goal: establish the boundary of RGB/RGBD policies without explicit G.

Conditions:

- RGB static+gripper;
- RGBD static+gripper;
- CALVIN baseline action head;
- flow/diffusion action head when ready.

### E2: GT-G Upper Bound

Goal: answer whether interaction geometry is useful if reliable.

Conditions:

- policy with GT geometry tokens;
- policy with GT geometry auxiliary loss only;
- compare to no-G under held-out tasks and low-data target adaptation.

### E3: Learned-G Failure Reproduction

Goal: reproduce the old IsaacLab lesson in CALVIN: hard learned-G can mislead.

Conditions:

- learned-G hard input;
- learned-G with controlled corruption;
- evaluate sensitivity to G error.

### E4: Reusable Shared-G

Goal: train G on source tasks, freeze or lightly tune it on target tasks, and measure transfer.

Conditions:

- task-specific G;
- shared G;
- frozen shared G;
- shared G + small target adapter.

### E5: Reliability-Aware G Injection

Goal: use G without over-trusting it.

Conditions:

- auxiliary-only G;
- gated G;
- cross-attention G;
- confidence-calibrated G.

### E6: Optional Online RL

Goal: test whether geometry improves exploration or fine-tuning after imitation.

Conditions:

- no-G flow policy + RL;
- shared-G flow policy + RL;
- geometry-aware reward/critic as optional branch.

## Metrics

### Policy Metrics

- success rate;
- chain length / long-horizon success;
- low-data AUC;
- target-task adaptation speed;
- robustness drop under perturbation;
- action smoothness and rollout stability.

### Geometry Metrics

- mask IoU;
- depth or center error;
- relative pose/vector error;
- contact or near-contact F1;
- articulated progress error;
- future-G prediction error;
- confidence calibration error.

### Causal/Sensitivity Metrics

- policy success under corrupted G;
- success gap between GT-G and learned-G;
- success gap between hard-G and gated-G;
- transfer gain from shared-G over task-specific-G;
- performance when G is frozen vs finetuned.

## First Two-Week Scope

1. Implement visualization scripts for CALVIN episodes and candidate G labels.
2. Build a task split manifest from CALVIN language/task annotations.
3. Define first candidate G vector from `robot_obs`, `scene_obs`, depth, and task family.
4. Run no-G RGBD baseline smoke/debug.
5. Train a tiny G predictor on debug data only to validate losses and visualizations.
6. Decide whether first GT-G uses scene-state vectors, rendered masks, or both.

## Decision Rules

If GT-G improves held-out or low-data target tasks, geometry is worth pursuing.

If learned-G hard conditioning hurts but auxiliary/gated G helps, the main method is reliability-aware reusable G.

If no-G RGBD flow policy already solves standard tasks, move evaluation toward held-out primitives, low-data transfer, occlusion, contact ambiguity, and camera perturbation.

If shared-G does not beat task-specific-G or no-G in transfer, the cross-task reuse hypothesis is weak and should be revised before expanding to real-world datasets.

## Key External References

- CALVIN: https://github.com/mees/calvin
- ReinFlow: https://arxiv.org/abs/2505.22094
- Mask World Model: https://arxiv.org/abs/2604.19683
- Flow Policy Optimization: https://arxiv.org/abs/2510.09976
- FlowPRO: https://arxiv.org/abs/2606.05468
- ForesightFlow: https://arxiv.org/abs/2606.04968
- OA-WAM: https://arxiv.org/abs/2605.06481
