# Paper Outline — Anchor-Conditioned Visual Guidance Fields

## Title (draft)

**Anchor and Field: A Visual Guidance Representation for Policy Learning across Tracking and Manipulation**

## Abstract bullets

- Single masks anchor target acquisition; fail on object relocation (lift post-grasp)
- Propose g_t = (M, G): anchor mask + guidance field for policy learning
- Controlled ablations B1 vs B3 on Isaac lift; HIEVT tracking extension
- WM optional for learning G — not required

## 1. Introduction

- Unified visual task interface question
- Isaac observation: grasp OK, wander after grasp
- Claim: policy-validated (M, G)

## 2. Related Work

- Mask state (EVT), bbox goals (HIEVT), fields (TarGF, DualGF, HumanoidPF)
- MWM as field learner (not our interface)
- VLAs / affordance (contrast)

## 3. Method

### 3.1 VisualPolicyGoal

- M_t anchor, G_t field types (object_vector, dense_2d, region_pull)
- a_t = pi(o_t, M_t, G_t)

### 3.2 Learning G

- Rule oracle (Phase 1)
- Demo trajectories (Phase 2)
- Optional WM derive (ablation)

### 3.3 Policy training

- Isaac DAgger / HIEVT offline RL — same pi, different g_t

## 4. Experiments

### 4.1 Lift (main)

- B0–B3 table: success, post_grasp_wander
- Qualitative trajectories

### 4.2 Tracking

- HIEVT bbox vs field

### 4.3 Learned field (B7)

- vs rule, vs WM (B8 if run)

## 5. Discussion

- Mask anchor vs full goal
- When WM helps
- Limits

## Main figure

B1 vs B3: success rate + post-grasp wander (lift)
