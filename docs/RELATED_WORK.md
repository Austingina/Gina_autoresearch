# Related Work Positioning

## Unified policies

- **VLA** (RT-2, OpenVLA, π0, GR00T): end-to-end image+language → action. Strong but couples perception errors with control errors.
- **Unified RL**: single offline RL policy for all skills — struggles on contact-rich manipulation (our observation).

## Goal-conditioned methods

- **GCRL / HER / GS-HER:** goal as state subset or predicate; we extend to `(pose, interaction_type)` with VFM grounding.
- **CoGHP:** chain of latent subgoals for long-horizon offline RL — complementary; we use explicit interpretable goals.
- **TTGS:** test-time graph search over subgoals — could wrap our skill router at inference.

## Affordance

- Affordance → trajectory (old view)
- Affordance → contact_point ≈ **goal** (our view)
- Affordance = special case of foundation goal representation

## Embodied tracking

- **EVT (ours, ECCV 2024):** VFM + offline RL for tracking — Stage 1 of paper arc
- **HIEVT:** instruction → goal → RL — precursor to FGR

## Imitation for manipulation

- **DAgger:** our choice for manipulation skills (lift, pick, pull, push)
- **Diffusion Policy:** alternative executor — cite as future skill backend

## Navigation

- **VLN / StreamVLN:** language → navigation; our VLN flywheel in unrealzoo-gym is a reference pipeline
- Could connect via `interaction_type=reach` goals

## Gap statement (for intro)

> Existing work unifies either perception and control (VLA) or control across tasks (unified RL), but not **perception into a shared goal space** with **heterogeneous skill executors**. We propose Foundation Goal Representation to decouple what to interact with from how to execute.

## Key citations to add

- Zhong et al. EVT ECCV 2024
- Brohan et al. RT-2
- Chi et al. Diffusion Policy
- OGBench / GS-HER
- CoGHP (ICLR 2026)
- Grounding DINO, SAM 2, DINOv2
