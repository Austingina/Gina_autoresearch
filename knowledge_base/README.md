# AutoResearch Knowledge Base

This knowledge base keeps the paper logic synchronized across the IsaacLab and CALVIN phases.

The central question is:

> When does a reusable G-like interaction geometry representation help cross-task visuomotor policy generalization, and when can a strong visual/action model learn the needed geometry without explicit G?

## Map

```text
00_project_overview.md
01_core_hypotheses.md
02_concepts/
  interaction_geometry.md
  learned_g_failure.md
  no_g_boundary.md
03_experiment_evidence/
  phase1_calvin_findings.md
  phase1_calvin_next_experiments.md
  phase1_calvin_visual_smoke.md
  phase1_calvin_schema_visual_pass.md
  phase1_calvin_g_predictor_smoke.md
  phase1_calvin_reusable_g_pipeline_dryrun.md
04_related_work/
  geo_vla_structure_review.md
  reinflow.md
  mask_world_model_robot_policy.md
  recent_flow_and_world_policy_2026.md
05_paper/
  story_line.md
  story_line_2026_update.md
06_experiment_plans/
  reusable_interaction_geometry_plan.md
decision_log.md
```

Use this folder as the source of truth for paper claims, evidence, risks, and experiment decisions. Framework logs and raw outputs live under `experiments/`.

## Current Reading Of The Project

The project should be maintained as one continuous paper direction:

- Phase 0 IsaacLab: early evidence that naive learned G can hurt.
- Phase 1 CALVIN: cleaner dual-camera benchmark to establish the no-G boundary.
- Reusable-G direction: train task-agnostic interaction geometry across source tasks and test transfer to held-out tasks.
- Action-decoder upgrade: LSTM to flow matching.
- Optional adaptation upgrade: imitation to online RL fine-tuning.

The paper novelty should remain centered on reusable interaction geometry under strong policies, not on inventing another generic world model or another flow-policy RL method.
