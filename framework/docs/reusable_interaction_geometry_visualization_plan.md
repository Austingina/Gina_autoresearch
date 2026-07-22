# Reusable Interaction Geometry Visualization Plan

Created: 2026-07-02

This document defines the manual visualization and inspection scripts needed before training reusable interaction geometry models.

## Output Conventions

All scripts should write to:

```text
artifacts/visual_checks/<timestamp>_<script_name>/
```

Each script should save:

- `manifest.json`: input paths, episode ids, random seed, git commit if available;
- `summary.md`: what was generated and what to inspect;
- PNG grids for quick inspection;
- MP4 videos for temporal checks when applicable.

## Script 1: `visualize_calvin_episode.py`

Purpose: inspect raw CALVIN data before adding any G logic.

Inputs:

- dataset root;
- split: training/validation;
- episode ids or random sample count;
- optional task filter.

Panels:

- `rgb_static`;
- `rgb_gripper`;
- `depth_static` colormap;
- `depth_gripper` colormap;
- robot state summary;
- action / relative action curve;
- language/task label if available.

Manual checks:

- RGB and depth are aligned and nonblank;
- gripper camera is useful near contact;
- task labels match visible behavior;
- episode boundaries are correct.

## Script 2: `make_calvin_task_split_manifest.py`

Purpose: create reproducible source/target task splits.

Inputs:

- CALVIN language annotation files;
- task YAML file;
- split recipe name.

Outputs:

- `splits/<recipe>.json`;
- task family counts;
- episode index ranges;
- leakage report.

Initial recipes:

- `object_transfer_red_blue_to_pink`;
- `primitive_transfer_push_lift_rotate_to_place_stack`;
- `articulation_transfer_slider_drawer`;
- `low_data_target_1_5_10_100`.

Manual checks:

- no target task appears in source split when testing held-out primitive/family;
- object/color transfer split keeps primitive distribution roughly comparable;
- low-data target subsets are deterministic.

## Script 3: `extract_candidate_g_labels.py`

Purpose: derive first-version G labels from available CALVIN observations.

Inputs:

- dataset root;
- split manifest;
- label mode: `scene_state`, `rgbd_proxy`, `sim_rendered_mask`, or `hybrid`.

Candidate G fields:

- end-effector pose and gripper state;
- object/articulated state from `scene_obs` when available;
- object-to-EE relative vectors;
- object-to-object relative vectors;
- near-contact/proximity flags;
- drawer/slider progress;
- future delta-G for horizons 1, 4, 8, 16.

Outputs:

- per-episode `.npz` or sharded arrays;
- `g_schema.json`;
- coverage and missing-value report.

Manual checks:

- object centers do not jump unexpectedly;
- contact/proximity labels fire near plausible moments;
- articulated progress matches drawer/slider motion;
- future-G target uses correct temporal offset.

## Script 4: `visualize_interaction_geometry.py`

Purpose: visually inspect candidate G labels.

Inputs:

- episode data;
- extracted G labels;
- optional camera calibration/projector if available.

Panels:

- RGB static/gripper;
- depth static/gripper;
- projected object/EE points if projection is available;
- relative vector table;
- contact/proximity heatmap over time;
- articulated progress curve;
- future-G target preview.

Manual checks:

- G labels correspond to visible task behavior;
- no obvious sign/axis inversion;
- no stale labels after object movement;
- EE/object proximity aligns with grasp/contact phases.

## Script 5: `visualize_g_prediction.py`

Purpose: debug G pretraining before using G in policy.

Inputs:

- trained G checkpoint;
- validation episodes;
- G label store.

Panels:

- predicted vs GT future G;
- error curves by horizon;
- confidence score if available;
- top failure cases sorted by error.

Manual checks:

- prediction quality degrades smoothly with horizon;
- confidence is lower on visibly ambiguous frames;
- failures cluster in interpretable cases rather than random noise.

## Script 6: `visualize_policy_rollouts.py`

Purpose: compare no-G, GT-G, learned-G, and gated-G policies.

Inputs:

- rollout directories;
- policy names;
- task filter.

Panels:

- synchronized side-by-side videos;
- success/failure marker;
- predicted G confidence overlay;
- action norm curve;
- task progress event markers.

Manual checks:

- learned-G failure modes are visible and explainable;
- gated-G reduces obvious over-trust failures;
- GT-G advantage appears in contact/occlusion/low-data cases.

## Script 7: `inspect_g_confidence_and_corruption.py`

Purpose: evaluate whether reliability-aware injection is justified.

Inputs:

- G predictions;
- policy rollouts;
- corruption schedule.

Views:

- success vs G corruption level;
- confidence vs true G error;
- examples where hard-G fails and gated-G recovers;
- examples where gating ignores useful G.

Manual checks:

- confidence is not a cosmetic scalar;
- gating decisions match actual reliability often enough to matter;
- corruption sensitivity is lower for gated-G than hard-G.

## First Implementation Order

1. `visualize_calvin_episode.py`
2. `make_calvin_task_split_manifest.py`
3. `extract_candidate_g_labels.py`
4. `visualize_interaction_geometry.py`
5. `visualize_g_prediction.py`
6. `visualize_policy_rollouts.py`
7. `inspect_g_confidence_and_corruption.py`

## First Smoke Command Targets

The first versions should run on the debug dataset:

```bash
DATA_ROOT=/mnt/nvme33n1/wuk_code/calvin_g/data/calvin_debug_dataset
OUT_ROOT=/mnt/nvme33n1/wuk_code/AutoResearch/artifacts/visual_checks
```

The scripts should use the H200-compatible CALVIN environment:

```bash
/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python
```

## Acceptance Criteria

Before any new G-policy training, we need at least:

- 20 manually inspected raw episode visualizations;
- one verified task split manifest;
- one verified candidate G schema;
- 20 manually inspected G-label overlays;
- one tiny G-prediction validation grid;
- one no-G vs G rollout comparison grid when policy checkpoints exist.
