# Phase 1 CALVIN Schema And Visual Pass

Date: 2026-07-02

## Goal

Move from a small smoke test to a schema-aware visual pass over all annotated sequences in the current CALVIN debug dataset.

## Scene Schema Verification

The visual/G-label scripts now read CALVIN scene schema from the dataset hydra config:

```text
training/.hydra/merged_config.yaml
validation/.hydra/merged_config.yaml
```

For the current debug dataset, the schema is:

```text
scene: calvin_scene_D
scene_obs[0:6]:
  base__slide
  base__drawer
  base__button
  base__switch
  lightbulb
  led

scene_obs[6:12]:  block_red xyz + euler orientation
scene_obs[12:18]: block_blue xyz + euler orientation
scene_obs[18:24]: block_pink xyz + euler orientation
```

This was cross-checked against CALVIN source:

- `calvin_env/scene/play_table_scene.py`: `get_obs()` concatenates door, button, switch, light, and movable-object states.
- `calvin_env/robot/robot.py`: `robot_obs` is TCP position, TCP euler orientation, gripper opening, arm joints, and gripper action.
- `calvin_models/calvin_agent/evaluation/utils.py`: initial-condition construction uses the same 24D `scene_obs` layout.

## Generated Outputs

Dataset summary:

```text
artifacts/dataset_summaries/calvin_debug_schema_summary
```

Split manifests:

```text
artifacts/task_splits/debug_object_transfer_red_blue_to_pink.json
artifacts/task_splits/debug_primitive_transfer_push_lift_rotate_to_place_stack.json
artifacts/task_splits/debug_articulation_transfer_slider_to_drawer.json
artifacts/task_splits/debug_articulation_transfer_drawer_to_slider.json
artifacts/task_splits/debug_low_data_all_tasks.json
```

Raw visual checks:

```text
artifacts/visual_checks/debug_training_raw_all
artifacts/visual_checks/debug_validation_raw_all
```

Schema-aware G labels:

```text
artifacts/g_labels/debug_training_all_schema_aware
```

Schema-aware G visual checks:

```text
artifacts/visual_checks/debug_training_g_all_schema_aware
```

## Dataset Summary

Training split:

- episode files: 2771
- annotations: 9
- annotation frames: 503
- task families: light 3, push 1, place 2, lift 3
- object colors: none 5, pink 1, red 2, blue 1

Validation split:

- episode files: 1675
- annotations: 8
- annotation frames: 472
- task families: lift 4, light 2, place 1, push 1
- object colors: red 1, none 3, blue 3, pink 1

## Split Recipe Result On Debug Data

```text
object_transfer_red_blue_to_pink: source 3, target 1
primitive_transfer_push_lift_rotate_to_place_stack: source 4, target 2
articulation_transfer_slider_to_drawer: source 0, target 0
articulation_transfer_drawer_to_slider: source 0, target 0
low_data_all_tasks: source 9, target 9
```

Interpretation: the current debug dataset is enough to validate code and visualization, but not enough for real transfer experiments. It does not contain drawer/slider annotations, so articulation transfer must wait for the full CALVIN dataset.

## G Label Result

The all-training debug G extraction produced:

```text
records: 72
object_names: block_red, block_blue, block_pink
scene_scalar_names: base__slide, base__drawer, base__button, base__switch, lightbulb, led
future horizons: 1, 4, 8, 16
```

The schema-aware overlay confirms that EEF-object distance and nearest-object identity evolve plausibly during lift/place sequences.

## Next Decisions

1. Treat the debug dataset as a pipeline validator only.
2. Download or point to full CALVIN data before making scientific claims.
3. On full CALVIN, regenerate all split manifests and require non-empty articulation transfer splits.
4. Before training G, manually review at least 20 raw checks and 20 G overlays from the full dataset.
5. Use `scene_obs` as first GT/proxy-G for CALVIN, while keeping RGBD-derived or rendered-mask G as a later robustness branch.
