# CALVIN Scene And Robot Observation Schema

Created: 2026-07-02

## Current Debug Dataset

The current debug dataset uses `calvin_scene_D`.

The schema-aware scripts read object order from:

```text
<dataset>/<split>/.hydra/merged_config.yaml
```

For the current dataset:

```text
scene_obs[0]  base__slide
scene_obs[1]  base__drawer
scene_obs[2]  base__button
scene_obs[3]  base__switch
scene_obs[4]  lightbulb
scene_obs[5]  led
scene_obs[6:12]   block_red  xyz + euler orientation
scene_obs[12:18]  block_blue xyz + euler orientation
scene_obs[18:24]  block_pink xyz + euler orientation
```

Important: object order can differ across CALVIN scenes. Do not hard-code red/blue/pink order for future datasets. Use the hydra scene config.

## Robot Observation

For euler observations, `robot_obs` is 15D:

```text
robot_obs[0:3]   tcp position xyz
robot_obs[3:6]   tcp orientation euler xyz
robot_obs[6]     gripper opening width
robot_obs[7:14]  arm joint states
robot_obs[14]    gripper action
```

## Source Files

CALVIN source files checked:

```text
calvin_env/scene/play_table_scene.py
calvin_env/robot/robot.py
calvin_models/calvin_agent/evaluation/utils.py
```

## Use In AutoResearch

For the first GT/proxy-G stage, use:

- TCP position/orientation and gripper state from `robot_obs`;
- articulated states and light states from `scene_obs[0:6]`;
- object positions/orientations from the hydra-defined object slices in `scene_obs`;
- relative object-to-EEF vectors and distances;
- near-contact flags from object-to-EEF distance thresholds;
- future deltas at horizons 1, 4, 8, and 16.

This is a proxy-G, not yet a final learned interaction geometry representation. It is suitable for validating whether reliable geometry helps policy learning before training a learned G module.
