# CALVIN EEF Trajectory Distribution Audit

This audit tests whether diverse segment starts still collapse to narrow, task-specific EEF motion templates.

## Per-task trajectory concentration

| Task | Segments | Frames p50 | Direction R | Angle p90 | Path CV | Aligned RMSE/path | End std norm | PCA effective rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `move_slider_left` | 151 | 65 | 0.774 | 75.9 | 0.170 | 0.150 | 0.040 | 2.58 |
| `move_slider_right` | 152 | 65 | 0.649 | 87.5 | 0.164 | 0.114 | 0.045 | 2.57 |
| `open_drawer` | 152 | 65 | 0.946 | 26.8 | 0.152 | 0.098 | 0.056 | 2.98 |
| `turn_off_led` | 150 | 65 | 0.521 | 107.9 | 0.326 | 0.206 | 0.031 | 3.19 |
| `turn_off_lightbulb` | 152 | 65 | 0.872 | 40.3 | 0.201 | 0.129 | 0.024 | 3.35 |
| `turn_on_led` | 150 | 65 | 0.491 | 99.7 | 0.291 | 0.213 | 0.033 | 3.33 |
| `turn_on_lightbulb` | 150 | 65 | 0.899 | 37.8 | 0.194 | 0.146 | 0.029 | 3.25 |

## Task predictability without language or G

| Feature | Balanced accuracy mean | Fold std | Chance |
| --- | ---: | ---: | ---: |
| `initial_eef_xyz` | 0.285 | 0.014 | 0.143 |
| `initial_robot_plus_scene_state` | 0.496 | 0.039 | 0.143 |
| `eef_displacement_pathlength_duration` | 0.611 | 0.035 | 0.143 |
| `start_aligned_eef_trajectory` | 0.658 | 0.025 | 0.143 |
| `displacement_normalized_eef_shape` | 0.650 | 0.039 | 0.143 |

## Reading the diagnostics

- High initial-state accuracy means the physical state itself leaks task identity, allowing a policy to ignore G.
- High start-aligned trajectory accuracy means each task has a recognizable motor template even after removing absolute EEF position.
- Direction R near 1, low aligned RMSE/path, and low path-length CV indicate a narrow canonical trajectory family.
- These are shortcut-risk diagnostics, not proof that a trained policy uses the shortcut; causal G corruption remains necessary.

The trajectory overlay is saved as `eef_trajectory_templates.png`.
