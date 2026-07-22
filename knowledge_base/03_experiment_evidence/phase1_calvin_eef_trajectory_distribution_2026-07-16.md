# Phase-1 CALVIN EEF Trajectory Distribution Audit

Date: 2026-07-16

## Question

The fixed CALVIN dataset has diverse segment starts, but that alone does not
exclude a task-template shortcut. This audit asks whether non-block tasks still
collapse to one narrow EEF trajectory per task, allowing a policy to infer and
replay task-specific motion without reading G.

## Scope and method

The audit scans all 1,057 training demonstrations from seven non-block tasks:

```text
open_drawer
move_slider_left / move_slider_right
turn_on_led / turn_off_led
turn_on_lightbulb / turn_off_lightbulb
```

Each complete EEF xyz trajectory is resampled to 32 normalized-time points.
The report measures endpoint spread, direction concentration, path-length
variation, start-aligned template RMSE, trajectory PCA rank, and five-fold
nearest-centroid task classification from several state/trajectory features.

## Results

| Task | Segments | Direction R | Angle p90 | Path CV | Aligned RMSE/path | End std norm | PCA effective rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `move_slider_left` | 151 | 0.774 | 75.9 deg | 0.170 | 0.150 | 0.040 m | 2.58 |
| `move_slider_right` | 152 | 0.649 | 87.5 deg | 0.164 | 0.114 | 0.045 m | 2.57 |
| `open_drawer` | 152 | 0.946 | 26.8 deg | 0.152 | 0.098 | 0.056 m | 2.98 |
| `turn_off_led` | 150 | 0.521 | 107.9 deg | 0.326 | 0.206 | 0.031 m | 3.19 |
| `turn_off_lightbulb` | 152 | 0.872 | 40.3 deg | 0.201 | 0.129 | 0.024 m | 3.35 |
| `turn_on_led` | 150 | 0.491 | 99.7 deg | 0.291 | 0.213 | 0.033 m | 3.33 |
| `turn_on_lightbulb` | 150 | 0.899 | 37.8 deg | 0.194 | 0.146 | 0.029 m | 3.25 |

Task predictability without language or G:

| Feature | Balanced accuracy | Chance |
| --- | ---: | ---: |
| Initial EEF xyz | 28.5% | 14.3% |
| Initial robot + scene state | 49.6% | 14.3% |
| EEF displacement + path length + duration | 61.1% | 14.3% |
| Start-aligned EEF trajectory | 65.8% | 14.3% |
| Displacement-normalized EEF shape | 65.0% | 14.3% |

## Interpretation

The demonstrations do **not** collapse to one duplicated trajectory. The
overlays show meaningful start offsets, approach variation, and different
local paths, especially for slider and LED tasks. The dataset therefore has
reasonable within-task motion diversity.

There is nevertheless a stable low-dimensional task trend. `open_drawer` is
the strongest example: its endpoint direction is highly concentrated and its
median start-aligned template error is only 9.8% of path length. A simple
classifier also recovers task identity from the physical initial state and
from start-aligned trajectory shape far above chance.

The main risk is consequently **not fixed-trajectory duplication**. It is a
lack of factorial conditional coverage: state offsets, mechanism state, goal
direction, contact mode, and expert strategy may vary but remain correlated.
Many physically similar states may still have only one demonstrated G/action
continuation, leaving BC/FM free to ignore G.

## Experimental consequence

The first G-token experiment keeps the D4 data, H16 target, first-action weight,
and split fixed. This is a valid architecture/identifiability diagnostic, but
not an environment-generalization claim. Its required causal controls remain
normal, zero, sign-flipped, and wrong-task G.

If token conditioning remains unstable, the next data intervention should be
state-goal overlap mining and balanced sampling across inverse task pairs,
followed by genuinely randomized simulator collection. Merely duplicating or
adding noise to the existing trajectories would not resolve the conditional
coverage problem.

## Artifacts

```text
remote:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-16_nonblock_eef_trajectory_distribution

local:
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-16_nonblock_eef_trajectory_distribution
```

The local and remote directories contain the full JSON/CSV report, Markdown
summary, and `eef_trajectory_templates.png` overlay.
