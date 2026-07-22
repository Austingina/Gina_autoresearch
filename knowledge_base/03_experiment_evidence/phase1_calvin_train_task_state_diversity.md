# Phase-1 CALVIN Train-Task State Diversity

Date: 2026-07-11

## Question

Does each train task receive randomized EEF starts and object arrangements, or
are all demonstrations trained from a nearly fixed state?

## Runtime Randomization

The current state chunk-BC trainer does not reset a simulator during training
and does not apply coordinate noise, object-pose jitter, feature dropout, image
augmentation, or online domain randomization. It reads fixed `.npz` trajectory
windows from CALVIN `task_D_D/training`.

DataLoader shuffling changes only the order of these fixed windows. With
`data_seed=0` and `split_seed=0`, every model seed receives the same supervised
examples.

## Exact Split

```text
available non-close segments: 2107
available windows:            48069
supervised train segments:    1896
supervised train windows:     43230
fixed validation segments:     211
chunk size:                      16
stride:                           2
```

## Segment-Start Diversity

All position uniqueness counts use xyz rounded to 1 mm.

| Task | Train segments | Windows | Unique EEF starts | EEF xyz span (m) | Red / blue / pink xyz unique |
| --- | ---: | ---: | ---: | --- | --- |
| `move_slider_left` | 137 | 3352 | 137 | 0.596 / 0.482 / 0.256 | 121 / 121 / 121 |
| `move_slider_right` | 142 | 3536 | 142 | 0.496 / 0.288 / 0.206 | 124 / 123 / 123 |
| `open_drawer` | 133 | 3325 | 133 | 0.403 / 0.276 / 0.174 | 110 / 111 / 108 |
| `push_blue_block_left` | 125 | 2784 | 123 | 0.585 / 0.438 / 0.274 | 49 / 64 / 50 |
| `push_blue_block_right` | 136 | 2852 | 133 | 0.534 / 0.481 / 0.235 | 52 / 74 / 53 |
| `push_into_drawer` | 134 | 2946 | 134 | 0.444 / 0.474 / 0.202 | 99 / 92 / 98 |
| `push_pink_block_left` | 134 | 2859 | 131 | 0.514 / 0.267 / 0.248 | 58 / 57 / 81 |
| `push_pink_block_right` | 140 | 2962 | 139 | 0.554 / 0.457 / 0.227 | 65 / 61 / 79 |
| `push_red_block_left` | 138 | 3226 | 138 | 0.550 / 0.476 / 0.256 | 79 / 59 / 57 |
| `push_red_block_right` | 134 | 3055 | 134 | 0.570 / 0.458 / 0.204 | 62 / 46 / 49 |
| `turn_off_led` | 134 | 2830 | 134 | 0.609 / 0.482 / 0.286 | 120 / 114 / 115 |
| `turn_off_lightbulb` | 137 | 3322 | 137 | 0.465 / 0.371 / 0.272 | 110 / 111 / 114 |
| `turn_on_led` | 139 | 3030 | 139 | 0.622 / 0.503 / 0.348 | 126 / 120 / 123 |
| `turn_on_lightbulb` | 133 | 3151 | 133 | 0.499 / 0.498 / 0.257 | 112 / 115 / 113 |

## Interpretation

The training set contains substantial EEF and scene-layout diversity. The seed
effect cannot be explained by different seeds seeing different initial states:
all seeds see the same 43,230 windows and the same train/validation split.

However, this is fixed offline diversity rather than independent factorial
randomization. Object poses, EEF starts, mechanism states, language labels, and
expert strategies can remain correlated. Repeating the same dataset for 50
epochs does not generate new combinations.

Most importantly for zero-shot close transfer, broad EEF/object diversity does
not provide the missing mechanism-direction combination: drawer demonstrations
contain `open_drawer`, while `close_drawer` remains absent. Thus the policy sees
many contexts but only one supervised drawer direction. This supports testing
sign/mechanism-equivariant G fusion rather than attributing the seed issue to a
fixed EEF start or fixed tabletop layout.

## Artifacts

```text
remote:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-11_train_task_state_diversity_split0_chunk16_stride2

local:
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-11_train_task_state_diversity_split0/
```
