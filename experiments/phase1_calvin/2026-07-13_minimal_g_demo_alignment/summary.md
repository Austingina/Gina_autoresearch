# Minimal G Demonstration Alignment

| Task | Windows | Far A cosine | Near F cosine | Near desired-flow positive | Far action norm | Near action norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| open_drawer | 3800 | 0.885 | 0.933 | 0.961 | 0.3404 | 0.3138 |
| move_slider_left | 3702 | 0.938 | 0.973 | 0.933 | 0.3676 | 0.2873 |
| move_slider_right | 3786 | 0.970 | 0.976 | 0.859 | 0.3431 | 0.2752 |

## Active blended field decomposition

The all-step row compares every future action in a chunk with the field observed at the chunk start.

| Task | Sample | Field cosine p50 | Positive parallel | Usable parallel | Orth/parallel p50 | Orth/parallel p90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| open_drawer | first_action | 0.707 | 0.929 | 0.902 | 0.881 | 3.373 |
| open_drawer | chunk_mean_action | 0.945 | 0.998 | 0.996 | 0.344 | 0.920 |
| open_drawer | all_chunk_steps_static_current_field | 0.929 | 0.982 | 0.974 | 0.384 | 1.491 |
| move_slider_left | first_action | 0.696 | 0.885 | 0.847 | 0.785 | 4.313 |
| move_slider_left | chunk_mean_action | 0.968 | 0.994 | 0.988 | 0.253 | 1.410 |
| move_slider_left | all_chunk_steps_static_current_field | 0.938 | 0.953 | 0.937 | 0.322 | 2.140 |
| move_slider_right | first_action | 0.675 | 0.808 | 0.764 | 0.645 | 4.387 |
| move_slider_right | chunk_mean_action | 0.976 | 0.981 | 0.971 | 0.213 | 1.532 |
| move_slider_right | all_chunk_steps_static_current_field | 0.938 | 0.931 | 0.908 | 0.314 | 2.083 |
