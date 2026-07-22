# Minimal G Demonstration Alignment

| Task | Windows | Far A cosine | Near F cosine | Near desired-flow positive | Far action norm | Near action norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| open_drawer | 5016 | 0.722 | 0.865 | 0.774 | 0.4145 | 0.3420 |
| move_slider_left | 4907 | 0.723 | 0.920 | 0.825 | 0.4621 | 0.3473 |
| move_slider_right | 5001 | 0.860 | 0.917 | 0.705 | 0.4724 | 0.3339 |

## Active blended field decomposition

The all-step row compares every future action in a chunk with the field observed at the chunk start.

| Task | Sample | Field cosine p50 | Positive parallel | Usable parallel | Orth/parallel p50 | Orth/parallel p90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| open_drawer | first_action | 0.769 | 0.914 | 0.892 | 0.706 | 2.855 |
| open_drawer | chunk_mean_action | 0.769 | 0.914 | 0.892 | 0.706 | 2.855 |
| open_drawer | all_chunk_steps_static_current_field | 0.769 | 0.914 | 0.892 | 0.706 | 2.855 |
| move_slider_left | first_action | 0.819 | 0.890 | 0.860 | 0.477 | 3.625 |
| move_slider_left | chunk_mean_action | 0.819 | 0.890 | 0.860 | 0.477 | 3.625 |
| move_slider_left | all_chunk_steps_static_current_field | 0.819 | 0.890 | 0.860 | 0.477 | 3.625 |
| move_slider_right | first_action | 0.833 | 0.833 | 0.798 | 0.399 | 3.669 |
| move_slider_right | chunk_mean_action | 0.833 | 0.833 | 0.798 | 0.399 | 3.669 |
| move_slider_right | all_chunk_steps_static_current_field | 0.833 | 0.833 | 0.798 | 0.399 | 3.669 |
