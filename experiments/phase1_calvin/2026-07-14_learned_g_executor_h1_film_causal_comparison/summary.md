# H1 FiLM G Causal Controls

Complete: `True`
Matched sequence indices: `True`

| G control | Seed successes (/100) | Mean SR | Pooled |
| --- | --- | ---: | ---: |
| normal | [19, 14] | 0.165 | 33/200 |
| zero | [0, 0] | 0.000 | 0/200 |
| sign_flip | [34, 3] | 0.185 | 37/200 |
| task_open_drawer | [0, 0] | 0.000 | 0/200 |
| zero_attraction | [0, 0] | 0.000 | 0/200 |
| zero_flow | [0, 0] | 0.000 | 0/200 |
| sign_flip_attraction | [0, 0] | 0.000 | 0/200 |

| A | B | Seed delta (B-A) | Per-seed A/B only | Pooled A only | Pooled B only | Exact p |
| --- | --- | --- | --- | ---: | ---: | ---: |
| normal | zero | [-0.19, -0.14] | ['3:19/0', '4:14/0'] | 33 | 0 | 2.328e-10 |
| normal | sign_flip | [0.15000000000000002, -0.11000000000000001] | ['3:12/27', '4:14/3'] | 26 | 30 | 0.6889 |
| normal | task_open_drawer | [-0.19, -0.14] | ['3:19/0', '4:14/0'] | 33 | 0 | 2.328e-10 |
| normal | zero_attraction | [-0.19, -0.14] | ['3:19/0', '4:14/0'] | 33 | 0 | 2.328e-10 |
| normal | zero_flow | [-0.19, -0.14] | ['3:19/0', '4:14/0'] | 33 | 0 | 2.328e-10 |
| normal | sign_flip_attraction | [-0.19, -0.14] | ['3:19/0', '4:14/0'] | 33 | 0 | 2.328e-10 |
