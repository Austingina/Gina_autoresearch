# Minimal G Direction-Coverage Matrix

| Condition | Completed seeds | Normal close | Worst seed | Zero | Sign flip | Wrong open G | Seen health |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 5/5 | 3.0% (5/5) | 0 | 0.0% (5/5) | 2.4% (5/5) | 0.4% (5/5) | 24.8% (5/5) |
| D1 | 5/5 | 0.4% (5/5) | 0 | 0.0% (5/5) | 7.2% (5/5) | 0.6% (5/5) | 39.8% (5/5) |
| D2 | 5/5 | 85.2% (5/5) | 51 | 0.0% (5/5) | 0.6% (5/5) | 16.8% (5/5) | 64.2% (5/5) |
| D4 | 5/5 | 2.2% (5/5) | 0 | 0.0% (5/5) | 3.6% (5/5) | 3.8% (5/5) | 51.1% (5/5) |
| D3 | 5/5 | 0.0% (5/5) | 0 | 0.0% (5/5) | 0.4% (5/5) | 0.4% (5/5) | 34.2% (5/5) |

## Seen-Task Families

| Condition | Articulation | Free-object push | All seen tasks |
| --- | ---: | ---: | ---: |
| D0 | 274/375 (73.1%) | 36/875 (4.1%) | 310/1250 (24.8%) |
| D1 | 303/375 (80.8%) | 194/875 (22.2%) | 497/1250 (39.8%) |
| D2 | 354/375 (94.4%) | 449/875 (51.3%) | 803/1250 (64.2%) |
| D4 | 288/375 (76.8%) | 351/875 (40.1%) | 639/1250 (51.1%) |
| D3 | 259/375 (69.1%) | 168/875 (19.2%) | 427/1250 (34.2%) |

## Paired Causal Controls

| Condition | Control | Seed pairs | Both | Normal only | Control only | Neither | Indices match |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| D0 | zero | 5 | 0 | 15 | 0 | 485 | True |
| D0 | sign_flip | 5 | 0 | 15 | 12 | 473 | True |
| D0 | task_open_drawer | 5 | 0 | 15 | 2 | 483 | True |
| D1 | zero | 5 | 0 | 2 | 0 | 498 | True |
| D1 | sign_flip | 5 | 1 | 1 | 35 | 463 | True |
| D1 | task_open_drawer | 5 | 0 | 2 | 3 | 495 | True |
| D2 | zero | 5 | 0 | 426 | 0 | 74 | True |
| D2 | sign_flip | 5 | 3 | 423 | 0 | 74 | True |
| D2 | task_open_drawer | 5 | 71 | 355 | 13 | 61 | True |
| D4 | zero | 5 | 0 | 11 | 0 | 489 | True |
| D4 | sign_flip | 5 | 0 | 11 | 18 | 471 | True |
| D4 | task_open_drawer | 5 | 0 | 11 | 19 | 470 | True |
| D3 | zero | 5 | 0 | 0 | 0 | 500 | True |
| D3 | sign_flip | 5 | 0 | 0 | 2 | 498 | True |
| D3 | task_open_drawer | 5 | 0 | 0 | 2 | 498 | True |
