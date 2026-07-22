# Phase 1 CALVIN Articulation-G FiLM V1

**Date:** 2026-07-03

First positive policy-level signal for Interaction Geometry in CALVIN.

## Result

Balanced first-task eval on articulation tasks, 10 official first-task sequences
per task:

| Mode | Balanced SR@1 |
| --- | ---: |
| `no_g` | 0.55 |
| `articulation_g_concat` | 0.35 |
| `articulation_g_film` | **0.675** |

Per-task:

| Mode | close_drawer | move_slider_left | move_slider_right | open_drawer |
| --- | ---: | ---: | ---: | ---: |
| `no_g` | 10/10 | 1/10 | **10/10** | 1/10 |
| `articulation_g_concat` | 10/10 | 1/10 | 2/10 | 1/10 |
| `articulation_g_film` | 10/10 | **3/10** | 8/10 | **6/10** |

## Takeaway

Raw dense G and concat G remain bad interfaces. A narrow task-family
articulation G used as FiLM modulation improves balanced task success, mainly
by improving `open_drawer` and `move_slider_left`, while slightly hurting the
already-easy `move_slider_right`.

This supports the revised story:

> Interaction Geometry should be a reusable task-family modulation signal, not
> a raw future-state vector appended to the policy input.

Full record:

- `docs/PHASE1_CALVIN_ARTICULATION_G_FILM_V1.md`
