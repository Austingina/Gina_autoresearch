# Open/Close Query-G Trajectory Analysis

Effective contact is the first step where cumulative drawer motion in the task direction reaches 2 mm.
Distances are in meters and action magnitudes are CALVIN relative-action units.

## Per-Policy Open/Close

| Policy | Task | Success | Effective contact | Min part dist | Min anchor dist | Pre action norm | Late desired-axis cmd | Max q progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_type_seed2 | close_drawer | 5/8 | 5/8 | 0.109 | 0.090 | 0.0093 | 0.0036 | 0.613 |
| with_type_seed2 | close_drawer | 0/8 | 0/8 | 0.209 | 0.257 | 0.0029 | 0.0005 | -0.000 |

## Pooled No-Token-Type Close

| Outcome | Count | Effective contact | Min part dist | Min anchor dist | Pre action norm | Pre G alignment | Late desired-axis cmd | Max q progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| success | 5 | 5/5 | 0.080 | 0.076 | 0.0104 | 0.199 | 0.0059 | 0.692 |
| failure_with_contact | 0 | 0/0 | n/a | n/a | n/a | n/a | n/a | n/a |
| failure_without_contact | 3 | 0/3 | 0.219 | 0.225 | 0.0056 | 0.495 | -0.0000 | 0.000 |
