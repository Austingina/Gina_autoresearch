# Open/Close Query-G Trajectory Analysis

Effective contact is the first step where cumulative drawer motion in the task direction reaches 2 mm.
Distances are in meters and action magnitudes are CALVIN relative-action units.

## Per-Policy Open/Close

| Policy | Task | Success | Effective contact | Min part dist | Min anchor dist | Pre action norm | Late desired-axis cmd | Max q progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_type_seed0_normal | close_drawer | 2/25 | 3/25 | 0.119 | 0.160 | 0.0045 | -0.0000 | 0.000 |
| no_type_seed0_sign_flip | close_drawer | 6/25 | 22/25 | 0.059 | 0.058 | 0.0059 | -0.0005 | 0.301 |
| no_type_seed2_normal | close_drawer | 3/25 | 4/25 | 0.220 | 0.223 | 0.0053 | 0.0001 | 0.000 |
| no_type_seed2_sign_flip | close_drawer | 4/25 | 4/25 | 0.225 | 0.233 | 0.0029 | 0.0001 | 0.000 |

## Pooled No-Token-Type Close

| Outcome | Count | Effective contact | Min part dist | Min anchor dist | Pre action norm | Pre G alignment | Late desired-axis cmd | Max q progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| success | 15 | 15/15 | 0.082 | 0.091 | 0.0067 | 0.564 | 0.0009 | 0.992 |
| failure_with_contact | 18 | 18/18 | 0.056 | 0.059 | 0.0055 | 0.609 | -0.0001 | 0.273 |
| failure_without_contact | 67 | 0/67 | 0.216 | 0.220 | 0.0040 | 0.119 | 0.0001 | 0.000 |
