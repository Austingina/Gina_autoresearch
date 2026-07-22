# Open/Close Query-G Trajectory Analysis

Effective contact is the first step where cumulative drawer motion in the task direction reaches 2 mm.
Distances are in meters and action magnitudes are CALVIN relative-action units.

## Per-Policy Open/Close

| Policy | Task | Success | Effective contact | Min part dist | Min anchor dist | Pre action norm | Late desired-axis cmd | Max q progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_type_seed0 | close_drawer | 0/25 | 3/25 | 0.113 | 0.150 | 0.0043 | -0.0003 | 0.000 |
| no_type_seed0 | open_drawer | 25/25 | 25/25 | 0.024 | 0.064 | 0.0072 | 0.0038 | 0.709 |
| no_type_seed1 | close_drawer | 0/25 | 1/25 | 0.308 | 0.324 | 0.0101 | 0.0023 | 0.000 |
| no_type_seed1 | open_drawer | 22/25 | 25/25 | 0.023 | 0.060 | 0.0082 | 0.0042 | 0.725 |
| no_type_seed2 | close_drawer | 1/25 | 4/25 | 0.235 | 0.229 | 0.0052 | 0.0001 | 0.000 |
| no_type_seed2 | open_drawer | 25/25 | 25/25 | 0.026 | 0.065 | 0.0091 | 0.0036 | 0.696 |
| no_type_seed3 | close_drawer | 0/25 | 0/25 | 0.182 | 0.222 | 0.0024 | 0.0001 | 0.000 |
| no_type_seed3 | open_drawer | 25/25 | 25/25 | 0.011 | 0.057 | 0.0078 | 0.0043 | 0.716 |
| no_type_seed4 | close_drawer | 0/25 | 0/25 | 0.236 | 0.231 | 0.0043 | 0.0002 | 0.000 |
| no_type_seed4 | open_drawer | 24/25 | 25/25 | 0.015 | 0.067 | 0.0070 | 0.0038 | 0.688 |
| with_type_seed0 | close_drawer | 3/25 | 5/25 | 0.272 | 0.296 | 0.0065 | -0.0004 | 0.000 |
| with_type_seed0 | open_drawer | 22/25 | 25/25 | 0.036 | 0.070 | 0.0080 | 0.0037 | 0.717 |
| with_type_seed2 | close_drawer | 0/25 | 0/25 | 0.205 | 0.256 | 0.0037 | 0.0003 | 0.000 |
| with_type_seed2 | open_drawer | 25/25 | 25/25 | 0.014 | 0.065 | 0.0090 | 0.0037 | 0.763 |

## Pooled No-Token-Type Close

| Outcome | Count | Effective contact | Min part dist | Min anchor dist | Pre action norm | Pre G alignment | Late desired-axis cmd | Max q progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| success | 1 | 1/1 | 0.080 | 0.076 | 0.0104 | 0.199 | 0.0058 | 0.657 |
| failure_with_contact | 7 | 7/7 | 0.118 | 0.170 | 0.0089 | 0.015 | 0.0009 | 0.382 |
| failure_without_contact | 117 | 0/117 | 0.228 | 0.229 | 0.0046 | -0.007 | 0.0001 | 0.000 |
