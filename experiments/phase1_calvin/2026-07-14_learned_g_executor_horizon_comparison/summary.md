# Learned G Executor Horizon Comparison

Complete: `True`
Matched sequence indices: `True`

## Close Drawer

| Horizon | Mode | Seed successes (/100) | Mean SR | SD | Pooled |
| --- | --- | --- | ---: | ---: | ---: |
| h1 | chunk_bc_lang | [0, 0, 0, 0, 0] | 0.000 | 0.000 | 0/500 |
| h1 | chunk_bc_minimal_g_concat | [0, 0, 0, 0, 0] | 0.000 | 0.000 | 0/500 |
| h1 | chunk_bc_minimal_g_film | [0, 0, 0, 19, 14] | 0.066 | 0.092 | 33/500 |
| h16 | chunk_bc_lang | [0, 0, 0, 0, 0] | 0.000 | 0.000 | 0/500 |
| h16 | chunk_bc_minimal_g_concat | [0, 0, 0, 0, 0] | 0.000 | 0.000 | 0/500 |
| h16 | chunk_bc_minimal_g_film | [0, 0, 0, 0, 0] | 0.000 | 0.000 | 0/500 |

## Paired Comparisons

| Scope | A | B | Mean seed delta (B-A) | A only | B only | Exact p |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| h1 | chunk_bc_lang | chunk_bc_minimal_g_concat | 0.000 | 0 | 0 | 1 |
| h1 | chunk_bc_lang | chunk_bc_minimal_g_film | 0.066 | 0 | 33 | 2.328e-10 |
| h1 | chunk_bc_minimal_g_concat | chunk_bc_minimal_g_film | 0.066 | 0 | 33 | 2.328e-10 |
| h16 | chunk_bc_lang | chunk_bc_minimal_g_concat | 0.000 | 0 | 0 | 1 |
| h16 | chunk_bc_lang | chunk_bc_minimal_g_film | 0.000 | 0 | 0 | 1 |
| h16 | chunk_bc_minimal_g_concat | chunk_bc_minimal_g_film | 0.000 | 0 | 0 | 1 |
| h16_vs_h1 | h1/chunk_bc_lang | h16/chunk_bc_lang | 0.000 | 0 | 0 | 1 |
| h16_vs_h1 | h1/chunk_bc_minimal_g_concat | h16/chunk_bc_minimal_g_concat | 0.000 | 0 | 0 | 1 |
| h16_vs_h1 | h1/chunk_bc_minimal_g_film | h16/chunk_bc_minimal_g_film | -0.066 | 33 | 0 | 2.328e-10 |
