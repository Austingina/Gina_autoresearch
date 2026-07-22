# E3 Factorial Checkerboard Multi-seed Summary

| Held-out task | G normal | F sign-flip | Matched no-G |
| --- | ---: | ---: | ---: |
| close_drawer | 0/300 (0%, 0%, 0%) | 4/300 (0%, 0%, 4%) | 0/300 (0%, 0%, 0%) |
| move_slider_left | 25/300 (14%, 10%, 1%) | 15/300 (1%, 2%, 12%) | 0/300 (0%, 0%, 0%) |
| push_red_block_right | 139/300 (46%, 56%, 37%) | 0/300 (0%, 0%, 0%) | 3/300 (0%, 3%, 0%) |
| push_blue_block_left | 147/300 (48%, 68%, 31%) | 2/300 (2%, 0%, 0%) | 1/300 (0%, 1%, 0%) |

## Source-task health

- seed0 with_g: 117/275 overall; 116/175 interaction subset
- seed0 matched_no_g: 16/275 overall; 16/175 interaction subset
- seed1 with_g: 111/275 overall; 110/175 interaction subset
- seed1 matched_no_g: 9/275 overall; 9/175 interaction subset
- seed2 with_g: 107/275 overall; 107/175 interaction subset
- seed2 matched_no_g: 17/275 overall; 17/175 interaction subset

### Normal-language source health

- seed0 with_g: 183/275 overall; 116/175 interaction subset
- seed1 with_g: 182/275 overall; 110/175 interaction subset
- seed2 with_g: 185/275 overall; 107/175 interaction subset

## Paired causal controls

- close_drawer, normal vs with_g_sign_flip: normal-only=0, control-only=4, both=0, neither=296, exact McNemar p=0.125
- close_drawer, normal vs matched_no_g: normal-only=0, control-only=0, both=0, neither=300, exact McNemar p=1
- move_slider_left, normal vs with_g_sign_flip: normal-only=24, control-only=14, both=1, neither=261, exact McNemar p=0.143
- move_slider_left, normal vs matched_no_g: normal-only=25, control-only=0, both=0, neither=275, exact McNemar p=5.96e-08
- push_red_block_right, normal vs with_g_sign_flip: normal-only=139, control-only=0, both=0, neither=161, exact McNemar p=2.87e-42
- push_red_block_right, normal vs matched_no_g: normal-only=136, control-only=0, both=3, neither=161, exact McNemar p=2.3e-41
- push_blue_block_left, normal vs with_g_sign_flip: normal-only=145, control-only=0, both=2, neither=153, exact McNemar p=4.48e-44
- push_blue_block_left, normal vs matched_no_g: normal-only=147, control-only=1, both=0, neither=152, exact McNemar p=8.35e-43
