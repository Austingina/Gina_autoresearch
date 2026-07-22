# Phase 1 CALVIN State-BC Task Metric V1

First rollout/task-metric comparison for no-G vs oracle-G.

Setup:

- dataset: CALVIN `task_D_D`
- policy: lightweight state-BC MLP
- input no-G: robot state, scene state, task one-hot
- input oracle-G: no-G input plus demo-derived `target_scene - current_scene`
- train records: 47350
- train segments: 3000
- eval: 10 long-horizon sequences, `EP_LEN=120`

Result:

| Mode | Val MSE | Avg Seq Len | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_g` | 0.03746 | 0.20 | 0.10 | 0.10 | 0.00 | 0.00 | 0.00 |
| `oracle_g` | 0.03655 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Interpretation:

- Oracle-G improves offline action MSE slightly.
- Oracle-G does not improve task rollout in this direct-concat state-BC setup.
- This reinforces that MSE is only a diagnostic gate; closed-loop CALVIN metrics
  must drive the paper claims.

Decision:

- Do not use direct dense scene-delta oracle-G as the main policy input.
- Next try should use task-family-specific, phase-gated, or auxiliary/FiLM-style
  G injection.

Artifacts:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_state_bc_task_D_D_no_g_vs_oracle_g_n10_h120
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/state_bc_task_D_D_no_g_vs_oracle_g_n10_h120_2026-07-03
```
