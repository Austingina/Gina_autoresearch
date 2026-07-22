# geo_vla Structure Review

Reference: `external/MacroRobotics/geo_vla`.

Observed organization:

```text
configs/         # task/model hyperparameters
envs/            # backend/task environments
features/        # geometry feature extraction
models/          # model families and shared modules
train.py         # single-model training
eval.py          # rollout/evaluation
run_experiments.py
h200/            # scaling launch scripts and runbook
results/         # csv/json/png summaries
libero_exp/      # benchmark-specific extension
```

Transfer to AutoResearch:

- Keep benchmark-specific code behind adapters: IsaacLab and CALVIN should be backends.
- Keep representation choices separate from policy/action-decoder choices.
- Every experiment should be config-driven and emit csv/json/logs/notes.
- Scaling scripts should live separately from one-off smoke tests.

AutoResearch mapping:

```text
framework/configs/       -> geo_vla/configs
framework/autoresearch/  -> geo_vla/envs/features/models equivalents
framework/scripts/       -> train/eval/run/h200 launchers
experiments/             -> geo_vla/results plus richer run manifests
knowledge_base/          -> docs and paper-facing interpretation
```
