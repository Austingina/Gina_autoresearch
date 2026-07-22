# Phase 1 CALVIN D Reusable-G V1 Full-D Results

**Date:** 2026-07-03  
**Server project:** `/mnt/nvme33n1/wuk_code/AutoResearch`  
**Dataset:** `/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D`  
**Local sync:** `/Users/wukui/Documents/AutoResearch/artifacts/server_sync/calvin_task_D_D_reusable_g_v1_archive_2026-07-03`

## Status

CALVIN task D_D was downloaded and extracted successfully.

| Item | Value |
| --- | ---: |
| `task_D_D.zip` | 166G |
| extracted dataset | 168G |
| training episodes | 512077 |
| validation episodes | 99022 |
| scene info fix | applied |

The reusable-G validation pipeline completed four source-to-target recipes:

| Recipe | Source records | Target records | Val MSE | Zero Val MSE | Pred / Zero |
| --- | ---: | ---: | ---: | ---: | ---: |
| `object_transfer_red_blue_to_pink` | 33728 | 16848 | 0.0006516 | 0.0003047 | 2.14x |
| `primitive_transfer_push_lift_rotate_to_place_stack` | 50576 | 12080 | 0.0007383 | 0.0005007 | 1.47x |
| `articulation_transfer_slider_to_drawer` | 4848 | 4848 | 0.0022844 | 0.0004277 | 5.34x |
| `articulation_transfer_drawer_to_slider` | 4848 | 4848 | 0.0064697 | 0.0003263 | 19.83x |

## Interpretation

This is a useful negative result. The current tiny G predictor does not yet provide reusable interaction geometry across held-out object, primitive, or articulation splits. In every recipe, a zero baseline is stronger on held-out target G labels.

Manual preview of `prediction_preview.png` suggests three failure modes:

1. **Object transfer:** early dimensions roughly follow local variation, but later dimensions collapse toward a mean/near-zero prediction. The model is partially learning the dominant distribution, not a reusable interaction structure.
2. **Primitive transfer:** predictions are smoother and lower-amplitude than ground truth. This is the least bad split, but it still loses to zero baseline.
3. **Articulation transfer:** predictions produce wrong sparse spikes or miss target sparse events. Raw coordinate G does not transfer cleanly between slider and drawer mechanisms.

The result supports the previous concern from Isaac experiments: a learned G can easily mislead the downstream policy when G error is structured. It also sharpens the research direction: reusable-G should not be framed as "predict raw future geometry in task coordinates" without stronger normalization or object-centric structure.

## Immediate Next Scope

The next experiment should not jump directly to flow-matching policy training. First, we need a better G representation and sanity checks:

1. **G-v2 representation:** convert raw G into object-centric / interaction-frame deltas.
   - Express target deltas relative to relevant object pose or articulated joint proxy.
   - Add per-channel clipping or robust normalization to suppress rare spike domination.
   - Split dense object-motion channels from sparse contact/articulation event channels.

2. **Baselines before policy:** evaluate reusable-G as representation prediction first.
   - `zero`
   - source mean
   - target train mean, if target supervision is allowed as an upper diagnostic
   - raw tiny predictor
   - normalized/clipped tiny predictor
   - object-centric predictor

3. **Visualization gates:** every G-v2 run must output:
   - prediction preview
   - per-channel error histogram
   - per-task/per-language breakdown
   - sampled G sequence overlays from source and target

4. **Only after G-v2 beats trivial baselines:** plug into the policy side.
   - Reuse CALVIN camera/data structure.
   - Prefer starting from existing flow-matching code if available.
   - Keep LSTM only as a diagnostic endpoint or low-cost baseline, not the main policy story.

## G-v2 Update: Canonical Feature Is Necessary

After the V1 result, the server extractor was updated to save `future_delta_scene_scalars_h*`, and the predictor was extended with:

- `target_mode=raw_with_scene`
- `target_mode=canonical_nearest_with_scene`
- `target_clip_quantile=0.99`
- `source_mean_*` baseline metrics
- `feature_mode=canonical_nearest`

The full-D G-v2 canonical-target run shows the core issue clearly:

| Recipe | Feature mode | Val MSE | Zero Val MSE | Source-mean Val MSE | Pred / Zero | Pred / Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `object_transfer_red_blue_to_pink` | `raw` | 0.0007285 | 0.0002929 | 0.0002897 | 2.49x | 2.51x |
| `object_transfer_red_blue_to_pink` | `canonical_nearest` | **0.0002534** | 0.0002929 | 0.0002897 | **0.87x** | **0.87x** |
| `primitive_transfer_push_lift_rotate_to_place_stack` | `raw` | 0.0007204 | 0.0004637 | 0.0004748 | 1.55x | 1.52x |
| `primitive_transfer_push_lift_rotate_to_place_stack` | `canonical_nearest` | 0.0005738 | 0.0004637 | 0.0004748 | 1.24x | 1.21x |
| `articulation_transfer_slider_to_drawer` | `raw` | 0.0023180 | 0.0002932 | 0.0003321 | 7.91x | 6.98x |
| `articulation_transfer_slider_to_drawer` | `canonical_nearest` | 0.0016182 | 0.0002932 | 0.0003321 | 5.52x | 4.87x |
| `articulation_transfer_drawer_to_slider` | `raw` | 0.0236947 | 0.0003087 | 0.0003769 | 76.76x | 62.87x |
| `articulation_transfer_drawer_to_slider` | `canonical_nearest` | 0.0040916 | 0.0003087 | 0.0003769 | 13.25x | 10.86x |

This is the first positive reusable-G signal on full CALVIN D: canonicalizing the target alone is not enough, but canonicalizing both input features and target around the nearest interaction object beats zero/source-mean on object transfer. The same canonicalization improves all other splits but does not cross the trivial baselines there.

Interpretation:

- **Object transfer:** interaction-frame G is promising. This is the cleanest candidate for a policy-side proof-of-concept.
- **Primitive transfer:** canonicalization reduces error but still loses to zero/source mean. The current nearest-object frame likely mixes heterogeneous primitives into one target distribution.
- **Articulation transfer:** canonicalization helps but is still far from enough. Raw nearest-object deltas do not represent drawer/slider state transitions; this branch needs articulation-specific scene-scalar/joint targets and event-style labels.

The paper story should therefore split G into at least two levels:

1. **Reusable object-centric interaction geometry:** nearest-object frame, reusable across object identity/color.
2. **Task/mechanism-specific geometry heads:** articulation and primitive phase/event heads, probably conditioned on language/task class.

Server jobs:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_task_D_D_g_v2_full_ablation_launcher
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-03_task_D_D_g_v2_canonical_feature_followup_launcher
```

The duplicate `raw_with_scene` phase was stopped after the canonical run because it would re-extract identical labels. Future raw/scene ablations should directly reuse existing G label directories.

Local G-v2 sync:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/calvin_task_D_D_g_v2_canonical_2026-07-03
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/calvin_task_D_D_g_v2_canonical_2026-07-03/g_v2_metrics_summary.csv
```

Remote lightweight archive:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/task_D_D_g_v2_canonical_metrics_viz_2026-07-03.tar.gz
```

## Next Experiment

The next round should avoid broad policy training until the representation gate is stronger.

Recommended next configs:

1. **Object-transfer policy smoke:** use `feature_mode=canonical_nearest`, `target_mode=canonical_nearest_with_scene`, `clip=0.99`; test whether predicted G helps a CALVIN policy on red/blue-to-pink transfer.
2. **Primitive phase split:** split push/lift/rotate vs place/stack into separate G heads or add task/language condition to the predictor; current single canonical target is too mixed.
3. **Articulation G-v3:** predict `future_delta_scene_scalars` / joint-relevant scalar channels as the primary target, with object deltas as auxiliary, and evaluate event detection instead of dense raw MSE alone.

## G-v3 Scene-Only Articulation Diagnostic

We tested an articulation-only diagnostic with `target_mode=scene_only`, using the already extracted G labels from the G-v2 run. This target predicts only `future_delta_scene_scalars_h8`.

| Recipe | Feature mode | Val MSE | Zero Val MSE | Pred / Zero |
| --- | --- | ---: | ---: | ---: |
| `articulation_transfer_slider_to_drawer` | `raw` | 0.0008569 | 1.0e-12 | 8.55e8x |
| `articulation_transfer_slider_to_drawer` | `canonical_nearest` | 0.0000696 | 1.0e-12 | 6.95e7x |
| `articulation_transfer_drawer_to_slider` | `raw` | 0.0040525 | 6.1e-9 | 6.59e5x |
| `articulation_transfer_drawer_to_slider` | `canonical_nearest` | 0.0001523 | 6.1e-9 | 2.48e4x |

This is a strong negative result for dense scene-delta regression: the held-out target scene scalar deltas are nearly zero under this split, so the zero baseline is overwhelmingly strong. The canonical feature still reduces predictor error, but the metric/target is not meaningful for reusable articulation G.

Updated interpretation:

- Articulation should not be evaluated primarily as dense future scene-scalar regression.
- The next articulation G should be event/phase-based: e.g. `open/close drawer`, `move slider left/right`, onset/completion flags, or signed progress toward the commanded articulation state.
- Language/task identity is likely necessary for articulation because the same geometry can imply opposite desired progress depending on command direction.

Local G-v3 sync:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/calvin_task_D_D_g_v3_scene_only_articulation_2026-07-03
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/calvin_task_D_D_g_v3_scene_only_articulation_2026-07-03/g_v3_scene_only_metrics_summary.csv
```

## Key Artifacts

Remote archive:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/task_D_D_reusable_g_v1_artifacts_2026-07-03.tar.gz
```

Local archive:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/calvin_task_D_D_reusable_g_v1_archive_2026-07-03/task_D_D_reusable_g_v1_artifacts_2026-07-03.tar.gz
```

Local extracted recipes:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/calvin_task_D_D_reusable_g_v1_archive_2026-07-03/task_D_D_reusable_g_v1_object_transfer_red_blue_to_pink
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/calvin_task_D_D_reusable_g_v1_archive_2026-07-03/task_D_D_reusable_g_v1_primitive_transfer_push_lift_rotate_to_place_stack
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/calvin_task_D_D_reusable_g_v1_archive_2026-07-03/task_D_D_reusable_g_v1_articulation_transfer_slider_to_drawer
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/calvin_task_D_D_reusable_g_v1_archive_2026-07-03/task_D_D_reusable_g_v1_articulation_transfer_drawer_to_slider
```
