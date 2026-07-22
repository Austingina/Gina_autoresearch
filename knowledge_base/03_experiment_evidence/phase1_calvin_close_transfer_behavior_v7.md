# Phase1 CALVIN Close Transfer Behavior Diagnosis - V7

Date: 2026-07-06

## Setting

- Train tasks: `open_drawer`, `move_slider_left`, `move_slider_right`
- Held-out eval task: `close_drawer`
- Task one-hot: zeroed for train/eval
- Policies compared: `no_g`, `articulation_g_v7_unified_field_concat`, `articulation_g_v7_unified_field_film`
- Eval horizon: 160
- Rollouts: 25

Local artifacts:

- Videos: `/Users/wukui/Documents/AutoResearch/artifacts/server_sync/transfer_open_left_right_to_close_2026-07-06/visual_rollouts/2026-07-06_transfer_close_visual_selected_rollouts`
- Step diagnostics: `/Users/wukui/Documents/AutoResearch/artifacts/server_sync/transfer_open_left_right_to_close_2026-07-06/visual_case_selection/articulation_trajectory_steps.json`
- Behavior curves: `/Users/wukui/Documents/AutoResearch/artifacts/server_sync/transfer_open_left_right_to_close_2026-07-06/visual_case_selection/close_behavior_curves_selected.png`

## Main Observation

The scalar success rate underestimates the qualitative difference between the policies.

- `no_g` shows behavior leakage from seen tasks. In close-drawer transfer videos, it often executes motions resembling slider or other seen-task behaviors instead of moving toward the drawer interaction region.
- `v7_concat` moves closer to the drawer/part than `no_g`, but still never enters the effective contact window and does not move the drawer joint.
- `v7_film` shows an early form of close-compatible generalization: it can approach the drawer, enter the near-part window, and sometimes move the drawer joint in the correct direction. Failures often look like spatial/contact-timing errors rather than completely wrong task intent.

## Quantitative Evidence

Aggregate diagnostics on 25 close-drawer rollouts:

| Variant | Success | Category Summary | Median Min Dist |
| --- | ---: | --- | ---: |
| `no_g_e50` | 0/25 | `never_near_part`: 25 | 0.327 |
| `v7_concat_e50` | 0/25 | `never_near_part`: 25 | 0.187 |
| `v7_film_e50` | 1/25 | `success`: 1, `near_insufficient_motion`: 1, `never_near_part`: 23 | 0.292 |

Selected rollout diagnostics:

| Variant / Rollout | Outcome | Min Dist | Near Steps | Final q | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `no_g_e50 / 05` | fail | 0.273 | 0 | 0.220 | Does not contact or move drawer |
| `v7_concat_e50 / 00` | fail | 0.153 | 0 | 0.220 | Better approach, no effective contact |
| `v7_film_e50 / 12` | success | 0.050 | 43 | 0.096 | Enters contact window and closes drawer |
| `v7_film_e50 / 20` | near insufficient motion | 0.030 | 42 | 0.149 | Contacts/approaches but does not close enough |

Mean approach cosine for selected cases:

- `no_g`: negative on representative selected cases, meaning actions are often not directed toward the drawer part.
- `v7_concat`: mildly positive, indicating approach improves but remains outside contact.
- `v7_film`: strongly positive before contact in rollout 12 and 20, indicating the policy is using the field to move toward the intended interaction region.

## Interpretation

This result is positive for the research story, but not yet a solved transfer result.

The important evidence is not `1/25` success alone. The important evidence is that G changes the policy failure mode:

1. Without G, the policy defaults to seen-task motion modes under a held-out task.
2. With concat G, the policy gains some spatial bias toward the target but does not reliably transform that into contact/control.
3. With FiLM G, the policy begins to express a reusable interaction behavior: approach the target part, enter contact, and move along the close-compatible direction.

The remaining bottleneck is likely contact timing and field-to-control calibration, not just high-level task ambiguity.

## Root-Cause Revision

The root cause should not be framed as a case-specific issue like "close earlier/later in rollout 20".

The more general issue is that the current v7 field is object-motion-centric: tokens are placed around the articulated part point, and the field mostly encodes the desired joint-motion direction plus corridor/backstop corrections. This tells the policy how the target part should move, but it does not robustly define the full end-effector interaction manifold:

- where the EEF can apply force,
- from which side/direction the force is effective,
- when an action should switch from approach to force application,
- and how to remain in contact while applying that force.

Additional rollout-level diagnostics:

| Variant | Entered Near Part | Root Bucket |
| --- | ---: | --- |
| `no_g_e50` | 0/25 | `no_effective_contact`: 25 |
| `v7_concat_e50` | 0/25 | `no_effective_contact`: 25 |
| `v7_film_e50` | 2/25 | `success`: 1, `contact_but_insufficient_progress`: 1, `no_effective_contact`: 23 |

This revises the earlier interpretation:

- `v7_film` does show a correct interaction mode in some cases.
- But it is not yet stable across cases.
- Therefore, the main failure is not only premature close-axis motion. The broader failure is unstable entrance into and maintenance of the effective contact manifold.

## Next Implication

The next G design should preserve the compact field idea, but explicitly evaluate whether it improves:

- Approach correctness: action direction toward the intended part before contact.
- Contact window entry: fraction of rollouts entering `near_part`.
- Contact-conditioned motion: action along the task-effective direction while in the near/contact window.
- Joint progress after contact: decrease in drawer q for `close_drawer`.

## Chunk BC/FM Axis-Contact Diagnostic

Date: 2026-07-07

Question:

```text
Do the close failures "satisfy" the drawer axis in a geometrically strange but
task-ineffective way?
```

Diagnostic:

```text
policies: chunk_bc_v7 e16, chunk_bc_v7 e50, chunk_fm_v7 e50
task: close_drawer
rollouts: 25 per policy
metrics: EEF-to-part distance, EEF-to-drawer-axis-line distance, EEF height
offset, action projection on desired drawer axis, desired q delta
```

Local synced artifact:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-07_transfer_chunk_bc_vs_fm_v7_h16/2026-07-07_close_axis_contact_diagnostic_chunk_bc_fm_v7/
```

Compact result:

| Policy | Success | Main buckets | Median min part dist | Median min axis-line dist | Median near-part steps |
| --- | ---: | --- | ---: | ---: | ---: |
| `bc_e16` | 1/25 | 16 near-part/no-progress, 6 axis-line/wrong-offset, 2 no-axis-contact | 0.034 m | 0.028 m | 50 |
| `bc_e50` | 0/25 | 19 near-part/no-progress, 6 no-axis-contact | 0.053 m | 0.053 m | 39 |
| `fm_e50` | 0/25 | 25 no-axis-contact | 0.180 m | 0.158 m | 0 |

Important detail:

- `fm_e50` looked close in top-down video, but state diagnostics say it never
  entered the geometry-defined close-drawer contact region. Its closest recorded
  state was still about 0.165 m from the drawer part point, about 0.148 m from
  the drawer axis line, and about 0.15 m above the part point. This suggests the
  top-down view can make the EEF look handle-adjacent while it is vertically or
  spatially outside the effective contact manifold.
- `bc_e50` is different: it often reaches the near-part region, but the action
  projection on the desired close axis is tiny around contact and drawer q
  progress remains insufficient. This is closer to "near the handle but not
  producing a useful push."

Updated interpretation:

```text
FM close failure: mostly invalid approach/contact geometry under the real
drawer part-point/axis geometry, hidden by the top-down render.

BC close failure: often valid proximity but poor post-contact actuation.

Therefore the bottleneck is not a single generic "push stage" issue. It splits
into (1) contact-manifold localization for FM and (2) contact-conditioned force
application for BC.
```

Avoid adding a detailed hand-engineered state bundle. The next iteration should target a better policy-facing field and better injection/training objective, not a larger set of symbolic terms.

## V7 G Scale Sweep

Date: 2026-07-08

Question:

```text
Were the close failures partly caused by the v7 field being too weak at
closed-loop inference time?
```

Setup:

```text
base checkpoints: chunk_bc_v7 e50, chunk_fm_v7 e50
train tasks: open_drawer, move_slider_left, move_slider_right
held-out task: close_drawer
eval rollouts: 25 close_drawer per scale
scales: 0, 0.5, 1, 2, 4
implementation: multiply articulation_g_v7_unified_field by g_scale before
condition normalization inside the policy wrapper
```

Local synced artifact:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-07_transfer_chunk_bc_vs_fm_v7_h16/2026-07-07_close_g_scale_sweep_chunk_bc_fm_v7_e50_n25_h160/
```

Close transfer result:

| Policy | g=0 | g=0.5 | g=1 | g=2 | g=4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bc_e50` | 0/25 | 0/25 | 0/25 | 19/25 | 0/25 |
| `fm_e50` | 0/25 | 0/25 | 0/25 | 0/25 | 0/25 |

Selected `bc_e50 g=2` videos:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-07_transfer_chunk_bc_vs_fm_v7_h16/2026-07-07_close_g_scale_sweep_chunk_bc_fm_v7_e50_n25_h160/visual_rollouts_bc_e50_g2_selected/
```

The successful selected cases include rollout 02, 04, and 10. The failed
selected cases include rollout 00, 03, 09, 15, and 21.

Axis/contact diagnostic for `bc_e50 g=2`:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-07_transfer_chunk_bc_vs_fm_v7_h16/2026-07-07_close_axis_contact_diagnostic_bc_e50_g2/
```

Compact diagnostic result:

| Policy | Success | Categories | Median min part dist | Median min axis-line dist | Median near-part steps |
| --- | ---: | --- | ---: | ---: | ---: |
| `bc_e50 g=2` | 19/25 | 19 success, 6 near-part/no-progress | 0.017 m | 0.014 m | 59 |

Important diagnostic detail:

- The `bc_e50 g=2` successes are not explained by a strange axis-satisfaction
  artifact. The diagnostic found no weird-axis-satisfied steps and no
  wrong-side-axis-command steps at the median level.
- The 6 failures are all `near_part_no_joint_progress`: the policy reaches the
  relevant part/axis neighborhood but does not create enough drawer joint
  progress.

Train-task non-regression check:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-07_transfer_chunk_bc_vs_fm_v7_h16/2026-07-08_train_task_eval_bc_e50_gscale1_2_n25_h160/
```

| Policy | Scale | Train total | Left | Right | Open |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bc_e50` | 1 | 71/75 | 22/25 | 24/25 | 25/25 |
| `bc_e50` | 2 | 5/75 | 1/25 | 3/25 | 1/25 |

Updated interpretation:

```text
The v7 field contains useful transferable information for close_drawer under a
deterministic chunk BC policy, but its effective gain is not calibrated.

The evidence is stronger than the previous 0/25 or 1/25 close result: with
g_scale=2, the same e50 BC checkpoint reaches 19/25 close success and the
geometry diagnostics confirm real near-part/near-axis contact.

However, the same global gain destroys the original train tasks. Therefore the
problem is not simply "make G stronger". The policy needs gain normalization,
state-conditioned gain, task-conditioned gain, training-time scale augmentation,
or a field injection mechanism that preserves the train-task operating point
while allowing stronger transfer guidance when the default field is too weak.
```

Immediate consequence:

```text
Do not discard the compact field representation yet. The new result shows that
v7 G is semantically useful for transfer, but the current policy treats its
magnitude as a fragile control knob. The next experiment should target G gain
calibration and robustness, not a larger hand-engineered G vector.
```

## Gated/Scale-Aug G2 Axis-Contact Diagnostic

Date: 2026-07-08

Question:

```text
Do the scale_aug/gated failures fail after reaching the handle, or do they fail
before establishing useful target-part contact?
```

Setup:

```text
same close_drawer initial states: 25 first-task rollouts
same g_scale: 2
policies:
  - original non-augmented chunk_bc_v7 e50
  - chunk_bc_v7_scale_aug e50
  - chunk_bc_v7_gated e50
diagnostic script:
  framework/scripts/policy_state_bc/diagnose_chunk_flow_axis_contact.py
```

Local artifact:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-07_transfer_chunk_bc_vs_fm_v7_h16/2026-07-08_close_axis_contact_diagnostic_bc_scaleaug_gated_g2_n25_h160/
```

Remote artifact:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-08_close_axis_contact_diagnostic_bc_scaleaug_gated_g2_n25_h160
```

Compact result:

| Policy | Success | Categories | Median min part dist | Median min axis dist | Median first part step | Median q progress after part |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `bc_e50_g2` | 19/25 | 19 success, 6 near-part/no-progress | 0.017 m | 0.014 m | 60 | 0.598 |
| `scale_aug_e50_g2` | 0/25 | 25 no-axis-contact | 0.241 m | 0.139 m | n/a | n/a |
| `gated_e50_g2` | 0/25 | 22 no-axis-contact, 3 near-axis/wrong-offset | 0.229 m | 0.147 m | n/a | n/a |

Action/joint projection detail:

| Policy | Median pre-contact action projection | Median post-contact action projection | Median post-contact q delta sum |
| --- | ---: | ---: | ---: |
| `bc_e50_g2` | -0.0033 | 0.0024 | 0.1321 |
| `scale_aug_e50_g2` | 0.00036 | n/a | n/a |
| `gated_e50_g2` | 0.00006 | n/a | n/a |

Important diagnostic detail:

- `bc_e50_g2` enters the part neighborhood around step 60 and then produces
  real drawer progress after contact.
- `scale_aug_e50_g2` never reaches near-part or near-axis contact in any of
  the 25 rollouts.
- `gated_e50_g2` almost never reaches near-axis contact. In the 3 near-axis
  cases, it is still at the wrong offset: median part-axis offset among those
  cases is about 0.15 m while near-part is never satisfied.

Updated interpretation:

```text
The gated/scale_aug failure is not primarily "reaches handle but fails to push".
It is earlier: the learned interface no longer maps G to target-part approach.

Scale augmentation appears to make the policy robust/invariant to G magnitude
on seen tasks, but this removes the narrow raw-concat extrapolation that made
bc_e50_g2 approach the drawer handle and close successfully.

The gated residual also does not solve semantic binding: it can preserve seen
tasks, but for held-out close it does not bind the desired drawer motion to a
specific spatial contact target.
```

Consequence:

```text
The next useful test is not another global G scale or scalar gate sweep. The
next interface should make G spatially queryable or tokenized around the target
part/current EEF, then test whether the policy can recover target-part approach
under close_drawer with zero-G/shuffled-G/sign-flipped-G controls.
```

## Language / Oracle Query-G Round 1

Date: 2026-07-08

Question:

```text
If language/task context is assumed to generate a reasonable oracle query-G,
can a deterministic chunk BC policy use that reusable G interface to improve
seen-task control and zero-shot transfer from open/slider tasks to close_drawer?
```

Setup:

```text
train tasks: open_drawer, move_slider_left, move_slider_right
held-out eval task: close_drawer
epochs: 50
rollouts: 25 per task
horizon: 160
modes:
  chunk_bc_state
  chunk_bc_lang
  chunk_bc_query_g
  chunk_bc_lang_query_g
```

`query_g` is an oracle task-conditioned field query, not a vision-only learned
G. It assumes the upstream language/task module has already selected the
intended interaction and generates four compact spatial tokens:

```text
EEF token, target part token, contact anchor token, side anchor token
```

Each token contains compact type, relative geometry, field direction, desired
object flow, distance/progress/contact confidence fields. This is intended as a
first test of whether G can act as a reusable policy-facing interface.

Code:

```text
/Users/wukui/Documents/AutoResearch/framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py
/Users/wukui/Documents/AutoResearch/framework/scripts/policy_state_bc/run_transfer_language_query_g_seed0.sh
```

Local synced artifacts:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-08_language_query_g_round1/
```

Remote artifacts:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-08_transfer_language_query_g_e50_n25_h160_seed0
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-08_language_query_g_train_task_eval_e50_n25_h160
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-08_language_query_g_close_axis_contact_e50_n25_h160
```

Held-out `close_drawer` result:

| Mode | Train-val chunk MSE | Close success |
| --- | ---: | ---: |
| `chunk_bc_state` | 0.0458 | 0/25 |
| `chunk_bc_lang` | 0.0418 | 0/25 |
| `chunk_bc_query_g` | 0.0389 | 0/25 |
| `chunk_bc_lang_query_g` | 0.0389 | 0/25 |

Seen-task result:

| Mode | Total | Left | Right | Open |
| --- | ---: | ---: | ---: | ---: |
| `state_e50` | 50/75 | 17/25 | 12/25 | 21/25 |
| `lang_e50` | 70/75 | 23/25 | 22/25 | 25/25 |
| `query_g_e50` | 73/75 | 24/25 | 24/25 | 25/25 |
| `lang_query_g_e50` | 70/75 | 21/25 | 24/25 | 25/25 |

Close axis/contact diagnostic:

| Mode | Success | Main category | Median min part dist | Median min axis-line dist | Median near-part steps |
| --- | ---: | --- | ---: | ---: | ---: |
| `state_e50` | 0/25 | 24 no-axis-contact, 1 near-axis wrong-offset | 0.327 m | 0.217 m | 0 |
| `lang_e50` | 0/25 | 25 no-axis-contact | 0.327 m | 0.271 m | 0 |
| `query_g_e50` | 0/25 | 25 no-axis-contact | 0.251 m | 0.218 m | 0 |
| `lang_query_g_e50` | 0/25 | 25 no-axis-contact | 0.269 m | 0.259 m | 0 |

Interpretation:

```text
Oracle query-G is not meaningless: it improves offline MSE and improves
seen-task control relative to state-only and language-only baselines.

However, the current flat deterministic chunk BC interface still cannot transfer
to close_drawer. In close, even query-G policies mostly fail before contact:
they do not enter the effective target-part/contact manifold.

Therefore this first query-G result supports "G contains useful information",
but falsifies the stronger claim that a simple concatenated oracle query-G is
already a sufficient reusable transfer interface.
```

Important distinction:

```text
This result is different from the previous raw v7 g_scale=2 result.

Raw v7 g_scale=2 achieved 19/25 close success but collapsed train tasks at the
same scale. The new query-G interface is stable and strong on seen tasks, but
does not produce held-out close contact. The immediate bottleneck is semantic
and spatial binding from the query tokens to target-part approach, not only
field magnitude.
```

## Small-Shot Close G-Binding Diagnostic

Date: 2026-07-08

Question:

```text
If zero-shot close is too harsh, does oracle query-G reduce the number of
close_drawer demos needed to acquire the close control mode?

More importantly, does the learned policy causally depend on query-G at eval
time, or does it simply memorize the few close demos?
```

Implementation:

```text
train tasks: open_drawer, move_slider_left, move_slider_right, plus k close_drawer demos
k: 1, 2, 5, 10, 25
epochs: 50
eval: 25 close_drawer first-task rollouts
modes:
  chunk_bc_state
  chunk_bc_lang
  chunk_bc_query_g
  chunk_bc_lang_query_g
```

Code additions:

```text
--task-segment-limits close_drawer=K
run_smallshot_language_query_g_seed0.sh
run_smallshot_query_g_corruption_eval_seed0.sh
```

Local/remote artifacts:

```text
remote:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-08_smallshot_language_query_g_e50_n25_h160_seed0
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-08_smallshot_query_g_corruption_eval_n25_h160_seed0

local sync pending:
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-08_smallshot_language_query_g/
```

Close success, 25 rollouts:

| k close demos | State | Lang | Query-G | Lang+Query-G |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0/25 | 0/25 | 0/25 | 0/25 |
| 1 | 0/25 | 0/25 | 23/25 | 21/25 |
| 2 | 0/25 | 11/25 | 25/25 | 25/25 |
| 5 | 0/25 | 25/25 | 25/25 | 25/25 |
| 10 | 1/25 | 25/25 | 25/25 | 25/25 |
| 25 | 7/25 | 25/25 | 25/25 | 25/25 |

Eval-time query-G corruption, 25 rollouts:

| k | Policy | Normal | Zero G | Sign-Flip G | Open-Drawer G |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `query_g` | 23/25 | 0/25 | 1/25 | 0/25 |
| 1 | `lang_query_g` | 21/25 | 0/25 | 7/25 | 0/25 |
| 2 | `query_g` | 25/25 | 0/25 | 0/25 | 0/25 |
| 2 | `lang_query_g` | 25/25 | 0/25 | 11/25 | 0/25 |
| 5 | `query_g` | 25/25 | 0/25 | 0/25 | 0/25 |
| 5 | `lang_query_g` | 25/25 | 0/25 | 0/25 | 0/25 |

Interpretation:

```text
This is the strongest evidence so far that query-G can act as a reusable
interaction interface.

With only 1 close demo, language alone cannot solve close_drawer, but query-G
reaches 23/25. With 2 close demos, language is partial at 11/25 while query-G
is already 25/25.

The corruption controls are decisive: zeroing query-G or replacing close query-G
with open_drawer query-G destroys success. Sign-flipping query-G also nearly
destroys success, especially for query-G-only policies.

Therefore, in this regime, the policy is not just memorizing a few close demos.
It is using the task-derived query-G signal causally.
```

Caveat:

```text
This is still same-environment CALVIN close transfer, not broad OOD
generalization. The 100-rollout confirmation below is now complete for k=1/2.
```

100-rollout confirmation:

| k close demos | State | Lang | Query-G | Lang+Query-G |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0/100 | 1/100 | 89/100 | 94/100 |
| 2 | 0/100 | 48/100 | 100/100 | 100/100 |

100-rollout corruption confirmation:

| k | Policy | Normal | Zero G | Sign-Flip G | Open-Drawer G |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `query_g` | 89/100 | 0/100 | 7/100 | 0/100 |
| 1 | `lang_query_g` | 94/100 | 0/100 | 43/100 | 0/100 |
| 2 | `query_g` | 100/100 | 0/100 | 0/100 | 0/100 |
| 2 | `lang_query_g` | 100/100 | 0/100 | 40/100 | 0/100 |

The `lang_query_g` sign-flip rows retain partial success because the language
one-hot still provides the correct close task identity. The decisive rows for
G-causality are zero-G and open-drawer-G: both collapse to 0/100 even when
language is present.

## Ablation C: Broader Primitive Scope for Zero-Shot Close

Question:

```text
Does k=0 zero-shot close fail because the train scope is too narrow, rather than
because the policy class cannot compose close-like behavior?
```

Date: 2026-07-09

Train tasks:

```text
open_drawer
move_slider_left
move_slider_right
push_into_drawer
push_blue_block_left
push_blue_block_right
push_pink_block_left
push_pink_block_right
push_red_block_left
push_red_block_right
turn_on_lightbulb
turn_off_lightbulb
turn_on_led
turn_off_led
```

Held-out eval task:

```text
close_drawer, 100 rollouts, ep_len 160
```

Result:

| Mode | Close Success |
| --- | ---: |
| `chunk_bc_state` | 0/100 |
| `chunk_bc_lang` | 100/100 |
| `chunk_bc_query_g` | 16/100 |
| `chunk_bc_lang_query_g` | 100/100 |

Training/eval notes:

```text
actual segments: 2107
actual windows: 48069
epochs: 50
chunk size: 16
stride: 2
```

Interpretation:

```text
This is strong evidence that the earlier zero-shot close failure was largely a
training-scope problem. With only open_drawer + slider left/right, the policy
did not see enough push-in / negative-axis / task-switching primitives. After
adding those non-close tasks, language-conditioned chunk BC solves held-out
close_drawer at 100/100.

State-only remains 0/100, so the success is not just visual proximity or scene
bias. Task context is essential.

Query-G-only reaches 16/100, better than state-only but far below language.
The current hand-built query-G does not contain enough task binding on its own.
This supports using language as the goal selector and G as a grounded reusable
interaction interface, rather than trying to make G replace language.
```

Artifacts:

```text
remote:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_e50_n100_h160_seed0
/mnt/nvme33n1/wuk_code/AutoResearch/logs/scope_primitive_zero_shot_close_20260709_wait.log

local:
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-09_scope_primitive_zero_shot_close/
```

## Ablation D: Eval-Time Language Control on Primitive-Scope Checkpoints

Question:

```text
Does the primitive-scope close success really depend on the close task context,
or is it a state/eval artifact? If query-G is present, can it override missing
or wrong language?
```

Date: 2026-07-09

Important implementation detail:

```text
The current "language" condition is a task one-hot, not a natural-language
embedding. Since close_drawer is held out from training in Ablation C, the
close_drawer one-hot dimension is not updated by supervised training. Therefore
this control checks dependency on the eval-time task key, but it does not yet
prove semantic language generalization.
```

Eval protocol:

```text
Checkpoints:
  primitive-scope chunk_bc_lang
  primitive-scope chunk_bc_lang_query_g

Eval task:
  close_drawer, 100 rollouts

Query-G control:
  normal

Language controls:
  normal
  zero
  task:open_drawer
  task:push_into_drawer
  task:move_slider_left
  task:move_slider_right
```

Result:

| Policy | normal | zero | open | push_into | slider_left | slider_right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chunk_bc_lang` | 100/100 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 |
| `chunk_bc_lang_query_g` | 100/100 | 0/100 | 0/100 | 0/100 | 0/100 | 1/100 |

Interpretation:

```text
The success is not a state-only or automatic-success artifact: zeroing the task
input collapses both language-conditioned policies.

It is also not simply "push_into_drawer solves close": replacing the task input
with push_into_drawer gives 0/100 for both policies.

The current query-G cannot independently recover the close behavior when the
language/task key is missing or wrong. This means that under the present concat
fusion and hand-built query-G, G is an auxiliary interaction feature, not a
dominant reusable control interface.

The next necessary control is seed replication and/or replacing task one-hot
with a real language/text embedding. Otherwise Ablation C should be treated as
evidence for task-key-conditioned transfer under broader scope, not semantic
zero-shot language generalization.
```

Artifacts:

```text
remote:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_language_control_n100
/mnt/nvme33n1/wuk_code/AutoResearch/logs/scope_primitive_zero_shot_close_language_control_20260709.log

local:
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-09_scope_primitive_zero_shot_close_language_control/
```

## Ablation E: Primitive-Scope Seed Replication

Question:

```text
Is the primitive-scope close result stable across training seeds?
```

Date: 2026-07-09

Protocol:

```text
Same train/eval setup as Ablation C.
Modes:
  chunk_bc_lang
  chunk_bc_lang_query_g

Seeds:
  0, 1, 2
```

Result:

| Seed | `chunk_bc_lang` | `chunk_bc_lang_query_g` |
| ---: | ---: | ---: |
| 0 | 100/100 | 100/100 |
| 1 | 100/100 | 100/100 |
| 2 | 100/100 | 0/100 |

Interpretation:

```text
Language/task-key-conditioned primitive-scope transfer is stable across the
three tested seeds.

Adding query-G by simple concat is not stable: seed2 collapses from 100/100
language-only to 0/100 with language+query-G, despite low validation MSE. This
means current query-G fusion can harm closed-loop behavior even when offline
chunk prediction looks good.

This strengthens the short-term conclusion:
  use language/task context as the reliable goal selector;
  do not claim the current concat query-G is a stable improvement;
  if G is kept, it needs a safer fusion mechanism, stronger regularization, or
  a design where G modulates an intent/affordance bottleneck rather than being
  concatenated as extra raw conditioning.
```

Artifacts:

```text
remote:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_e50_n100_h160_seed1
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_e50_n100_h160_seed2
/mnt/nvme33n1/wuk_code/AutoResearch/logs/scope_primitive_zero_shot_close_multiseed_seed1_seed2_20260709.log

local:
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-09_scope_primitive_zero_shot_close_multiseed/
```

## Ablation F: Seed2 Lang vs Lang+Query-G Failure Diagnosis

Question:

```text
For seed2, why does chunk_bc_lang solve close_drawer 100/100 while
chunk_bc_lang_query_g collapses to 0/100?
```

Date: 2026-07-09

Policy checkpoints:

```text
seed2_lang:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_e50_n100_h160_seed2/chunk_bc_lang/policy.pt

seed2_lang_query_g:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_e50_n100_h160_seed2/chunk_bc_lang_query_g/policy.pt
```

Axis/contact diagnostic, 100 close rollouts:

| Policy | SR | Category | min dist to part p50 | min dist to axis-line p50 | joint progress p50 |
| --- | ---: | --- | ---: | ---: | ---: |
| `seed2_lang` | 100/100 | success: 100 | 0.065 m | 0.062 m | 0.162 |
| `seed2_lang_query_g` | 0/100 | no_axis_contact: 100 | 0.327 m | 0.271 m | 0.000 |

Rollout-0 first-step behavior:

```text
seed2_lang:
  action xyz = [+0.02, -0.02, -0.02]
  distance to part decreases from 0.327 to 0.065 by step 11
  eef final xyz approx [0.262, -0.278, 0.360]
  drawer joint progresses by about 0.162 desired units

seed2_lang_query_g:
  action xyz = [+0.02, +0.02, +0.02]
  distance to part increases from 0.327 to 0.525 by step 11
  eef final xyz approx [0.260, -0.225, 1.259]
  drawer joint progress is 0
```

Query-G control on the seed2 lang+query-G checkpoint:

| Query-G control | Close success |
| --- | ---: |
| normal close query-G | 0/100 |
| zero query-G | 0/100 |
| sign-flip query-G | 0/100 |
| open-drawer query-G | 0/100 |

Interpretation:

```text
The seed2 failure is not a contact-timing failure after reaching the handle.
It fails at the approach stage: lang+query-G moves the EEF away from the drawer
interaction region, especially in y/z, and never establishes useful axis
contact.

The eval-time query-G controls show that the failure is not caused only by the
specific close query-G value at inference time. Zeroing or replacing G does not
recover the language-only behavior. The concat-G training has changed the
policy's internal control allocation; the model no longer preserves the robust
language-conditioned approach manifold.

This is direct evidence against continuing with raw concat G as the main
interface. G needs a safer fusion scheme or objective: gated/regularized
modulation, FiLM with identity initialization, an intent bottleneck, or training
losses that explicitly preserve approach/contact behavior.
```

Videos:

```text
local:
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-09_seed2_lang_vs_lang_query_g_diag/2026-07-09_seed2_lang_vs_lang_query_g_videos/

remote:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_seed2_lang_vs_lang_query_g_videos
```

Diagnostics:

```text
local:
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-09_seed2_lang_vs_lang_query_g_diag/
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-09_seed2_lang_query_g_query_control/

remote:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_seed2_lang_vs_lang_query_g_axis_diag_n100
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_seed2_lang_query_g_query_control_n100
```

## Ablation G: Task One-Hot Normalization Artifact

Question:

```text
Why does held-out close_drawer reach 100/100 with chunk_bc_lang, even though
the "language" input is only a task one-hot and close_drawer is not in the
training task set?
```

Date: 2026-07-09

Finding:

```text
The current chunk policy normalizes every condition dimension as:

  cond_n = (cond - cond_mean) / cond_std
  cond_std = train_std + 1e-6

For a held-out task one-hot dimension such as close_drawer, the training value
is always 0. Therefore:

  cond_mean(close_drawer) = 0
  cond_std(close_drawer)  = 1e-6
  eval close_drawer one-hot value 1 becomes about 1e6
```

Checkpoint verification:

| Checkpoint | close one-hot mean | close one-hot std | normalized eval value |
| --- | ---: | ---: | ---: |
| seed0 `chunk_bc_lang` | 0.0 | 1e-6 | ~1,000,000 |
| seed2 `chunk_bc_lang` | 0.0 | 1e-6 | ~1,000,000 |
| seed2 `chunk_bc_lang_query_g` | 0.0 | 1e-6 | ~1,000,000 |

Interpretation:

```text
The primitive-scope chunk_bc_lang 100/100 result is not valid evidence of
semantic language-conditioned zero-shot transfer. It is dominated by an
out-of-distribution one-hot normalization artifact.

Because the close_drawer task dimension is never trained but is amplified to
~1e6 at eval, small random differences in first-layer initialization and in the
trained downstream weights can produce saturated, seed-dependent closed-loop
policies.

This also explains why seed2 lang+query-G can collapse while seed0/1 do not:
raw concat G changes the learned control allocation under a numerically
unstable, untrained task-key dimension. Offline validation MSE does not reveal
this because the validation split contains only seen train tasks.
```

Code fix:

```text
For chunk_bc_lang and chunk_bc_lang_query_g, task one-hot dimensions should not
be standardized by train variance. They are now assigned:

  cond_mean[task_slice] = 0
  cond_std[task_slice] = 1

This prevents held-out one-hot dimensions from exploding at eval.
```

Status:

```text
All primitive-scope results using task one-hot normalization should be treated
as diagnostic artifacts until rerun with the fixed normalization or replaced by
a real language/text embedding.
```

Fixed-normalization rerun:

```text
Date: 2026-07-09

Change:
  task one-hot condition dims use mean 0 and std 1.
  robot/scene/query-G dims keep the original train-set standardization.

Train tasks:
  open_drawer
  move_slider_left
  move_slider_right
  push_into_drawer
  push_blue_block_left / right
  push_pink_block_left / right
  push_red_block_left / right
  turn_on_lightbulb / turn_off_lightbulb
  turn_on_led / turn_off_led

Eval task:
  close_drawer only

Training:
  chunk size 16, stride 2, epochs 50, batch size 512, hidden 512, layers 4

Eval:
  100 close_drawer rollouts per seed, episode length 160
```

Results:

| Seed | `chunk_bc_lang` | `chunk_bc_lang_query_g` |
| ---: | ---: | ---: |
| 0 | 0/100 | 19/100 |
| 1 | 0/100 | 0/100 |
| 2 | 0/100 | 0/100 |

Training metrics at epoch 50:

| Seed | Mode | val chunk MSE | first gripper acc |
| ---: | --- | ---: | ---: |
| 0 | `chunk_bc_lang` | 0.0467 | 0.9715 |
| 0 | `chunk_bc_lang_query_g` | 0.0479 | 0.9692 |
| 1 | `chunk_bc_lang` | 0.0462 | 0.9629 |
| 1 | `chunk_bc_lang_query_g` | 0.0487 | 0.9576 |
| 2 | `chunk_bc_lang` | 0.0483 | 0.9661 |
| 2 | `chunk_bc_lang_query_g` | 0.0458 | 0.9661 |

Interpretation after fix:

```text
The previous primitive-scope chunk_bc_lang 100/100 result is invalidated.
With the one-hot normalization artifact removed, task one-hot alone gives
0/300 close_drawer zero-shot success.

The query-G interface gives a nonzero but unstable signal:
  seed0 improves from 0/100 to 19/100,
  seed1 and seed2 remain 0/100.

This is not yet evidence of robust zero-shot transfer. It is evidence that
query-G can sometimes bias the closed-loop policy toward the held-out close
behavior, but the current raw concat fusion and current task-key conditioning
are not stable enough for a main result.

Offline validation MSE is not predictive for this OOD transfer setting: all
six checkpoints have similar val MSE and gripper accuracy, while only one
query-G checkpoint obtains any close success.
```

Artifacts:

```text
remote log:
/mnt/nvme33n1/wuk_code/AutoResearch/logs/scope_primitive_zero_shot_close_fixed_tasknorm_seed0_seed1_seed2_20260709.log

remote outputs:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_fixed_tasknorm_e50_n100_h160_seed0
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_fixed_tasknorm_e50_n100_h160_seed1
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_fixed_tasknorm_e50_n100_h160_seed2

local sync:
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-09_scope_primitive_zero_shot_close_fixed_tasknorm/metrics_only/
```

## Ablation H: Paired-Seed Causal Audit with Disjoint Close Pools

Question:

```text
Is the seed0 close success caused by reusable query-G conditioning, or by
changes in the demonstration split / optimization randomness / eval cases?
```

Date started: 2026-07-10

Training protocol:

```text
Held-out task:
  close_drawer (zero train segments and zero train windows)

Train tasks (all retained):
  open_drawer
  move_slider_left / move_slider_right
  push_into_drawer
  push_blue_block_left / right
  push_pink_block_left / right
  push_red_block_left / right
  turn_on_lightbulb / turn_off_lightbulb
  turn_on_led / turn_off_led

Compared policies:
  chunk_bc_lang
  chunk_bc_lang_query_g

Train seeds:
  0, 1, 2, 3, 4

Controlled randomness:
  data_seed = 0
  split_seed = 0
  train_seed = 0..4

Training:
  epochs 50
  checkpoints 10, 20, 30, 40, 50
  chunk size 16, stride 2
  batch size 512, hidden 512, layers 4
```

Close evaluation pools:

```text
Both pools are selected from the same deterministic get_sequences(50000) pool.

diagnostic validation:
  first 100 matching close_drawer cases
  eval_sequence_offset = 0

independent test:
  next 300 matching close_drawer cases
  eval_sequence_offset = 100

The evaluator records selected_sequence_indices in every summary so overlap can
be audited directly. Close validation is not used to select a favorable epoch;
the predeclared final checkpoint is epoch 50.
```

Final-checkpoint causal controls on the fixed validation pool:

```text
normal close query-G
zero query-G
sign-flipped query-G
wrong-task query-G generated as open_drawer from the same current state
```

Why there is no naive vector shuffle:

```text
Online evaluation has batch size one and query-G is recomputed from the current
EEF/object state. Randomly permuting its 76 dimensions would create a malformed
feature vector rather than a valid mismatched interaction field. The wrong-task
control preserves the representation schema and geometry while corrupting task
binding; sign flip separately corrupts the intended direction.
```

Decision criteria:

```text
1. correct G must exceed language-only in median success across seeds;
2. correct G must exceed zero, sign-flip, and wrong-task G;
3. the benefit must reproduce on the disjoint test pool;
4. selected close sequence indices must be identical across policies/seeds and
   disjoint between validation and test;
5. close remains absent from every run_meta training-task/window count.
```

Implementation and queue status:

```text
server scheduler PID: 1852921
maximum parallel jobs: 2
launch threshold: <=20 GB used memory and <=20% GPU utilization

scheduler log:
/mnt/nvme33n1/wuk_code/AutoResearch/logs/2026-07-10_scope_primitive_zero_shot_close_paired_full_scheduler.log

per-seed logs/status:
/mnt/nvme33n1/wuk_code/AutoResearch/logs/2026-07-10_scope_primitive_zero_shot_close_paired/

planned outputs:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-10_scope_primitive_zero_shot_close_paired_e50_n100_h160_seed{0..4}
```

Completed: 2026-07-10 20:52 server time

Protocol audit:

```text
All five runs:
  close_drawer train segments = 0
  close_drawer train windows  = 0
  data_seed = 0
  split_seed = 0

All validation summaries:
  identical 100 selected sequence indices

All independent-test summaries:
  identical 300 selected sequence indices

validation/test index overlap:
  0
```

Normal-G results:

| Seed | `lang` validation | `lang+query-G` validation | `lang` test | `lang+query-G` test |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0/100 | 18/100 | 0/300 | 53/300 |
| 1 | 0/100 | 0/100 | 0/300 | 0/300 |
| 2 | 0/100 | 0/100 | 0/300 | 0/300 |
| 3 | 0/100 | 0/100 | 0/300 | 1/300 |
| 4 | 0/100 | 0/100 | 0/300 | 0/300 |

Query-G causal controls on validation:

| Seed | correct close G | zero G | sign-flip G | open-drawer G |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 18/100 | 0/100 | 1/100 | 0/100 |
| 1 | 0/100 | 0/100 | 0/100 | 0/100 |
| 2 | 0/100 | 0/100 | 0/100 | 3/100 |
| 3 | 0/100 | 0/100 | 17/100 | 22/100 |
| 4 | 0/100 | 0/100 | 0/100 | 0/100 |

Conclusion:

```text
The seed0 behavior is real rather than validation-case luck: 18/100 validation
and 53/300 disjoint-test success are statistically consistent. Within seed0,
correct G is causally important because zero, sign-flip, and wrong-task G reduce
success to 0, 1, and 0.

However, the reusable semantics are not stable across optimization seeds. The
five-seed median is 0. Seed3 succeeds more often with sign-flipped or open-task
G than with correct close G, showing that raw concat does not enforce a common
mapping from G direction/task binding to action direction.

The current held-out task one-hot is also intrinsically problematic: its close
dimension receives no supervised update, so its first-layer weights retain a
seed-dependent random contribution. Fixed normalization removed the previous
1e6 amplification, but it cannot make an unseen one-hot semantically meaningful.

Therefore this round supports a narrow claim:
  a useful zero-shot close control mode exists and can be selected by query-G;
  the current language-one-hot + raw-concat interface does not learn that mode
  reliably or with seed-consistent G semantics.
```

## Ablation I: Query-G-Only Paired-Seed Isolation

Question:

```text
Does removing the untrained held-out close one-hot make reusable query-G
transfer stable, or is the instability already present in the G-to-action map?
```

Protocol:

```text
Train scope, data/split seeds, checkpoints, validation pool, independent test
pool, and G controls are identical to Ablation H.

Only policy condition changes:
  state + query-G
  no language/task one-hot input

Seeds:
  0, 1, 2, 3, 4
```

Status:

```text
cancelled before any seed/job started
```

Cancellation reason:

```text
The current query-G builder is defined only for drawer/slider articulation tasks.
For block-push and light tasks it returns all-zero G. Removing language while
retaining the complete 14-task scope would therefore make opposite block-push
tasks indistinguishable and introduce a new multimodal-conflict confound. This
would not cleanly isolate the unseen close one-hot.
```

## Ablation J: Eval-Time Held-Out Language-Key Removal

Question:

```text
Is cross-seed close instability caused mainly by the untrained close task-one-hot
weight, while the correctly generated close query-G is already sufficient?
```

Protocol:

```text
Reuse the five Ablation-H chunk_bc_lang_query_g epoch-50 checkpoints.
Do not retrain and do not change the train scope.

At close eval:
  language_control = zero
  query_g_control = normal

Evaluate on the exact same pools:
  validation offset 0, n=100
  independent test offset 100, n=300
```

This intervention removes only the held-out close one-hot at inference. It
preserves the trained network, state input, correct state-conditioned query-G,
initial conditions, episode length, and task success criterion.

Status:

```text
completed on 2026-07-11 09:28 server time
maximum parallel jobs: 2
launch threshold: <=132 GB used memory (about >=12 GB free) and <=90% GPU utilization

scheduler log:
/mnt/nvme33n1/wuk_code/AutoResearch/logs/2026-07-11_scope_primitive_zero_shot_close_language_zero_relaxed_scheduler.log

per-seed logs/status:
/mnt/nvme33n1/wuk_code/AutoResearch/logs/2026-07-11_scope_primitive_zero_shot_close_language_zero/
```

Decision:

```text
If language-zero makes correct-G transfer consistent across seeds, the primary
bottleneck is the unseen one-hot representation. Replace task IDs with a frozen
semantic language embedding and keep G as the grounded control interface.

If language-zero remains seed/sign inconsistent, the instability lies in the
raw-concat G-to-action mapping itself. The next model should enforce directional
alignment through an identity-initialized residual/gated interface rather than
continuing to tune one-hot language.
```

Results:

| Seed | Normal-lang val | Zero-lang val | Normal-lang test | Zero-lang test |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 18/100 | 21/100 | 53/300 | 66/300 |
| 1 | 0/100 | 0/100 | 0/300 | 0/300 |
| 2 | 0/100 | 0/100 | 0/300 | 0/300 |
| 3 | 0/100 | 0/100 | 1/300 | 1/300 |
| 4 | 0/100 | 0/100 | 0/300 | 0/300 |

Paired-case audit:

```text
All normal-lang and zero-lang comparisons use identical sequence indices.

Seed0 validation (100):
  success under both: 16
  normal-lang only:   2
  zero-lang only:     5

Seed0 independent test (300):
  success under both: 51
  normal-lang only:   2
  zero-lang only:     15
  exact paired two-sided p ~= 0.00235

Seed1/2/4:
  no success under either language condition

Seed3 test:
  the same single case succeeds under both conditions
```

Conclusion:

```text
The held-out close one-hot is mildly harmful for the one seed that already has
a useful G-control manifold: removing it improves seed0 from 53/300 to 66/300,
with 15 newly successful cases versus 2 lost cases.

However, language-key removal does not recover any failed seed and leaves the
five-seed median at zero. Therefore the untrained close one-hot is not the root
cause of cross-seed instability.

The dominant bottleneck is now localized to the learned G-to-action mapping:
raw concat permits different optimization seeds to assign inconsistent control
semantics to the same directional/task-conditioned G. The next experiment should
change the fusion/learning constraint, not continue tuning the language key.
```

## Ablation K: Initialization Seed vs DataLoader Seed

Question:

```text
Does the successful seed0 G-control manifold come from parameter initialization
or from minibatch ordering during optimization?
```

Code audit:

```text
For chunk_bc_lang_query_g, the previous --seed controlled:
  torch / CUDA parameter initialization
  DataLoader shuffle generator

It did not randomize G features, normalization, the collected demonstrations,
or the train/validation split. data_seed and split_seed were already fixed to 0.
```

The code now exposes independent `--init-seed` and `--loader-seed` arguments.
Backward compatibility is preserved: both default to `--seed`.

Factorial sweep:

```text
fixed loader=0, vary init=0..4:
  i0_l0, i1_l0, i2_l0, i3_l0, i4_l0

fixed init=0, vary loader=0..4:
  i0_l0, i0_l1, i0_l2, i0_l3, i0_l4

total unique combinations: 9
```

All other settings match Ablation H: complete 14-task non-close scope, 50
epochs, fixed data/split seeds, epoch 10/20/30/40/50 checkpoints, identical
100-case close validation, and both normal-language and zero-language eval.

Status:

```text
completed on 2026-07-11 12:43 server time
maximum parallel jobs: 2

scheduler log:
/mnt/nvme33n1/wuk_code/AutoResearch/logs/2026-07-11_scope_primitive_seed_factor_scheduler.log

per-combination logs/status:
/mnt/nvme33n1/wuk_code/AutoResearch/logs/2026-07-11_scope_primitive_seed_factor/
```

Decision rule:

```text
If i0_l1..4 remain successful while i1..4_l0 fail, initialization determines
the basin. If i1..4_l0 recover while i0_l1..4 vary, minibatch ordering dominates.
If both sweeps vary, the objective has multiple weakly separated solutions and
requires an architectural/causal alignment constraint.
```

Results:

| Combination | Normal-language close | Zero-language close | Val chunk MSE |
| --- | ---: | ---: | ---: |
| `i0_l0` | 18/100 | 21/100 | 0.04790 |
| `i1_l0` | 0/100 | 0/100 | 0.04873 |
| `i2_l0` | 1/100 | 1/100 | 0.04760 |
| `i3_l0` | 0/100 | 0/100 | 0.04692 |
| `i4_l0` | 20/100 | 21/100 | 0.04874 |
| `i0_l1` | 0/100 | 0/100 | 0.04901 |
| `i0_l2` | 0/100 | 0/100 | 0.04845 |
| `i0_l3` | 0/100 | 0/100 | 0.04774 |
| `i0_l4` | 0/100 | 0/100 | 0.04755 |

Interpretation:

```text
Both random sources affect basin selection.

Initialization effect with loader fixed to 0:
  init0 and init4 discover a close-capable policy;
  init1/init3 fail and init2 is only borderline.

Loader-order effect with initialization fixed to 0:
  only loader0 succeeds; loader1..4 all fail.

The original seed4 pair (init4, loader4) failed, while (init4, loader0) reaches
21/100. Therefore seed4 does not lack representational capacity; a different
minibatch path with the same initialization reaches a successful solution.

Offline val MSE remains similar for every combination and does not predict
closed-loop OOD transfer. The objective admits multiple train-equivalent but
OOD-inconsistent G-to-action solutions.
```

`i4_l0` confirmation completed on 2026-07-12:

| Evaluation | Correct G | Zero G | Sign-flipped G | Open-drawer G |
| --- | ---: | ---: | ---: | ---: |
| Fixed validation cases, language zero | 21/100 | 0/100 | 0/100 | 0/100 |
| Disjoint test cases (offset 100), language zero | 81/300 | - | - | - |

The disjoint-test rate (27%) is consistent with, and slightly higher than, the
fixed validation rate (21%). The result is therefore not an artifact of the
original 100 close initial states. Correct G is causally necessary for the
successful behavior: removing it, reversing its direction, or replacing it
with the open-drawer query eliminates all 100 successes.

This closes the seed audit. A useful and semantically correct G-conditioned
control solution can be reproduced with two distinct initializations (`i0_l0`
and `i4_l0`), but it is not selected reliably. Changing either initialization
or minibatch order commonly produces a train-equivalent solution with zero
close transfer, despite nearly identical validation MSE. The current failure is
therefore optimization/identifiability instability in raw-concat BC, rather
than randomized input features, insufficient representational capacity, or a
spurious evaluation subset.

Confirmation artifacts:

```text
remote:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-12_seed_factor_i4_l0_causal_val_n100
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-12_seed_factor_i4_l0_test_offset100_n300

local:
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-12_seed_factor_feature_ablation_results/i4_l0_confirmation
```

## Ablation L: Query-G Feature-Group Localization

Static audit:

```text
Compared effective first-layer weights for five paired checkpoints over:
  token_type, rel_eef, rel_part, field_vec, desired_flow,
  dist_norm, residual_norm, contact_conf
```

Finding:

```text
Normalized first-layer RMS magnitudes are almost identical across success/failure
seeds. For example:
  desired_flow: approximately 0.0456 to 0.0464
  field_vec:    approximately 0.0514 to 0.0519

Therefore seed0 does not succeed merely by assigning a larger first-layer norm
to one G group. The difference must lie in weight orientation, ReLU activation
patterns, and downstream feature composition.
```

Normalization audit:

```text
21 of 76 G dimensions have std approximately 1e-6, but they are structural
constants: unused token-type slots, q=eef/q=part zero offsets, and horizontal
axis zero-z terms. These dimensions remain structurally constant for close and
do not currently indicate another held-out normalization explosion. Raw-space
weight rankings that divide by their tiny std are invalid and are not used as
causal evidence.
```

Causal rollout ablation:

```text
Policy: successful seed0 checkpoint
Language: zero
Eval: same 100 close validation cases

For each group, set its normalized query-G dimensions to zero (training mean):
  token_type
  rel_eef
  rel_part
  field_vec
  desired_flow
  dist_norm
  residual_norm
  contact_conf
```

Status:

```text
completed on 2026-07-11
log:
/mnt/nvme33n1/wuk_code/AutoResearch/logs/2026-07-11_query_g_seed0_feature_group_ablation.log
```

Results:

| Query-G condition | Close success |
| --- | ---: |
| normal | 21/100 |
| mask `token_type` | 48/100 |
| mask `rel_eef` | 0/100 |
| mask `rel_part` | 0/100 |
| mask `field_vec` | 6/100 |
| mask `desired_flow` | 0/100 |
| mask `dist_norm` | 3/100 |
| mask `residual_norm` | 19/100 |
| mask `contact_conf` | 25/100 |

Interpretation:

```text
Seed0's successful close mode causally depends on both relative interaction
geometry and directional field information:
  rel_eef, rel_part, and desired_flow are individually necessary;
  field_vec and dist_norm are strongly important.

residual_norm is largely redundant in this fixed close evaluation, and
contact_conf is not necessary. Removing token_type improves success, indicating
that fixed slot/type identity currently adds interference rather than useful
transfer structure.

Group effects are not additive because the policy is a nonlinear raw-concat
MLP, but the ablation rejects the hypothesis that seed0 succeeds through an
unrelated state shortcut. Its useful behavior relies on the intended G geometry
and flow content.
```

## Ablation M: Retrain Without Query-G Token Type

Question:

```text
Does removing the redundant fixed-slot token-role one-hot improve the
cross-seed stability of zero-shot close transfer?
```

Interface change:

```text
old query-G: 4 tokens x 19 dims = 76 dims
  [token_type(4), rel_eef(3), rel_part(3), field_vec(3),
   desired_flow(3), dist_norm(1), residual_norm(1), contact_conf(1)]

new query-G: 4 tokens x 15 dims = 60 dims
  [rel_eef(3), rel_part(3), field_vec(3), desired_flow(3),
   dist_norm(1), residual_norm(1), contact_conf(1)]
```

This is a from-scratch retraining ablation, not the test-time normalized-mean
mask used in Ablation L. New checkpoints store
`query_g_include_token_type=false`; old checkpoints default to the original
76-dimensional layout for backward compatibility.

Controlled settings:

```text
train tasks: same 14 non-close tasks as Ablations H/K
training windows: same fixed data_seed=0 collection
train/validation split: fixed split_seed=0
epochs: 50
seeds: 0,1,2,3,4 (initialization and DataLoader order vary together)
primary eval: 100 fixed close cases with language zero
secondary eval: same close cases with normal language
follow-up: correct/zero/sign-flip/open-task G controls for successful seeds
```

Status on 2026-07-12:

```text
all five training/evaluation jobs completed automatically after GPUs became free
scheduler log:
/mnt/nvme33n1/wuk_code/AutoResearch/logs/2026-07-12_scope_primitive_no_token_type_scheduler.log

per-seed logs/status:
/mnt/nvme33n1/wuk_code/AutoResearch/logs/2026-07-12_scope_primitive_no_token_type_multiseed/

training outputs:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-12_scope_primitive_no_token_type_e50_n100_h160_seed{0..4}
```

Initial results:

| Seed | Normal-language close | Zero-language close | Val chunk MSE |
| ---: | ---: | ---: | ---: |
| 0 | 0/100 | 3/100 | 0.04974 |
| 1 | 0/100 | 0/100 | 0.04951 |
| 2 | 10/100 | 8/100 | 0.04901 |
| 3 | 0/100 | 0/100 | 0.04742 |
| 4 | 0/100 | 0/100 | 0.04731 |

The five-seed median remains zero, so the new 60-D interface is not stable.
However, this run is not a strict causal token-type ablation: reducing the
condition from 149 to 133 dimensions changes the first-layer fan-in,
initialization scale, RNG consumption, and consequently all downstream initial
weights. Seed0 and seed2 therefore cannot be paired directly with their 149-D
baseline counterparts. These results evaluate a compact interface redesign,
not the isolated effect of token type.

60-D confirmation completed:

```text
seed0, language zero, fixed 100:
  correct 3, zero 0, sign-flip 22, open-task 0
seed2, language zero, fixed 100:
  correct 8, zero 0, sign-flip 13, open-task 1

disjoint 300 cases at sequence offset 100:
  seed0 correct G: 14/300
  seed2 correct G: 31/300

measured GPU memory per rollout process: approximately 0.5-1.3 GB
```

Strict paired control launched on 2026-07-12:

```text
Keep the original 149-D policy input and 76-D query-G layout.
Set all 16 token-type coordinates to raw zero during both training and eval.
The resulting normalized token-type input is exactly zero.

This preserves for each seed:
  network shape and fan-in
  model initialization RNG sequence
  DataLoader order
  data and train/validation split

flag: --zero-query-g-token-type
scheduler PID: 3557314
GPUs: 4,5,6,7; maximum four parallel seeds
log:
/mnt/nvme33n1/wuk_code/AutoResearch/logs/2026-07-12_scope_primitive_zero_token_type_scheduler.log
```

Only this fixed-dimensional experiment can determine whether removing the
token-type signal changes cross-seed zero-shot transfer under matched
optimization seeds.

Strict paired results:

| Seed | Normal-language close | Zero-language close | Val chunk MSE |
| ---: | ---: | ---: | ---: |
| 0 | 0/100 | 0/100 | 0.04752 |
| 1 | 2/100 | 3/100 | 0.04876 |
| 2 | 0/100 | 0/100 | 0.04897 |
| 3 | 7/100 | 8/100 | 0.04634 |
| 4 | 1/100 | 1/100 | 0.04861 |

Keeping the 149-D architecture while removing only the token-type signal does
not stabilize transfer. The zero-language median rises only from 0 to 1/100,
and the useful basin moves from baseline seed0 to seed3. Token type therefore
changes basin selection but is not the root cause of G-to-action
identifiability failure.

Decision rule:

```text
If the five-seed median and number of correct-G causal successes improve over
the 76-D baseline, token type was a material shortcut/optimization confound.
If only individual seeds improve while the median remains zero, token type is
harmful but not the root identifiability problem. If performance degrades, the
binary articulation-family route was useful despite being non-reusable.
```

## Ablation N: Open/Close EEF, G-Direction, and Effective-Contact Diagnosis

Question:

```text
Do no-token-type close failures lack the correct push direction, fail to reach
the handle/contact side, or reach contact but apply insufficient motion?
```

Protocol:

```text
policies: 60-D no-token-type seed0..4
tasks: seen open_drawer and zero-shot close_drawer
cases: 25 per task and seed, normal language and normal G
effective contact: first step where cumulative drawer motion in the desired
                   task direction reaches 2 mm

Additional comparisons:
  with-token-type seed0/seed2 on the same states
  seed0/seed2 normal vs sign-flipped G with language zero
  the exact eight seed2 rollouts rendered in the visual comparison
```

Physical direction audit:

```text
drawer axis ~= [0.043, -0.996, 0.078]
open desired axis is +axis: mostly -Y
close desired axis is -axis: mostly +Y

query-G is physically correct:
  open median desired_flow_y  = -0.996
  close median desired_flow_y = +0.996
  desired_flow dot desired_axis = 1.000 for both
```

Seen-open versus zero-shot-close pooled over five no-token-type seeds:

| Metric | Open | Close |
| --- | ---: | ---: |
| Success | 121/125 | 1/125 |
| Effective contact | 125/125 | 8/125 |
| Median minimum EEF-part distance | 0.019 m | 0.221 m |
| Median pre-contact EEF/G-field cosine | 0.596 | -0.003 |
| Median late desired-axis action | 0.00392 | 0.00011 |
| Median maximum normalized joint progress | 0.711 | 0.000 |

Close failure decomposition:

| Outcome | Count | Median part distance | Median anchor distance | Pre-action norm | Late close-axis action | Max joint progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Success | 1 | 0.080 m | 0.076 m | 0.0104 | 0.00579 | 0.657 |
| Failed after effective contact | 7 | 0.118 m | 0.170 m | 0.0089 | 0.00085 | 0.382 |
| Failed without effective contact | 117 | 0.228 m | 0.229 m | 0.0046 | 0.00011 | 0.000 |

The dominant gate is contact localization. Most close policies express a tiny
positive push tendency but never bring the EEF close enough for it to affect
the drawer. Among trajectories that move the joint, insufficient sustained
post-contact motion is a second bottleneck.

Exact seed2 visual cases:

```text
successful cases 7,18,34,37,69:
  minimum part distance:   0.080-0.123 m
  minimum anchor distance: 0.059-0.103 m
  effective-contact EEF axis offset: -0.024 to -0.065 m
  late close-axis action:  +0.00148 to +0.00757

failed cases 0,1,8:
  minimum part distance:   0.217-0.219 m
  minimum anchor distance: 0.225-0.229 m
  late close-axis action:  approximately 0
```

All five successes first place the EEF on the negative side of the close axis,
near the nominal `-radius` contact side, and then switch to a positive close-axis
push. The failed visual cases partially follow the early direction but remain
about 22 cm from the fitted handle; their action amplitude is roughly half the
successful pre-contact amplitude and collapses to zero before effective
contact. High action/G cosine alone is not sufficient because a small motion
can point correctly while failing to close the spatial error.

The successful trajectories are not yet good feedback policies. They often use
large, oscillatory motions and succeed late (effective contact around steps
82-125). Their success demonstrates a correct-side approach-and-push basin,
not precise or robust control. One visually successful rollout uses lower
post-contact action magnitude, confirming that large magnitude alone is not the
common cause; correct-side contact geometry plus sustained positive push is.

Sign-flip diagnosis:

| Policy | Effective contact | Median anchor distance | Success |
| --- | ---: | ---: | ---: |
| seed0 normal G | 3/25 | 0.160 m | 2/25 |
| seed0 sign-flipped G | 22/25 | 0.058 m | 6/25 |
| seed2 normal G | 4/25 | 0.223 m | 3/25 |
| seed2 sign-flipped G | 4/25 | 0.233 m | 4/25 |

This control does not mean the flipped field is physically correct. The current
sign-flip intervention reverses `field_vec`, `desired_flow`, and residual while
retaining the close-task anchor positions, creating an internally inconsistent
G. Its large seed0 improvement shows that raw concat uses anchor geometry and
vector sign as separate, seed-dependent routing signals. It does not learn a
single equivariant rule in which reversing desired object flow predictably
reverses post-contact control.

Conclusion:

```text
G direction generation is correct.
The learned policy fails to reuse it as a two-stage control interface:
  pre-contact: follow the EEF-to-contact field and reach the correct side;
  post-contact: switch to sustained motion along desired object flow.

Across seeds, the primary failure is pre-contact field decoding/localization.
For near-successes, the secondary failure is post-contact gain and duration.
The seed instability comes from unconstrained mixing of anchor geometry,
field direction, desired flow, state, and language in raw concat.
```

Local analysis artifacts:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-13_no_token_type_open_close_deep_analysis/
```

## Ablation O: minimal coherent G and structured executor (launched 2026-07-13)

The preceding failures motivate reducing oracle G to two spatial vectors rather
than adding another collection of contact-stage features:

```text
G_min = [A, F] in R^6

A = contact_anchor - EEF        # contact-attraction displacement
F = desired object flow         # object motion requested by language/query
```

`contact_anchor` is generated from the current object state, fitted articulation
geometry, and the language-conditioned desired motion. It is therefore not an
independent absolute anchor competing with the field. Non-articulation tasks
receive zero `G_min` and continue to use state plus language.

Demonstration audit over all open/slider training segments (`11,288` windows,
chunk 16, stride 2) supports a continuous approach-to-interaction transition:

| Task | Windows | Far action/A cosine p50 | Near action/F cosine p50 | Near positive F projection |
| --- | ---: | ---: | ---: | ---: |
| open_drawer | 3,800 | 0.885 | 0.933 | 96.1% |
| move_slider_left | 3,702 | 0.938 | 0.973 | 93.3% |
| move_slider_right | 3,786 | 0.970 | 0.976 | 85.9% |

This is evidence for two reusable control roles, not for a discrete stage
label. The structured executor therefore uses a smooth readiness function of
`||A||`, FiLM conditioning by `G_min`, and a sign-preserving low-frequency
residual that blends normalized `A` into `F`. The rest of the chunk remains
learned from behavior cloning.

Primary seed0 comparison:

```text
same 48,069 windows from 14 non-close tasks
same data/split/init seed, chunk 16, stride 2, 50 epochs
same state and language input

1. chunk_bc_lang_minimal_g_concat
2. chunk_bc_lang_minimal_g_structured

test: the same 100 zero-shot close_drawer initial states, episode length 160
```

Run locations:

```text
script:
  framework/scripts/policy_state_bc/run_minimal_g_concat_structured_seed0.sh
log:
  /mnt/nvme33n1/wuk_code/AutoResearch/logs/2026-07-13_minimal_g_concat_structured_seed0.log
output:
  /mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-13_minimal_g_concat_structured_e50_n100_h160_seed0
```

Seed0 primary result:

| Executor | Validation chunk MSE | Seen open (25) | Zero-shot close (100) |
| --- | ---: | ---: | ---: |
| 6-D raw concat | 0.05074 | 21/25 | 0/100 |
| FiLM + additive field residual | 0.04733 | 25/25 | 0/100 |
| Projected field executor | **0.04639** | **25/25** | **14/100** |

The additive residual retained nonzero learned gains (`A=0.216`, `F=0.209`),
so its failure is not explained by G being ignored at the scalar-gain level.
Close trajectory decomposition instead shows cancellation in the control loop:

```text
additive structured close, 25/25 failures:
  median pre-contact prior projection on desired axis:   -0.1203
  median learned-branch projection:                     +0.0935
  median final projection:                              -0.00057
  median minimum contact-anchor distance:                0.261 m

additive structured seen open, 25/25 successes:
  median minimum part distance:                          0.015 m
  median post-contact desired-axis action:               0.00452
```

The prior points toward the pre-contact side, while an unconstrained learned
branch can oppose it. This makes "explicit G residual" insufficient: the
network can learn a cancellation solution on seen tasks whose balance does not
transfer when object flow reverses.

The projected executor removes this degree of freedom. For valid articulation
G, it predicts a nonnegative field-parallel gain and projects the remaining
learned translation into the plane orthogonal to the active field direction:

```text
d = normalize((1-r) normalize(A) + r normalize(F))
r = sigmoid((0.10 - ||A||) / 0.025)

a_xyz = positive_gain(state, language, G) * d
        + project_orthogonal(a_learned, d)
```

It does not add stage one-hot or contact-state labels. The smooth readiness is
derived only from the reusable attraction displacement. The policy retains
learned orientation, gripper, timing, and field-orthogonal corrections, but it
cannot cancel the G-defined parallel direction. The seed0 gain learned from
`0.20` initialization to a validation mean of `0.305`.

Interpretation after seed0:

```text
Supported:
  A/F is a useful compact interface when the executor respects its control
  subspace; this changes zero-shot close from 0/100 to 14/100 without reducing
  seen-open success.

Not yet supported:
  cross-seed stability, learned-from-video G, broad task transfer, or a final
  comparison with a mainstream image-conditioned flow-matching policy.
```

Projected seed1-4 were launched with the identical data/split protocol and 100
close rollouts per seed. Their results are required before promoting the seed0
result to a paper claim.

Direct-language projected five-seed result:

| Init seed | Close success / 100 | Validation MSE | Validation field gain |
| ---: | ---: | ---: | ---: |
| 0 | 14 | 0.04639 | 0.305 |
| 1 | 0 | 0.04804 | 0.307 |
| 2 | 11 | 0.04683 | 0.313 |
| 3 | 0 | 0.04759 | 0.312 |
| 4 | 2 | 0.04873 | 0.310 |
| Mean | 5.4 | 0.04752 | 0.309 |

Projected control improves the average over raw/additive decoding, but does not
remove seed instability. Nearly identical validation MSE and field gain do not
predict zero-shot success.

One remaining seed-dependent input is the direct language one-hot. The unseen
`close_drawer` dimension is never active in training, so its executor input
weights retain an initialization-dependent value. On the seed0 checkpoint,
zeroing only direct language at test time while retaining normal query-G lowers
close success from `14/100` to `5/100`. This proves the untrained language path
affects zero-shot control, although post-hoc zeroing is itself a distribution
shift and is not a clean fix.

The corrected hierarchy masks direct language only when minimal G is valid:

```text
language -> query/minimal G -> articulation executor

if G is valid:
  executor input = state + zeroed direct-language + G
else:
  executor input = state + language + zero G
```

This preserves task identity for non-articulation training tasks whose current
G is zero, while preventing unseen articulation task IDs from entering the
low-level controller. From-scratch masked-language seed0 obtains validation MSE
`0.04623` and close `15/100`. Seed1-4 are required to determine whether the
change reduces variance rather than merely moving one seed.

Masked-language projected five-seed result:

| Init seed | Close success / 100 | Validation MSE |
| ---: | ---: | ---: |
| 0 | 15 | 0.04623 |
| 1 | 6 | 0.04835 |
| 2 | 0 | 0.04758 |
| 3 | 50 | 0.04693 |
| 4 | 1 | 0.04880 |
| Mean | 14.4 | 0.04758 |

The median is only `6/100` and the sample standard deviation is `20.77`
percentage points. Removing the unseen direct-language path increases mean
transfer (`5.4% -> 14.4%`) but does not stabilize the executor. It therefore
identifies one leakage path without explaining the dominant optimization
variance.

Trajectory diagnosis on the same first 25 close initial states isolates the
remaining difference:

| Metric | Masked seed2 | Masked seed3 |
| --- | ---: | ---: |
| Success | 0/25 | 13/25 |
| No effective contact | 25/25 | 3/25 |
| Median minimum EEF-part distance | 0.279 m | 0.068 m |
| Median minimum EEF-anchor distance | 0.332 m | 0.110 m |
| Median action/active-field cosine | 0.112 | 0.426 |
| Median EEF-motion/active-field cosine | -0.231 | 0.130 |

Their validation losses and scalar field gains are similar. The behavior
difference is therefore not primarily whether a nonzero gain was learned; it
is whether the closed-loop EEF motion contracts the attraction error. The
unbounded field-orthogonal branch can dominate the positive parallel branch:
the executor cannot reverse G, but it can move almost sideways to it and fail
to approach the handle.

Important diagnostic correction: CALVIN's relative-action conversion mutates
the input NumPy action in place and multiplies translation by `0.02`. Earlier
axis-action magnitudes read after `env.step` are physical per-step deltas, while
the stored projected prior and orthogonal components are normalized policy
commands. Those quantities must not be subtracted directly. Distance,
success, progress, and cosine results remain valid because the uniform scaling
does not change direction. The diagnostic now copies the command before
calling the environment.

An offline decomposition over the `11,288` open/slider training windows gives
the data-supported range for field-orthogonal correction:

| Task | First-action field cosine p50 | Positive parallel | Orth/parallel p50 | Orth/parallel p90 |
| --- | ---: | ---: | ---: | ---: |
| open_drawer | 0.707 | 92.9% | 0.881 | 3.373 |
| move_slider_left | 0.696 | 88.5% | 0.785 | 4.313 |
| move_slider_right | 0.675 | 80.8% | 0.645 | 4.387 |

The seed2 close cosine of `0.112` corresponds to an orthogonal/parallel ratio
near nine, outside the main demonstration regime. The next executor therefore
uses a train-selected bound `rho=2`:

```text
||a_orthogonal|| <= rho * a_parallel,  rho = 2
cos(a_xyz, d) >= 1 / sqrt(1 + rho^2) = 0.447
```

This retains field-orthogonal corrections, does not constrain non-articulation
samples with zero G, and does not use close labels or close rollouts to select
the value. The bounded masked-language seed0 run is recorded at:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/
2026-07-13_minimal_g_bounded_projected_rho2_e50_n100_h160_seed0
```

Bounded masked-language seed0 result:

| Validation MSE | Seen open / 25 | Zero-shot close / 100 |
| ---: | ---: | ---: |
| 0.04695 | 25 | 20 |

The corresponding open/close trajectory audit shows that the bound is
selective rather than globally shrinking behavior:

| Metric | Seen open | Zero-shot close |
| --- | ---: | ---: |
| Success / 25 | 25 | 3 |
| Minimum EEF-part distance p50 | 0.013 m | 0.067 m |
| Pre-contact action/field cosine p50 | 0.781 | 0.518 |
| Pre-contact EEF-motion/field cosine p50 | 0.764 | 0.481 |
| Pre-contact field gain p50 | 0.318 | 0.110 |
| Pre-contact orthogonal scale p50 | 1.000 | 0.789 |
| Command-clipped fraction p50 | 0.000 | 0.000 |

The constraint is effectively inactive on the seen open trajectories and
activates on close, where it improves approach geometry without relying on
component clipping. The low close field gain and late first-contact step
(`127.5` median among contacting rollouts) remain bottlenecks. Five-seed
evaluation is still required before deciding whether the bound reduces
optimization variance or only improves seed0.

Bounded masked-language five-seed result on the identical close initial states:

| Init seed | Unbounded / 100 | Bounded rho=2 / 100 | Bounded validation MSE |
| ---: | ---: | ---: | ---: |
| 0 | 15 | 20 | 0.04695 |
| 1 | 6 | 49 | 0.04977 |
| 2 | 0 | 2 | 0.04728 |
| 3 | 50 | 69 | 0.04804 |
| 4 | 1 | 82 | 0.04918 |
| Mean | 14.4 | 44.4 | 0.04824 |
| Median | 6 | 49 | 0.04804 |

The selected sequence indices match exactly for every seed. Across all 500
paired `(seed, initial state)` evaluations, `179` cases change from failure to
success and `29` change from success to failure (two-sided exact McNemar
`p=1.41e-27`). The bound therefore produces a real aggregate improvement, not
an initial-state sampling artifact. Absolute seed variance remains substantial
(`2-82%`), so the executor is improved but not yet stable.

The bounded seed1/seed2 trajectory contrast localizes that residual variance:

| Metric, close first 25 | Seed1 | Seed2 |
| --- | ---: | ---: |
| Success | 13/25 | 0/25 |
| Minimum EEF-part distance p50 | 0.047 m | 0.071 m |
| Minimum EEF-anchor distance p50 | 0.083 m | 0.121 m |
| First near-part step p50 | 73 | 121 |
| Pre-contact action/field cosine p50 | 0.837 | 0.787 |
| Pre-contact EEF-motion/field cosine p50 | 0.705 | 0.651 |
| Pre-contact field gain p50 | 0.203 | 0.094 |
| Progress after first near-part p50 | 0.543 | 0.00015 |

After the direction bound, both seeds decode the field direction reasonably
well. Seed2 instead learns roughly half the approach gain, reaches the part too
late, often misses the contact anchor, and remains in the attraction-dominated
regime rather than sustaining object flow. This exposes a second leakage path:
the scalar gain head currently reads a hidden state modulated by signed
Cartesian G and absolute state, so reversing the requested motion can still
produce an initialization-dependent amplitude even though direction is
structurally protected.

The next executor keeps the same six-dimensional G and `rho=2` control cone,
but predicts gain only from sign/rotation-invariant geometry:

```text
gain = gain_net(||A|| / readiness_distance, ||F||, readiness)
```

This makes direction reversal share an amplitude law while retaining learned
orientation, gripper behavior, and bounded field-orthogonal correction. It is
a policy-interface change, not a new task-specific G term.

Five-seed seen-task health check for the bounded executor (`25` rollouts per
task and seed):

| Init seed | Open drawer | Slider left | Slider right |
| ---: | ---: | ---: | ---: |
| 0 | 25/25 | 19/25 | 25/25 |
| 1 | 25/25 | 25/25 | 25/25 |
| 2 | 25/25 | 24/25 | 22/25 |
| 3 | 25/25 | 24/25 | 25/25 |
| 4 | 25/25 | 25/25 | 25/25 |
| Total | 125/125 | 117/125 | 122/125 |

The aggregate is `364/375 = 97.1%`. The close improvement therefore does not
come from a broad collapse or replacement of the seen articulation policies.

Invariant-gain five-seed result:

| Init seed | Bounded / 100 | Invariant-gain bounded / 100 | Invariant validation MSE |
| ---: | ---: | ---: | ---: |
| 0 | 20 | 62 | 0.04561 |
| 1 | 49 | 75 | 0.04772 |
| 2 | 2 | 0 | 0.04747 |
| 3 | 69 | 73 | 0.04895 |
| 4 | 82 | 28 | 0.05010 |
| Mean | 44.4 | 47.6 | 0.04797 |
| Median | 49 | 62 | 0.04772 |

On the 500 paired rollouts, the invariant head changes `102` failures to
successes and `86` successes to failures (exact McNemar `p=0.274`). It raises
seed0/1 but collapses the strong seed4 solution and does not rescue seed2.
Therefore signed hidden-state dependence of gain is a real mechanism in the
seed1/seed2 contrast, but it is not the only source of basin instability. The
invariant-gain variant is diagnostic evidence and should not replace the
bounded baseline.

## Ablation P: align chunk training with receding-horizon execution

The current evaluator predicts a 16-action chunk but executes only
`chunk[0]`; it then observes the next state and replans. Training nevertheless
weights all 16 future actions equally. The only action that enters the control
loop therefore contributes about `1/16` of the action loss, while the shared
trunk is dominated by 15 actions that are never executed under the current
evaluation protocol.

This is not a G representation issue. It is a policy objective/execution
mismatch that can amplify initialization dependence and also makes the current
flow-matching baseline unlike mainstream action-chunk policies that execute or
temporally ensemble a horizon. The next bounded-policy experiment keeps G,
language masking, architecture, data, and rollout fixed, but uses:

```text
first_action_loss_weight = 16
future_action_loss_weight = 1
```

After normalization, the executed first action contributes approximately half
of the chunk objective and the other 15 steps contribute the other half. The
unweighted validation chunk MSE and a new first-action validation MSE are both
reported. This value follows directly from chunk length and is not selected on
close success.

First-action-weighted bounded five-seed result:

| Init seed | Bounded / 100 | First-weight 16 / 100 | Val chunk MSE | Val first-action MSE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 20 | 87 | 0.04672 | 0.03295 |
| 1 | 49 | 59 | 0.04644 | 0.03300 |
| 2 | 2 | 85 | 0.04645 | 0.03134 |
| 3 | 69 | 100 | 0.04916 | 0.03247 |
| 4 | 82 | 100 | 0.04738 | 0.03383 |
| Mean | 44.4 | 86.2 | 0.04723 | 0.03272 |
| Median | 49 | 87 | 0.04672 | 0.03295 |

The close sample standard deviation falls from `33.29` to `16.75` percentage
points, and the worst seed rises from `2%` to `59%`. Sequence indices match for
all seeds. Across 500 paired rollouts, `233` failures become successes and only
`24` successes become failures (exact McNemar `p=3.53e-44`). Full-chunk
validation MSE remains in the same range, so the large control improvement is
not explained by a globally easier fit or changed data.

This is the strongest short-term evidence so far:

```text
minimal reusable A/F field
+ language-to-query hierarchy
+ field-respecting bounded executor
+ objective aligned with the action actually executed
```

is sufficient for robust zero-shot open-to-close transfer in this state-oracle
setting. The result supports G as a reusable control interface, while also
showing that previous seed instability was substantially a policy training
artifact rather than evidence that G itself was unusable. Seen-task health and
image/learned-G transfer remain separate required gates.

First-action-weighted seen-task health check:

| Init seed | Open drawer | Slider left | Slider right |
| ---: | ---: | ---: | ---: |
| 0 | 25/25 | 20/25 | 25/25 |
| 1 | 25/25 | 20/25 | 25/25 |
| 2 | 25/25 | 24/25 | 21/25 |
| 3 | 25/25 | 21/25 | 25/25 |
| 4 | 20/25 | 24/25 | 23/25 |
| Total | 120/125 | 109/125 | 119/125 |

The aggregate is `348/375 = 92.8%`, versus `364/375 = 97.1%` for the
unweighted bounded policy. On paired seen rollouts, `8` failures become
successes and `24` successes become failures (exact McNemar `p=0.0070`). Thus
the close/stability gain carries a measurable `4.3` percentage-point seen-task
cost, concentrated in slider-left and seed4 open. This is still a healthy
policy rather than a collapse, but it argues for replacing the ad hoc
first-action weighting with faithful action-horizon execution or temporal
ensembling when migrating to a mainstream chunk/flow policy.

Two matched five-seed controls were run before attributing the 86.2% close
result to G rather than only to objective repair:

```text
language-only + first-action weight 16
unbounded projected minimal-G + first-action weight 16
```

Both reuse the same training windows, seeds, and close initial states.

Matched-control result:

| Init seed | Language only + first-w16 | Unbounded G + first-w16 | Bounded G + first-w16 |
| ---: | ---: | ---: | ---: |
| 0 | 0/100 | 26/100 | 87/100 |
| 1 | 0/100 | 59/100 | 59/100 |
| 2 | 0/100 | 99/100 | 85/100 |
| 3 | 0/100 | 30/100 | 100/100 |
| 4 | 0/100 | 40/100 | 100/100 |
| Mean | 0.0 | 50.8 | 86.2 |
| Median | 0 | 40 | 87 |
| Sample std. | 0.0 | 29.81 | 16.75 |

The selected sequence indices match exactly for all three methods and all five
seeds. Relative to language-only, unbounded G changes `254` failures to
successes and no successes to failures (two-sided exact McNemar
`p=6.91e-77`). Relative to unbounded G, the bounded executor changes `216`
failures to successes and `39` successes to failures (`p=6.89e-31`). The
per-seed bounded-versus-unbounded transitions are `(63,2)`, `(22,22)`,
`(1,15)`, `(70,0)`, and `(60,0)`, where each pair is
`(failure-to-success, success-to-failure)`.

These controls separate three mechanisms that had previously been entangled:

1. **G supplies held-out task control information.** Language-only remains at
   `0/500` even after the executed action receives half of the chunk loss.
   Objective repair alone cannot infer the unseen close direction from the
   training tasks.
2. **The first-action objective makes G usable by the actual receding-horizon
   controller.** Unbounded G rises from the earlier `14.4%` mean to `50.8%`.
   A substantial part of the previous seed instability came from optimizing
   future chunk entries that the evaluator never executed.
3. **The bounded control cone makes reuse substantially more reliable.** It
   raises mean success from `50.8%` to `86.2%`, worst-seed success from `26%`
   to `59%`, and reduces sample standard deviation by `13.06` points. It is not
   monotonically better for every optimization seed: seed2 falls from `99%` to
   `85%`. Its supported claim is improved aggregate reliability, not that every
   unconstrained solution is inferior.

The current result supports the paper's *oracle interface* claim: a compact,
task-conditioned `G=[A,F]` can carry reusable interaction intent from seen
open/slider/push tasks into a held-out close task, provided the executor is
trained and structurally coupled to the interface. It does **not** yet support
the scaling claim. State-derived oracle G bypasses visual grounding, language
query generation, occlusion, and prediction error; those remain explicit
gates rather than details to hide in the same number.

## Next policy gate: faithful chunk execution

`first_action_loss_weight=16` is a useful causal diagnosis, but it is an ad hoc
match to an evaluator that discards 15/16 predicted actions. The next policy
comparison should keep the successful six-dimensional G and bounded executor,
then replace this workaround with one of the execution schemes used by modern
chunked BC and flow policies:

```text
receding-horizon chunk prediction
+ execute a short action horizon or temporally ensemble overlapping chunks
+ condition the action vector field on language, observation, and G
+ retain a bounded residual around the G-induced translational direction
```

The immediate gate is not a larger G. Compare `first-w16/replan-1` against
`uniform-loss/execute-k` and, if available, temporal ensembling on the same
five seeds. Require close mean and worst-seed success to remain near the
current result while recovering the `4.3` point seen-task cost. Only after
that gate should the state oracle be replaced by image-grounded or learned G,
with oracle/corrupted/dropped-G controls preserved.

This gate was run with the same uniform-loss bounded checkpoints and the same
500 close initial states. `execute-4` holds a predicted chunk open-loop for
four environment steps. `ensemble-d0.25` predicts a new chunk and recomputes G
at every step, then exponentially averages all still-valid predictions for
the current timestep, favoring newer predictions.

| Init seed | Uniform/replan-1 | Uniform/execute-4 | Uniform/ensemble-d0.25 | First-w16/replan-1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 20/100 | 4/100 | 38/100 | 87/100 |
| 1 | 49/100 | 49/100 | 29/100 | 59/100 |
| 2 | 2/100 | 10/100 | 18/100 | 85/100 |
| 3 | 69/100 | 62/100 | 88/100 | 100/100 |
| 4 | 82/100 | 92/100 | 70/100 | 100/100 |
| Mean | 44.4 | 43.4 | 48.6 | 86.2 |
| Median | 49 | 49 | 38 | 87 |
| Sample std. | 33.29 | 36.77 | 29.34 | 16.75 |

All selected sequence indices match. Relative to uniform/replan-1,
uniform/execute-4 changes `67` failures to successes and `72` successes to
failures (exact McNemar `p=0.735`). Temporal ensembling changes `94` failures
to successes and `73` successes to failures (`p=0.121`). Neither execution
change is a significant net improvement. First-w16 remains decisively better
than temporal ensembling: `210` failures become successes and `22` successes
become failures (`p=1.13e-39`).

The negative result refines the policy diagnosis. Future chunk entries are
supervised under the demonstration observation and G at the chunk start. They
are not equivalent to a control target queried later under a shifted
closed-loop state:

- Open-loop execute-4 makes the attraction vector stale for three steps and
  increases seed variance.
- Temporal ensembling recomputes the current G, but still mixes actions
  predicted under older observations and older fields. It modestly improves
  the mean and variance while redistributing success basins, but does not fix
  the undertrained current-state action.
- First-w16 repairs the mismatch at training time: the action conditioned on
  the current observation and current G receives enough supervision to remain
  stable after replanning.

Therefore the current mechanism baseline remains bounded G + first-w16 +
replan-1. For migration to a mainstream chunked or flow policy, execution and
training must be designed together. Temporal ensembling should not be bolted
onto this checkpoint only at inference. The migrated model should train under
its intended action-position weighting or an ensemble-consistency objective,
while conditioning the action vector field on the latest observation,
language, and G. The oracle-interface gate is now strong enough to begin that
migration; further MLP executor variants are lower priority than learned-G
grounding and a credible conditional flow-policy implementation.
