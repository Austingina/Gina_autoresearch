# CALVIN V7 G Interface Review With Notion KB

Date: 2026-07-08

Status: design review after local Notion KB refresh.

## KB Update

Local Notion mirror was refreshed from the read-only `Paper Tracking / Papers`
database.

```text
rows mirrored: 102
snapshot: knowledge_base/00_external/notion/notion_papers_snapshot.jsonl
enriched: knowledge_base/00_external/notion/notion_papers_enriched.jsonl
index: knowledge_base/00_external/notion/notion_papers_index.md
```

Enrichment was run with `--skip-api`, so local deterministic IDs and manual
public lookup fields were refreshed, but arXiv API verification remains
pending for rows not already manually verified.

Primary-source spot checks used for this review:

| Paper | Source | Verified design signal |
| --- | --- | --- |
| RT-Affordance | https://arxiv.org/abs/2411.02704 | Affordance is an intermediate representation capturing robot poses at key task stages and conditioning a hierarchical policy. |
| AffordDP | https://arxiv.org/abs/2412.03142 | Uses 3D contact points plus post-contact trajectories and incorporates affordance guidance during diffusion sampling. |
| FlowBot++ | https://arxiv.org/abs/2306.12893 | Predicts dense per-point motion and articulation parameters for articulated-object manipulation. |
| 3DFlowAction | https://arxiv.org/abs/2506.06199 | Treats desired 3D object flow as an embodiment-agnostic clue and uses it as a constraint for action optimization/planning. |
| TarGF | https://arxiv.org/abs/2209.00853 | Learns a target gradient field and uses it as reference control, reward, and residual-policy signal. |
| MaskWAM | https://arxiv.org/abs/2606.13515 | Uses masks as explicit inputs and prediction targets to provide object-centric spatial anchors. |

## Trigger

The latest CALVIN result sharpened the failure mode:

```text
non-augmented chunk_bc_v7 e50:
  close g=1: 0/25
  close g=2: 19/25
  train tasks g=2: 5/75

scale_aug / gated v7 e50:
  train tasks g=1: 69/75 and 71/75
  train tasks g=2: 75/75 and 71/75
  close g=0/0.5/1/2/4: all 0/25
```

The question is not whether v7 G contains any useful signal. The original
`g=2` result suggests it can push behavior in a useful direction. The question
is why the learned policy interface cannot use that signal as a stable,
transferable task condition.

## Relevant Notion/Paper Hints

The Notion KB points to a consistent pattern:

| Paper family | Useful hint | Warning for current design |
| --- | --- | --- |
| RT-Affordance | Intermediate affordances work when they specify key robot poses/stages in a lightweight but actionable way. | A continuous G vector appended to an MLP input is weaker than an explicit plan/token that the policy can attend to. |
| AffordDP / A0 | Contact point plus post-contact trajectory is an effective transferable affordance for diffusion policies. | Successful affordance methods preserve object/contact structure and often use guidance during sampling, not passive concatenation. |
| FlowBot++ / 3DFlowAction / VIP | Object or point flow can be embodiment-agnostic and useful across tasks/robots. | Flow becomes useful when it is tied to planning, constraints, visual instruction, or action optimization. It is not just another state feature. |
| TarGF | A field can be used as reference control, reward, or residual policy signal. | Field methods usually affect the control objective directly. Our current G has no direct action-level energy or reward role. |
| MaskWAM / World Action Models | Object masks and future object-centric prediction reduce language/spatial ambiguity. | Mask/object anchors are important. A flat state-vector policy loses the spatial binding between target object, current EEF, and desired future object motion. |
| Spectral / hierarchical policy hints | Low-frequency intent and high-frequency contact correction can be separated. | We currently ask one deterministic chunk MLP to infer both task intent and contact timing from weak conditioning. |

## Diagnosis

### 1. Scale augmentation likely teaches G-invariance, not G-semantics

`chunk_bc_v7_scale_aug` sees the same demonstration action under multiple
scaled versions of the same G. The easiest supervised solution is to become
robust to G magnitude and rely on state/history for seen tasks.

That explains the result:

```text
scale_aug train tasks g=2: 75/75
scale_aug close g=2: 0/25
```

This is not a generic policy failure. It is specifically a semantic-binding
failure: G perturbations are treated as harmless nuisance variation rather than
as a task-switching condition.

### 2. The current split is too strong for vanilla BC to learn controllable G

Training on `open_drawer`, `move_slider_left`, and `move_slider_right` while
testing zero-shot `close_drawer` removes the only supervised evidence that the
same drawer context can require the opposite drawer-axis behavior.

With task one-hot zeroed, the policy sees drawer states mostly through
`open_drawer`. A BC objective has no counterfactual pressure to learn:

```text
same/similar observation + different desired object motion -> different action
```

The original non-augmented `g=2` close success is therefore likely a narrow
out-of-distribution extrapolation of the raw concat interface. It is useful as
a clue, but not a stable learned interface.

### 3. Flat vector injection loses entity correspondence

The paper hints consistently preserve structure:

```text
mask / contact point / point flow / object flow / affordance plan / field query
```

Our current state-BC interface is closer to:

```text
MLP([proprio/state, flattened G]) -> action chunk
```

This does not force the decoder to answer:

```text
Which object is targeted?
Where should the EEF make contact?
What object motion should occur after contact?
Is the current EEF motion aligned with that object motion?
```

The model can fit train tasks without learning these bindings.

### 4. G currently does not directly constrain action generation

AffordDP, TarGF, 3DFlowAction, and FlowBot++ all make the intermediate
representation operational: it guides sampling, planning, reference control, or
optimization. Current v7 G is only an input feature. If the action decoder
chooses to ignore or average it, the closed-loop system has no fallback.

### 5. Contact timing is probably the immediate behavioral failure, but not the root cause

The visual rollouts showed that both BC and FM can approach the drawer handle,
then fail at the push/contact phase. That is compatible with the above
diagnosis:

```text
early approach can be learned from state distribution;
post-contact object-motion control requires explicit binding between
desired object motion and EEF action.
```

The next design should improve that binding generally, not add a drawer-specific
stage label.

## Design Decision

Stop treating v7 G scale/gate as the main axis. The next module should test
whether G can function as an explicit goal/field-query condition.

Use the following interface family:

```text
policy(action_chunk | state, target_mask_or_part, queried_G_tokens)
```

where `queried_G_tokens` are not a large hand-engineered term list. They should
be compact, spatially bound samples of the reusable interaction field:

```text
G_at_target_part      : desired motion of the interacted object/part
G_at_current_eef      : field value near the current end-effector
G_contact_anchor      : compact anchor from mask/part geometry
G_progress_scalar     : continuous remaining object-motion progress, if available
```

The important change is not adding many variables. The important change is
making G a spatial goal token that the decoder must bind to target entity and
action generation.

## Minimum Falsification Ablations

### A. Trajectory comparison before new training

Compare these policies on the same close initial states:

```text
original non-augmented chunk_bc_v7 e50, g=2
scale_aug e50, g=2
gated e50, g=2
```

Metrics:

```text
min EEF-to-handle distance
first near-contact step
drawer joint progress after near-contact
action projection on drawer axis before contact
action projection on drawer axis after contact
```

Expected falsification:

```text
If scale_aug/gated reach contact as well as original g=2 but still fail to move
the drawer, the missing piece is post-contact action/object-motion binding.

If scale_aug/gated fail to reach contact, the missing piece is spatial anchoring
or target-part selection.
```

Result on 2026-07-08:

```text
Ablation A completed.

bc_e50_g2:
  close success: 19/25
  median min part distance: 0.017 m
  median first near-part step: 60
  median q progress after first near-part: 0.598

scale_aug_e50_g2:
  close success: 0/25
  categories: 25/25 no-axis-contact
  median min part distance: 0.241 m

gated_e50_g2:
  close success: 0/25
  categories: 22/25 no-axis-contact, 3/25 near-axis wrong-offset
  median min part distance: 0.229 m
```

Interpretation:

```text
The confirmed bottleneck for scale_aug/gated is spatial anchoring/target-part
selection, not post-contact drawer actuation. These policies preserve seen-task
behavior but fail to route held-out close G toward the drawer handle/contact
manifold.
```

### B. Counterfactual G-binding diagnostic

The current zero-shot held-out close split is useful but too harsh for deciding
whether G is learnable. Add a small-shot close diagnostic:

```text
train: open_drawer, move_slider_left, move_slider_right, plus k close_drawer demos
k: 0, 5, 10, 25, all
compare: no-G vs raw concat G vs goal-token/query G
test: held-out close initial states, n>=100 when promising
controls: zero-G, shuffled-G, sign-flipped-G
```

A positive G interface should show:

```text
higher close sample efficiency than no-G
clear performance drop under shuffled/sign-flipped G
no train-task collapse at g=1
```

This is a better gate than asking BC to learn a fully unseen opposite action
from three related tasks only.

### C. Mainstream policy integration gate

Only after B is positive, integrate with a stronger decoder:

```text
conditioned flow/diffusion policy:
  image/state encoder
  target mask or part token
  G query token encoder
  action chunk decoder
  classifier-free G dropout
  optional G-alignment energy during sampling
```

This aligns with AffordDP/TarGF/3DFlowAction-style use of intermediate
representations as guidance or constraints rather than passive features.

## What This Means For The Paper Story

The current story should not be:

```text
Any G vector concatenated to a policy improves transfer.
```

The defensible story is:

```text
Reusable interaction fields become useful when they are spatially bound to the
target entity and exposed to the policy through a controllable goal interface.
```

This keeps the original cross-task reuse thesis, but moves the technical
novelty from "G exists" to "what policy interface makes G causally usable."

## Next Implementation Step

Implement Ablation A first. It is the smallest diagnostic and uses existing
checkpoints.

Then implement Ablation B with the goal-token/query-G variant. Do not continue
with only `g_scale`, `gate_max`, or scalar gate sweeps unless Ablation A shows
that the only remaining gap is gain calibration.

## Ablation B.0 Result: Language / Oracle Query-G

Date: 2026-07-08

Implemented a first query-G version in the existing deterministic chunk BC
framework:

```text
state only
state + language one-hot
state + oracle query-G
state + language one-hot + oracle query-G
```

The oracle query-G is task/language-conditioned by construction. This matches
the current project assumption:

```text
language + observation -> G generator -> query-G tokens -> policy
```

The experiment does not claim that query-G can solve task ambiguity without
language. It tests whether a reasonable task-derived G can act as a reusable
low-level interface for the policy.

Result summary:

| Mode | Seen train tasks | Held-out close |
| --- | ---: | ---: |
| `state_e50` | 50/75 | 0/25 |
| `lang_e50` | 70/75 | 0/25 |
| `query_g_e50` | 73/75 | 0/25 |
| `lang_query_g_e50` | 70/75 | 0/25 |

Close contact diagnostic:

```text
state_e50:
  median min part distance: 0.327 m
  categories: 24/25 no-axis-contact, 1/25 near-axis wrong-offset

query_g_e50:
  median min part distance: 0.251 m
  categories: 25/25 no-axis-contact

lang_query_g_e50:
  median min part distance: 0.269 m
  categories: 25/25 no-axis-contact
```

Interpretation:

```text
Query-G improves seen-task control and offline fit, so the token content is not
pure noise.

But zero-shot close transfer still fails before effective contact. The current
policy can use query-G as an additional feature on seen tasks, but it does not
yet learn a compositional rule of the form:

  same/similar drawer scene + different desired object flow -> different
  target-contact approach and post-contact action

This is a semantic/spatial binding failure, not just a data-fitting failure.
```

Design consequence:

```text
Do not conclude that reusable G is invalid. Conclude that flat concat query-G
inside a deterministic MLP chunk BC is not a sufficient policy interface.

The next useful step is a sample-efficiency G-binding diagnostic, not another
zero-shot-only run:

  train: open, left, right + k close demos
  k: 0, 5, 10, 25, all
  compare: no-G, lang, query-G, lang+query-G
  controls: zero-G, shuffled-G, sign-flipped-G

The criterion is whether query-G reduces the number of close demos needed to
enter contact and produce drawer progress, while degrading under corrupted G.
```

Mainstream integration gate:

```text
Only move to a stronger flow/diffusion policy once the small-shot diagnostic
shows causal G usage. Then integrate G at the image/part-token/action-decoder
interface, with G dropout/corruption controls and optional guidance energy.
```

## Ablation B.1 Result: Small-Shot G-Binding

Date: 2026-07-08

Result:

```text
close demos k=1:
  state:          0/25
  language:       0/25
  query-G:       23/25
  lang+query-G:  21/25

close demos k=2:
  state:          0/25
  language:      11/25
  query-G:       25/25
  lang+query-G:  25/25

close demos k>=5:
  language, query-G, and lang+query-G all saturate at 25/25
```

Corruption controls:

```text
k=1 query-G:
  normal:          23/25
  zero-G:           0/25
  sign-flip-G:      1/25
  open-drawer-G:    0/25

k=2 query-G:
  normal:          25/25
  zero-G:           0/25
  sign-flip-G:      0/25
  open-drawer-G:    0/25
```

Design interpretation:

```text
The useful claim is no longer "query-G solves fully zero-shot close".

The stronger and more defensible claim is:

  A compact, task-derived interaction query-G dramatically reduces the number
  of target-task demonstrations needed for a policy to acquire the correct
  contact/control mode, and the learned behavior causally depends on G.

This fits the project thesis better than a pure zero-shot framing. It says G is
a reusable interaction interface that language can instantiate, not a complete
replacement for task-specific policy adaptation.
```

What this means for scope:

```text
Keep language in the final system. Language selects/generates the appropriate
query-G. Query-G then supplies the reusable geometry/control interface.

The next scaling question becomes:
  Can query-G be learned/generated from language + observation/video at scale,
  and can the policy keep the same low-shot advantage when G is predicted rather
  than oracle-generated?
```

Immediate next checks:

```text
1. Finish n=100 confirmation for k=1/2.
2. Add visual rollouts for k=1 query-G normal vs zero/open/sign-flip.
3. Run train-task non-regression for k=1/2 query-G policies.
4. If n=100 confirms, start replacing oracle query-G with a learned G generator.
```

Update after n=100 partial confirmation:

```text
k=1:
  state:          0/100
  language:       1/100
  query-G:       89/100
  lang+query-G:  94/100

k=2:
  state:          0/100
  language:      48/100
  query-G:      100/100
  lang+query-G: 100/100

k=1 query-G corruption:
  normal:         89/100
  zero-G:          0/100
  sign-flip-G:     7/100
  open-drawer-G:   0/100

k=2 query-G corruption:
  normal:        100/100
  zero-G:          0/100
  sign-flip-G:     0/100
  open-drawer-G:   0/100

k=2 lang+query-G corruption:
  normal:        100/100
  zero-G:          0/100
  sign-flip-G:    40/100
  open-drawer-G:   0/100
```

This is now strong enough to move the main story from a speculative G interface
to a concrete sample-efficiency claim:

```text
Given minimal target-task exposure, query-G provides a reusable interaction
condition that language alone does not provide, and corruption controls show
the policy causally depends on that condition.
```

## Ablation C: Primitive-Scope Zero-Shot Close

Date: 2026-07-09

Motivation:

```text
The k=1/2 result shows query-G strongly reduces target-task sample complexity,
but it does not explain why k=0 zero-shot close fails.

The next hypothesis is that zero-shot fails because the train scope is too
narrow: open_drawer + slider left/right does not contain enough push-in,
reverse-axis, or negative-control primitives for the policy to compose close.
```

Experiment:

```text
train tasks:
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

held-out task:
  close_drawer

rollouts:
  100 close_drawer

modes:
  chunk_bc_state
  chunk_bc_lang
  chunk_bc_query_g
  chunk_bc_lang_query_g
```

Important caveat:

```text
The current oracle query-G generator only covers articulation tasks:
open_drawer, close_drawer, move_slider_left, move_slider_right.

For the added block-push and switch/button tasks, query-G is zero. Therefore
this ablation primarily tests whether broader action/task scope plus language
helps zero-shot close. The most informative rows are chunk_bc_lang and
chunk_bc_lang_query_g.
```

Result:

```text
chunk_bc_state:        0/100 close_drawer
chunk_bc_lang:       100/100 close_drawer
chunk_bc_query_g:     16/100 close_drawer
chunk_bc_lang_query_g:100/100 close_drawer
```

Interpretation:

```text
The previous k=0 zero-shot failure was not an intrinsic impossibility of
chunked BC or language-conditioned transfer. It was mainly caused by the narrow
training scope: open_drawer + slider left/right did not expose enough push-in,
negative-axis, and task-switching primitives.

Once broader non-close tasks are added, language alone is sufficient to select a
close-like control mode and reaches 100/100 on held-out close_drawer.

Query-G-only improves over state-only but remains weak at 16/100, which means
the current hand-built query-G does not carry enough task context by itself.
This supports the revised interface view:

  language provides task/goal binding;
  G should provide reusable grounded interaction structure;
  policy uses both to resolve execution.

The result weakens a pure "G replaces language" story, but strengthens a
"language-to-G / language-plus-G reusable interface" story.
```

Follow-up language-control result:

```text
Eval-time language controls on the primitive-scope checkpoints:

chunk_bc_lang:
  normal close key:        100/100
  zero language:             0/100
  task:open_drawer:          0/100
  task:push_into_drawer:     0/100
  task:move_slider_left:     0/100
  task:move_slider_right:    0/100

chunk_bc_lang_query_g:
  normal close key:        100/100
  zero language:             0/100
  task:open_drawer:          0/100
  task:push_into_drawer:     0/100
  task:move_slider_left:     0/100
  task:move_slider_right:    1/100
```

Design implication:

```text
The current G cannot override missing or wrong language. Even when query-G is
the close_drawer query-G, the lang+query-G policy collapses if the task one-hot
is zeroed or replaced.

Therefore the current concat interface should not be described as "G is already
the reusable control interface". A more accurate claim is:

  broader task scope gives the policy the required motor primitives;
  language/task context selects the intended primitive;
  current query-G is useful in small-shot regimes, but not yet sufficient as an
  independent task-binding interface.

Because the current language input is a task one-hot and close_drawer is held
out, the 100/100 primitive-scope result needs seed replication or a true text
embedding before being treated as semantic zero-shot language transfer.
```

Seed replication:

```text
Primitive-scope close, 100 rollouts:

seed 0:
  chunk_bc_lang:         100/100
  chunk_bc_lang_query_g: 100/100

seed 1:
  chunk_bc_lang:         100/100
  chunk_bc_lang_query_g: 100/100

seed 2:
  chunk_bc_lang:         100/100
  chunk_bc_lang_query_g:   0/100
```

Revised design constraint:

```text
Language/task-key-conditioned broad-scope BC is stable in the current CALVIN
setup. Concat query-G is not stable; it can collapse closed-loop performance
without an obvious offline MSE warning.

Therefore the next G design should not simply add more G dimensions. The
problem is likely the fusion/control allocation:

  language/key selects the intended behavior robustly;
  concat G can perturb the learned control manifold;
  G needs to be either learned as an intent bottleneck, gated conservatively, or
  supervised/regularized by an objective that reflects closed-loop contact
  timing rather than only chunk MSE.
```

Seed2 failure diagnosis:

```text
For seed2, chunk_bc_lang succeeds 100/100 but chunk_bc_lang_query_g fails 0/100.

Axis/contact diagnostics show:
  seed2_lang min dist to part p50:       0.065 m
  seed2_lang_query_g min dist p50:      0.327 m
  seed2_lang joint progress p50:        0.162
  seed2_lang_query_g joint progress:    0.000

Rollout-0 first steps:
  seed2_lang action xyz:        [+0.02, -0.02, -0.02]
  seed2_lang_query_g action xyz:[+0.02, +0.02, +0.02]

The failing lang+query-G policy moves upward/away from the drawer instead of
descending toward the handle. This is an approach-manifold failure, not a
post-contact axis-control failure.

Eval-time controls on the failed checkpoint:
  normal close G:   0/100
  zero G:           0/100
  sign-flip G:      0/100
  open-drawer G:    0/100

Thus the failure is not just that the close query-G value is wrong at inference.
Training with raw concat G changed the policy's internal control allocation so
that removing or corrupting G at eval cannot recover the stable language-only
behavior.
```

Critical correction: task one-hot normalization artifact

```text
The primitive-scope 100/100 chunk_bc_lang result is not yet valid evidence for
semantic language-conditioned zero-shot transfer.

Reason:
  close_drawer is held out from training.
  The close_drawer one-hot dimension is therefore always 0 in train.
  The old normalizer used cond_std = train_std + 1e-6 for all condition dims.
  At eval, close_drawer one-hot = 1 becomes about 1e6 after normalization.

Verified in checkpoints:
  close one-hot mean = 0
  close one-hot std  = 1e-6
  normalized eval value ~= 1e6

This explains the strange seed sensitivity. The held-out task key is an
untrained, massively amplified input dimension. Depending on initialization and
the rest of the learned network, it can accidentally produce useful saturated
actions or catastrophically wrong approach behavior.
```

Design implication:

```text
The immediate next baseline must rerun with fixed task-key normalization:
  task one-hot dims use mean 0 and std 1, not train variance.

Longer term, task one-hot should be replaced by a real text embedding or a
language encoder. Otherwise "language" experiments only test task-key behavior.

The previous seed2 concat-G collapse remains useful as a failure diagnosis for
raw concat under unstable conditioning, but it should not be used as final
evidence about G until the normalization bug is removed.
```

Fixed-normalization rerun result:

```text
Date: 2026-07-09

Setup:
  train on primitive-scope non-close tasks
  hold out close_drawer
  eval close_drawer with 100 rollouts per seed
  epochs 50, chunk size 16, episode length 160

Results:
  seed0 chunk_bc_lang:         0/100
  seed0 chunk_bc_lang_query_g: 19/100
  seed1 chunk_bc_lang:         0/100
  seed1 chunk_bc_lang_query_g: 0/100
  seed2 chunk_bc_lang:         0/100
  seed2 chunk_bc_lang_query_g: 0/100

Corrected conclusion:
  Task one-hot alone does not zero-shot transfer to held-out close_drawer in
  this setup after the normalization artifact is removed.

  Query-G has a weak positive signal in one seed, but it is not robust across
  seeds. The current evidence supports "G can bias the policy toward the
  missing interaction primitive" rather than "G already gives stable zero-shot
  transfer".

  The similar train/val MSE across all checkpoints but very different OOD
  rollout outcomes confirms that closed-loop task success and trajectory
  diagnosis must remain the primary metric for this line of work.
```

Implication for next design step:

```text
Do not continue interpreting task one-hot as language.

The next controlled comparison should separate three questions:
  1. Does real language/text embedding provide compositional information for
     held-out task names such as close_drawer?
  2. Does query-G improve over language-only when the language channel is not
     a one-hot artifact?
  3. Is raw concat the wrong fusion mechanism for query-G, given the observed
     seed instability?

Recommended next implementation:
  language/text embedding + no-G
  language/text embedding + query-G concat
  language/text embedding + gated/FILM query-G

Keep the same primitive-scope split and 100-rollout close eval so this rerun
remains directly comparable.
```

Artifacts:

```text
remote log:
/mnt/nvme33n1/wuk_code/AutoResearch/logs/scope_primitive_zero_shot_close_20260709_wait.log
/mnt/nvme33n1/wuk_code/AutoResearch/logs/scope_primitive_zero_shot_close_language_control_20260709.log
/mnt/nvme33n1/wuk_code/AutoResearch/logs/scope_primitive_zero_shot_close_multiseed_seed1_seed2_20260709.log

remote output:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_e50_n100_h160_seed0
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_language_control_n100
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_e50_n100_h160_seed1
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-09_scope_primitive_zero_shot_close_e50_n100_h160_seed2

local sync:
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-09_scope_primitive_zero_shot_close/
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-09_scope_primitive_zero_shot_close_language_control/
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-09_scope_primitive_zero_shot_close_multiseed/
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-09_scope_primitive_zero_shot_close_fixed_tasknorm/metrics_only/
```
