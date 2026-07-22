# CALVIN Minimal-G Direction-Coverage Matrix (2026-07-15)

## Question

The previous expanded-scope policy was misleading: block pushes,
`push_into_drawer`, and light tasks were present in the BC dataset, but the
minimal query builder returned zero G for all of them. Under the language-to-G
hierarchy, these examples trained a language policy rather than the reusable G
executor. Nonzero `G=[A,F]` supervision came only from `open_drawer` and the two
slider directions.

This matrix tests whether poor close transfer and unstable flow-sign binding
come from narrow directional support rather than from the six-dimensional G
itself. A whole-dataset endpoint survey later exposed an important distinction:
three-dimensional linear rank does not imply support for the held-out signed
direction.

## G Contract

The representation is unchanged:

```text
G = [A(3), F(3)]
A = contact_anchor - current_eef
F = desired object-flow vector
```

Language/task context selects the queried interaction. When a valid G exists,
the direct task one-hot is masked from the executor. No stage one-hot, active
joint ID, desired EEF trajectory, or task-specific control phase is added.

For named block pushes, the CALVIN task contract provides exact world-x object
flow and object identity. For `push_into_drawer`, language intentionally does
not specify a color and any block is valid. The query uses the block nearest to
the current EEF; this is an observation-only grounding rule, not a future-state
label. The target region and task-level flow direction are fitted from
demonstration endpoints and stored in the checkpoint.

## Pre-training Audit

The first audit exposed an invalid `push_into_drawer` heuristic: selecting the
block nearest to the terminal drawer region matched the block moved by the
demonstrator in only `46.7%` of 150 segments. It was rejected before training.

Observation-only selector audit:

| Selector | Segment first frame | All sampled frames | Early | Middle | Late |
| --- | ---: | ---: | ---: | ---: | ---: |
| nearest drawer target, 3D | 46.7% | 54.3% | 46.2% | 48.2% | 67.4% |
| nearest EEF, 3D | **74.0%** | **92.0%** | **87.3%** | **99.0%** | **90.7%** |
| shortest EEF-block-target path, 3D | 64.7% | 86.8% | 74.3% | 96.1% | 91.5% |

Nearest EEF is used because it has the strongest current-observation agreement
and naturally locks after approach begins. A disagreement with the recorded
demonstrator at the first frame is not necessarily a wrong task choice because
CALVIN accepts any block for this task. Its remaining label inconsistency is a
known limitation and is reported separately from named-object tasks.

The initial coverage audit showed:

```text
old articulation-only scope: effective direction rank 2
expanded translational scope: effective direction rank 3
```

All six named block push tasks changed from zero G to 100% nonzero G. After
changing the selector, `push_into_drawer` also has 100% nonzero G windows, with
`74.0%` endpoint agreement against the recorded demonstrator. Its near-anchor
first-action projection onto F is positive in `68.8%` of windows, so it is kept
as a deliberately noisier, generic-object task rather than reported as equally
clean as the named block pushes.

## Matrix

D0--D2 and D4 use the same training segments, stride, 50 epochs, state
observations, seed protocol, and held-out close initial states. D3 changes only
the training set by adding `place_in_slider`. Every condition excludes
`close_drawer`.

| ID | G support | Executor | Chunk/objective | Purpose |
| --- | --- | --- | --- | --- |
| D0 | articulation only; primitive tasks receive zero G | FiLM | H1, uniform | matched sparse-support baseline for D1/D2 |
| D1 | articulation + translational primitives | FiLM | H1, uniform | isolate directional-support expansion |
| D2 | same expanded support | bounded projected field-coupled BC | H16, first-action weight 16, replan 1 | test whether expanded support also helps the prior-plus-residual executor |
| D3 | D1 support + `place_in_slider` | FiLM | H1, uniform | add +drawer-y flow in a different task while close remains held out |
| D4 | same expanded support as D1/D2 | FiLM | H16, first-action weight 16, replan 1 | isolate D2's field coupling from chunk/loss changes |

Training tasks:

```text
open_drawer
move_slider_left
move_slider_right
push_into_drawer
push_{red,blue,pink}_block_{left,right}
```

D3 adds only:

```text
place_in_slider
```

D0 is intentionally not an exact rerun of the 2026-07-14 H1 FiLM result
`[0,0,0,19,14]/100`. That earlier experiment also trained on
`turn_{on,off}_{lightbulb,led}`, four zero-G tasks that change the shared state
trunk and task sampling distribution. They are omitted from both D0 and D1 so
that their difference isolates the G builder/scope on identical data. The old
result remains historical evidence; D0 is the valid matched control for this
matrix.

The endpoint survey shows why D3 is necessary:

| Task family | Dominant object flow |
| --- | --- |
| open drawer | -y in 100% of moving segments |
| push into drawer | -y/-z in 100% of segments |
| named block pushes | +/-x |
| place in slider | **+y in 100% of 151 segments** |
| held-out close drawer | +y in 100% of moving segments |

Thus D1 expands vector rank but still asks FiLM to extrapolate the sign needed
by close. D3 is a directional signed-coverage control: the +y supervision
comes from placing an object in the slider, not from any close demonstration.
It is not a contact-mode-matched control. `place_in_slider` contains
grasp/place dynamics and its builder anchors the block center, whereas close is
a sustained articulated push. D3 can test whether seeing the held-out signed
object-flow direction is sufficient, but cannot isolate direction from contact
mode or anchor construction.

D2 is a mechanism control, not the final proposed policy. It explicitly
couples translation to G and therefore tests the closed-loop low-frequency
prior hypothesis. D0 versus D1 is the clean representation-coverage comparison.
D4 is the strict executor control for D2: data, G, H16 target, first-action
weight, and inference horizon are matched, while ordinary FiLM replaces the
bounded field-coupled output.

## Evaluation Protocol

For each condition, train seeds `0..4` and evaluate the identical 100
`close_drawer` initial states under:

```text
normal G
zero G
flow sign flip
wrong open-drawer G
```

Also evaluate 25 rollouts per seen translational task. Preserve all 100 close
rollouts; do not shorten evaluation because a partial estimate looks stable.
CALVIN/PyBullet evaluation is serialized through one renderer and resumes from
`eval_progress.json` after a driver abort; policy training may still run on a
different GPU concurrently.

Primary gates:

1. D1 normal close measures whether generic component coverage alone improves D0.
2. D3 normal close improves D1 across at least four of five seeds and has
   nonzero worst-seed success.
3. D3 correct normal G beats sign-flipped G across at least four of five seeds.
4. Seen-task aggregate health remains at least 90%.
5. Normal and corrupted runs use identical selected sequence indices.

If D1 improves D0 but fails sign binding, coverage helps interpolation but
ordinary FiLM still does not learn the physical G semantics. If D2 alone is
strong, the result supports G as a reusable closed-loop action prior, not a
condition-only token. If neither improves, the next change should target
coordinate-equivariant training or task composition rather than enlarging G.

## Normal-G Results

All 25 normal-G evaluations are complete on the same 100 held-out
`close_drawer` initial states.

| Condition | Seed successes (/100) | Mean SR | Sample SD | Worst seed |
| --- | --- | ---: | ---: | ---: |
| D0 sparse FiLM | `[0,15,0,0,0]` | 3.0% | 6.7% | 0% |
| D1 expanded-direction FiLM | `[2,0,0,0,0]` | 0.4% | 0.9% | 0% |
| D4 expanded-direction FiLM H16 | `[2,0,5,4,0]` | 2.2% | 2.3% | 0% |
| D2 expanded field-coupled | `[95,83,51,100,97]` | **85.2%** | 20.2% | **51%** |
| D3 FiLM + cross-task `+y` | `[0,0,0,0,0]` | 0.0% | 0.0% | 0% |

Mean final first-action validation MSE is `0.027467` (D0), `0.026389`
(D1), `0.024973` (D4), `0.026304` (D2), and `0.024749` (D3). D4 is the strict
control for D2: it uses the same data, expanded G, H16 chunks, first-action
weight 16, and replan-1 inference, while replacing bounded field coupling with
ordinary FiLM. D4 fits the first action better offline but reaches only
`11/500` close successes versus D2's `426/500`, an `83.0` percentage-point gap.
D3 also has low offline error but zero close success. The transfer difference
is therefore a closed-loop integration effect, not evidence that D2 simply fit
the demonstrations more accurately or benefited from the chunk/loss setting.

These results reject two simple explanations for unstable close transfer:

1. Moving from rank-2 articulation G to rank-3 translational G does not make
   an ordinary FiLM executor reusable; D1 is lower than its matched D0 control.
2. The FiLM failure is not explained only by the absence of the held-out
   signed direction. D3 adds 151 `place_in_slider` demonstrations whose object
   flow has positive drawer-y in every segment, yet all five close policies
   remain at zero.

D2 differs architecturally, not representationally: it constructs a positive
translation component along the attraction/flow field and restricts the norm
of the learned orthogonal correction. Its nonzero worst-seed result shows that
the six-dimensional G contains enough task-specific geometry to solve held-out
close when used as a closed-loop control prior. It does **not** show that this
hand-structured executor is the final scalable policy. The current method
problem is to retain D2's weak geometric contract inside a mainstream learned
residual flow/diffusion policy without hard-coding the full action.

D4 closes the main matrix confound. Across all five seeds, matched H16 FiLM has
zero worst-seed success and remains in the same low-success regime as H1 FiLM.
Thus neither action chunking nor first-action weighting explains D2. Its causal
controls are also complete: zero G gives `0/500`, sign flip gives `18/500`, and
wrong-open G gives `19/500` versus normal `11/500`. Normal successes have zero
overlap with either corrupted condition. D4 therefore retains FiLM's unstable
mode-code behavior under the exact D2 chunk/objective setting. Its seen-task
health is also complete at `639/1250 = 51.1%`, below D2's
`803/1250 = 64.2%`.

### Ordinary FiLM causal controls

D0, D1, and D4 completed all five seeds under the same matched close initial
states:

| Condition | Normal | Zero | Flow sign flip | Wrong open G |
| --- | ---: | ---: | ---: | ---: |
| D0 sparse FiLM | 15/500 | 0/500 | 12/500 | 2/500 |
| D1 expanded FiLM | 2/500 | 0/500 | **36/500** | 3/500 |
| D4 expanded FiLM H16 | 11/500 | 0/500 | **18/500** | **19/500** |

D0's normal and sign-flip successes have zero rollout overlap: the paired table
contains 15 normal-only and 12 flip-only successes. D1 is more decisive: it has
35 flip-only, one normal-only, one both-success, and 463 both-failure outcomes.
All selected sequence indices match exactly.

Zero G removes every D0/D1 success, so G influences both policies. It does not
follow that they learned reusable physical semantics. D1's physically correct
close flow is +drawer-y, while `[A,-F]` points -drawer-y and is much closer to
the seen open branch. The 18x pooled improvement under the wrong sign shows
that expanded rank alone makes FiLM interpolate a familiar conditioning code;
it does not bind vector sign to object motion. This is the opposite of D2,
where normal beats sign flip for every seed and by 423 paired normal-only versus
zero flip-only successes.

D4 gives the same diagnosis under D2-matched H16 training. Its paired tables
contain 11 normal-only and 18 flip-only outcomes, or 11 normal-only and 19
wrong-open-only outcomes, with no shared successes. More temporal targets and
first-action weighting change neither transfer stability nor physical sign
binding in an ordinary conditional executor.

### Flow-sign control

The five-seed D2 sign-flip control is complete on the same initial states:

| Control | Seed successes (/100) | Pooled |
| --- | --- | ---: |
| normal G | `[95,83,51,100,97]` | 426/500 |
| `[A,-F]` | `[0,0,0,0,3]` | 3/500 |

The paired rollout table contains 423 normal-only successes, zero flip-only
successes, three both-successes, and 74 both-failures. The auxiliary exact
McNemar value is `p=9.23e-128`. This pooled value describes matched rollout
transitions and does not replace the five-seed result; all five independently
trained policies are worse under the flipped flow.

This control establishes stable causal flow-sign semantics for D2. It sharply
contrasts with the earlier unconstrained H1 FiLM result, where flow flip could
improve one seed. The remaining D2 causal controls and all four seen-task
evaluations are reported below.

### Wrong-task G control

The five-seed wrong-open-G control is also complete:

| Control | Seed successes (/100) | Mean SR | Pooled |
| --- | --- | ---: | ---: |
| normal close G | `[95,83,51,100,97]` | 85.2% | 426/500 |
| open-drawer G | `[9,6,14,15,40]` | 16.8% | 84/500 |

Matched outcomes contain 355 normal-only successes, 13 wrong-only successes,
71 both-successes, and 61 both-failures. The nonzero wrong-only count matters:
the residual policy can occasionally overcome an incorrect field, so G does
not fully determine the action. Nevertheless, correct task-selected G changes
the success distribution strongly and improves every independently trained
seed.

### Zero-G control and seen health

Zero-G is complete at `[0,0,0,0,0]/100`. Together, the three interventions
show that D2 needs a nonzero, correctly signed, task-selected query G; the
field-coupled architecture does not close the drawer unconditionally.

All five seen-task evaluations are complete at
`[160,150,183,144,166]/250 = 803/1250 = 64.2%`, below the registered 90%
health gate:

| Seen family | Success |
| --- | ---: |
| drawer/slider articulation | 354/375 (94.4%) |
| named block pushes + push into drawer | 449/875 (51.3%) |

The family split identifies a representation-builder mismatch rather than a
uniform policy collapse. For articulation, A points to a geometry-fitted handle
contact and F follows a constrained joint axis. Named pushes do use a crude
surface-side approximation, `A = block_center - 0.05 * flow_dir - EEF`, rather
than the block center itself. However, the fixed `0.05 m` offset and common
readiness radius are not calibrated to the actual block half-extent, gripper
geometry, or demonstrated contact transition. They can switch the controller
from attraction to flow before physical contact. `place_in_slider` directly
uses the block center and additionally has grasp/place dynamics. D2 therefore
establishes the low-frequency-prior mechanism but fails as a universal final
executor. The same family split appears across all five optimization seeds.

### Cross-condition common-task health

All five conditions completed the same `10 common tasks x 25 rollouts x 5
seeds` protocol:

| Condition | All seen tasks | Drawer/slider articulation | Free-object push |
| --- | ---: | ---: | ---: |
| D0 sparse FiLM | 310/1250 (24.8%) | 274/375 (73.1%) | 36/875 (4.1%) |
| D1 expanded FiLM | 497/1250 (39.8%) | 303/375 (80.8%) | 194/875 (22.2%) |
| D2 field-coupled | **803/1250 (64.2%)** | **354/375 (94.4%)** | **449/875 (51.3%)** |
| D4 expanded FiLM H16 | 639/1250 (51.1%) | 288/375 (76.8%) | 351/875 (40.1%) |
| D3 FiLM + `place_in_slider` | 427/1250 (34.2%) | 259/375 (69.1%) | 168/875 (19.2%) |

This common-task table does not include D3's additional training task,
`place_in_slider`. A completeness audit attempted a separate first-task
evaluation, but CALVIN exposes no matching first-task sequence because the task
requires a block to be grasped by a preceding lift/grasp task. The official
evaluator correctly raises `No first-task eval sequences matched filter`.
Evaluating place requires a chain protocol with its predecessor and cannot be
inserted as an independent reset task. The common 10-task table is therefore
the final matched health comparison; standalone D3 place health is unavailable
under this protocol and does not affect any held-out close result.

D1's improvement over D0 proves that expanding the nonzero-G builder does help
the policy fit the added primitive tasks. It does not transfer to held-out
close: normal close falls from D0's already weak `3.0%` to `0.4%`. D2 improves
both seen families and held-out close, so its advantage is not a trade in which
hard field coupling sacrifices the training tasks to solve one test task.

D4 gives the strict chunk/objective comparison. Its five seen seeds are
`[130,130,135,115,129]/250`; per-task totals are `124/125` open drawer,
`88/125` slider left, `76/125` slider right, and `351/875` across the seven
free-object push evaluations. D4 improves common-task fitting over H1 D1 but
remains below D2 on articulation and free-object tasks, while close stays at
`11/500`. H16 supervision helps ordinary FiLM imitate more seen behavior but
does not create reusable physical G semantics.

D3 is lower than D1 on both common seen families and remains `0/500` on close. Adding
the missing +y object-flow sign in a grasp/place task is therefore not
sufficient and can introduce negative transfer when one FiLM executor is asked
to identify contact mode from the same compact field. This does not prove that
+y demonstrations are intrinsically harmful; it bounds the claim to the
current center-anchor builder, language-masking rule, and contact-mode-mismatched
data. A valid next positive control needs matched push/contact dynamics without
including close itself.

## Primitive Contact-Transition Diagnosis

A data-only audit locates the first frame where the active block has moved at
least `5 mm`, then uses the closest pre-motion EEF-to-current-anchor frame as a
contact proxy. This is inferred from motion and is not a simulator contact
sensor.

| Task family | Contact-proxy EEF projection on F (p50) | Fixed-anchor error (p50) | Readiness at first motion (p50) | Frames readiness>0.5 before motion (p50) |
| --- | ---: | ---: | ---: | ---: |
| blue left/right | `-0.007/-0.010 m` | `0.045/0.041 m` | `0.884/0.881` | `19/15` |
| pink left/right | `-0.040/-0.041 m` | `0.019/0.018 m` | `0.955/0.960` | `15/19` |
| red left/right | `-0.034/-0.032 m` | `0.028/0.026 m` | `0.948/0.951` | `16/20.5` |
| push into drawer | `-0.046 m` | `0.021 m` | `0.944` | `23` |

The demonstrated EEF offset is not represented by one universal `-0.05 m`
contact rule, even among the colored blocks. More importantly, the common
`0.10 m` readiness threshold begins blending toward object flow 15--23 frames
before the object actually moves. This explains the video symptom previously
seen on close failures in a task-general way: the controller can begin the
manipulation direction before physical contact.

The next builder should preserve compact `G=[A,F]` but learn or geometrically
estimate A on an executable contact manifold and predict a contact/readiness
confidence from observation history. This confidence belongs in the policy
integration/gating mechanism; it should not be added as another task-specific
G semantic term.

## Artifacts

- Final generated table:
  [summary.md](../../artifacts/server_sync/2026-07-15_minimal_g_direction_coverage_matrix/2026-07-15_minimal_g_direction_coverage_matrix_summary/summary.md)
- Full per-seed/per-rollout aggregate:
  [matrix_summary.json](../../artifacts/server_sync/2026-07-15_minimal_g_direction_coverage_matrix/2026-07-15_minimal_g_direction_coverage_matrix_summary/matrix_summary.json)

Server audit:

```text
experiments/phase1_calvin/2026-07-15_minimal_g_direction_coverage_audit/
```

Server experiment stems:

```text
2026-07-15_D0_sparse_direction_coverage_film_h1_seed{0..4}
2026-07-15_D1_expanded_direction_coverage_film_h1_seed{0..4}
2026-07-15_D2_expanded_direction_coverage_field_coupled_h16_seed{0..4}
2026-07-15_D3_signed_direction_coverage_film_h1_seed{0..4}
```

Master and per-job logs:

```text
logs/2026-07-15_minimal_g_direction_coverage_matrix/
```
