# Phase-1 CALVIN G-Token and Inverse-Pair Experiment

Date: 2026-07-16

## Question

Can a rule-free conditional policy learn to parse compact reusable
`G=[A(3),F(3)]` directly, and does explicitly supervising near-state
inverse-goal action differences improve zero-shot `close_drawer` transfer?

## Policies

E0 uses separate learned anchor and flow token encoders, a state token, and
action-query cross-attention. E0-BC decodes an H16 action chunk directly;
E0-FM predicts the conditional flow velocity of an H16 action chunk. There is
no action projection, readiness threshold, field gain, or inference-time
geometric rule. For tasks with valid oracle G, language is used upstream to
construct G and is masked from the executor input.

E1 keeps the E0 architecture, data, split, H16 target, first-action weight 16,
and 50-epoch schedule. It adds a training-only inverse-pair difference loss.
For near-state opposite-goal samples, the predicted action difference must
match the expert action difference. Inference remains identical to E0.

E1 mines pairs with EEF distance below 5 cm and target mechanism/object-state
difference below 10% of its observed range. Train-pair counts are:

| Family | Unique matched pairs |
| --- | ---: |
| slider left/right | 1,593 |
| red block left/right | 4,058 |
| blue block left/right | 3,706 |
| pink block left/right | 3,169 |

Families are balanced to 16,232 pair samples; 128 pairs are used per update and
the pair-loss weight is 0.5.

## Offline fit

| Policy | Val chunk MSE | Val first-action MSE | Final pair-difference loss |
| --- | ---: | ---: | ---: |
| E0 BC | 0.04336 | 0.02705 | n/a |
| E1 BC | 0.04415 | 0.02926 | 0.116 |
| E0 FM | 0.06649 | 0.03756 | n/a |
| E1 FM | 0.06379 | 0.03843 | 0.173 |

E1 does not obtain its behavior by improving ordinary BC MSE. Its pair loss
does converge strongly: BC falls from 1.476 to 0.116.

## Same-state G audit

The validation audit holds state fixed and replaces only G with the matched
opposite-goal G. It then compares the resulting first-action difference with
the expert opposite-goal action difference.

| Policy | Slider cosine | Red block | Blue block | Pink block |
| --- | ---: | ---: | ---: | ---: |
| E0 BC | 0.810 | 0.666 | 0.831 | 0.616 |
| E1 BC | 0.876 | 0.679 | 0.824 | 0.486 |
| E0 FM | 0.753 | 0.628 | 0.798 | 0.478 |
| E1 FM | 0.882 | 0.642 | 0.844 | 0.565 |

E0 already parses G on seen support. For E0-BC, normal/zero/flow-flipped G
give overall first-action MSE `0.02705/0.05114/0.05714`. E1 strengthens
slider sensitivity and several FM families, but is not uniformly better:
E1-BC regresses on pink-block direction.

## Task metric

All rows use the same first 100 CALVIN `close_drawer` initial states and
episode length 160. State-only observation skips unused camera rendering but
keeps simulator dynamics, policy state, task oracle, and success criteria
unchanged.

| Policy/control | Execute horizon | Success |
| --- | ---: | ---: |
| E0 BC, normal G | 1 | 0/100 |
| E1 BC, normal G | 1 | 2/100 |
| E1 BC, zero G | 1 | 0/100 |
| E1 BC, F sign-flip | 1 | 0/100 |
| E1 BC, wrong open-drawer G | 1 | 0/100 |
| E0 FM, normal G | 16 | 0/100 |
| E1 FM, normal G | 16 | 0/100 |
| E0 FM, normal G | 1 | 0/100 |
| E1 FM, normal G | 1 | 0/100 |

E1-BC succeeds only on validation sequence indices 683 and 982. The two cases
pass basic G-corruption controls, so they are G-dependent successes, but 2% is
too small to support a method claim. FM fails with both complete-chunk and
one-step execution. The H1 evaluations were split into four disjoint 25-case
shards and formally aggregated; their 100 sequence indices exactly match the
BC protocol.

## Interpretation

The experiment rejects the simple diagnosis that the token policy ignores G.
It reads G and changes actions in the correct relative direction on seen
slider/block support. The remaining failure is cross-family executor transfer:
the learned mapping from a desired interaction direction to contact-producing
control does not reliably compose with the unseen open-drawer state and drawer
contact dynamics.

The FM result is not evidence against flow matching in general. This baseline
has a lightweight state encoder, only the current CALVIN dataset, a small
action-token Transformer, deterministic Euler integration, and no visual
pretraining. It establishes that merely replacing deterministic chunk BC with
conditional flow matching does not solve the G transfer problem under the
current data and interface.

## Seen-task health

Each of the 14 training tasks is evaluated on 25 first-task sequences. BC uses
H1 replanning and FM uses H16 chunk execution.

| Policy | Articulation (75) | Block/push (175) | Light/LED (100) | Total (350) |
| --- | ---: | ---: | ---: | ---: |
| E0 BC | 45 | 55 | 68 | 168 (48.0%) |
| E1 BC | 56 | 72 | 69 | 197 (56.3%) |
| E0 FM | 75 | 111 | 97 | 283 (80.9%) |
| E1 FM | 74 | 80 | 93 | 247 (70.6%) |

E0-FM is perfect on `open_drawer`, `move_slider_left`, and
`move_slider_right` (`25/25` each), and nearly perfect on lights/LEDs. The
conditional FM implementation is therefore a credible seen-task executor.
Its `0/100` close result is a transfer failure, not evidence that FM training
or sampling is globally broken.

Inverse-pair loss is architecture-dependent. It improves BC seen health by
8.3 points, but reduces FM seen health by 10.3 points, concentrated in block
pushes (`111/175 -> 80/175`). Do not increase this auxiliary weight. It can
distort the learned action manifold even while improving same-state G
sensitivity.

The strongest remaining diagnosis is missing conditional composition:
`drawer already open + EEF far/approach + close-signed sustained contact`.
Open-drawer data contains closed/far approach and open/near contact states;
slider data contains both direction signs but different mechanism/contact
dynamics. None supplies the full held-out conjunction.

Inverse-pair supervision is useful as a diagnostic and modest local
regularizer, but it is not the next main method. It reinforces directions that
already exist in slider/block data and does not supply the missing shared
contact-state support for closing a drawer. The next data intervention should
create matched approach/contact states across multiple mechanisms and motion
signs, rather than further tuning the pair-loss weight.

## Reverse-interaction feasibility gate

CALVIN relative actions have direct control semantics: xyz translation is
normalized by `0.02 m`, Euler-angle change by `0.05 rad`, and the final value
is the binary gripper command. This makes an open-drawer transition a
candidate source of a reversed close-drawer transition, but contact dynamics
must be tested rather than assumed reversible.

Twenty-five open-drawer demonstrations were reset at their exact
post-demonstration robot and scene states and replayed backward. Two controls
were evaluated with CALVIN's official relative `close_drawer` threshold of
`-0.12 m`:

| Reverse controller | Close success | Median close displacement | Median TCP position RMSE |
| --- | ---: | ---: | ---: |
| Negate recorded relative xyz/Euler action | 25/25 | 0.165 m | 7.4 mm |
| Track the preceding recorded TCP pose | 25/25 | 0.171 m | 1.5 mm |

The direct action inversion is sufficient in every sampled segment. Reverse
labels therefore do not require an inference-time push rule or an oracle
contact controller. Gripper commands target the preceding recorded state's
binary gripper state; source and reverse windows share one segment id so they
cannot cross the train/validation group split.

This result sharpens the missing-support diagnosis. It does not count as
clean zero-shot close transfer: reversed open data injects same-mechanism
close-action support. It is instead a self-supervised inverse-interaction
augmentation and a controlled test of whether E0's failure is caused by the
missing `open drawer + close-signed contact` branch. E2 keeps the E0 policy,
14 original tasks, split, H16 target, and 50-epoch schedule fixed while adding
only 3,800 reverse-open windows from the same 152 source segments.

| E2 policy/control | Close success | Seen-task health |
| --- | ---: | ---: |
| BC H1, normal G | 74/100 | 216/350 (61.7%) |
| BC H1, F sign-flip | 1/100 | n/a |
| FM H16, normal G | 0/100 | 255/350 (72.9%) |
| FM H16, F sign-flip | 0/100 | n/a |
| FM H1, normal G | 1/100 | n/a |

The paired BC normal/flip table is 73 normal-only, 0 flip-only, 1 both, and
26 neither. Reverse interaction data therefore converts E0-BC from `0/100`
to `74/100`, and the improvement depends on the correct physical direction
of G. This strongly supports missing same-mechanism inverse-contact support as
the main E0-BC bottleneck.

FM exposes a different failure. On 475 held-out reverse-close validation
windows its normal first-action MSE is `0.0199` versus BC's `0.0159`; zero and
flow-flipped G worsen FM to `0.0527/0.0542`. Its predicted first xyz action has
median cosine `0.963` to the target, and its mean projection along F is
`0.166`, exactly matching the expert mean. Thus FM fits the inverse data and
direction offline but fails in closed loop under both H1 and H16. Fixed-noise,
continuous-noise-stream, intermediate-horizon, and temporal-ensemble audits
reject simple evaluator tuning as the explanation:

| FM rollout setting | Close success |
| --- | ---: |
| H1, fixed noise seed 17 | 1/100 |
| H2 / H4 / H8 / H16 | 4 / 0 / 0 / 0 out of 100 |
| H1, continuous noise stream across episodes | 4/100 |
| H1, temporal ensemble | 0/100 |

Fixed H1 noise seeds 18--21 give `1/25, 0/25, 0/25, 1/25` on the
same initial states. Reusing one noise seed per episode is an evaluator flaw,
but correcting it does not close the BC/FM gap.

Axis/contact diagnosis on the same 25 close states gives BC `17/25` and FM
`0/25`. FM median minimum distance to the drawer part/axis line is
`13.45/5.82 cm`, versus BC `6.84/2.09 cm`; median first near-axis step is
`81` versus `36`. FM issues a median 43 desired-axis commands without joint
progress, versus 5 for BC. It often begins actuation before lateral/height
alignment creates effective contact. This is a temporal contact-ordering
failure, not a one-step flow-direction failure.

E2 does not uniformly preserve source-task health. Relative to E0, BC rises
from `168/350` to `216/350`, while FM falls from `283/350` to `255/350`, mainly
on block pushes. Future augmentation must use task-balanced sampling instead
of simply increasing the articulation fraction.

The compact G is sufficient for this branch: BC learns approach-then-actuate
from the same state and G inputs. Do not add stage/contact details to G to fix
the lightweight FM result. Retire further seed/horizon tuning and migrate the
interface into a credible flow/diffusion policy with observation history,
task-balanced sampling, and a proven temporal action backbone.

## Artifacts

```text
training:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-16_E0_g_token_bc_fm_matched_d4_seed0
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-16_E1_g_token_inverse_pair_w0p5_seed0

condition sensitivity:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-16_E0_E1_g_token_condition_sensitivity

state-only task evaluations:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-16_E0_g_token_bc_normal_close_h1_stateonly_seed0
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-16_E1_g_token_bc_normal_close_h1_stateonly_seed0

reverse feasibility and E2 training:
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-16_open_drawer_time_reversal_n25_seed0
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-16_E2_g_token_reverse_open_seed0
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-16_E2_g_token_condition_sensitivity
/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-16_E2_bc_fm_close_axis_contact_diag_n25
```
