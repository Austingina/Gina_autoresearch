# CALVIN interaction-volume G audit (2026-08-14)

## Question

Why did the matched behavior-G policy improve offline MSE without improving
held-out block-push success, and what G can be retrained without encoding one
demonstration-specific contact strategy?

## Protocol

- Source-only fit tasks: `push_red_block_left`, `push_blue_block_right`,
  `push_pink_block_left`, and `push_pink_block_right`.
- Frozen audit tasks: `push_blue_block_left` and `push_red_block_right`.
- Data: 114 non-overlapping official CALVIN ABC segments selected by HTTP range
  from `task_ABC_D.zip`; 96 source segments (24 per task), 18 held-out segments
  (9 per task), and 38 segments in each A/B/C scene.
- Scene geometry is loaded from each official scene YAML. This matters because
  movable-object ordering and block sizes differ across A/B/C.
- Object-motion onset is the first frame with at least 2 mm directed object
  progress. Held-out rows never participate in clearance fitting.

## Findings

The old training and online G were semantically mismatched. Training used the
demonstration EEF contact pose and zeroed attraction after inferred object
motion. Online G pointed to the block center and zeroed attraction only within
1 cm. In ten traced rollouts, the online controller never entered this
post-contact state, although 63.3% of converted training frames were in it.

A flow-opposed canonical surface anchor is not a sufficiently reusable point
target. Its source-fitted tool clearance is stable at 14.15 mm by aggregate
task median (source-task spread 2.26 mm; held-out max gap 0.79 mm), but the
strict scene-by-task gate fails. In scene B, pink-right demonstrations contain
both upstream-side and top/side contact strategies. A single canonical point
therefore has 34.3 mm validation p90 and 36.1 mm held-out p90 contact error.
This is demonstration multimodality, not a yaw or scene-indexing bug.

An isotropically expanded oriented-object interaction volume retains the same
six-dimensional interface:

```text
G = [delta_to_interaction_volume_xyz, desired_object_flow_xyz]
```

The source-only p90 expansion is 18.01 mm. It covers 91.7% of source validation
contacts and 94.4% of held-out contacts. At motion onset, validation and
held-out p90 residual distance to the volume are both zero. This representation
does not select a unique grasp/push point and accommodates the observed
multi-strategy demonstrations.

The volume is deliberately permissive: validation enters it a median 5 frames
before motion (p90 36.2), and 50% enter more than five frames early. Thus it is
an approach-region condition, not a contact detector. The phase transition is
instead defined by observable object progress: attraction becomes sticky zero
after the first directed 2 mm motion, while object flow remains active.

## Decision

Do not swap volume G into the old contact-anchor checkpoint. The old policy was
never trained on this field geometry. Relabel source trajectories and train a
matched executor from scratch.

The first gate uses proprioception plus G, no scene state, no language, and no
task one-hot. It trains a no-G proprio baseline and a minimal-G token policy on
the same 96 source demonstrations. Evaluation in scene D first measures the
four seen combinations (scene-transfer health), then the two held-out
color-direction combinations. Held-out G normal/zero/flow-flip each use 50
standard initial states per task. The primary run uses only scale 1 and no G
dropout; scale augmentation is deferred until the representation itself passes.

This 96-demo gate is a representation and policy-integration test, not a final
scaling result. Scaling remains blocked until source scene-transfer health is
nontrivial, normal G beats no-G on held-out recombinations, and zero/flip
controls establish causal dependence across multiple seeds.

## Three-seed offline policy audit

The first matched run uses 96 source demonstrations, 2,170 stride-2 H16
windows, proprioception only, no language/task/scene identifier, and scale-1 G
without dropout. The no-G MLP, unrestricted token-G transformer, and
factorized token-G policy use the same split and action targets. The
factorized policy lets G condition only xyz translation; rotation and gripper
are predicted from proprioception alone because the current six-dimensional G
contains no angular or gripper intent.

Across seeds 0/1/2, the factorized policies reach their best validation
first-action MSE at epochs 58/36/44 (`0.03108/0.06122/0.03692`). The matched
no-G checkpoints are selected at epochs 31/27/23. Training loss continues
falling while validation error rises afterward, so 100 epochs are sufficient
for this gate and the best-validation checkpoint must be used.

On the two fully held-out color-direction combinations (18 demonstrations,
387 windows), factorized G improves total first-action MSE from no-G `0.04857`
to `0.04418`. Its predicted flow projection is `0.11668`, close to the expert
`0.13030`, compared with no-G `0.04510`. Zero G lowers it to `0.01999` and
sign-flipped G reverses it to `-0.09927`. Normal-to-zero action response is
`0.22811` L2 and occurs entirely in xyz: rotation L2 and gripper absolute
response are exactly zero by construction and measurement.

After directed object motion begins, the held-out xyz MSE improves from no-G
`0.03672` to `0.02637`. Predicted flow projection rises from `0.08152` to
`0.24137` against expert `0.27778`; zero and sign-flip produce `0.00581` and
`-0.22322`. This is direct evidence that a learned policy can parse and
recombine the compact flow interface on unseen color-direction pairs without
a hand-coded G-to-action projection.

Before entering the interaction volume, G also raises attraction-direction
projection from `0.04771` to `0.09697` against expert `0.11964`. However,
held-out xyz MSE changes from `0.02973` to `0.03349`: flow-parallel error is
essentially unchanged (`0.01495` to `0.01475`), while flow-orthogonal error
increases (`0.01477` to `0.01874`). Zero-attraction and zero-flow diagnostics
give xyz MSE `0.03416` and `0.04108`, so neither token can simply be removed.
Task-axis decomposition is also nonuniform: blue-left adds mostly z error,
whereas red-right retains x-amplitude error. Thus the remaining approach issue
is not weak global G magnitude; a global scale change is unlikely to fix both
held-out tasks. Online traces must instead measure interaction-volume entry,
object-motion onset, and axis-specific drift.

The unrestricted token policy previously achieved similar xyz behavior but
failed at the whole-action level on held-out data: gripper MSE was `0.24675`,
versus `0.09618` for factorized G and `0.09966` for no-G. Factorization removes
this cross-head shortcut while preserving causal translational control. The
unrestricted model's held-out normal-to-zero response is not merely correlated
with bad gripper fit: changing G alters xyz by `0.17087` L2, rotation by
`0.10521` L2, and gripper by `0.13367` absolute value. The factorized responses
are `0.22811`, exactly zero, and exactly zero. This is an interface-scope prior,
not an oracle contact rule: no anchor-to-action projection, stage one-hot, or
simulator contact signal is supplied.

The three-seed replication passes the offline representation/integration gate:

- Held-out total MSE is no-G/factorized-G
  `0.04857/0.04418`, `0.04830/0.04282`, and `0.04873/0.04086`.
- Held-out xyz MSE is `0.03265/0.03048`, `0.03234/0.02736`, and
  `0.03215/0.02532`. All three seeds improve; the mean reduction is 14.4%.
- Post-motion xyz MSE is `0.03672/0.02637`, `0.03673/0.02364`, and
  `0.03718/0.02007`. All three improve; the mean reduction is 36.7%.
- Post-motion normal/zero/sign-flip flow projection is respectively
  `0.24137/0.00581/-0.22322`, `0.18696/-0.00942/-0.20063`, and
  `0.20302/-0.00406/-0.21784`. The expert value is `0.27778`.
- Held-out normal-to-zero action response is `0.22811/0.16698/0.19204` L2,
  entirely in xyz in every seed. Rotation and gripper response remain exactly
  zero.

This still does not establish task success. The next mandatory check is
matched scene-D online rollouts for source and held-out tasks under
normal/zero/sign-flipped G. Scaling and visual pretraining remain blocked until
that closed-loop gate passes across seeds.

## Seed-0 scene-D closed-loop gate

The first closed-loop seed does not pass M3A. The matched no-G policy is
`0/100` on the four source combinations and `0/100` on the two held-out
combinations. The unrestricted token-G policy is `30/100` on source tasks and
`30/100` on held-out tasks, but every success is in the positive-x direction:

| Split/task | Normal G | Zero G | Sign-flipped G |
| --- | ---: | ---: | ---: |
| source blue-right | 14/25 | - | - |
| source pink-right | 16/25 | - | - |
| source pink-left | 0/25 | - | - |
| source red-left | 0/25 | - | - |
| held-out red-right | 30/50 | 0/50 | 0/50 |
| held-out blue-left | 0/50 | 0/50 | 0/50 |

Thus red-right provides real paired causal evidence, but the result cannot be
reported as a bidirectional reusable field. It is carried by one direction and
fails the predeclared requirement that every held-out cell be nonzero.

The factorized token policy fails more strongly. Normal G is `0/100` on source
and `0/100` on held-out tasks. Its zero-G control happens to reach `4/50` on
blue-left and zero elsewhere; sign flip is zero everywhere. Although
factorization improved demonstration MSE and removed G-to-gripper leakage, it
removes useful direction-dependent coordination between translation, wrist
orientation, and gripper state. A compact translational G does not imply that
G should be prevented from conditioning non-xyz action heads. Do not continue
the factorized executor.

Rollout traces localize the unrestricted model's failure to approach/flow
composition. In successful blue-right traces, mean action projection on object
flow is approximately `0.002` before object motion and `0.344--0.348` after the
2 mm motion switch. Failed blue-right traces instead project about `0.123`
along flow before contact, pass within roughly 4 cm of the block center, never
move the object, and then drift away. Failed blue-left traces similarly project
`0.084--0.093` along the requested flow while attraction is still nonzero;
they only reach 6.1--6.4 cm from the block center and then leave. The field is
not globally too weak: the policy executes substantial desired-axis motion at
the wrong time.

The next gate keeps the same six-dimensional G and unrestricted full-action
decoder, but tests objective alignment before adding a new architecture. The
current H16 model replans after one action while all 16 target actions receive
equal loss. Windows near interaction onset can therefore make future flow
motion dominate the action chunk even though only the first output is executed.
Run matched `H1/replan1` and `H16/first-action-weight16/replan1` token policies
with normal/zero/sign-flip controls. If either restores both directions, the
representation remains viable and the failure is an action-objective mismatch.
If both remain one-sided, replace raw token composition with a learned
approach/interaction gate; do not add a stage token to G or a hard action
projection.

## Objective and target-position identifiability follow-up

Objective alignment changes the source behavior but does not solve
recombination. The `H1/replan1` token policy reaches `23/100` source successes:
pink-left `12/25`, pink-right `4/25`, red-left `7/25`, and blue-right `0/25`.
The `H16/first-action-weight16/replan1` policy reaches `14/100`: pink-left
`4/25`, pink-right `9/25`, red-left `1/25`, and blue-right `0/25`. Both are
`0/100` on normal-G held-out recombinations. H1 removes the original strictly
rightward success collapse, so the first-action objective matters, but it is
not the missing reusable-control mechanism. Increasing only the first-action
weight is worse than training the action actually executed.

A full-rank interaction-frame decoder aligned with object flow also fails the
source scene-D gate at `0/100`. Its held-out offline post-motion xyz MSE is
better than the token decoder, but before-volume xyz MSE is worse, and
zero-attraction remains indistinguishable from normal attraction. Expressing
actions in a flow-aligned frame is therefore insufficient when the training
objective does not require the network to localize the target.

Two state-scope diagnostics isolate that failure. Removing only the first
three absolute TCP-position values is a weak intervention because the retained
arm joints still determine TCP position. On held-out windows, its normal and
zero-attraction xyz MSE is nevertheless already indistinguishable
(`0.027089/0.027095`), while zero-flow rises to `0.040896`. The stricter
`eef_orientation_gripper` scope removes both absolute TCP position and all arm
joints; it retains only TCP orientation, gripper state, and G. Even when
attraction is the only target-position input, normal and zero-attraction remain
nearly identical on held-out windows (`0.029301/0.029351`). Zero-flow rises to
`0.038090`, and no-G is `0.040444`.

The current policy can parse desired flow, but the per-step BC objective does
not identify attraction. Demonstration actions are sufficiently stereotyped
given flow and local phase cues that low offline error does not require knowing
which target position generated the action. This is conditional collapse, not
evidence that attraction should be removed from G. Do not spend online rollout
budget on either state-scope diagnostic: neither causally uses attraction.
First test whether broader A/B/C position and trajectory coverage makes
attraction necessary. If it does not, the next integration must create an
explicitly target-relative learning problem while keeping the six-dimensional
interface coarse; another global G scale, action-chunk weight, or flow-only
coordinate frame is not justified.

## Reproduction

- Range extractor:
  `framework/scripts/data/extract_calvin_remote_calibration_subset.py`
- Source-only calibration:
  `framework/scripts/data/calibrate_calvin_block_surface_g.py`
- Matched relabel/cache builder:
  `framework/scripts/data/build_calvin_interaction_volume_cache.py`
- Offline phase and causal audit:
  `framework/scripts/data/audit_interaction_volume_policy_offline.py`
- Training/evaluation gate:
  `framework/scripts/run_stage3_interaction_volume_matched_gate.sh`
