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
