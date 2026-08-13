# AutoResearch Progress Journal

**Active goal:** CALVIN reusable interaction G — establish causal dependence,
then migrate the compact `G=[A,F]` interface into a credible conditional
flow/diffusion policy.

**Legacy Isaac track:** S1 random-cube B1/B7 remains retained below as
preliminary evidence; it is not the active experiment queue.

---

## E3 Factorial Zero-Shot Composition Gate — 2026-07-16

- Built a leave-one-combination-out checkerboard around the unchanged compact
  `G=[A(3),F(3)]`. Twelve source tasks provide every tested object/mechanism
  and flow direction separately; `close_drawer`, `move_slider_left`,
  `push_red_block_right`, and `push_blue_block_left` are completely absent
  from policy training. Training uses 1,807 segments and 39,961 H16 windows.
- Audited the implementation after training: the upstream query-G cache is
  four 19-D tokens, but `make_minimal_articulation_g` extracts only A/F and
  the policy enforces a six-dimensional G input. Token type and other detailed
  query features do not enter the executor.
- Completed three independent 50-epoch seeds and 3,600 held-out rollouts.
  Zero-shot red-right is `139/300` with normal G versus `0/300` sign-flip and
  `3/300` matched no-G. Blue-left is `147/300` versus `2/300` and `1/300`.
  Every seed is positive. Paired exact McNemar p-values are below `1e-40` for
  normal G against both controls on both block cells.
- This is the first robust support for the primary claim: compact G can
  zero-shot recombine object identity and direction inside a shared
  block-pushing interaction grammar. It is stronger than same-task scene OOD
  because each target object-direction pair is absent from training.
- Cross-mechanism transfer remains unresolved. Slider-left is `25/300` normal,
  `15/300` flip, and `0/300` no-G, but seed 2 reverses the normal/flip ordering.
  Close-drawer is `0/300` normal, `4/300` flip, and `0/300` no-G. Direction
  coverage alone is therefore insufficient to identify mechanism-specific
  contact control.
- Offline held-out actions explain the old flip-sign observation. For close,
  flip G has lower first-action MSE than normal G in all seeds; the source
  executor conflates desired object flow with instantaneous EEF motion and has
  not learned the drawer-specific object-flow-to-contact-action mapping.
- Source interaction health is `333/525 = 63.4%` with G versus
  `42/525 = 8.0%` matched no-G. The earlier total including light/LED tasks is
  not a valid health estimate because language was deliberately zeroed for
  tasks whose G is empty. Normal-language evaluation restores the four
  light/LED tasks to `217/300 = 72.3%` and gives three-seed total source health
  `550/825 = 66.7%`. The executor is functional but still below a credible
  final-policy quality gate.
- Freeze E3 as the compact-G causal checkpoint. Keep G compact and move next
  to a history-conditioned mainstream diffusion/flow executor, balanced
  factor sampling, stronger language/goal baselines, and a wider checkerboard
  spanning multiple objects, layouts, and articulation mechanisms.
- Full record:
  [knowledge_base/03_experiment_evidence/phase1_calvin_factorial_checkerboard_e3_2026-07-16.md](knowledge_base/03_experiment_evidence/phase1_calvin_factorial_checkerboard_e3_2026-07-16.md)

---

## G-token policy and trajectory-distribution gate — 2026-07-16

- Audited all 1,057 demonstrations from seven non-block tasks using complete,
  32-step-resampled EEF xyz trajectories. The data does not collapse to one
  duplicated trajectory: slider and LED tasks show substantial offsets and
  path variation. `open_drawer` is the strongest canonical trend
  (`direction R=0.946`, angle p90 `26.8 deg`, aligned RMSE/path `0.098`).
- Task identity remains partially recoverable without language or G:
  initial robot+scene state reaches `49.6%` balanced seven-way accuracy and a
  start-aligned EEF trajectory reaches `65.8%`, versus `14.3%` chance. The
  corrected risk is insufficient state x goal factorization, not fixed-path
  duplication.
- A direct inverse-goal audit confirms this: compact state predicts slider
  left/right at `98.4%`, and predicts lightbulb and LED on/off at `100%`.
  Slider still has `25.0--33.1%` near-state cross-goal overlap with strongly
  opposite matched actions (cosine p50 about `-0.92`), but lightbulb has none
  and LED has almost none. Drawer has no inverse close supervision. Most
  non-block tasks therefore do not force the policy to consult G.
- The three block left/right pairs are the best existing G-identifiability
  source: near-state opposite-goal overlap is `56.6--84.7%`, with negative
  matched-action cosine medians in every direction. State-only goal accuracy
  remains `75.8--86.2%`, so these hard pairs should be mined and co-sampled
  rather than left rare inside ordinary i.i.d. batches.
- Implemented rule-free `chunk_bc_minimal_g_tokens` and
  `chunk_fm_minimal_g_tokens`. Both encode raw `A` and `F` as separate learned
  tokens and use action-token cross-attention; neither projects actions onto
  G, computes readiness, or adds a handcrafted field prior.
- Forward/backward, checkpoint save, checkpoint reload, and single-step
  inference smoke tests pass for both models. The matched seed-0 E0 run uses
  the same 14 tasks, split, H16 target, and first-action weight 16 as D4. Both
  models completed 50 epochs. BC validation chunk/first MSE is
  `0.04336/0.02705`; FM is `0.06649/0.03756`. These are offline fit metrics,
  not task evidence.
- Full data audit:
  [knowledge_base/03_experiment_evidence/phase1_calvin_eef_trajectory_distribution_2026-07-16.md](knowledge_base/03_experiment_evidence/phase1_calvin_eef_trajectory_distribution_2026-07-16.md)
- E0/E1 task result: rule-free E0 BC is `0/100` on held-out close. Training
  with a matched inverse-pair difference loss gives E1 BC `2/100`; the same
  E1 checkpoint gives zero/sign-flip/wrong-open `0/100`. E0/E1 FM execute-16
  and execute-1 are all `0/100`. The FM-H1 runs use four validated disjoint
  shards whose aggregate sequence list exactly matches the BC 100-case list.
- Same-state G replacement shows that E0 already parses G on seen support:
  first-action difference cosine is `0.81` on slider and `0.62--0.83` on the
  three block pairs. Zero and flow-flipped G roughly double E0-BC first-action
  MSE. E1 improves slider BC/FM sensitivity to about `0.88` but is not uniform
  across blocks. The bottleneck is cross-family contact-control transfer, not
  wholesale G neglect.
- State-only CALVIN evaluation now bypasses unused camera rendering while
  retaining dynamics and task-oracle metrics. This removes the repeated EGL
  aborts and completes 100 rollouts in minutes.
- Seen-task health resolves the BC/FM baseline concern. Across 14 train tasks
  x 25 rollouts, E0 BC is `168/350 = 48.0%`, while E0 FM-H16 is
  `283/350 = 80.9%`. E0 FM is `25/25` on open drawer and both slider
  directions, yet remains `0/100` on close under H1 and H16. Conditional FM is
  healthy on seen support; the failure is held-out state/contact composition.
- E1 pair loss raises BC seen health to `197/350 = 56.3%` but lowers FM to
  `247/350 = 70.6%`, especially on block pushes. Do not tune this weight
  upward: pairwise G sensitivity and task control quality are not equivalent.
- Tested whether open-drawer trajectories can self-supervise the missing
  close-contact branch. Starting from exact post-open states, negating the
  recorded relative xyz/Euler actions succeeds on `25/25` official close
  trials (median close displacement `0.165 m`, median TCP RMSE `7.4 mm`). A
  preceding-pose feedback replay is also `25/25`. The reversed contact path is
  physically valid and direct reversed labels are reliable; no handcrafted
  push controller is needed.
- This is a diagnostic/self-supervised augmentation, not clean zero-shot
  evidence, because it injects same-drawer close-action support. E2 keeps the
  E0 architecture, 14 source tasks, split, and schedule fixed and adds only
  3,800 reverse-open windows from 152 source segments. E2-BC reaches `74/100`
  close versus E0's `0/100`; F sign-flip gives `1/100`, with paired outcomes
  73 normal-only, 0 flip-only, 1 both, and 26 neither. Correct G is causal.
- E2-FM remains `0/100` at H16 and `1/100` at H1 despite fitting the synthetic
  close validation windows: first-action MSE is `0.0199`, median xyz cosine to
  expert is `0.963`, and predicted/expert mean projection along F is
  `0.166/0.166`. The FM failure is closed-loop temporal/sampling behavior, not
  missing one-step direction learning. Sampling audits reject simple
  inference tuning: H2/H4/H8 are `4/0/0`, a continuous noise stream is
  `4/100`, and H1 temporal ensemble is `0/100`.
- E2 BC-vs-FM contact diagnosis on the same 25 close states is `17/25` versus
  `0/25`. FM median minimum distance to the drawer part/axis line is
  `13.45/5.82 cm`, versus BC `6.84/2.09 cm`, and first near-axis step is
  `81` versus `36`. FM issues a median 43 desired-axis commands without joint
  progress versus BC's 5: it actuates before completing lateral/height contact
  alignment. Since BC learns correct ordering from the same compact G, do not
  add stage details to G; retire lightweight FM tuning and move G into a
  mainstream history-conditioned temporal policy.
- E2 seen health is BC `216/350 = 61.7%` and FM-H16
  `255/350 = 72.9%`, versus E0 `168/350` and `283/350`. Reverse augmentation
  helps BC overall but shifts FM away from block pushes, so future training
  must task-balance source and inverse windows.
- Full E0/E1 record:
  [knowledge_base/03_experiment_evidence/phase1_calvin_g_token_inverse_pair_2026-07-16.md](knowledge_base/03_experiment_evidence/phase1_calvin_g_token_inverse_pair_2026-07-16.md)

---

## Server Storage Cleanup — 2026-07-15

- Removed reproducible historical checkpoints, rendered media, raw step
  traces, derived episode caches, archives, and oversized logs after writing a
  path-level deletion manifest.
- AutoResearch shrank from `6.2 GiB` to `109 MiB`; source, configs,
  summary/metrics JSON, rollout aggregates, CSV, knowledge-base records, and
  compact log tails remain available.
- Removed the obsolete `852 MiB` CALVIN debug checkpoint and `15.4 GiB` of
  RoboTwin frame cache while retaining the canonical CALVIN dataset, RoboTwin
  assets, and compact trajectory metadata.
- Purged `17.1 GB` of pip download cache. Root availability increased from
  `85 GiB` to `116 GiB`.
- Full record:
  [knowledge_base/03_experiment_evidence/server_storage_cleanup_2026-07-15.md](knowledge_base/03_experiment_evidence/server_storage_cleanup_2026-07-15.md)

---

## Active Direction-Coverage Matrix — 2026-07-15

- Implemented a unified translational minimal-G scope for articulation tasks,
  six named block-left/right pushes, and `push_into_drawer`; the representation
  remains `G=[A(3),F(3)]`.
- Coverage audit confirms the old scope gave primitive tasks zero G and rank-2
  direction support. The expanded scope gives all named block pushes nonzero G
  and rank-3 support.
- Whole-task endpoint audit shows rank 3 is still insufficient: D1 contains
  -drawer-y and +/-x but not the held-out +drawer-y sign. `place_in_slider`
  supplies +y object flow in 100% of 151 segments, so D3 adds only this task
  while keeping close completely held out.
- Rejected nearest-drawer active-object grounding for `push_into_drawer`
  (`46.7%` endpoint agreement). Nearest-EEF grounding gives `74.0%` endpoint,
  `92.0%` all-frame, and `99.0%` middle-phase agreement without future-state
  access; the task accepts any block.
- D0/D1/D2/D3/D4 completed five-seed, 50-epoch training and all normal-G
  `close_drawer` evaluations (`100` rollouts per seed). D0 sparse FiLM gives
  `[0,15,0,0,0]`; D1 expanded-direction FiLM gives `[2,0,0,0,0]`; D3 adds
  cross-task `+y` coverage via `place_in_slider` but gives `[0,0,0,0,0]`.
- D2 uses the same expanded G/data as D1 but couples action translation to a
  positive field component plus bounded learned orthogonal residual. It gives
  `[95,83,51,100,97]` (`85.2%` mean, `51%` worst seed). This establishes that
  compact `G=[A,F]` is sufficient as a reusable control prior; ordinary FiLM
  does not learn the geometric coupling merely from broader direction data.
- D2 five-seed sign-flip is complete at `[0,0,0,0,3]` versus normal
  `[95,83,51,100,97]`. Across the same 500 initial states, there are 423
  normal-only, 0 flip-only, 3 both-success, and 74 both-failure transitions
  (auxiliary exact McNemar `p=9.23e-128`). Correct flow sign is causal across
  every seed.
- D2 wrong-open-G is complete at `[9,6,14,15,40]` (`16.8%`) versus normal
  `85.2%`. Matched outcomes contain 355 normal-only, 13 wrong-only, 71 both,
  and 61 neither. A learned residual can occasionally overcome the incorrect
  prior, but correct task-selected G is the dominant success contributor in
  every seed. Zero-G is also complete at `[0,0,0,0,0]`, so D2 requires a
  nonzero, correctly signed, task-selected G rather than merely its policy
  architecture. Five-seed seen-task health is
  `[160,150,183,144,166]/250 = 64.2%`: articulation is `354/375 = 94.4%`,
  while block/push tasks are `449/875 = 51.3%`. The current primitive
  fixed-offset contact builder therefore fails the 90% health gate even though
  held-out drawer transfer is strong. Named pushes use
  `block_center - 0.05 * flow_dir`, but this offset/readiness scale is not
  calibrated to the true block surface, gripper geometry, or demonstrated
  contact timing.
- All D0--D4 common 10-task seen evaluations are complete. Aggregate success
  is D0 `310/1250 = 24.8%`, D1 `497/1250 = 39.8%`, D2
  `803/1250 = 64.2%`, D3 `427/1250 = 34.2%`, and D4
  `639/1250 = 51.1%`. D1 improves primitive fitting over D0 but not held-out
  close, while D2 improves both seen health and held-out transfer. D3's
  cross-task +y data does not help close and reduces seen health relative to
  D1. Interpret D3 as a direction-sufficiency test, not a clean contact-mode
  positive control: `place_in_slider` is grasp/place, uses a center anchor, and
  differs from sustained articulated pushing.
- Completeness audit found that D3's common-task seen directory did not include
  its extra training task `place_in_slider`. A supplemental evaluation was
  attempted, but CALVIN has no first-task sequence for a task that presupposes
  an already-grasped block; the evaluator raises `No first-task eval sequences
  matched filter`. Standalone place health is invalid under this protocol and
  would require a predecessor grasp/lift chain. The common 10-task comparison
  is retained as the final matched health table; close results are unaffected.
- D1 and D2 have nearly identical five-seed mean first-action validation MSE
  (`0.026389` vs `0.026304`) on identical data/G, while held-out close is
  `0.4%` vs `85.2%`. D3 has the lowest MSE (`0.024749`) but zero close success.
  The matrix isolates policy/G integration as the main cause of transfer; an
  offline regression-loss explanation is inconsistent with the results.
- D4 was added after a matrix confound audit. It matches D2's expanded data/G,
  H16 chunks, first-action weight 16, and replan-1 inference but uses ordinary
  FiLM with no bounded field projection. It isolates the D1--D2 executor change
  from the simultaneous chunk/objective change.
- D4 normal close is complete at `[2,0,5,4,0]/100 = 2.2%` versus D2
  `[95,83,51,100,97]/100 = 85.2%`. D4's mean first-action val MSE is lower
  (`0.024973` vs `0.026304`). Because data, G, H16, first-action weight, and
  replan horizon are matched, the `83.0 pp` rollout gap isolates bounded
  field-coupled action parameterization as the decisive D2 factor. D4's common
  seen-task seeds are `[130,130,135,115,129]/250`, or
  `639/1250 = 51.1%`: articulation is `288/375 = 76.8%` and free-object push is
  `351/875 = 40.1%`. D2 is also better on both seen families, so its close gain
  is not obtained by sacrificing training-task behavior.
- D4 causal controls are complete: normal/zero/sign-flip/wrong-open are
  `11/0/18/19` out of 500. Paired normal-vs-flip and normal-vs-wrong tables have
  11 normal-only, 18/19 corruption-only, and zero shared successes. Exact
  H16/first-w16 matching therefore preserves FiLM's unstable mode-code
  semantics; it does not reproduce D2's physical sign binding.
- D0 and D1 five-seed causal controls are complete. D0 gives normal/zero/flip/
  wrong-open `15/0/12/2` out of 500; normal and flip successes have zero paired
  overlap. D1 gives `2/0/36/3`; its paired normal-vs-flip table has 35
  flip-only, one normal-only, one both, and 463 neither. Expanded direction
  coverage therefore strengthens a seen-code interpolation shortcut under
  ordinary FiLM instead of teaching the physical flow sign. D2 reverses this
  pattern with 423 normal-only and zero flip-only outcomes.
- A data-only contact-transition audit confirms the mechanism. Before first
  object motion, demonstrated EEF projection along F has task/color medians
  from `-0.007 m` to `-0.046 m`, inconsistent with one fixed `-0.05 m`
  anchor. The controller crosses readiness `0.5` a median `15-23` frames
  before object motion and reaches readiness `0.88-0.96` at first motion.
  Contact is inferred from object motion, not a simulator contact sensor.
- Full protocol:
  [knowledge_base/03_experiment_evidence/phase1_calvin_minimal_g_direction_coverage_matrix_2026-07-15.md](knowledge_base/03_experiment_evidence/phase1_calvin_minimal_g_direction_coverage_matrix_2026-07-15.md)

---

## Active CALVIN Gate — 2026-07-14

- Completed 30 matched `close_drawer` evaluations for H1/H16 and
  language/concat/FiLM over seeds 0-4, 100 rollouts each.
- Only H1 FiLM is nonzero: `[0,0,0,19,14]/100`, mean `6.6%`. H1 concat,
  language, and all H16 conditions are `0/500`.
- Validation MSE does not predict rollout success. Full execute-16 removes all
  H1 FiLM successes, so the active policy route is short-horizon replanning.
- Interpretation boundary: compact G has policy-usable signal, but ordinary
  learned integration is seed-unstable; the previous `86.2%` bounded result
  also relied materially on structural coupling.
- H1 FiLM causal controls are complete on successful seeds 3/4. Normal gives
  `[19,14]`, zero/wrong-task/A-only/F-only/anchor-flip all give `[0,0]`, but
  flow-flip gives `[34,3]`. G is causal, yet raw FiLM does not preserve a
  stable physical sign semantics across seeds.
- Seen-support analysis explains the anomaly: held-out close normal G has
  median standardized nearest-seen distance `2.914`, while flow-flip is
  closer to the seen open branch at `1.776`; wrong-open is `0.382` but is
  physically incorrect. Ordinary FiLM is interpolating a seen-like code
  instead of extrapolating the correct inverse direction.
- The first semantic/equivariant adapter gate is complete at
  `[0,0,17,0,5]/100` close success, mean `4.4%`; it does not improve FiLM and
  fails the nonzero worst-seed gate.
- Axis diagnostics identify residual leakage rather than missing G magnitude:
  failed seed0 has pre-contact G-basis/learned/total desired-axis components
  `-0.136/+0.129/-0.011`, while successful seed2 has
  `-0.159/-0.014/-0.172`. The residual cancels the correct approach field in
  failed seeds.
- Training-scope audit also finds that the added block/light/push tasks receive
  zero minimal articulation G, so they do not teach the G executor reverse
  semantics; nonzero G supervision remains open drawer plus bidirectional
  slider only.
- The generic soft residual-leakage ablation gives `[8,4,9,0,4]/100`, mean
  `5.0%`. It spreads a few successes across more seeds than the unrestricted
  semantic adapter, but retains a zero worst seed and remains below FiLM's
  `6.6%`; do not tune this loss weight further.
- The full-rank G-local-frame decoder is also complete and gives `0/500` close
  success despite seen validation MSE `0.029-0.031`. It fails before approach:
  sampled policies never improve the initial `0.327 m` part distance and end
  `1.10-1.24 m` away with negative action/field cosine.
- Consolidated diagnosis: close creates the unseen state composition
  `(drawer open, EEF far)`. Open demos contain `(drawer closed, EEF far)` early
  and `(drawer open, EEF near)` late. Conditioning, signed vector bases, soft
  leakage loss, and decoder-only local coordinates do not make the MLP compose
  these factors reliably.
- **Next method boundary:** G must enter as a closed-loop low-frequency policy
  prior (far-field approach from A, near-field object flow from F), while the
  learned policy models execution residuals. For a mainstream flow policy,
  implement this as a base action vector field plus learned conditional-flow
  residual, with zero/corrupted-G controls. Do not enlarge G or tune scales.
- **Decision gate:** no mainstream flow migration unless normal G becomes
  cross-seed stable and later beats flow-flip/zero/wrong-task controls.
- Full record:
  [knowledge_base/03_experiment_evidence/phase1_calvin_learned_g_executor_horizon_v1.md](knowledge_base/03_experiment_evidence/phase1_calvin_learned_g_executor_horizon_v1.md)

---

## Phase 1 CALVIN — 2026-07-03

- CALVIN `task_D_D` full dataset is downloaded/extracted on the server: 512077 training episodes, 99022 validation episodes.
- Reusable-G V1 completed four held-out recipes: object transfer, primitive transfer, slider→drawer, drawer→slider.
- Current tiny raw-coordinate G predictor loses to zero baseline on every held-out target split; strongest failure is articulation transfer.
- G-v2 canonical interaction-frame result: `canonical_nearest` feature + `canonical_nearest_with_scene` target beats zero/source-mean on object transfer (`pred/zero=0.87`), improves primitive/articulation but does not beat trivial baselines there.
- Next split: object-centric G can move toward policy smoke; primitive/articulation need task-conditioned or mechanism-specific G heads.
- Object-transfer action-proxy smoke: oracle-G strongly helps (`-29.75%` val MSE vs no-G), but predicted-G hurts (`+6.62%` vs no-G). Current predicted reusable-G should not be used as a direct policy input yet.
- CALVIN official task-metric wrapper is now available and smoke-tested. Zero-policy and checkpoint-policy smoke both produced official `avg_seq_len`, `chain_sr`, and per-task success JSON; this is the primary evaluation layer for future no-G/oracle-G/predicted-G policy claims.
- First Task D_D rollout comparison with lightweight state-BC: no-G beats direct oracle-G on task metrics despite oracle-G having slightly lower offline action MSE (`no_g avg_seq_len=0.20, SR@1=10%, SR@2=10%; oracle_g all 0`). This argues against raw dense scene-delta G concatenation as the policy interface.
- Articulation-family G follow-up: raw/concat G still hurts, but narrow rule-oracle articulation G used as FiLM modulation gives the first positive task-metric signal on balanced first-task eval (`no_g SR@1=0.55`, `articulation_g_film SR@1=0.675`; `open_drawer` improves `1/10 -> 6/10`, `move_slider_left` improves `1/10 -> 3/10`, while `move_slider_right` drops `10/10 -> 8/10`). This supports treating G as task-family modulation rather than a raw future-state vector.
- Full local record: [docs/PHASE1_CALVIN_REUSABLE_G_V1_FULL_D.md](docs/PHASE1_CALVIN_REUSABLE_G_V1_FULL_D.md)
- Policy-proxy record: [docs/PHASE1_CALVIN_OBJECT_TRANSFER_POLICY_PROXY_V1.md](docs/PHASE1_CALVIN_OBJECT_TRANSFER_POLICY_PROXY_V1.md)
- Task-metric protocol: [docs/PHASE1_CALVIN_TASK_METRIC_PROTOCOL.md](docs/PHASE1_CALVIN_TASK_METRIC_PROTOCOL.md)
- State-BC task metric record: [docs/PHASE1_CALVIN_STATE_BC_TASK_METRIC_V1.md](docs/PHASE1_CALVIN_STATE_BC_TASK_METRIC_V1.md)
- Articulation-G FiLM record: [docs/PHASE1_CALVIN_ARTICULATION_G_FILM_V1.md](docs/PHASE1_CALVIN_ARTICULATION_G_FILM_V1.md)
- Representation survey: [docs/TASK_RELEVANT_REPRESENTATION_SURVEY_V1.md](docs/TASK_RELEVANT_REPRESENTATION_SURVEY_V1.md). Main decision: G should be typed task-family interaction geometry (`entity + affordance + relation + progress + flow + contact`) and should enter policies through FiLM/gating/attention routing, not raw concat.
- Oracle-G v2 articulation record: [docs/PHASE1_CALVIN_ORACLE_G_V2_ARTICULATION.md](docs/PHASE1_CALVIN_ORACLE_G_V2_ARTICULATION.md). Demo-estimated geometry is stable, and n25 balanced eval shows `v2_concat` becomes useful (`0.65` vs no-G `0.56`), with `v1_film` the best pre-split mode (`0.66`). Do not globally reject concat; diagnose v2 `move_slider_left` collapse before learned-G training.
- Left/right v2 diagnostic: slider residual geometry is mostly symmetric (`+0.782` left vs `-0.778` right normalized residual; similar affordance norms and stages), but the signed residual-motion vector interacts with an asymmetric action distribution. Next fix should test a less prescriptive split representation such as `[axis_unit, signed_residual_scalar, affordance, stage]`.
- Split-G follow-up: `v2_split_concat` is now the best seed-0 balanced mode (`0.70`), recovering `move_slider_left` to no-G level (`6/25`) while preserving `move_slider_right` (`25/25`) and `open_drawer` (`14/25`). `v2_split_film` improves to `0.61` but still has left `0/25`. Exported 12 qualitative videos for no-G, v2 concat, and v2 FiLM across the four articulation tasks.
- No-G convergence audit: retrained no-G for 50 epochs. MSE reaches a near plateau in the last 10 epochs (`0.02349-0.02390`), but `move_slider_left` only improves `6/25 -> 7/25`; `open_drawer` improves `2/25 -> 9/25`, and `move_slider_right` drops `23/25 -> 18/25`. This argues that left failure is not just no-G undertraining. Saved successful `move_slider_left` videos for no-G, old v2 concat, and v2 split concat under `artifacts/server_sync/oracle_g_v2_articulation_2026-07-05/2026-07-05_move_slider_left_success_rollouts`.
- V2 50-epoch audit: retrained `v2_concat`, `v2_film`, `v2_split_concat`, and `v2_split_film` for 50 epochs. All MSEs drop to about `0.023`, but task success drops vs e8 (`v2_concat 0.65 -> 0.58`, `v2_film 0.57 -> 0.33`, `v2_split_concat 0.70 -> 0.61`, `v2_split_film 0.61 -> 0.47`). This removes the training-length bias concern but strengthens the warning that offline BC MSE is not reliable for checkpoint/model selection.
- Checkpoint sweep completed for `no_g`, `v2_concat`, and `v2_split_concat` at epochs 4/8/16/32/50 with the same balanced n25 protocol. Best no-G is e32 (`0.61`), while best oracle-G reaches `0.70` (`v2_concat@4`, `v2_split_concat@8`). The improvement is real in aggregate but task-skewed: G strongly helps `move_slider_right` and `open_drawer`, while `move_slider_left` remains weak and sometimes worse than no-G. Conclusion: G is useful but current articulation-G is not yet a robust cross-direction representation; model selection must use rollout checkpoints, not final MSE.
- Left dynamics diagnostic: replayed best checkpoints on the same first 25 left/right validation states with per-step slider progress, fitted-point distance, contact-stage proxy, and action-axis logs. For G policies, failed left rollouts usually reach the fitted affordance (`failed min_dist p50` about `1-1.5 cm`) and use the correct action direction, but produce almost zero slider progress. Right succeeds (`25/25`) under the same G interfaces. Clear conclusion: left is a contact-mode failure, not an approach failure or residual-sign failure. Current G lacks side-of-handle / push-vs-pull / force-maintenance information.
- Contact-pose G-v3 implemented and validated. Expression: `[(p_part(q)+median_demo(eef-p_part(q))-eef), desired_motion_dir, residual_mag, stage]`, where the contact offset is estimated only from demo frames that actually move the target joint in the desired direction. Geometry statistics show slider left/right effective offsets are both tiny (`left [+2.6,-0.2,+1.2] mm`, `right [+0.2,+0.4,-2.3] mm`), so left is not mainly a wrong target-point problem. Rollout sweep: best `v3_contact_concat@4` reaches `0.67` with left `13/25`, right `25/25`, close `25/25`, but open only `4/25`; `v3_contact_film` is worse. Conclusion: contact-pose helps left but is not a robust unified G; next expression should model contact mode/actuation dynamics rather than only a single effective contact point.
- Mechanism-twist G-v4 tested after critique that `effective_contact_delta` and `stage_onehot` are too low-level. Expression: `[signed_linear_twist(3), signed_angular_twist(3), residual_magnitude(1)]`, with no contact point and no hard stage. Result is negative: best `v4_mechanism_film@4` reaches only `0.57`, and best concat only `0.55`, below no-G best `0.61` and v2/v3. This bounds the representation space: contact-pose G is too execution-specific, but global mechanism twist is too abstract and loses object-local affordance. For large-video self-supervision, the main target should be task-conditioned object-local interaction/motion fields; 7D twist can be a diagnostic decoder, not the core Foundation G.
- Potential-field G-v5 implemented after revisiting attraction/repulsion as a more scalable G. Expression is 6 flattened object-local tokens, each `[anchor_delta(3), attractive_flow(3), repulsive_flow(3), attract_weight, repel_weight]`: five target-joint local anchors carry desired object flow, and one non-target articulation token supplies repulsion. Same n25/h160 checkpoint sweep shows a major positive result: `v5_potential_concat@4` reaches `0.85` with close `25/25`, left `10/25`, right `25/25`, open `25/25`, beating no-G (`0.61`), v2 (`0.70`), v3 (`0.67`), and v4 (`0.57`). Later checkpoints collapse despite lower MSE, and FiLM is much weaker (`best 0.64`). Next priority: seed repeat, causal ablations (zero/shuffle/repulsion-off/attraction-off), and qualitative video comparison for `v5_concat@4`.
- Render-overlay sanity check exposed that the v5 repulsion token is semantically dirty: for slider tasks it is placed on the drawer handle, and for drawer tasks on the slider, i.e. object exclusion rather than interaction geometry. Two cleaner variants were implemented and tested: `v5_attraction_only` removes the non-target repulsion, and `v6_interaction_field` makes repulsion target-local around the handle/part corridor. Both render correctly on CALVIN `rgb_static`. Task-metric sweep: best `v5_attraction_only_concat@4` reaches `0.79` (`25/25, 4/25, 25/25, 25/25`), and best `v6_interaction_field_concat@4` also reaches `0.79`; `v6@50` improves left to `14/25` but drops right/open, total `0.76`. Conclusion: target-local repulsion is visually/semantically cleaner but not yet a robust positive control signal; old v5's `0.85` may include a shortcut bias from non-target object exclusion. Repulsion should stay as a hypothesis, not a claimed solved component.

---

## S1 — random cube (legacy Isaac track)

### Pipeline
```bash
# 1. Collect demos
STAGE=s1_random_cube RANDOMIZE_CUBE=1 NUM_DEMOS=100 bash scripts/collect_lift_demos.sh

# 2. Step 1 — G_θ (mask in, GT subgoal label; auto subgoal for s1)
bash scripts/run_s1_step1.sh

# 3. Step 2 — DAgger
MAX_ITERS=300 bash scripts/run_s1_dagger_smoke.sh B1
MAX_ITERS=300 bash scripts/run_s1_dagger_smoke.sh B7
```

### Eval metrics
- **Primary:** `success@2cm` (existing `success_rate`)
- **Auxiliary:** `success@5cm` — cube lifted + `min_dist_to_goal ≤ 0.05`

---

## Day 0 — 2026-06-18 (session start)

### Done
- [x] Data isolation (`data/`, import/collect scripts)
- [x] Step 1: `MaskFieldCNN` trained — **val field_alignment 0.9845** (pass ≥0.75)
- [x] IsaacLab: `guidance_field_bridge.py` — append G as +2 visual channels (10ch total)
- [x] `run_s0_dagger_comparison.sh` — B1 vs B7 harness
- [x] v3 HDF5 copied to isolated `data/lift/s0_fixed/demos/`
- [x] **Eval diagnostics** — EE/cube trajectories + failure-mode taxonomy (`lift_episode_diagnostics.py`)
- [x] **B1 DAgger complete** — 100 iters, buffer 179 ep / 221k steps, teacher lift 100%
- [x] **B7 DAgger complete** — 100 iters; diagnostic eval + trajectory plots
- [x] **B7 diagnostic re-analysis** (20 ep, 10ch load fix)

### In progress
- [ ] B7 vs B1 success gap ≥ 15%（当前两者均为 0%）
- [ ] 调参：更长 DAgger / β decay / G 幅度或 phase gating

### B7 root-cause analysis (diagnostic eval, fixed 10ch load)

**Run:** `runs/s0_B7_20260622/eval_diagnostics/` (20 ep, `student_best.pt` @ iter 100)

| Metric | B1 | B7 |
|--------|-----|-----|
| eval success | 0% | 0% |
| grasp rate | 85% | **90%** |
| mean min EE–cube dist | 1.4 mm | 2.5 cm |
| mean post-grasp wander | 10.4 cm | **6.9 cm** |
| typical final_dist_goal (lifted ep) | 11–36 cm | **3.2–7.0 cm** |
| failure mode | 100% `lifted_not_at_goal` | 100% `lifted_not_at_goal` |

**B7 相对 B1 的变化：**
- G 场显著缩短 relocation 误差：多数 episode 最终距 goal **3–7 cm**（B1 为 11–36 cm）
- 2/20 episode 弱 lift（max_z≈0.055，同 B1 边界情况）
- 未出现 `grasp_then_dropped`；抬起后 cube 保持在 ~0.31–0.35 m
- **仍未 crossing 2 cm success 阈值** → 归类为 `lifted_not_at_goal`

**训练内 eval vs 完整诊断 eval 差异：**
- 训练 eval（iter 100）：grasp 50%, wander 6 cm（带 early_done，episode 可能提前截断）
- 完整诊断（1250 步）：grasp 90%, wander 6.9 cm → 以诊断为准做轨迹分析

**结论：** G_θ 方向正确（relocation 明显改善），但 100 iter smoke + 全局广播 G 仍不足以精确到达 goal；且 approach 精度略降（EE–cube 2.5 cm vs 1.4 mm）。

**Artifacts:**
- `runs/s0_B7_20260622/eval_diagnostics/eval_analysis.md`
- `runs/s0_B7_20260622/eval_diagnostics/trajectory_plot.png`

### Visualization

**Isaac GUI 实时观看（推荐）：**
```bash
# B1
bash scripts/play_student.sh B1

# B7（自动加载 G_θ + 10ch）
bash scripts/play_student.sh B7
```

**离线轨迹图（无需 Isaac，含 GOAL ★ + 2cm 成功圈 + 距离标注）：**
```bash
# 单个 run
python3 scripts/plot_eval_trajectories.py runs/s0_B1_restart/eval_diagnostics/eval_diagnostics.json
python3 scripts/plot_eval_trajectories.py runs/s0_B7_20260622/eval_diagnostics/eval_diagnostics.json

# B1 vs B7 并排对比（同 episode id）
python3 scripts/plot_eval_trajectories.py \
  runs/s0_B1_restart/eval_diagnostics/eval_diagnostics.json \
  --compare runs/s0_B7_20260622/eval_diagnostics/eval_diagnostics.json \
  --episodes 1 4 8 \
  --output runs/trajectory_compare_b1_b7.png
```

俯视图图例：**★红色=GOAL**，**绿色虚线圆=2cm 成功区**，**紫色虚线=cube 终点→goal**，数字为 3D 距离(cm)。

**原始 Isaac play（更多参数）：**
```bash
cd /home/l/code/IsaacLab
CHECKPOINT=/home/l/code/AutoResearch/runs/s0_B7_20260622/student_best.pt \
HEADLESS=0 MAX_EPISODES=10 AUTORESEARCH_ROOT=/home/l/code/AutoResearch \
./scripts/play_lift_cube_student_vision_binary_gripper.sh --show_opencv
```

### B1 root-cause analysis (diagnostic eval)

**Run:** `runs/s0_B1_restart/eval_diagnostics/` (20 episodes, `student_best.pt` @ iter 100)

| Metric | Value |
|--------|-------|
| eval success | **0%** |
| grasp rate | **85%** |
| mean min EE–cube dist | **1.4 mm** |
| mean post-grasp wander | **10.4 cm** |
| mean max cube z | ~0.27 m (table ≈ 0.04 m) |

**Failure mode breakdown (100% of episodes):**

| Mode | Count | Interpretation |
|------|-------|----------------|
| `lifted_not_at_goal` | 20/20 | Cube lifted off table but never relocated to goal |

**NOT the primary failure:**
- ~~`never_reached_cube`~~ — EE consistently reaches cube (min dist ≈ 1 mm)
- ~~`never_grasped`~~ — only 3/20 weak lifts (max_z ≈ 0.055 m, borderline)
- ~~`grasp_then_dropped`~~ — no episodes; final cube_z stays elevated (~0.25 m)

**Trajectory reading (EE vs cube vs goal):**
1. **Approach (steps 1–~200):** EE moves from home toward cube; EE–cube dist drops from ~0.33 m → <0.01 m.
2. **Grasp + partial lift (~200–600):** gripper closes; cube_z rises to 0.23–0.29 m (well above table).
3. **Post-grasp wander (600–1250):** cube hovers ~10–36 cm from goal in xy; `dist_to_goal` never crosses 0.02 m threshold → timeout.

**Conclusion:** B1 (anchor mask only) learns approach + grasp from DAgger teacher demos but **cannot relocate cube to goal** without guidance field. This is exactly the hypothesis B7 tests.

**Artifacts:**
- `runs/s0_B1_restart/eval_diagnostics/eval_diagnostics.json` — full trajectories
- `runs/s0_B1_restart/eval_diagnostics/eval_analysis.md` — human summary

### Issues encountered
| Issue | Resolution |
|-------|------------|
| `mask_flow` supervision → align ~0.03 | Switched to **`state_goal`**: cube→goal direction in x-z from `student_state`; mask still input-only at inference |
| Weak signal from mask centroid Δ | Documented in PROGRESS; mask_flow kept as ablation option |
| Eval only reported 0% success, no failure reason | Added diagnostic eval + wired `collect_diagnostics` into training eval |
| B7 ckpt meta `visual_channels=8` but weights 10ch | Infer channels from conv weights; fixed checkpoint save for future runs |
| `TERM=dumb` breaks isaaclab.sh | Force `TERM=xterm-256color` in wrapper scripts |

### Reflections
- Step 1 target: predict demo relocation direction from **masks only**; proprio used only as training label (not leaked to student at test)
- B1 result validates research framing: mask anchors grasp, G needed for relocation
- Next: B7 should reduce `post_grasp_wander_m` and convert `lifted_not_at_goal` → `success`

---

## Day 1 — 2026-06-23 (S1 clean collect + disk cleanup)

### Done
- [x] **Clean S1 collect** (`run_s1_collect_clean.sh`): 100 ep, 100% success, 240px mask+depth, no teacher noise, topdown_close
- [x] **Disk cleanup** after Cursor crash: repo 520 GB → **2.2 GB**; archive at `/home/l/code/data/AutoResearch_Data` (~516 GB)
- [x] S1 DAgger B1/B7 both finished 300 iters (eval@300: 0% SR; 16/20 `never_reached_cube` — trained with **stale** G_θ)

### Archive policy
- **Moved:** `dagger_aggregate.pt`, `student_final.pt`, `student_iter_200.pt` (embedded 76 GB buffers) for S0/S1 runs
- **Kept in repo:** `student_best.pt`, `student_iter_300.pt`, logs, eval JSON, TensorBoard
- **Data:** clean240 + old noisy HDF5 → archive; symlinks in `demos/` and `demos_clean240/`
- **Deleted:** empty/partial failed HDF5 (`111054`, `111256`)
- Manifest: `/home/l/code/data/AutoResearch_Data/ARCHIVE_MANIFEST.md`

### Blocked
- [ ] **G_θ retrain on clean240** — `learn_guidance_field.py` OOM-killed (~84 GB RSS loading all samples). Checkpoint still old `guidance_field_s1_random_cube.pt` (0.723, noisy 84px). Need lazy HDF5 loading or RAM subsample.

### Next
```bash
# After fixing loader memory — retrain Step 1
DEMO_DIR=data/lift/s1_random_cube/demos_clean240 IMAGE_SIZE=240 \
  bash scripts/run_s1_step1.sh --batch-size 16

# Verify G on clean teacher
bash scripts/play_teacher_g_check.sh
```

---

## Metrics log

| Date | Step | field_alignment (val) | B1 eval SR | B7 eval SR | Notes |
|------|------|----------------------|------------|------------|-------|
| 2026-06-18 | Step 1 | **0.9845** | — | — | state_goal, v3+smoke, epoch 79 |
| 2026-06-19 | Step 2 B1 | — | **0%** | — | 100 iters; failure=lifted_not_at_goal (85% grasp) |
| 2026-06-22 | Step 2 B7 | — | 0% | **0%** | aux @5cm backfilled |
| 2026-06-22 | **S1** | — | — | — | subgoal field + dagger scripts ready |
| 2026-06-23 | S1 collect | — | — | — | clean240, 100% success, archived 17 GB |
| 2026-06-23 | S1 Step 2 | — | 0% | 0% | 300 iters each; stale G; buffers archived |

---

## Commands

### Diagnostic eval (any checkpoint)
```bash
CHECKPOINT=runs/s0_B1_restart/student_best.pt \
OUTPUT_DIR=runs/s0_B1_restart/eval_diagnostics \
bash scripts/run_diagnostic_eval.sh
```

### DAgger progress
```bash
tail -f runs/s0_B7_*/progress.txt
python3 scripts/dagger_progress.py runs/s0_B7_*/
```

### Launch B7
```bash
FIELD_CKPT=checkpoints/lift/guidance_field_s0_fixed.pt \
  bash scripts/run_s0_dagger_smoke.sh B7
```
# Research-priority update -- 2026-07-16

- Primary target remains **zero-shot compositional interaction transfer**:
  every interaction factor is seen, but selected mechanism/state/goal
  combinations are held out and instantiated through compact G.
- Same-task scene/initial-state OOD and `k={1,2,5,10,25}` demonstration
  adaptation are fallback claims, not replacements for the first target.
- E2 BC `74/100` normal versus `1/100` F-flip establishes conditional
  sufficiency and causal G use after inverse-contact support, but is not
  target-free zero-shot evidence.
- Next data step: construct a factorial interaction matrix and multiple
  leave-one-combination-out splits. Do not add target-cell reverse labels.
- Next policy step: use a credible history-conditioned temporal policy with
  task-balanced sampling; retire further lightweight-FM horizon/seed tuning.
- Decision note:
  `knowledge_base/05_design_decisions/calvin_g_generalization_claim_ladder_2026-07-16.md`

# E4 inverse-closure gate -- 2026-07-17

- Audited CALVIN support: only `open_drawer` and `move_slider_right` provide
  physically valid same-mechanism reverse-contact data for the held-out close
  and left-slider tasks. Reverse replay succeeds at 25/25 and 24--25/25;
  reverse block replay is only 3/25 and 5/25 and is not used for training.
- Added generic inverse-pair augmentation, frozen-base residual adapters, and
  a learned state+G two-expert router. No task-specific inference controller
  is used.
- Found a temporal contact effect: H1/H2/H4 on the same single residual policy
  give `[20,17,40,34]`, `[32,33,50,28]`, and `[26,28,59,29]`. H2 improves
  sustained articulation contact; H4 over-amplifies block motion.
- Final seed-0 setting is `moe_h64_sw7`, G scale 1, horizon 2. Four held-out
  tasks at 100 rollouts each are `[51,32,47,44]`, so every task is non-zero and
  the max-min gap is **19**, passing the predefined `<20` gate.
- Causal controls on the same states: zero G is `[0,0,0,0]`; direction-flipped
  G is `[0,1,0,0]`. The policy depends on the directional semantics of G.
- This is not clean target-free zero-shot for close/slider because their labels
  come from self-supervised inverse source trajectories. The two block targets
  remain held-out action combinations. Treat E4 as inverse closure/support
  expansion plus causal reusable-interface evidence.
- Full record:
  `knowledge_base/03_experiment_evidence/phase1_calvin_inverse_closure_e4_2026-07-17.md`
- Lightweight local evidence:
  `artifacts/phase1_calvin/2026-07-17_E4_gate_moe_h64_sw7_seed0/`

# Official ABC-to-D scene gate and pretrained vision -- 2026-07-19

- Downloaded and checksum-verified official CALVIN `task_ABC_D.zip` (518 GB).
  Built a 207 MB state/action/language subset with 2,101 block-push segments
  and 51,594 frames for held-out color-direction recombination in scene D.
- Replaced the invalid held-out task one-hot baseline with official 384-D
  CALVIN MiniLM embeddings in both training and evaluation.
- Direct MiniLM+G fusion fails (`0/200`) even though full-state MiniLM/no-G is
  `44/200`. Explicit state+G without direct executor language is causal:
  full-state normal/zero/flip is `35/200`, `5/200`, `0/200`; proprio is
  `47/200`, `0/200`, `0/200`.
- The correct architecture boundary is now explicit: language is upstream for
  predicting G; the low-level executor consumes visual/proprio history plus G.
  Direct language+G remains a fusion ablation.
- Entered pretrained visual stage with frozen `facebook/dinov2-small` shared
  across static and gripper cameras. Cached all 51,594 frames as two 384-D
  float16 CLS features and passed offline-training plus online-rollout smoke.
- Visual seed-0 training completed. Final offline chunk/first-action MSE is
  `0.04217/0.02312` for visual+oracle-G and `0.05548/0.02734` for
  visual+MiniLM/no-G.
- The first visual evaluation jobs aborted in an unused tactile-camera
  `getCameraImage` call. Evaluation now instantiates only the official static
  and gripper cameras; a two-reset smoke test passes.
- Renderer profiling shows DINOv2+policy costs `0.01--0.09 s/step`. The first
  NVIDIA attempt was black/slow because `CUDA_VISIBLE_DEVICES` remapped CUDA
  ids while EGL received a physical id. The D gate used complete llvmpipe
  frames at `~0.053--0.056 s/env-step`. The corrected GPU protocol leaves all
  devices visible and matches policy `cuda:N` to physical EGL id N; it returns
  complete frames with NVIDIA-vs-llvmpipe MAE `1.32` static and `0.76` gripper.
- Visual D scene gate completed on 2026-07-21 with the same 100 official
  sequence indices per held-out task. Oracle G gives blue-left/red-right
  `1/100, 11/100` (`12/200` total), while zero G and sign-flipped G are both
  `0/200`. The matched MiniLM/no-G executor gives `3/100, 0/100` (`3/200`).
  Normal G has 12 normal-only and zero control-only paired successes against
  both causal controls (two-sided exact McNemar `p=4.88e-4`). Against language
  only it has 12 G-only versus 3 language-only (`p=0.0352`).
- This passes a weak causal-use gate but fails the robust visual scene gate:
  the absolute success rate is low, blue/red behavior remains asymmetric, and
  visual+G falls from privileged-state+G's `35/200` to `12/200`. Run
  source-scene visual closed-loop health next to separate executor weakness
  from A/B/C-to-D transfer before changing compact G or starting learned G.
- Source reconstruction found official fixed scene ranges and A/B/C configs.
  Scene A online reset matches stored source RGB almost exactly under corrected
  NVIDIA EGL (median DINO cosine `0.99981` static, `0.99961` gripper); B/C
  still show a config reproduction gap and are excluded for now. Scene-A
  health completed at 25 starts x four source tasks: visual+G is `72/100` and
  visual+MiniLM/no-G is `62/100`. Per-task G/language results are `21/15`
  blue-right, `15/18` pink-left, `18/17` pink-right, and `18/12` red-left.
  Paired outcomes are 22 G-only, 12 language-only, 50 both, and 16 neither
  (`p=0.121`). Both executors are healthy on exact source support; the D drop
  is therefore primarily a visual scene/composition transfer problem.
- Corrected-NVIDIA D reruns completed in separate `_nvidia_egl` directories.
  Oracle G reaches blue-left/red-right `14/100, 18/100` (`32/200`), versus
  zero G `1/200`, sign-flipped G `1/200`, and MiniLM/no-G `6/200`. Paired
  normal-only/control-only counts are `32/1` against both zero and flip
  (`p=7.92e-9`) and `31/5` against language-only (`p=1.29e-5`). The corrected
  renderer improves absolute success over llvmpipe but does not change the
  conclusion: G is causally useful, while A-to-D visual/composition transfer
  remains weak.
- Eval logs: `/mnt/nvme33n1/wuk_code/AutoResearch/logs/M1_abc_d_scene_gate_none/`
  and `/mnt/nvme33n1/wuk_code/AutoResearch/logs/M1_abc_d_scene_gate_minilm/`.
- Full record:
  `knowledge_base/03_experiment_evidence/phase1_calvin_abc_d_scene_gate_visual_transition_2026-07-19.md`

# Matched behavior-G recombination gate -- 2026-08-13

- Downloaded and SHA256-verified the complete ModelScope CALVIN ABC-D LeRobot
  mirror. The six block-push tasks provide 3,357 episodes and 199,292 frames;
  all frames have frozen dual-camera DINOv2-small features.
- Replaced the earlier train/online G mismatch with the same coarse 6-D
  behavior interface on both sides: `G=[attraction_to_block_center(3),
  semantic_object_flow(3)]`. The executor receives visual features, proprio,
  and G, but no direct language or task identifier.
- Trained on four combinations (`red-left`, `blue-right`, `pink-left`,
  `pink-right`) and held out `blue-left` and `red-right`. The full seed-0 run
  contains 50,032 chunk windows (45,016 train / 5,016 validation), chunk size
  16, stride 2, and 50 epochs.
- Offline learning did not stop early: the best validation chunk and
  first-action MSE both occur at epoch 50 (`0.03810` and `0.02426`), with
  first-action gripper accuracy `0.9649`.
- On the same 100 held-out starts, normal G succeeds `9/100`; zero G and
  flow-sign-flipped G both succeed `0/100`. Each paired comparison has nine
  normal-only and zero control-only successes (two-sided exact McNemar
  `p=0.00390625`). This passes the seed-0 causal-dependence gate.
- Absolute recombination is not mature: normal G is `1/50` on blue-left and
  `8/50` on red-right. Treat this as evidence that the executor parses and
  uses matched G, not as scaling-ready generalization.
- On the temporary BNU node (10 vCPU, one V100-32GB), compact window assembly
  took about 39 minutes, GPU training about 9 minutes, and each 100-rollout
  llvmpipe evaluation about 46 minutes. Data loading and CPU rendering, not
  GPU memory, dominate wall time.
- Before learned-G scaling: add a state/action/G-only reusable training cache,
  evaluate source-task closed-loop health, and run seeds 1--4 with matched
  normal/zero/sign-flip controls. Recommended next node is 32 vCPU, four
  V100-32GB (or four A100-40GB), and at least 128 GiB RAM.

# Behavior-G training-stability gate -- 2026-08-13

- Migrated the method gate to a two-V100 BNU allocation and built a verified
  42 MiB packed cache for all 199,292 state/action/G rows. Per-seed input
  assembly fell from about 39 minutes to roughly one minute; CPU file opening,
  not GPU capacity, caused the previous startup bottleneck.
- A five-seed audit exposed optimizer/checkpoint instability in the original
  `lr=1e-3` protocol. Best validation first-action MSE across seeds 0--4 is
  `[0.024261,0.020625,0.024709,0.030968,0.025488]`, with best epochs
  `[50,45,18,15,19]`. Four of five final checkpoints are worse than their own
  best checkpoint, by as much as 42.1%. Final-checkpoint seed variance must not
  be interpreted as G variance.
- A seed-0 learning-rate diagnostic selected AdamW `lr=3e-4` plus the
  minimum-validation-first-action checkpoint. `lr=3e-4` reaches `0.012165`
  best and `0.012732` final first-action MSE; `lr=1e-4` reaches `0.012225` best
  but regresses to `0.014101` final.
- The corrected `lr=3e-4` five-seed run is stable: best first-action MSE is
  `[0.012165,0.013433,0.012133,0.015233,0.013120]` (mean `0.013217`, standard
  deviation `0.001265`, CV `9.6%`), and best epochs are `[42,41,40,50,48]`.
  This is about 48% lower than the old five-seed mean and no seed peaks early.
- The original seed-0 final checkpoint reaches only `74/400 = 18.5%` on its
  four training tasks: blue-right `8/100`, pink-left `26/100`, pink-right
  `22/100`, and red-left `18/100`. Thus its `9/100` held-out score combines
  weak closed-loop execution with a recombination gap; it cannot by itself
  measure the compositional limit of G.
- Full matched closed-loop evaluation of the stable seed-0 best-validation
  checkpoint is running on the same held-out starts with normal, zero, and
  sign-flipped G, alongside a 400-rollout source-task health audit. Learned-G
  or larger-data scaling remains gated on task success, causal controls, and
  cross-seed stability rather than offline MSE.
