# Phase1 CALVIN Flow-Matching Data Feasibility

Date: 2026-07-07

## Question

Before running `flowmatching + G`, check whether the current CALVIN data scale
can support such training, and whether conditional flow matching is a mature
framework or would need to be built from scratch.

## Current Data Scale

Server data root:

```text
/mnt/nvme33n1/wuk_code/AutoResearch/phase1_calvin/data/task_D_D
```

Full CALVIN D_D training split:

```text
segments: 5124
episode files: 512077
single-frame windows: 308918
```

Articulation-only subset:

```text
tasks: open_drawer, close_drawer, move_slider_left, move_slider_right
segments: 606
frames: 39037
stride-2 state-BC samples: 19818
```

Transfer split:

```text
train: open_drawer + move_slider_left + move_slider_right
segments: 455
frames: 29397
stride-2 state-BC samples: 14924

test: close_drawer
segments: 151
frames: 9640
```

Action-chunk windows, no cross-segment chunking:

| Dataset | H=16 stride-1 | H=16 stride-2 | H=32 stride-1 | H=32 stride-2 | H=48 stride-2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full D_D | 232058 | 116284 | 150074 | 75292 | 36981 |
| articulation4 | 29947 | 14977 | 20251 | 10129 | 5328 |
| transfer_train3 | 22572 | 11288 | 15292 | 7648 | 4032 |
| heldout_close | 7375 | 3689 | 4959 | 2481 | 1296 |

## Feasibility Judgment

### From-scratch foundation-scale flow policy

Not feasible with the articulation subset.

The effective independent data is only a few hundred language segments, despite
having 10k-30k overlapping frame/chunk windows. This is enough for controlled
ablation, but not enough to train a large flow/diffusion policy that robustly
learns contact decoding and task transfer from scratch.

### Small state/action conditional flow matching

Feasible as a diagnostic experiment.

Recommended scope:

- state input: `robot_obs + scene_obs + optional task zeros + G`
- action target: relative action chunk, H=16 or H=32
- conditioning: concat or FiLM/cross-attention over G tokens
- no image encoder at first
- segment-level split, not random frame split, for validation

This can test whether a stronger action decoder can use v7 object-flow better
than the current single-step MLP. It should not be interpreted as the final
scaling result.

### Fine-tuning large pretrained VLA/FM models

Potentially feasible, but not the right first step.

OpenPI reports pi0/pi0.5 base models pretrained on 10k+ hours of robot data and
provides fine-tuning examples. It also reports high memory requirements:
LoRA fine-tuning >22.5 GB GPU, full fine-tuning >70 GB.

The engineering cost is nontrivial:

- convert CALVIN to LeRobot/openpi format;
- map CALVIN observations/actions into the expected policy schema;
- decide how to inject custom analytic G, which is not a default input;
- integrate CALVIN evaluation loop.

This should be a later stage after the state/action diagnostic shows that
object-flow G benefits a stronger decoder.

## Framework Landscape

Conditional flow/diffusion policy is mature as a paradigm:

- Diffusion Policy models a visuomotor policy as conditional action diffusion
  and uses receding-horizon action chunks.
- FlowPolicy provides official consistency-flow-matching code conditioned on
  point clouds and robot state.
- LeRobot provides a standardized dataset format and policy zoo including
  Diffusion, Pi0, Pi0Fast, Pi0.5, and SmolVLA.
- OpenPI provides pi0/pi0.5 flow-matching VLA training and fine-tuning code.

But `flowmatching + our custom G` is not plug-and-play in the current CALVIN
state-BC code. Adding G means either:

1. implement a small conditional FM policy locally; or
2. modify LeRobot/openpi/FlowPolicy input schemas to include G tokens.

Therefore this is not pure imagination, but also not a one-command mature
baseline in our exact setting.

## Recommended Experiment Ladder

Do not jump directly to large VLA fine-tuning.

1. **Chunked deterministic BC + G**
   - action chunk H=16/32
   - tests whether receding-horizon chunking improves contact timing.

2. **Small conditional flow/diffusion state policy + G**
   - same state/G inputs, action chunk target
   - tests whether multimodal generative action decoding helps beyond MLP.

3. **LeRobot Diffusion / Pi0-style integration**
   - only if step 2 shows positive signal.

4. **OpenPI / pretrained VLA fine-tune**
   - later scaling path; requires dataset conversion and G input design.

## Key Decision

V8 interaction-manifold G should not be the default next step.

First test:

```text
Can a stronger action decoder use v7 object-flow G?
```

If `flow/diffusion + v7` improves held-out close, keep object-flow G and focus
on policy/scaling.

If `flow/diffusion + v7` still fails while `v8` helps, then the representation
is under-specified and interaction-manifold G is justified.

## Follow-up Result: Chunk BC vs Small FM + v7

Date: 2026-07-07

Experiment:

```text
train tasks: open_drawer, move_slider_left, move_slider_right
test task: close_drawer
G: v7 unified interaction field
task one-hot: zeroed
chunk size: 16
stride: 2
train windows: 11288
segments: 455
epochs: 50
eval: 25 balanced first-task close_drawer rollouts, ep_len 160
checkpoints: 4, 8, 16, 32, 50
```

Local synced artifact:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-07_transfer_chunk_bc_vs_fm_v7_h16/
```

Result:

| Mode | Epoch | Val chunk MSE | Close success |
| --- | ---: | ---: | ---: |
| chunk_bc_v7 | 4 | 0.0412 | 0/25 |
| chunk_bc_v7 | 8 | 0.0394 | 0/25 |
| chunk_bc_v7 | 16 | 0.0393 | 1/25 |
| chunk_bc_v7 | 32 | 0.0402 | 0/25 |
| chunk_bc_v7 | 50 | 0.0417 | 0/25 |
| chunk_fm_v7 | 4 | 0.2338 | 0/25 |
| chunk_fm_v7 | 8 | 0.2224 | 0/25 |
| chunk_fm_v7 | 16 | 0.1702 | 0/25 |
| chunk_fm_v7 | 32 | 0.0896 | 0/25 |
| chunk_fm_v7 | 50 | 0.0686 | 0/25 |

Interpretation:

- Deterministic chunked BC did not turn v7 into reliable close-drawer transfer.
  Its only nonzero checkpoint was 1/25 at epoch 16, while later checkpoints
  regressed to 0/25 despite low validation chunk MSE.
- The small conditional flow-matching policy learned the action chunk
  distribution progressively in MSE terms, but did not improve closed-loop
  transfer on close_drawer.
- This negative result should not be read as "flow matching is ineffective" in
  general. In this low-dimensional MLP setting, replacing single-step BC with
  chunked deterministic BC or small FM is insufficient to make v7 G robustly
  reusable for the held-out close task.

Updated decision:

```text
The next bottleneck is likely not only action decoder strength.
Need diagnose whether v7 G gives the policy enough contact/action-affordance
information for close, and whether the close transfer split is semantically too
far from open/left/right under the current low-dimensional observation.
```

## Follow-up Result: Train-Task Check for FM

Date: 2026-07-07

Question:

```text
Does chunk_fm_v7 fail because the flow-matching policy was trained/sampled
incorrectly, or does it mainly fail on the held-out close transfer task?
```

Experiment:

```text
checkpoints: chunk_bc_v7 epoch 50, chunk_fm_v7 epoch 50
eval tasks: open_drawer, move_slider_left, move_slider_right
rollouts: 25 per task, 75 total per policy
ep_len: 160
G: v7 unified interaction field
task one-hot: zeroed
```

Local synced artifact:

```text
/Users/wukui/Documents/AutoResearch/artifacts/server_sync/2026-07-07_transfer_chunk_bc_vs_fm_v7_h16/2026-07-07_train_task_eval_chunk_bc_vs_fm_v7_e50_n25_h160/
```

Result:

| Mode | Overall | Left | Right | Open |
| --- | ---: | ---: | ---: | ---: |
| chunk_bc_v7 e50 | 71/75 | 22/25 | 24/25 | 25/25 |
| chunk_fm_v7 e50 | 74/75 | 24/25 | 25/25 | 25/25 |

Interpretation:

- The FM policy is not generally broken. On the train task distribution it is
  at least as strong as the deterministic chunked BC baseline.
- The previous `0/25` held-out close result should be interpreted as a
  transfer/generalization failure, not as evidence that FM training or CALVIN
  checkpoint loading is invalid.
- FM rollout videos revealed an important stochasticity caveat: because the
  policy samples action chunks from a flow model, qualitative videos can change
  with sampling order/seed. Visual comparisons must record policy alias,
  checkpoint, rollout index, and sampling order or seed.

Updated decision:

```text
Next diagnose why v7 + FM succeeds on open/left/right but does not reuse to
close. The issue is likely in the reusable G semantics, train/test task
composition, or stochastic action selection under distribution shift, not basic
FM optimization.
```
