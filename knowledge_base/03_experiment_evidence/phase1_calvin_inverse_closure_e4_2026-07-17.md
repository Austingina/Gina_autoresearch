# CALVIN E4: inverse closure and balanced held-out transfer

Date: 2026-07-17

## Question

Can one state policy use the same compact query-G interface to obtain non-zero,
reasonably balanced control on four held-out interaction combinations?

- `close_drawer`
- `move_slider_left`
- `push_red_block_right`
- `push_blue_block_left`

The stage gate is 100 rollouts per task, every task above zero, and
`max(success)-min(success) < 20`.

## Support audit

CALVIN has no second close-drawer mechanism. `open_drawer` is the only source
task with the same mechanism and contact path in the opposite direction.
`move_slider_right` is the corresponding source for left-slider motion. Block
pushes share directional factors but not sustained handle contact.

Physical reverse replay verified that time-reversing the articulation source
trajectories is valid:

| Source -> held-out | Reverse replay success |
| --- | ---: |
| `open_drawer -> close_drawer` | 25/25 |
| `move_slider_right -> move_slider_left` | 24/25 to 25/25 |
| `push_red_block_left -> right` | 3/25 |
| `push_blue_block_right -> left` | 5/25 |

Therefore E4 adds inverse labels only for drawer and slider. It does not use
reverse block labels. This is self-supervised inverse support, not clean
target-free zero-shot evidence for the two articulation tasks.

## Final policy

Checkpoint label: `moe_h64_sw7`, seed 0.

- Frozen E3 state+G base policy.
- Two 64-wide residual experts, associated with `close_drawer` and
  `move_slider_left` through a learned state+G router.
- Source trajectories distill the frozen teacher into both experts.
- Reverse source pairs: 100 `open_drawer` segments and 100
  `move_slider_right` segments, yielding 2,500 close and 2,486 left-slider
  windows.
- Slider inverse loss weight: 7; source distillation and reverse-source loss
  weights: 1.
- Chunk size 16, first-action weight 16, 50 epochs, batch 512, Adam learning
  rate `1e-3`.
- Query-G token type removed; language zeroed for this oracle-query-G test.
- One shared inference setting for all tasks: G scale 1, execute two actions
  from each predicted chunk, then replan.

No task-specific action rule, gain, horizon, or hand-authored controller is
used at inference.

## Final result

| Query-G control | Close drawer | Slider left | Red right | Blue left | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | **51/100** | **32/100** | **47/100** | **44/100** | **19** |
| zero | 0/100 | 0/100 | 0/100 | 0/100 | 0 |
| sign flip | 0/100 | 1/100 | 0/100 | 0/100 | 1 |

The normal-G row passes the predefined gate. Zeroing G destroys all four
tasks. Flipping the direction-bearing attraction, flow, and residual features
while preserving token count and non-zero magnitude also destroys performance.
The policy therefore depends on the directional semantics of G, not merely
the presence of an auxiliary token.

## Temporal diagnosis

For the earlier single residual adapter (`res_h64_p1_r1`):

| Execution horizon | Close | Slider | Red | Blue | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 17 | 40 | 34 | 23 |
| 2 | 32 | 33 | 50 | 28 | 22 |
| 4 | 26 | 28 | 59 | 29 | 33 |

Executing two steps improves sustained drawer/slider contact, but four steps
amplifies open-loop block motion and increases imbalance. The result is not
"longer chunks are better". Two steps are a useful contact-coherence point
for this policy, while closed-loop replanning remains necessary.

## Architecture and weighting audit

| Candidate, shared horizon | Close | Slider | Red | Blue | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| single residual, H1 | 20 | 17 | 40 | 34 | 23 |
| single residual, H2 | 32 | 33 | 50 | 28 | 22 |
| two experts, slider weight 4, H1 | 32 | 21 | 41 | 42 | 21 |
| two experts, slider weight 4, H2 | 39 | 27 | 48 | 39 | 21 |
| two experts, slider weight 7, H2 | **51** | **32** | **47** | **44** | **19** |
| two experts, slider weight 8, H2 | 53 | 32 | 50 | 41 | 21 |

The response to weighting is discontinuous: weight 7 passes, while weight 8
raises drawer without raising slider. These are contact-policy mode changes,
not a smooth gain curve. A three-expert variant with a source no-op expert was
also rejected: it removed useful source approach/contact behavior from the
articulation experts and produced a 56/8 drawer/slider split on the first 25
trials.

## Conclusions

1. The original all-zero articulation result was primarily missing
   same-mechanism inverse-contact support, not evidence that directional G is
   useless.
2. G alone is insufficient. The policy needs action/contact support that lets
   it interpret G in the relevant control regime.
3. A learned state+G router plus mechanism residuals reduces interference
   while retaining a shared base policy.
4. Temporal execution is part of the interface: per-step replanning can break
   sustained contact, while excessive open-loop execution harms block pushes.
5. The causal controls strongly support reusable directional conditioning, but
   this stage does not yet prove video-learned G or seed-level robustness.

## Next gate

- Repeat the selected configuration on seeds 1 and 2 before treating the
  exact success rates as stable.
- Replace oracle query-G with language-conditioned learned G and repeat normal,
  zero, and sign-flip controls.
- Add true target-free held-out mechanism/object combinations. Keep inverse
  closure as a support-expansion ablation, not the headline zero-shot claim.
- Move the two-step contact-coherence idea into a mainstream history-conditioned
  policy instead of treating horizon as a permanent hand-selected constant.

## Artifacts

Local lightweight mirror:
`artifacts/phase1_calvin/2026-07-17_E4_gate_moe_h64_sw7_seed0/`

Remote checkpoint and full run:
`/mnt/nvme33n1/wuk_code/AutoResearch/experiments/phase1_calvin/2026-07-17_E4_inverse_finetune_moe_h64_sw7_seed0/`

