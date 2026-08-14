# AutoResearch Progress — Index

**Last reconciled:** 2026-08-14 (local workstation ↔ BNU `/public/home/zhongfangwei/wuk_code/AutoResearch_method_gate`).

**Active goal:** Phase 1 — CALVIN reusable interaction G (`G=[A,F]`), causal dependence → conditional flow/diffusion executor.

**Legacy track:** Phase 0 — Isaac Lab S1 lift DAgger (AnchorMask + G_θ). Kept for preliminary evidence; not the active queue.

---

## Where to read next

| Journal | Path |
|---------|------|
| **Phase 1 (active)** | [docs/progress_phase1_calvin.md](docs/progress_phase1_calvin.md) |
| Phase 0 Isaac | [docs/progress_phase0_isaac.md](docs/progress_phase0_isaac.md) |
| Isaac findings ratchet | [findings.md](findings.md) |
| Rebuild from Git | [bootstrap.md](bootstrap.md) |
| Dataset provenance | [dataset.md](dataset.md) |

## Snapshot (do not replace journals)

### Phase 1 — latest gates

- **2026-08-14** Interaction-volume target-pair gate on expanded A/B/C source
  data: ordinary scaled token BC still ignores attraction A, but a
  same-flow/different-target action-difference objective makes A causally
  necessary. The epoch-50 probe reaches held-out blue-left/red-right
  `9/50, 4/50`; zero-A, zero-F, all-zero, and sign-flipped controls are each
  `0/100` on identical starts. Matched no-G is `0/100` on source and held-out.
  This is the first bidirectional reusable-interface result, but absolute
  normal success is only `13/100` and the source total is `15/100`. Scaling is
  still blocked on control recovery plus seed replication. A weak-pair
  continuation from epoch 12 reaches `17/100` held-out; zero-A, zero-F, and
  all-zero are each `0/100`, while sign flip is `6/100` on disjoint starts
  (`p=0.0347`). It is a better but still one-sided Pareto point. The active
  correction restricts target-pair supervision to first-step xyz because the
  pair criterion does not justify the previous rotation/gripper difference
  penalty; the full action remains supervised by ordinary BC.
- **2026-08-13** Training-stability audit on the matched behavior-G gate:
  packing all `199,292` state/action/G rows reduced per-run input assembly from
  about 39 minutes to about one minute. With the original AdamW `lr=1e-3`,
  four of five final checkpoints regressed from their validation optimum; best
  epochs ranged from 15 to 50. The corrected `lr=3e-4` five-seed run has best
  first-action MSE `0.01322 +/- 0.00127` (CV `9.6%`) and best epochs at 40--50.
  Closed-loop best-checkpoint evaluation is in progress; offline MSE alone is
  not a learned-G scaling gate.
- **2026-08-14** Stable seed-0 closes at `7/100` held-out versus old `9/100`
  on identical starts despite halving offline MSE; zero G remains `0/100`.
  A 1,600-step trace finds the online 1 cm block-center contact switch fires
  zero times, whereas training G is post-contact on 63.3% of frames. The next
  gate uses orientation- and asset-aware object-surface anchors with a shared
  embodiment clearance before any learned-G scaling.
- **2026-08-13** Matched behavior-G seed-0 recombination gate: visual+G
  `9/100` on held-out blue-left/red-right, versus zero/flow-flip `0/100,
  0/100` on identical starts (`p=0.0039` paired). G is causally used, but the
  asymmetric `1/50, 8/50` absolute result is not scaling-ready. The original
  checkpoint's four seen tasks reach only `74/400`, so its held-out score
  mixes weak execution with a recombination gap rather than isolating G alone.
- **2026-07-22** Corrected-NVIDIA ABC→D **visual** scene gate: oracle G `32/200` vs zero/flip `1/200, 1/200` and MiniLM/no-G `6/200`. Scene-A health remains G `72/100` vs MiniLM `62/100`. G is causal, but absolute scene/composition transfer is still too weak for scaling learned G.
- **2026-07-17** E4 inverse-closure: residual MoE variants; not clean target-free ZS for close/slider.
- **2026-07-16** **E3** factorial checkerboard: red-right `139/300` vs flip/no-G ~0; blue-left `147/300`. Freeze as compact-G causal checkpoint; cross-mechanism still open.

Evidence trees: `experiments/phase1_calvin/`, `knowledge_base/03_experiment_evidence/`.

### Phase 0 — Isaac (local, 2026-07-22)

- Simulators nested under `simulators/isaaclab` (legacy symlink `/home/l/code/IsaacLab`).
- B1 fixed-cube bring-up finished earlier; strict `success@2cm` still **0%** (failure moved toward grasp/hold). See `findings.md` / phase-0 journal.

## Hosts

| Host | Root |
|------|------|
| Local | `/home/l/code/AutoResearch` |
| H200 | `/mnt/nvme33n1/wuk_code/AutoResearch` |
| BNU | `/public/home/zhongfangwei/wuk_code/AutoResearch` |

Details: [docs/HOSTS.md](docs/HOSTS.md).
