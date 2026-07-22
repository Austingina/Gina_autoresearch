# AutoResearch Progress — Index

**Last reconciled:** 2026-07-22 (local workstation ↔ H200 `/mnt/nvme33n1/wuk_code/AutoResearch`).

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

Details: [docs/HOSTS.md](docs/HOSTS.md).
