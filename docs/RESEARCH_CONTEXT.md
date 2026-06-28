# Research Context

## Problem

What **visual representation** can serve as a shared **policy task interface** across tracking, navigation, and manipulation?

Not affordance learning. Not unified whole-body policy. **Policy-learnable visual goal.**

## Staged conclusion

1. **Single mask** is a strong **visual anchor** (what/where) — supports tracking, approach, grasp acquisition.
2. It **under-specifies** object relocation / state transition (lift post-grasp wander in IsaacLab).
3. **TargetMask** (ghost future mask) is intuitive but often **not a real scene entity** → task engineering risk.
4. **Guidance field G** is the policy-facing fix: action-agnostic, visual, reward-computable.
5. **WM** (e.g. MWM) can learn mask dynamics but is an **optional field learner** — not the final interface.

## Policy contract

```python
a_t = pi(o_t, M_t, G_t)
```

| Symbol | Meaning | Examples |
|--------|---------|----------|
| M | Anchor mask | person, cube, handle |
| G | Guidance field | upward (lift), toward region (push), view-centering (track) |

## Lineage

| Work | Contribution to (M, G) | Code |
|------|------------------------|------|
| **EVT** | M from VFM segmentation | `$EVT_ROOT` |
| **HIEVT** | region goal ≈ G with bbox + IoU | `$HIEVT_ROOT` |
| **TarGF** | G as target gradient field (score-matching) | [GitHub](https://github.com/AaronAnima/TarGF) → `$TARGF_ROOT` |
| **DualGF** | target + support fields; image navi | `$DUALGF_ROOT` |
| **HumanoidPF** | field as obs + reward for RL | paper |
| **MWM** | learns future mask dynamics → optional G source | paper |
| **Isaac vision DAgger** | B1 fails post-grasp → motivates B7 | `$ISAACLAB_ROOT` |

## Hypotheses (policy-centric)

- **H1:** Step 1 — G_θ learns demo mask dynamics on **S0 fixed** (field_alignment ≥ 0.75).
- **H2:** Step 2 — B7 (M+G_θ) >> B1 (M only) on lift success and post-grasp wander @ S0.
- **H3:** **S1** — G_θ generalizes to random cube positions (field + policy within 85% of S0).
- **H4:** **S2** — G_θ handles random lift goals when goal context is available.
- **H5:** HIEVT tracking — same (M,G) interface vs bbox channel baseline.

## Curriculum

See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): **S0 fixed → S1 random cube → S2 random goal → S3 OOD**.

## References

See [REFERENCES.md](REFERENCES.md) for full paper/repo table.
