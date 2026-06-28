# Future Research Routes (Deferred)

**Status:** Recorded for later — **not** in the active experiment plan until S1 lift validation (Tier 0–2) completes. See [findings.md](../findings.md) for current checklist.

---

## Motivation

Lift cube (S0/S1) is a **minimal interface validation** for `a_t = π(o_t, M_t, G_t)`. Longer term, ask whether a shared **Foundation G** can combine with RL/DAgger across manipulation and tracking (open cabinet, press button, throw-to-basket, etc.).

---

## 1. Foundation G — target form (beyond 2D lift)

Current S0/S1 uses **2D unit vector** in robot x–z (task-specific projection). Proposed canonical form:

```python
G_t = (d, m, ψ, κ)
  d ∈ ℝ³     # anchor-centric desired displacement direction (robot base frame)
  m ∈ ℝ⁺     # optional magnitude / progress
  ψ          # interaction primitive: reach, actuate (lift: contact_latched)
  κ          # constraint geometry: vertical_free | slide | revolute | normal_press
```

**Lift instance (active):** see [INTERACTION_GEOMETRY.md](INTERACTION_GEOMETRY.md) — ACTUATE = vertical above cube, not fixed xy goal.

Legacy note: earlier docs used `(τ, φ)`; **ψ** replaces task-specific phase naming for cross-task sharing; **κ** replaces implicit “which direction after grasp”.

| Level | Form | Role |
|-------|------|------|
| L0 | `d ∈ ℝ³` object vector | lift, push, throw, approach |
| L1 | dense `G: ℝ² → ℝ³` | occlusion, multi-object |
| L2 | `(Δp, ΔR)` / twist | cabinet, knob, oriented contact |

**Policy interface (target):** `π(o, M, G_3ch_broadcast_or_dense, τ_embed, φ)` — not the current 2D broadcast only.

**Migration path:** S1 lift ψ (REACH/ACTUATE) + κ=vertical_free → drawer/door/button κ → multi-task pretrain.

---

## 2. Cross-task generalization — what is shared

**Shared (one family G_θ):**

- Interface `(M, G, τ)`; visual backbone; fusion into policy obs (+ optional reward)
- Training signal: demo subgoal direction (supervised) or score field (TarGF-style)

**Must be conditioned (not one static G):**

- **τ / φ** — same visuals, different subgoal (EE→cube vs cube→goal)
- Contact timing, force, gripper — **π**, not G
- Multi-step horizons — sequence of subgoals / phase machine, not one vector for full task
- **M** from VFM/instruction (EVT line) — separate from G learner

**Task examples (same output space, different τ):**

| Task | Anchor M | τ sequence | G* intuition (3D) |
|------|----------|------------|-------------------|
| Lift | cube | reach→grasp→relocate | EE→cube → cube→goal |
| Throw to basket | object | grasp→throw | toward release + ballistic dir |
| Open cabinet | handle/door | pull→rotate | pull normal → hinge tangent |
| Press button | button | reach→press | EE→button → −normal |
| Track | target | track | toward view region (2D proj. of d) |

---

## 3. Known gaps in current lift implementation (inform future design)

Recorded from analysis — motivates explicit `(τ, φ)` and 3D, but **fix first on lift via Tier 0–1 validation**:

1. **Implicit phase switching** — labels are phase-aware (`subgoal`); network input/inference are not → ambiguity at grasp boundary.
2. **Goal not in mask** — only cube=white; goal invisible in segmentation; post-grasp `cube→goal` labels use proprio GT at train time.
3. **Global broadcast G** — loses spatial structure; may limit 2 cm precision.
4. **S1 0% ≠ hypothesis falsified** — confounds: stale G (0.72, 6ch, noisy demos), approach failure (`never_reached_cube`), resolution mismatch.

---

## 4. Large-scale training paradigms (candidates)

### A. Multi-task supervised G (extends current pipeline)

```text
Loss: Σ_tasks w_τ · (1 - cos(G_θ(M, depth, τ, φ), G*))
Data: multi-task sim demos + optional human video
```

Easiest bridge from `learn_guidance_field.py`.

### B. TarGF-style conditional score matching

Learn `s_θ(x, t, c) ≈ ∇_x log p(x | c)`; inference `G ∝ score` (project to ℝ³).

- TarGF already conditions on **category + geometry** ([`CondScoreModelGNN`](../../TarGF/networks/score_nets.py)); DualGF [`ScoreModelMLPImg`](../../DualGF/Algorithms/Navi/SDE.py) adds **image encoder**.
- Original TarGF: **terminal arrangement distribution** (low-dim state). We need **trajectory phase–indexed** or **instantaneous subgoal** fields on visual/anchor state.

### C. G pretrain + task RL finetune

```text
Stage 1  Foundation G_θ  (multi-task demo / DSM)
Stage 2  π_τ or π(o,M,G,τ)  (DAgger/RL per task or multi-task)
Stage 3  Small adapters (optional)
```

### D. G as obs + reward (HumanoidPF line)

`r_align = cos(v_anchor, G)` for contact-rich tasks.

---

## 5. Text embedding as condition (TarGF / score matching)

**Question:** Can text condition G for dynamic subgoal switching?

| Claim | Verdict |
|-------|---------|
| Score matching supports arbitrary condition `c` | **Yes** — conditional DSM is standard |
| Text embedding as `c` | **Yes** — replace TarGF `embed_category` with CLIP/T5; same loss |
| Text alone switches EE→cube→goal within one task | **Weak** — same sentence ambiguous pre/post grasp |
| Recommended condition | **`c = concat(e_text, embed(τ), φ)`** |

**Two ways to get dynamic subgoals via score matching:**

1. **Joint condition** — `p(x | e_text, τ, φ)`; group demos by (language, phase); DSM per group.
2. **Hierarchical language** — sub-instructions per phase ("move to cube" / "move cube to goal"); each defines a conditional distribution.

**EVT / HIEVT / TarGF stack:**

```text
text → M          (EVT: VFM segmentation)
text → region     (HIEVT: bbox goal)
text, τ, φ → G    (TarGF-style conditional score → 3D d)
π(o, M, G, τ)     (Isaac / offline RL)
```

**Classifier-free guidance** (optional at scale): train with random dropout of text/τ; infer with guidance scale `w`.

---

## 6. What G should not be expected to solve alone

- Full multi-step planning (open then pick) → skill graph / planner above G
- Raw language grounding without M → EVT/instruction module
- Contact force, gripper timing → π + reward / force sensing
- S2+ random goal without goal observability → goal region mask, bbox, or proprio to policy

---

## 7. Suggested timeline (after lift validation)

```text
[ACTIVE NOW]  Tier 0–2 S1 lift validation (supervised G, phase ablation)
[Later]       Lift 3D + explicit (τ, φ); policy 3ch G
[Later]       2–3 task sim multi-task supervised G (lift + push + button)
[Later]       Conditional DSM pilot on anchor pose or visual latent
[Later]       Text + τ joint condition; multi-task pretrain; paper-scale
```

**Do not open** until S1 exit: clean240 G retrain, phase-wise metrics, G-impl vs G-phase ablation, B1 vs B7 with phase-split eval.

---

## References

- [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md) — `(M, G)` contract
- [REFERENCES.md](REFERENCES.md) — TarGF, DualGF, HIEVT, EVT paths
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) — curriculum S0→S3 (active scope)

---

*Recorded: 2026-06-23 from design discussion (3D G, multi-task, TarGF + text conditioning).*
