# Paper Strategy & Stage Roadmap

**Decision (2026-06-25):** Stage 1 (mask + direction field on lift) is **NOT a standalone paper**. It is the foundation experiment. **Paper 1 ships after Stage 2 (Interaction Geometry across ≥2 tasks).**

Target horizon: 6–12 months, aiming CoRL / RSS / ICRA / IROS (Paper 1 ≥ CoRL/RSS competitive).

---

## Why not publish Stage 1 alone

Reviewers in the 2025–2026 foundation-model era weigh **pretraining scale, OOD, cross-task, world models** over single-task policy tricks. `Mask + direction vector` on lift alone reads as **task-specific engineering** ("why not goal position / language / VIP?") → at best Workshop / mid ICRA-IROS.

Stage 1 only becomes a contribution when framed as the **lift instance of a shared Interaction Geometry** (Stage 2).

---

## Five-stage research arc

| Stage | Representation | Story | Standalone venue (if forced) |
|-------|----------------|-------|------------------------------|
| 1. Direction field G | mask + `d ∈ ℝ³` | state-evolution signal helps manipulation | Poster / mid (—) |
| **2. Interaction Geometry** | shared `(ψ, κ)`: REACH→ACTUATE, κ=slide/revolute/normal | tasks share interaction structure, not goals | **CoRL Oral / RSS** |
| 3. Phase-aware G | unified task progression / skill structure | the hard part is *when to switch*, not *where to go* | HRL / long-horizon |
| 4. Universal Goal Space | Push / Pull / Rotate / Relocate composable | learn shared structure → skill token / robot language | Foundation robotics |
| 5. Progress Field (VIP-style) | learn *what state is better*, not *where next*; weak/no task labels | task-agnostic progress from video; bridges VLA + WM | **Highest ceiling** |

Tension: Stages 1–2 (direction field) need κ labels (door axis, drawer rail, button normal). Stage 5 (progress field) drops task labels → fits foundation-model era best, but is a different data/engineering scale.

---

## Publication plan (revised)

### Paper 1 — Interaction Geometry  *(first submission)*
- **Includes** Stage 1 as the lift instance + Stage 2 cross-task evidence.
- Claim: Lift + ≥1 articulated task (drawer **or** button) share `G_θ(visual, ψ, κ)`; `(ψ, κ)` beats task-onehot for transfer.
- Ships only after Stage 2 cross-task ablation passes.
- Target: **CoRL Oral / RSS**.

### Paper 2 — Phase-aware / Universal Goal Space
- Stages 3–4: unify task progression; composable Push/Pull/Rotate/Relocate goal space.

### Paper 3 — Progress Field  *(biggest bet)*
- Stage 5: learn "next better state" from large video, task-agnostic. VIP-inspired.
- Positioned between VLA (learns actions) and World Models (learns future pixels): **learns task progress itself**.

---

## Current execution focus → Stage 1 closure (feeds Paper 1)

Stage 1 is a **prerequisite milestone, not a deliverable paper**. Finish it cleanly so Stage 2 has a trustworthy baseline.

Remaining Stage 1 experiments (active sprint):

1. **B3 fresh restart** — eval phase-reset fixes + `lift_goal_above_cube` (vertical ACTUATE). Confirm grasp>0 at first evals and adaptive β decays after warmup.
2. **Step 1 G_θ retrain on clean240** — 3D output + vertical ACTUATE labels; report REACH / ACTUATE align split.
3. **G-phase** — add ψ (`contact_latched`) conditioning to `G_θ`; ablate vs G-impl (no ψ).
4. **Step 2 @ S1** — B1 vs B7-impl vs B7-phase vs B3; split eval: approach (grasp) vs lift height (not xy wander).

Explicitly deferred during this sprint: drawer/door/button envs, TarGF, text condition, world model, progress field. Hooks kept in interface (`G` swappable for progress `P`; `ψ` stays a condition).

---

## References
- [INTERACTION_GEOMETRY.md](INTERACTION_GEOMETRY.md) — (ψ, κ) design, lift instance
- [FUTURE_RESEARCH_ROUTES.md](FUTURE_RESEARCH_ROUTES.md) — Foundation G, TarGF, progress field
- [findings.md](../findings.md) — Stage 1 checklist + run log

*Recorded 2026-06-25 from expert strategy discussion (5-stage arc; Paper 1 = Stage 2).*
