# Interaction Geometry — G Conditioning & Cross-Task Plan

**Status:** Active design. **Lift** is the first concrete instance; drawer / door / button follow the same `(ψ, κ)` interface.

---

## Claim (next stage)

Lift, drawer, door, and button share one **Interaction Geometry**:

```text
approach (REACH) → engage contact → actuate along constraint (ACTUATE)
```

The G state manifold is similar across tasks: **direction changes with ψ and κ**, not with task name alone.

---

## Condition: `(ψ, κ)` not task ID

| Symbol | Meaning | Lift instance |
|--------|---------|---------------|
| **ψ** | Interaction primitive | `REACH` (EE→contact) / `ACTUATE` (move anchor) |
| **κ** | Constraint geometry | `vertical_free` — straight up above cube |
| **d** | G output | Unit vector in robot base frame (ℝ³) |

**Avoid** task one-hot as the primary condition — it does not prove geometric sharing.

**Lift ψ mapping (current implementation):**

| ψ | Contact latch | G* (attractor form) |
|---|---------------|---------------------|
| `REACH` | not latched | `normalize(cube − ee)` |
| `ACTUATE` | latched | `normalize(attractor − cube)` where attractor = **cube xy + (cube_z + Δz)** |

### Why lift ACTUATE goal is “above cube”, not a hidden xy goal

Previous setup: post-grasp G* = `cube → fixed_command_goal` (e.g. `(0.5, 0, 0.375)`).

Problems:

- Goal is **invisible in mask** — student must infer relocation from proprio at train time only.
- Horizontal relocation to a fixed point **increases wander and convergence difficulty** (S0 `lifted_not_at_goal`).
- Not aligned with drawer/door/button where ACTUATE is **along a local constraint**, not “go to a far point”.

**New lift ACTUATE rule (κ = vertical_free):**

```python
attractor = (cube_x, cube_y, cube_z + Δz)   # default Δz = 0.20 m
G*_ACTUATE = normalize(attractor − cube)   # ≈ (0, 0, 1) in base frame
```

Episode success: cube **lifted** to target height above table, at **same xy as spawn** (within threshold). Command `object_pose` is synced to attractor on env reset when `lift_goal_above_cube=True`.

---

## G learning plan (lift first)

### Step 1 — `G_θ(visual, ψ)`

| Item | Spec |
|------|------|
| Input | mask+depth (8ch), 240px |
| Condition | `ψ` via `contact_latched` broadcast (1ch) — lift special case of REACH/ACTUATE |
| Output | `d ∈ ℝ³` (upgrade from 2D x-z) |
| Label | `goals/interaction_geometry.py` attractor rules (ignore stale command in old HDF5 for ACTUATE) |
| Metrics | pre-grasp align, post-grasp (vertical) align, optional buffer OOD align |
| Data | clean240 demos + later DAgger buffer mix |

### Step 2 — policy

- B3: oracle `(ψ, κ)` via `OracleRuleGuidanceField` + phase reset fixes
- B7: learned `G_θ` + same ψ gating
- Success: grasp + lift to height (not relocate-to-marker)

### Migration to multi-task

| Task | REACH attractor | ACTUATE κ | κ vector |
|------|-----------------|-----------|----------|
| Lift | cube center | `vertical_free` | +z |
| Drawer | handle | `slide` | joint slide axis |
| Door | handle | `revolute` | hinge tangent |
| Button | button center | `normal_press` | −surface normal |

```text
Loss: Σ_tasks w · (1 − cos(G_θ(visual, ψ, κ), G*))
```

Text / language → future `concat(e_text, embed(ψ), κ)` — not required for lift pass.

---

## Ablations (Interaction Geometry stage)

1. `G(visual)` — no ψ
2. `G(visual, grasp_latched)` — lift ψ only
3. `G(visual, ψ, κ)` — target
4. `G(visual, task_onehot)` — negative control (should not beat ψ,κ for transfer claim)

---

## Code map

| Piece | Path |
|-------|------|
| Attractor / label rules | `goals/interaction_geometry.py` |
| Step 1 dataset labels | `goals/field_data.py` → `subgoal` uses interaction_geometry |
| Oracle B3 runtime G | `IsaacLab/.../guidance_field_bridge.py` |
| Goal sync on reset | `IsaacLab/.../mdp/interaction_geometry.py` → `sync_lift_goal_above_cube` |
| Env flag | `apply_lift_teacher_student_env_randomization(..., lift_goal_above_cube=True)` |

---

## References

- [FUTURE_RESEARCH_ROUTES.md](FUTURE_RESEARCH_ROUTES.md) — Foundation G, TarGF, multi-task timeline
- [findings.md](../findings.md) — S1 active checklist
- [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) — Step 1 / Step 2 order

*Recorded: 2026-06-25 — (ψ, κ) conditioning, lift vertical ACTUATE goal.*
