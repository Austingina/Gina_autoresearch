# Experiment Plan — Curriculum from Easy to Hard

**Principle:** Fix variables first, prove Step 1 → Step 2 on the easiest setting, then add randomization only after the previous stage passes exit criteria.

---

## North star

| # | Question | How we answer |
|---|----------|---------------|
| Q1 | Can **G_θ** be learned from demo mask trajectories? | Step 1: `field_alignment` on val |
| Q2 | Does **(M, G_θ)** help policy vs **M** alone? | Step 2: B7 vs B1, `success_rate` + `post_grasp_wander_m` |
| Q3 | Does **G_θ** generalize when cube layout changes? | Curriculum S0→S1→S2 |
| Q4 | Same interface on **tracking**? | HIEVT parallel track |

---

## Curriculum stages (Isaac lift)

| Stage | ID | Cube position | Lift goal | DR / notes | Exit before next stage |
|-------|-----|---------------|-----------|------------|------------------------|
| **S0** | `s0_fixed` | **Fixed** (default spawn) | **Fixed** `(0.5, 0, 0.375)` | No randomization | Step1 val align ≥0.75; B7 lift SR > B1 +15%; wander ↓ |
| **S1** | `s1_random_cube` | **Random** xy on table | Fixed lift goal | `randomize_cube_spawn=True` | Field align ≥0.70; B7 within 85% of S0 B7 SR |
| **S2** | `s2_random_goal` | Random xy | **Random** lift goal in range | + `randomize_lift_goal=True` | B7 still >> B1 |
| **S3** | `ood_object` | Random + **novel color/size** | Random | Asset / material swap | Supplement / Month 2 |

Isaac flags (see `train_lift_cube_dagger_vision_binary_gripper.py` / env cfg):

```bash
# S0 — default, no extra flags
# S1 — random cube
--randomize_cube_spawn   # or env equivalent
# S2 — random cube + goal
--randomize_lift_goal
```

**Rule:** Do not open S1 until S0 Step 1+2 pass. Do not open S2 until S1 field generalization passes.

---

## Two-step pipeline (every stage)

```text
┌─────────────────────────────────────────────────────────┐
│ Step 1  Learn G_θ on demos (train split of stage)       │
│         scripts/learn_guidance_field.py                 │
│         OUT: checkpoints/guidance_field_{stage}.pt      │
│         METRIC: field_alignment (val)                   │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2  Train/eval π(o, M, G_θ)  —  B1 vs B7 (+ B3 UB) │
│         Isaac vision DAgger + inject G channels         │
│         METRIC: success_rate, post_grasp_wander_m       │
└─────────────────────────────────────────────────────────┘
```

---

## Stage S0 (Week 1–2) — **START HERE**

### S0-Step 1: Learn field

| Item | Detail |
|------|--------|
| Data | Isaac oracle HDF5 **copied** to `data/lift/s0_fixed/demos/` | Import or `collect_lift_demos.sh` |
| Supervision | `G*_t = norm(centroid(M_{t+1}) - centroid(M_t))`; emphasize post-grasp upward |
| Model | `FieldPredictor` in `experiment.py` (agent tunes) |
| Val | 20% held-out episodes |
| **Pass** | `field_alignment` ≥ **0.75** on post-grasp timesteps |

```bash
./scripts/init_workspace.sh
python scripts/import_isaac_demos.py \
  --source $ISAACLAB_ROOT/datasets/lift_cube_lenient_teacher_maskdepth_smoke.hdf5 \
  --stage s0_fixed

python scripts/learn_guidance_field.py \
  --task lift --stage s0_fixed
# output: checkpoints/lift/guidance_field_s0_fixed.pt
```

### S0-Step 2: Policy

| Baseline | `guidance_mode` | Purpose |
|----------|-----------------|---------|
| B1 | `anchor_only` | No G — expect grasp OK, wander high |
| B7 | `anchor_field_demo` | Load `guidance_field_s0.pt` |
| B3 | `anchor_field_rule` | Oracle ceiling only |

| **Pass** | B7 success − B1 ≥ **15%**; B7 `post_grasp_wander_m` ≤ **50%** of B1 |

### S0 debug shortcut (optional, not paper)

- Run B3 once to verify policy pipeline accepts G channels before B7 converges.

---

## Stage S1 (Week 3) — Random cube position

| Change | Field learner | Policy |
|--------|---------------|--------|
| Train demos include random cube xy | Retrain G_θ on S1 demos | Re-run B1 vs B7 same architecture |
| Test on **held-out** cube positions | `field_alignment` on val | SR + wander |

**Hypothesis:** G_θ encodes **object-centric upward / toward-goal** direction, not memorized pixel location.

**Fail action:** More demo diversity or phase-aware field (pre_grasp vs post_grasp heads).

---

## Stage S2 (Week 4) — Random cube + random lift goal

Field must condition on **goal context** (e.g. lift target in proprio for teacher only → student gets G_θ that points toward current goal region).

**Pass:** B7 >> B1 on SR; paper table row for generalization.

---

## HIEVT track (parallel after S0-Step 1 template works)

| Stage | Setting | Step 1 | Step 2 |
|-------|---------|--------|--------|
| T0 | Fixed `(angle, distance)` grid | Learn region-pull G from bbox demos | Offline RL vs bbox-only (HIEVT baseline) |
| T1 | Random instruction / bbox | Generalize G_θ | Same |

Repo: `$HIEVT_ROOT/train_offline_goal_conditioned_bbox.py`

---

## Baselines summary

| ID | When | Role |
|----|------|------|
| B0 | Optional | No M, no G |
| B1 | Every stage | **Main negative** |
| B3 | Debug / ceiling | Oracle G, not main method |
| B7 | Every stage | **Main method** |
| B8 | Month 2 optional | WM-derived G |

---

## Metrics cheat sheet

| Metric | Step | Stage |
|--------|------|-------|
| `field_alignment` | 1 | All |
| `success_rate` | 2 | All |
| `post_grasp_wander_m` | 2 | S0+ (lift) |
| `grasp_success_rate` | 2 | Ensure B7 doesn't hurt grasp |

---

## Paper figures (target)

1. **(M, G)** schematic
2. S0: learned G visualization on lift demo
3. **Main:** B1 vs B7 vs B3 bars @ S0, then S1, S2 (generalization curve)
4. HIEVT T0: bbox vs field interface

---

## Timeline (8 weeks)

| Week | Stage | Deliverable |
|------|-------|-------------|
| 1 | S0 Step 1 | `guidance_field_s0.pt`, field align ≥0.75 |
| 2 | S0 Step 2 | B1 vs B7 pass; wander plot |
| 3 | S1 | Random cube field + policy |
| 4 | S2 | Random goal |
| 5 | HIEVT T0 | Tracking unified (M,G) |
| 6 | S3 / OOD | Object appearance |
| 7 | Ablations | B3 ceiling, B8 optional |
| 8 | Paper | Draft |
