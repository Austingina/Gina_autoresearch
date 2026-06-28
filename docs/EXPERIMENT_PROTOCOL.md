# Experiment Protocol — Two Steps: Learn Field, Then Policy

## Order (non-negotiable)

```text
Step 1  Learn G_θ from demonstration mask trajectories
        → field_alignment on val set

Step 2  Train π(o, M, G_θ); compare B1 vs B7
        → success_rate, post_grasp_wander_m

B3 (oracle rule field) = upper bound sanity check, NOT the main method
```

Representation is only complete when **both** steps succeed.

---

## Step 1 — Learn guidance field

### Supervision signal (S1: `subgoal` + Interaction Geometry)

Phase-aware **ψ** labels from demo proprio (not available to student at test):

```python
# ψ=REACH (phase < 0.5)
G*_REACH = normalize(cube - ee)                    # ℝ³

# ψ=ACTUATE (latched) — lift κ=vertical_free
attractor = (cube_x, cube_y, cube_z + Δz)          # default Δz=0.20 m
G*_ACTUATE = normalize(attractor - cube)           # ≈ (0, 0, 1)
```

Do **not** use fixed hidden xy command goal for ACTUATE (hurts convergence; goal not in mask).

S0 ablation `state_goal`: post-grasp only, cube→goal in x-z (legacy).

```bash
python scripts/learn_guidance_field.py --task lift --stage s0_fixed --supervision state_goal
```

### Data sources

| Task | Source | Path hint |
|------|--------|-----------|
| lift | **Copied** Isaac maskdepth HDF5 | `AutoResearch/data/lift/<stage>/demos/` |
| lift | DAgger aggregate ep | successful rollouts with mask+depth |
| track | HIEVT `.pt` | mask + goal_bbox trajectories |

### Script

```bash
python scripts/import_isaac_demos.py --source $ISAACLAB_ROOT/datasets/<file>.hdf5 --stage s0_fixed
python scripts/learn_guidance_field.py --task lift --stage s0_fixed
```

### Metrics (Step 1 ratchet)

| Metric | Target |
|--------|--------|
| `field_alignment` | cosine(G_θ, G*) — **primary for Step 1** |
| `field_mse` | optional |
| phase-wise error | pre_grasp vs post_grasp (lift) |

### Exit criterion

Val `field_alignment` ≥ 0.75 on lift post-grasp phase before Step 2.

---

## Step 2 — Policy learning with learned G

Fixed: same π architecture, DAgger/RL hyperparameters.

| ID | guidance_mode | When |
|----|---------------|------|
| B1 | `anchor_only` | baseline |
| B7 | `anchor_field_demo` | **main** — uses G_θ from Step 1 |
| B3 | `anchor_field_rule` | oracle ceiling |

### Metrics (Step 2 ratchet)

| Metric | Notes |
|--------|-------|
| `success_rate` | primary |
| `post_grasp_wander_m` | lift killer metric |
| `sample_efficiency` | optional |

### Success criteria

- B7 success >> B1
- B7 wander << B1
- B7 within ~90% of B3 (oracle) — proves learned G is sufficient

---

## Why not start with B3 (oracle field)?

Oracle G proves **mechanism** in one afternoon but does not prove:

### Field learning references (Step 1)

| Source | What to borrow |
|--------|----------------|
| **TarGF** `score_matching/` | Score-matching loss for gradient field; `--mode train_gf` |
| **DualGF** `NaviSDEImg.py` | Image-rendered state → field (closest visual analogue) |
| **Our S0 minimal** | `G*_t = normalize(centroid(M_{t+1}) - centroid(M_t))` on Isaac mask demos |

TarGF/DualGF train on **state/particle** or **rendered image** distributions; we start with **2D mask centroid flow** on fixed-cube demos, then scale toward score-matching if needed.
- G is not task-specific engineering
- Paper contribution is a **representation**, not a sim cheat

Use B3 only to debug policy pipeline before G_θ is ready.

---

## Paper figure

1. Step 1: learned field visualization on lift demos
2. Step 2: B1 vs B7 vs B3 bars (success + wander)
