# Two-Month Roadmap (aligned with EXPERIMENT_PLAN.md)

> **Publication strategy → [PAPER_ROADMAP_STRATEGY.md](PAPER_ROADMAP_STRATEGY.md).** Stage 1 (mask + direction field) is **not** a standalone paper. **Paper 1 ships after Stage 2 (Interaction Geometry, ≥2 tasks).** The two-month plan below is execution detail for finishing the Stage 1 foundation experiment.

**Claim:** Learnable anchor-conditioned guidance field `G_θ` improves policy learning vs mask-only; generalizes under curriculum S0→S1→S2.

## Month 1

| Week | Stage | Step | Deliverable |
|------|-------|------|-------------|
| 1 | **S0 fixed** | 1 | Learn G_θ; val `field_alignment` ≥ 0.75 |
| 2 | **S0 fixed** | 2 | B7 vs B1; post-grasp wander ↓ |
| 3 | **S1 random cube** | 1+2 | Field + policy generalize to new xy |
| 4 | **S2 random goal** | 1+2 | Field + policy with random lift target |

## Month 2

| Week | Stage | Deliverable |
|------|-------|-------------|
| 5 | HIEVT T0 | Same (M,G) on tracking |
| 6 | S3 OOD object | Color/size transfer |
| 7 | Ablations | B3 ceiling; optional B8 |
| 8 | Paper | Draft + figures |

## Exit criteria (project)

- [ ] S0: Step 1 + Step 2 pass (see EXPERIMENT_PLAN.md)
- [ ] S1: B7 within 85% of S0 success at new cube positions
- [ ] S2: B7 still >> B1
- [ ] HIEVT: field interface matches or beats bbox channel baseline

## Out of scope

- Full MWM training as main method
- Pull/push envs (supplement)
- Unified loco-manipulation demo
