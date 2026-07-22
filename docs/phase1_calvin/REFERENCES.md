# References & Codebases — What to Read and Reuse

One-page map: **paper idea → our (M,G) role → local path**.

---

## 1. Core research question

> What visual representation `g_t` makes **policy learning** easier across tracking and manipulation?

```python
a_t = pi(o_t, M_t, G_t)    # M = anchor mask, G = guidance field
```

---

## 2. Our lineage (must cite + reuse code)

| Work | Paper / link | What they do | We reuse | Local path |
|------|--------------|--------------|----------|------------|
| **EVT** | [ECCV 2024](https://arxiv.org/abs/2404.09857) | VFM text-mask + offline RL tracking | **M** extraction (segmentation anchor) | `$EVT_ROOT` = `/home/l/code/Offline_RL_Active_Tracking` |
| **HIEVT** | OpenReview / your submission | Instruction → bbox goal + mask IoU RL | **G** as region-pull; tracking Step 1/2 template | `$HIEVT_ROOT` = `/home/l/code/Hierarchical-Instruction-aware-Embodied-Visual-Tracking-master` |
| **TarGF** | [NeurIPS 2022](https://openreview.net/forum?id=Euv1nXN98P3) · [GitHub](https://github.com/AaronAnima/TarGF) | Learn **target gradient field** from examples via score-matching; no explicit goal | **Step 1** objective + score network training loop | `$TARGF_ROOT` → clone [AaronAnima/TarGF](https://github.com/AaronAnima/TarGF) (default `/home/l/code/TarGF`) |
| **DualGF** | [OpenReview](https://openreview.net/forum?id=nVYND1kLOug) | **Target + support** gradient fields; nav, tracking, rearrange | Step 1 dual-field ablation; image-based navi field (`NaviSDEImg`) | `$DUALGF_ROOT` = `/home/l/code/DualGF` |
| **Isaac lift (ours)** | `vision_motor_control_exp.md` | Vision DAgger 8ch mask+depth; proprio 100% lift | **Step 2** policy; B1 failure = motivation | `$ISAACLAB_ROOT` |

### HIEVT key files

| File | Use |
|------|-----|
| `train_offline_goal_conditioned_bbox.py` | Mask+bbox channel policy, IoU reward |
| `DataCollection_Goal_condition.py` | Demo format: `goal`, `goal_bbox` |
| `agent_CNN_LSTM_GoalCondition_bbox.py` | CQL-SAC goal-conditioned agent |

### Isaac lift key files

| File | Use |
|------|-----|
| `scripts/reinforcement_learning/train_lift_cube_dagger_vision_binary_gripper.py` | Step 2 student training |
| `scripts/tools/record_oracle_goal_conditioned_demos.py` | Step 1 demo HDF5 (`goal_position`, phases) |
| `source/.../lift/mdp/teacher_student/vision_observations.py` | Mask+depth **M** |
| `source/.../lift/mdp/teacher_student/eval.py` | Eval + wander metrics (wire) |
| `vision_motor_control_exp.md` | Current status, ablations, DR flags |

### TarGF key files ([AaronAnima/TarGF](https://github.com/AaronAnima/TarGF))

| File / dir | Use |
|------------|-----|
| `main.py --mode train_gf` | Train target score / gradient field network |
| `score_matching/` | Score-matching loss — **primary Step 1 reference** |
| `networks/` | Score network architectures |
| `configs/targf_*.py` | Task configs (ball rearrange, room) |
| `planners/` | ORCA / gradient-based action from learned field |
| `main.py --mode train_rl` | Optional: RL with TarGF reward (contrast; we use DAgger) |

```bash
# Clone if not present
git clone https://github.com/AaronAnima/TarGF $TARGF_ROOT
python main.py --config configs/targfballbase.py --workdir Ball_Score --mode train_gf
```

**Mapping to us:** TarGF learns G from demo **state** distributions; we learn G_θ from demo **mask centroid flow** `G*_t = norm(c(M_{t+1}) - c(M_t))` — same “field without explicit terminal goal” idea, simpler 2D visual supervision for S0.

### DualGF key files (`$DUALGF_ROOT`)

| File / dir | Use |
|------------|-----|
| `sde_run.sh` | Train target (`--score_mode tar`) or support (`sup`) fields |
| `dual_run.sh` | Plan / eval with dual fields |
| `Runners/BallRunners/BallSDE.py` | Ball rearrange — target field training |
| `Runners/NaviRunners/NaviSDE.py` | Navigation target field |
| `Runners/NaviRunners/NaviSDEImg.py` | **Image-based** navi field — closest to our visual G |
| `Runners/NaviRunners/NaviEvalPolicyImg.py` | Policy eval with image field |
| `Algorithms/*/SDE.py` | Core score-matching trainers |
| `Algorithms/Maze/DualPRM.py` | Dual-field planning (target + support) |

**Mapping to us:** DualGF **target field** ≈ our G (toward goal distribution); **support field** = optional Month 2+ for obstacles / clutter. For S0 lift, start with target-field only (TarGF-style); add support field only if needed.

---

## 3. Related (contrast / optional ablation)

| Work | Link | Relation to us |
|------|------|----------------|
| **Mask World Model** | [arXiv 2604.19683](https://arxiv.org/html/2604.19683v1) | Future mask dynamics → **optional B8** field learner, not our interface |
| **3DFlowAction** | [arXiv 2506.06199](https://arxiv.org/abs/2506.06199) | 3D object-flow world model for cross-embodiment manipulation; strong boundary for "G as flow" story |
| **Im2Flow2Act** | [project](https://im-flow-act.github.io/) | 2D object flow as cross-domain manipulation interface from human videos + robot play |
| **HumanoidPF** | [arXiv 2601.16035](https://arxiv.org/html/2601.16035) | Field as RL obs+reward; navigation traversal |
| **Giving Robots a Hand** | [project](https://giving-robots-a-hand.github.io/) | Human video for post-grasp phases; demo data for **G** |
| **GeCCo / Contact** | [arXiv 2509.17582](https://arxiv.org/html/2509.17582) | Contact goals — future loco-manip, not Month 1 |
| **CLIPort / Transporter** | CoRL 2021 | Heatmap goals — manipulation only baseline |
| **VLAs** | RT-2, OpenVLA | End-to-end contrast; we decouple **M,G** from π |

---

## 4. Autoresearch pattern

| Resource | Use |
|----------|-----|
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | `program.md` + ratchet loop |
| `unrealzoo-gym/.../VLN_AUTORESEARCH.md` | Multi-step flywheel reference (eval→train→compare) |

---

## 5. What NOT to confuse

| Their focus | Our focus |
|-------------|-----------|
| Better manipulation policy | Better **policy-facing visual goal** `g_t` |
| World model SOTA | Learnable **G_θ** from demos |
| Unified VLA | Unified **(M, G)** across track + lift |
| Affordance / contact point | **Visual evolution** of anchor entity |
