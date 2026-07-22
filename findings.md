# Autoresearch Findings Log

## Data policy

All demos: `data/lift/<stage>/demos/` — see `docs/DATA_ISOLATION.md`.

**Large artifacts** live under `/home/l/code/data/AutoResearch_Data` (see `ARCHIVE_MANIFEST.md` there). Repo uses symlinks for active HDF5; DAgger replay buffers are archived — only `student_best.pt` (~4 MB) kept for eval.

## Current stage: **S1 random cube** (S0 soft-closed)

## S0 exit checklist (soft pass — relocation validated)

### Step 2 (policy B7 vs B1 @ S0)
- [x] B1/B7 DAgger 100 iters + diagnostic eval
- [x] B7 wander 10.4→6.9 cm; final_dist 3–7 cm vs B1 11–36 cm
- [ ] B7 success@2cm − B1 ≥ 15% (strict metric not met; proceed with aux @5cm)

## Active plan — S1 validation tiers (do this first)

**Scope:** Supervised `MaskFieldCNN` + lift only. **Interaction Geometry** design → [docs/INTERACTION_GEOMETRY.md](docs/INTERACTION_GEOMETRY.md). Multi-task / TarGF deferred until S1 passes.

| Tier | Goal | Key actions |
|------|------|-------------|
| **0** | Clean Step 1 | clean240 + 8ch + `subgoal` (ψ REACH/ACTUATE), lazy loader; 3D G + vertical ACTUATE; phase-wise align |
| **1** | Phase ablation | **G-impl** vs **G-phase** (+ψ/`contact_latched` input); compare pre/post-grasp align |
| **2** | Step 2 | B1 vs B7-impl vs B7-phase @ S1; split eval: approach (grasp) vs lift height (not xy wander) |

**S1 Step 1 pass (proposed):** val align ≥ 0.70; ACTUATE (vertical) ≥ 0.75; REACH ≥ 0.65.

**Deferred:** [docs/FUTURE_RESEARCH_ROUTES.md](docs/FUTURE_RESEARCH_ROUTES.md) — drawer/door/button κ, multi-task G, TarGF.

### Task base update (2026-06-28) — infra back on IsaacLab

**Available tasks:** `lift cube`, `open drawer`, `open door`. **`button` dropped** (`normal_press` κ is contact/force-heavy, weak transfer — out of scope for now).

**Revolute deferred:** current `open door` implementation **pulls in a straight line, not an arc** → it is *not* genuinely revolute yet. True rotational scenes = future work (see FUTURE_RESEARCH_ROUTES). Keep door tagged as `κ=slide` now, **upgradable→revolute** later without re-collecting (object identity unchanged).

**Geometry axis right now = 2 classes:**
- `free-translate` : lift cube (vertical_free)
- `constrained-straight-pull` : open door + open drawer (both straight-line → **same κ**)

**Implication for transfer story:**
- `door ↔ drawer` = clean **same-geometry, different-object positive control** (was confounded when I assumed door=revolute; now clean).
- `lift ↔ open` = cross-class: expect **REACH shared / ACTUATE diverges** (vertical-free vs straight-pull) → needs (ψ,κ) conditioning.
- Weakest part of the thesis (rotation-vs-translation) is exactly the missing revolute → revolute is the eventual upgrade that makes the claim strong.

**Immediate plan = representation-level (open-loop) transfer first** — does NOT depend on the still-imperfect lift closed-loop (B7 not yet > B1). See "Immediate transfer sprint" below.

---

## S1 exit checklist

### Data
- [x] **clean240** collect — 100 ep, 100% success, no teacher noise, topdown_close, 240×240 mask+depth
  - Archive: `AutoResearch_Data/data/lift/s1_random_cube/s1_random_cube_20260623_clean240.hdf5` (17 GB)
  - Symlink: `data/lift/s1_random_cube/demos_clean240/`
- [x] Old noisy 4 GB demos → archived (`20260622_170659`)

### Step 1 (learn G_θ, ψ-conditioned `subgoal`)

- [ ] Val `field_alignment` ≥ **0.70** on **clean240** (retrain pending — prior 0.723 was noisy 84px mask-only)
- Input: mask+depth (8ch, 240px) + **ψ** via `contact_latched` (planned for G-phase)
- Label (**Interaction Geometry lift instance**):
  - **ψ=REACH** (phase < 0.5): `normalize(cube − ee)` in ℝ³
  - **ψ=ACTUATE** (latched): `normalize(attractor − cube)` where attractor = **cube xy + Δz** (default Δz=0.20 m) — **not** fixed hidden xy goal
- Implementation: `goals/interaction_geometry.py`; old HDF5 `student_state[7:10]` command ignored for ACTUATE labels
- **OOM note:** lazy HDF5 loader only on clean240

### OOM incident log + launch guardrails (for future runs)

Observed OOM/system-kill cases:
- **Field training preload OOM**: `scripts/learn_guidance_field.py` with full-RAM preload on clean240 hit ~80-84 GB RSS and was killed.
- **B7 DAgger run killed**: `runs/s1_B7_cam240_zonly_adapt_20260624_093352/launch.log` ended with `./isaaclab.sh: line 561: ... Killed` at `DAgger 14/2000`.
- **B7 DAgger run killed**: `runs/s1_B7_cam240_xy_adapt_20260624_095325/launch.log` ended with `./isaaclab.sh: line 561: ... Killed` at `DAgger 14/2000`.
- **B7 DAgger run killed (latest)**: `runs/s1_B7_cam240_xy_adapt_20260624_100643/launch.log` ended with `./isaaclab.sh: line 561: ... Killed` at `DAgger 14/2000`.
- **B7 DAgger run killed (safe profile)**: `runs/s1_B7_cam240_xy_adapt_20260624_102028/launch.log` ended with `./isaaclab.sh: line 561: ... Killed` at `DAgger 69/2000`.

What looked risky before kills:
- Machine state in logs repeatedly showed swap almost exhausted (`Free Page/Swap` around `1-4 MB` out of `2047 MB`) before/around startup.
- Even with lighter safe profile, startup still showed low swap headroom (`Free Page/Swap: 18 MB / 2047 MB`) and later hit kill at iter 69.
- Kills happened after training started normally (iter 1-14), so these were not config-parse failures.
- For B7 at kill time, aggregate buffer was still small (around 16 episodes), so these events are unlikely caused by replay cap alone.

Default anti-OOM launch profile (prefer this first, then scale up):
- Keep **single Isaac process** only (do not run concurrent train/play/eval jobs).
- Use smaller rollout load first: `--num_envs 4` (or 6 max) and `--collect_steps 128`.
- Keep vision load aligned but avoid extra pressure: `--camera_height 240 --camera_width 240 --image_size 240 --topdown_close`.
- Keep eval lighter during bring-up: `--num_eval_envs 2 --eval_episodes 3` (restore 4/5 after stability).
- Keep field training in **lazy HDF5** mode; do not enable full preload on clean240.

Scale-up policy:
- Only increase from `4 -> 6 -> 8` envs after a run survives at least to first eval (`iter 25`) without memory kill.
- If any `Killed` recurs, immediately step back one notch (envs/collect_steps/eval load) before retrying.
- **2026-06-24:** swap expanded to **48 GB**; B3 with disk aggregate backend survived **666/2000 iters** without kill (see B3 section below).

### Step 2 B3 — oracle GT 3D G (`anchor_field_rule`, stopped @ iter 666)

**Run:** `runs/s1_B3_cam240_xy_gt3d_20260624_155743/` — user stopped manually.

**Setup:** Same safe profile as B7 fresh (240px topdown_close, xy cube DR, adaptive β, disk buffer) but **no G_θ checkpoint** — visual = mask+depth **+ 3ch GT 3D rule field** (EE→cube pre-grasp, cube→goal post-grasp). B3 = oracle ceiling to verify the policy pipeline accepts G channels.

| Signal | Value | Note |
|--------|-------|------|
| Stopped at | iter **666** / 2000 | ~17.4 h |
| Training β | **1.0 entire run** | adaptive β never decayed |
| Buffer | 244 ep / 276k steps, **88 teacher-success ep** | rollout collection OK |
| Training eval (iter 25→625) | **success=0%, grasp=0%**, all `never_reached_cube` | **broken metric**, not policy ceiling |
| Play @ `student_best.pt` (post phase-reset fix) | grasp **OK** most eps; **4/12 lift success** (ep 4,7,9,10) | student *can* approach + grasp + relocate |

**Root cause — teacher phase latch not reset on eval/play episode boundaries:**
- `infer_teacher_phase()` latches LIFT after grasp; GT G then points cube→goal instead of EE→cube.
- Training **rollout** already reset on done; **periodic eval** and **play** did not → eval envs entered with stale LIFT → `never_reached_cube` even when student improved.
- **Adaptive β** uses eval success EMA → `ema=0` → `β≈1.0` for all 666 iters → student never executed during training despite low loss (~25).

**Fixes (IsaacLab, restart required):**
1. `eval.py` — `reset_teacher_phase_state(env, done_ids)` on episode done (all eval paths).
2. `play_lift_cube_student_vision_binary_gripper.py` — same on episode done.
3. `train_lift_cube_dagger_vision_binary_gripper.py` — `reset_teacher_phase_state(env)` **before** periodic eval.

**Interpretation (does accurate G → grasp + lift?):** **Yes, with the right caveats.**
- With **oracle GT G** and correct phase gating, π can learn the full chain: approach cube → grasp → lift/relocate to goal (play + teacher buffer success ep support this).
- B3 does **not** prove learned G_θ (B7) is sufficient — only that the **DAgger + vision + G-channel pipeline** works when G is perfect.
- This run **under-trained the student** (β=1.0) because eval fed false zeros into adaptive β; restart with fixes before comparing B3 ceiling vs B7.

**Next:** Fresh B3 restart (`run_s1_b3_fresh.sh`) with eval fixes + **lift_goal_above_cube** (vertical ACTUATE).

### Step 2 B3 — vertical ACTUATE restart (`anchor_field_rule`, stopped @ iter 200)

**Run:** `runs/s1_B3_cam240_xy_gt3d_20260625_100836/` — stopped after `student_iter_200.pt` + Eval 200 were written.

**Setup:** Same safe profile, but with Interaction Geometry lift goal: `lift_goal_above_cube=True`, κ=`vertical_free`, ACTUATE G≈(0,0,1). Eval/play phase reset fixes active.

| Signal | Value | Note |
|--------|-------|------|
| Stopped at | iter **200** / 2000 | checkpoint + buffer saved |
| Buffer | 80 ep / 91.2k steps, **29 teacher-success ep** | rollout collection OK |
| Loss | 771.6 → **267.7** | imitation learning healthy |
| Eval 200 | strict success=0.000, **success@5cm=0.667**, **grasp=1.000** | pipeline reaches/grasp/lifts but misses strict 2 cm |
| Failure mode | `lifted_not_at_goal:3` | no longer `never_reached_cube` |
| β | stayed **1.0** | because adaptive input still used strict success@2cm |

**Interpretation:** B3 vertical pipeline is validated enough for Stage 1. Continuing to 2000 is low value until β decay uses a bring-up metric (`success@5cm` / lift proxy) instead of strict 2 cm. Move on to vertical clean240 demo collection + G_θ retrain.

### Interaction Geometry — lift as first (ψ, κ) instance (2026-06-25)

Full design: [docs/INTERACTION_GEOMETRY.md](docs/INTERACTION_GEOMETRY.md).

**Claim (next stage):** Lift / drawer / door / button share `REACH → ACTUATE` with task-specific **κ** (constraint). Lift uses **κ=vertical_free** (+z above cube).

**Why change ACTUATE goal from fixed xy marker to above-cube:**
- Old post-grasp G* = cube→fixed command (e.g. 0.5, 0, 0.375) — goal invisible in mask, adds horizontal wander / convergence difficulty.
- New: attractor = `(cube_x, cube_y, cube_z + 0.20)` → G ≈ (0,0,1); success = lift height at spawn xy.
- Env: `lift_goal_above_cube=True` (default) syncs `object_pose` on reset via `sync_lift_goal_above_cube`.

**G learning path:** lift ψ special case first → 3D `G_θ(visual, ψ)` → multi-task with κ (slide / revolute / normal_press).

### Stage 1 sprint runbook (vertical-lift Interaction Geometry, single Isaac process)

Publication: Stage 1 is **not** a standalone paper → [docs/PAPER_ROADMAP_STRATEGY.md](docs/PAPER_ROADMAP_STRATEGY.md) (Paper 1 = after Stage 2).

Order (one Isaac job at a time — OOM guardrail):

1. **B3 oracle (running)** — `bash scripts/run_s1_b3_fresh.sh` — vertical ACTUATE + eval phase-reset fixes. Watch iter 50–100: grasp>0 and `success@5cm`/lift proxy improve.
2. **Re-collect clean240 vertical** (after B3 frees GPU) — `bash scripts/collect_s1_clean240_vertical.sh` → symlink as `demos_clean240v`.
3. **Retrain G_θ (3D)** — two ablations:
   - G-impl: `python scripts/learn_guidance_field.py --task lift --stage s1_random_cube --supervision subgoal --out-dim 3 --num-workers 4`
   - G-phase: add `--g-phase` (ψ = contact_latched conditioning).
   - Reports `val_align`, `REACH`, `ACTUATE` split; pass: align≥0.70, ACTUATE≥0.75, REACH≥0.65.
4. **Step 2 @ S1** — `bash scripts/run_s1_b7_fresh.sh` (B7-impl), B7-phase (G-phase ckpt), B1, vs B3 ceiling. Split eval: grasp (approach) vs vertical lift height.

Code wired this sprint: `goals/interaction_geometry.py`, `MaskFieldCNN(out_dim, cond_dim)`, `learn_guidance_field.py` (`--out-dim/--g-phase` + REACH/ACTUATE split), `LearnedGuidanceField` (3D + ψ via teacher_phase), `LiftGoalAboveCubeCommand` (vertical goal), collect `--lift_goal_above_cube`.

**B7 monitoring requirement:** TensorBoard now logs learned-vs-GT G deviation during policy training/eval:
- `train_g/cos_gt`, `train_g/angle_deg`, `train_g/l2`, `train_g/reach_cos_gt`, `train_g/actuate_cos_gt`
- `eval_g/cos_gt`, `eval_g/angle_deg`, `eval_g/l2`, `eval_g/reach_cos_gt`, `eval_g/actuate_cos_gt`
- Purpose: distinguish policy failure from `G_θ` OOD drift once β<1 and student states leave the teacher trajectory manifold.

**Adaptive β strategy (post-B3 finding):**
- Do **not** drive β decay from strict `success@2cm` during Stage 1 bring-up; it is too brittle and kept β=1.0 despite grasp/lift progress.
- Use a bring-up metric for β input: prefer `success@5cm`; fallback proxy = `grasp && lifted` / vertical-lift success. Keep strict `success@2cm` as the final report metric.
- Interpretation: β should start letting the student execute once it consistently reaches/grasp/lifts, instead of waiting for 2 cm precision.

### Immediate transfer sprint (2026-06-28) — open-loop first, two non-competing tracks

Rationale: lift closed-loop not yet proven (B7 ≯ B1). Get a transfer signal that does **not** need closed-loop. Track A is pure offline G work (no Isaac) and does not compete with Track B for the single Isaac process.

**Track A — representation-level transfer (primary, no Isaac after collection):**
- [ ] A1 collect `open` (door+drawer) demos in **G-format** (ee_mask+ee_depth+third_person_mask+third_person_depth+student_state+goals/phase), safe profile, single Isaac job. *Blocker to verify first:* does `open` demo `student_state` carry handle pos + a pull reference so ACTUATE dir is computable? (lift uses `student_state[0:3]=ee, 4:7=cube`).
- [ ] A2 add `open` G* labels in `goals/field_data.py`: REACH=`unit(handle−ee)`; ACTUATE=`unit(pull displacement)` straight-line; tag `κ=slide` (door upgradable→revolute).
- [ ] A3 train 3 specialists with `learn_guidance_field.py`: lift-only, open-only(door+drawer), joint **(ψ,κ)-conditioned** (`--g-phase`, extend cond to carry κ).
- [ ] A4 cross-eval matrix, **phase-split REACH/ACTUATE**: lift↔open (expect REACH high / ACTUATE low zero-shot, recovered by conditioning) + door↔drawer (expect ACTUATE high = positive control). → first transfer table.

**Track B — lift closed-loop gate (parallel, pre-existing, Isaac):**
- [ ] B1 re-run B3/B7 with **β bring-up metric = success@5cm** (not strict@2cm) so β decays and student executes.
- [ ] Gate: **B7 (learned G) ≥ B1** on grasp+vertical-lift@5cm. Transfer claim's policy half needs this.

### Lift B7 correction plan (2026-06-28)

Before any cross-task transfer claim, recover B7 on IsaacLab lift. Current correction targets:

1. **Initial visibility first.** Prior S1 B7 failed largely as `never_reached_cube`; random cube positions can start outside the wrist/EE camera view. Fix the initial robot/EE pose and cube randomization so the initial `ee_mask` contains the white cube mask under uniform cube randomization.
2. **ACTUATE G follows the EE frame, not a fixed world +Z simplification.** For grasped lift, the guidance should follow the direction opposite the current EE approach/forward direction (the retreat/lift direction), preserving any needed XY component. Do not hard-code ACTUATE to `[0, 0, 1]` until this is revalidated.
3. **B7 gate before transfer.** Train/retrain G only after the visibility check passes; then rerun B3/B7 with a bring-up metric (`success@5cm` or grasp+lift proxy) before collecting drawer/door transfer evidence.

Initial-pose gate result:
- Rejected `robot_x_offset=-0.20`: initial EE mask visibility passed, but teacher missed a far cube case.
- Rejected `robot_x_offset=-0.25` alone: 3x3 grid over default cube randomization had visibility 9/9 but teacher lift 7/9.
- Accepted `robot_x_offset=-0.10` plus `panda_joint6=2.7`: 3x3 grid over full default cube randomization (`x=[-0.1,0.1]`, `y=[-0.25,0.25]`) had initial EE white-mask visibility 9/9 and teacher lift success 9/9 (`success_lift_height=0.12 m`, median xy drift about 0.038 m).

### Step 2 (policy B1 vs B7 @ S1)
- [ ] `--randomize_cube`; lift success = **grasp + vertical lift** (goal above cube, not xy relocate)
- [ ] Report **success@2cm** (height+xy at spawn) + grasp rate

### S1 B1 without G — EE-visible ablation (2026-06-29/30)

**Goal:** Fair B1 vs B7 comparison under accepted EE-visible init (`robot_x_offset=-0.10`, `panda_joint6=2.7`, 240px topdown_close, random cube xy). Pure vision 8ch (`anchor_only`), no learned G.

**Runs:**
| run | status | note |
|-----|--------|------|
| `s1_B1_visible_ee_forward_20260629_172231` | **invalid** | Anti-hang partial **discard** → buffer stuck at 0, KL/BCE logged as 0 |
| `s1_B1_visible_ee_forward_20260629_175108` | finished 2000 iters | **flush+reset** at cap → buffer 162 ep but `ok=0`; fragment-only training; eval **0%** |
| `s1_B1_visible_ee_forward_20260630_094041` | **in progress** (~948/2000 @ 2026-06-30) | **segment flush + env continue** → `ok=88`, teacher rollout grasp/lift; eval still **0%** strict SR |

**Buffer logic evolution (IsaacLab DAgger):**
| version | at `partial_cap` | env reset | outcome |
|---------|------------------|-----------|---------|
| original (old B1 `20260622`) | none | only on episode done | `ok=13`, rollout `grasp=1.0` |
| discard (`172231`) | drop data | yes | buffer=0 |
| flush+reset (`175108`) | save 256-step segment | yes | buffer grows, `ok=0`, only APPROACH |
| **segment continue (`094041`)** | save segment, shift tensors left | **no** (eval envs reset before eval only) | full teacher trajectories as multi-segment logical episodes |

**Key findings — run `094041` (segment continue, current):**
1. **Training data fixed.** iter 10+: `buffer ok=88/250` (35% success pool), rollout `grasp=1.0 lift=1.0` every ~10 iters when teacher completes; `bce` active (gripper learning close). `arm_kl` 790→~24 (higher than `175108` because data includes grasp/lift phases).
2. **Eval still 0% strict success** @ iter 940, but **intermittent student progress**: eval@710 and @790 had `grasp=0.333`, `success@5cm=0.333`, `pre_grasp_ee_cube≈0.08 m` (vs typical 0.15–0.25 m). Failure modes shifted from pure `never_reached_cube` to mix of `grasp_then_dropped`, `lifted_not_at_goal`.
3. **Eval protocol (same as training DR):** uniform random cube `x∈[-0.1,0.1]`, `y∈[-0.25,0.25]`; lift goal = cube+xy +0.20 m; student deterministic; **only 3 episodes** per eval (`EVAL_EPISODES=3`) → high variance (1/3 grasp = 33% then back to 0%).
4. **Adaptive β side effect:** eval@710 `success@5cm=0.333` dropped β to **0.8** (iter 711+, `teacher_exec≈0.81`); EMA decay brought β back to ~0.999 by iter 940. Single lucky eval should not gate β with n=3.

### S1 B1 mosaic layout ablation (2026-06-30) — planned / in flight

**Motivation:** 8ch channel-stacked input + global avg pool loses spatial structure; random cube needs ee/top geometry + short-term motion. Mosaic layout gives CNN explicit inductive bias before LSTM.

**Visual layout (`--visual_layout=mosaic`):**
```text
         [t-2]   [t-1]   [t]     ← 3 frames horizontal (W=3H)
ee       H×W     H×W     H×W     ← mask+depth on C=2
top      H×W     H×W     H×W
         ↕ 2H total height
Policy CNN input: (2, 2H, 3W) per step; LSTM retained for longer context.
```

**Compared to baseline B1 (`visual_layout=channels`):**
| knob | baseline `094041` / `204236` | **mosaic v1** |
|------|------------------------------|---------------|
| input | (8, H, W) channel stack | **(2, 2H, 3W)** spatial mosaic |
| temporal | LSTM only | **3-frame mosaic + LSTM** |
| β | adaptive (stuck @1.0) | **linear decay** warmup 50 / decay 500 |
| throughput | 4×128 or 8×512 | **8×256** (disk buffer) |
| eval | 10 ep / 50 iter | same |

**Launch:**
```bash
bash scripts/run_s1_b1_mosaic.sh
# overrides: NUM_ENVS=8 COLLECT_STEPS=256 BETA_ADAPTIVE=0 ...
```

**Exit / compare vs baseline @ iter 200/500/1000:**
- `pre_grasp_ee_cube` ↓ vs `204236` (target <0.12 m sustained)
- `grasp` or `success@5cm` > 0 on ≥2 consecutive evals
- failure modes: `never_reached_cube` <70%

**Not in v1:** B7/G_θ mosaic (needs G field input redesign); resume from 8ch ckpt (incompatible conv weights).

**Code:** IsaacLab `vision_policy_obs.py` (`VisualMosaicHistory`, `stack_policy_visual_torch`); train flag `--visual_layout mosaic`.
5. **B1 without G — interim conclusion:** Buffer/teacher pipeline is healthy after segment continue; pure vision student **approaches** cube on good episodes but **cannot reliably grasp/hold/lift** without G. Supports running B7 with same infra before claiming G hurts.

**Key findings — run `175108` (flush+reset, superseded):**
1. Fragment-only buffer (`ok=0`); student mimicked APPROACH only (`bce→0`).
2. β locked at 1.0 entire run.

**Infra fixes shipped (IsaacLab + AutoResearch scripts):**
- IsaacLab: `_flush_capped_partial_rollouts` — persist segment at cap, `_shift_episode_buffers_left`, keep env running; `reset_after_flush=True` only for eval envs before eval.
- AutoResearch: `scripts/run_s1_b1_fresh.sh`; `play_student.sh` EE init; `max_store_episode_steps` + `eval_phase_stall_steps`; buffer prune/rebuild tools.

**Next guardrails (B1 finish + B7):**
- Let `094041` complete 2000 iters; then diagnostic eval with **20+ episodes**.
- Raise `EVAL_EPISODES` (e.g. 20) before trusting grasp/success curves.
- Consider `BETA_ADAPTIVE=0` + linear decay, or higher EMA / min eval count for β input.
- Launch B7 with same segment-continue IsaacLab + `run_s1_b7_fresh.sh`.

### Fixed-cube B1 bring-up (2026-07-13…16) — EE-visible + fixed mask

**Goal:** Stabilize pure-vision B1 before returning to random-cube / B7. Match play geometry during train+eval.

**Shared setup:** `robot_x_offset=-0.10`, `panda_joint6=2.7`, fixed cube, fixed mask palette, 8ch channel stack @64px, fast teacher, goal above cube +0.20m, `EVAL_EPISODES=20`, full episodes (`partial_cap=0`, `max_store=1250`).

**Runs:**
| run | β / gripper | status | final failure | note |
|-----|-------------|--------|---------------|------|
| `s0_B1_exact_repro_20260713_085702` | adaptive@success, slow teacher | done 100 | `grasp_then_dropped` | did not recover old S0 85% grasp under train-eval |
| `...fixedcube_eevis_fixedmask_20260713_142031` | adaptive@success | stopped ~465 | `never_reached_cube` | β=1; buffer frozen 175; pre~11cm |
| `...linbeta_20260713_163917` | **linear** warmup100/decay800 | **done 2000** | **`lifted_not_at_goal`** | pre 32→**2.4cm**; approach unlocked |
| `...gripw50_hold15_20260715_155957` | linear + **BCE×50** + **hold=15** | **done 2000** | **`grasp_then_dropped`** | earlier close; peak grasp **15%** @1950 |

**What worked:**
1. **Linear β** — after warmup, guaranteed student mixing; buffer grows past teacher-only demos; covariate shift reduced enough for approach.
2. **Fixed appearance/geometry** — removes mask-color / cube-xy confounders while debugging policy.
3. **Higher gripper BCE weight + longer CLOSE hold** — shifts failure from “never close” to “close then drop”.

**What failed / remaining gaps:**
1. **Strict success@2cm still 0%** on all fixed-cube long runs.
2. **Gripper architecture** — dual-head with encoder `no_grad` during BCE; arm_kl dominates (~100–1000× BCE magnitude even after ×50). Close is learned weakly; hold/lift not stable.
3. **Late β=0 pollution** — once student-only collect fills buffer with failures, loss rises and behavior oscillates (`never_reached` ↔ `dropped` ↔ `lifted`).
4. **Play/eval setting gap** — play default `arm_scale=0.5` vs train `1.0`; play lacks eval phase-stall. Can look worse than eval; still, iter 1900 eval itself was `never_reached_cube` (pre≈19cm). Prefer `student_iter_1800` / diagnostics-aligned play CLI.

**Play (aligned):**
```bash
STAGE=s0 CHECKPOINT=runs/s1_B1_fixedcube_eevis_gripw50_hold15_20260715_155957/student_iter_1800.pt \
bash scripts/play_student.sh B1
# Prefer arm_scale=1.0 when calling play_lift_cube_student_vision_binary_gripper.py directly.
```

**Interpretation for research goal:**
- Fixed-cube B1 now reaches the **same failure family as early S0 diagnostics** (post-grasp), not a visibility/DR pathology.
- B7 / G_θ comparison is still blocked until B1 can **hold + lift@5cm** more than intermittently.
- Highest-leverage next knobs: gripper BCE through shared encoder (or close class weighting), slower/hold-aware teacher during ACTUATE, keep linear β with a floor (avoid pure β=0 flooding).

## Best runs

| stage | step | split | field_alignment | success@2cm | success@5cm | wander | note |
|-------|------|-------|-----------------|-------------|-------------|--------|------|
| s0_fixed | 1 | s0_fixed | 0.9845 | — | — | — | state_goal |
| s0_fixed | 2 B1 | s0_fixed | — | 0% | **0%** | 0.104 m | lifted_not_at_goal |
| s0_fixed | 2 B7 | s0_fixed | — | 0% | **70%** | 0.069 m | final_dist 3-7cm |
| s1_random_cube | 1 | s1 | **0.723** | — | — | — | **stale** — noisy demos, mask-only 84px; clean240 retrain pending |
| s1_random_cube | 2 B1 | s1 | — | 0% | 0% | — | 300 iters @ iter300; 16/20 never_reached_cube |
| s1_random_cube | 2 B1 | s1 EE-visible | — | 0% | 0% | — | `175108` flush+reset; ok=0; fragment buffer |
| s1_random_cube | 2 B1 | s1 EE-visible | — | 0%* | 0–33%† | — | `094041` segment continue; eval n=3 noisy |
| fixedcube EE-vis | 2 B1 | fixed + linβ | — | **0%** | 0% | — | `linbeta_...163917` 2000; late lifted_not_at_goal; pre→2.4cm |
| fixedcube EE-vis | 2 B1 | gripw50 hold15 | — | **0%** | grasp†15%@1950 | — | `gripw50_...155957` 2000; grasp_then_dropped dominant |
| s1_random_cube | 2 B7 | s1 | — | 0% | 0% | — | 300 iters @ iter300; 16/20 never_reached_cube (old G) |
| s1_random_cube | 2 **B3** | s1 | — | **0% train eval** / **~33% play** | — | — | oracle GT 3D G; iter 666 stopped; eval latch bug → β=1.0; play 4/12 success after fix |

## Disk cleanup (2026-06-23)

Repo **520 GB → 2.2 GB**. Moved ~516 GB to `/home/l/code/data/AutoResearch_Data`:

- S1/S0 DAgger `dagger_aggregate.pt` + bloated `student_*` (76 GB buffers in `student_iter_200`)
- S1 HDF5 (clean240 + old noisy) via symlink
- Deleted failed partial collects (111054, 111256)

## Disk cleanup (2026-06-25)

Removed historical large replay/buffer artifacts while preserving small logs, TensorBoard traces, diagnostics, and compact checkpoints:

- Deleted old B3 disk-backed `dagger_episodes/` from:
  - `runs/s1_B3_cam240_xy_gt3d_20260624_155743/`
  - `runs/s1_B3_cam240_xy_gt3d_20260625_100836/`
- Deleted archived obsolete run buffers under `AutoResearch_Data/runs/`.
- Deleted old B7 huge `dagger_aggregate.pt` / bloated `student_final.pt` from stale S1 B7 runs.
- Result: `AutoResearch/runs` ≈ **549G → 490M**; `AutoResearch_Data/runs` ≈ **496G → 4K**.
- Kept current vertical clean240 data (`14G`), active G training outputs, small logs, eval diagnostics, and compact `student_*.pt` checkpoints.

## Journal

See [PROGRESS.md](PROGRESS.md) for issues, reflections, daily log.
