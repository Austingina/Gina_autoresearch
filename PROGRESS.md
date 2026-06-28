# AutoResearch Progress Journal

**Goal:** S1 random cube — phase-aware G_θ (`subgoal`) + B1 vs B7.

**Focus:** Tier 0→2 validation only. Long-term routes (3D G, TarGF, text condition) → [docs/FUTURE_RESEARCH_ROUTES.md](docs/FUTURE_RESEARCH_ROUTES.md) (**deferred**).

---

## S1 — random cube (current)

### Pipeline
```bash
# 1. Collect demos
STAGE=s1_random_cube RANDOMIZE_CUBE=1 NUM_DEMOS=100 bash scripts/collect_lift_demos.sh

# 2. Step 1 — G_θ (mask in, GT subgoal label; auto subgoal for s1)
bash scripts/run_s1_step1.sh

# 3. Step 2 — DAgger
MAX_ITERS=300 bash scripts/run_s1_dagger_smoke.sh B1
MAX_ITERS=300 bash scripts/run_s1_dagger_smoke.sh B7
```

### Eval metrics
- **Primary:** `success@2cm` (existing `success_rate`)
- **Auxiliary:** `success@5cm` — cube lifted + `min_dist_to_goal ≤ 0.05`

---

## Day 0 — 2026-06-18 (session start)

### Done
- [x] Data isolation (`data/`, import/collect scripts)
- [x] Step 1: `MaskFieldCNN` trained — **val field_alignment 0.9845** (pass ≥0.75)
- [x] IsaacLab: `guidance_field_bridge.py` — append G as +2 visual channels (10ch total)
- [x] `run_s0_dagger_comparison.sh` — B1 vs B7 harness
- [x] v3 HDF5 copied to isolated `data/lift/s0_fixed/demos/`
- [x] **Eval diagnostics** — EE/cube trajectories + failure-mode taxonomy (`lift_episode_diagnostics.py`)
- [x] **B1 DAgger complete** — 100 iters, buffer 179 ep / 221k steps, teacher lift 100%
- [x] **B7 DAgger complete** — 100 iters; diagnostic eval + trajectory plots
- [x] **B7 diagnostic re-analysis** (20 ep, 10ch load fix)

### In progress
- [ ] B7 vs B1 success gap ≥ 15%（当前两者均为 0%）
- [ ] 调参：更长 DAgger / β decay / G 幅度或 phase gating

### B7 root-cause analysis (diagnostic eval, fixed 10ch load)

**Run:** `runs/s0_B7_20260622/eval_diagnostics/` (20 ep, `student_best.pt` @ iter 100)

| Metric | B1 | B7 |
|--------|-----|-----|
| eval success | 0% | 0% |
| grasp rate | 85% | **90%** |
| mean min EE–cube dist | 1.4 mm | 2.5 cm |
| mean post-grasp wander | 10.4 cm | **6.9 cm** |
| typical final_dist_goal (lifted ep) | 11–36 cm | **3.2–7.0 cm** |
| failure mode | 100% `lifted_not_at_goal` | 100% `lifted_not_at_goal` |

**B7 相对 B1 的变化：**
- G 场显著缩短 relocation 误差：多数 episode 最终距 goal **3–7 cm**（B1 为 11–36 cm）
- 2/20 episode 弱 lift（max_z≈0.055，同 B1 边界情况）
- 未出现 `grasp_then_dropped`；抬起后 cube 保持在 ~0.31–0.35 m
- **仍未 crossing 2 cm success 阈值** → 归类为 `lifted_not_at_goal`

**训练内 eval vs 完整诊断 eval 差异：**
- 训练 eval（iter 100）：grasp 50%, wander 6 cm（带 early_done，episode 可能提前截断）
- 完整诊断（1250 步）：grasp 90%, wander 6.9 cm → 以诊断为准做轨迹分析

**结论：** G_θ 方向正确（relocation 明显改善），但 100 iter smoke + 全局广播 G 仍不足以精确到达 goal；且 approach 精度略降（EE–cube 2.5 cm vs 1.4 mm）。

**Artifacts:**
- `runs/s0_B7_20260622/eval_diagnostics/eval_analysis.md`
- `runs/s0_B7_20260622/eval_diagnostics/trajectory_plot.png`

### Visualization

**Isaac GUI 实时观看（推荐）：**
```bash
# B1
bash scripts/play_student.sh B1

# B7（自动加载 G_θ + 10ch）
bash scripts/play_student.sh B7
```

**离线轨迹图（无需 Isaac，含 GOAL ★ + 2cm 成功圈 + 距离标注）：**
```bash
# 单个 run
python3 scripts/plot_eval_trajectories.py runs/s0_B1_restart/eval_diagnostics/eval_diagnostics.json
python3 scripts/plot_eval_trajectories.py runs/s0_B7_20260622/eval_diagnostics/eval_diagnostics.json

# B1 vs B7 并排对比（同 episode id）
python3 scripts/plot_eval_trajectories.py \
  runs/s0_B1_restart/eval_diagnostics/eval_diagnostics.json \
  --compare runs/s0_B7_20260622/eval_diagnostics/eval_diagnostics.json \
  --episodes 1 4 8 \
  --output runs/trajectory_compare_b1_b7.png
```

俯视图图例：**★红色=GOAL**，**绿色虚线圆=2cm 成功区**，**紫色虚线=cube 终点→goal**，数字为 3D 距离(cm)。

**原始 Isaac play（更多参数）：**
```bash
cd /home/l/code/IsaacLab
CHECKPOINT=/home/l/code/AutoResearch/runs/s0_B7_20260622/student_best.pt \
HEADLESS=0 MAX_EPISODES=10 AUTORESEARCH_ROOT=/home/l/code/AutoResearch \
./scripts/play_lift_cube_student_vision_binary_gripper.sh --show_opencv
```

### B1 root-cause analysis (diagnostic eval)

**Run:** `runs/s0_B1_restart/eval_diagnostics/` (20 episodes, `student_best.pt` @ iter 100)

| Metric | Value |
|--------|-------|
| eval success | **0%** |
| grasp rate | **85%** |
| mean min EE–cube dist | **1.4 mm** |
| mean post-grasp wander | **10.4 cm** |
| mean max cube z | ~0.27 m (table ≈ 0.04 m) |

**Failure mode breakdown (100% of episodes):**

| Mode | Count | Interpretation |
|------|-------|----------------|
| `lifted_not_at_goal` | 20/20 | Cube lifted off table but never relocated to goal |

**NOT the primary failure:**
- ~~`never_reached_cube`~~ — EE consistently reaches cube (min dist ≈ 1 mm)
- ~~`never_grasped`~~ — only 3/20 weak lifts (max_z ≈ 0.055 m, borderline)
- ~~`grasp_then_dropped`~~ — no episodes; final cube_z stays elevated (~0.25 m)

**Trajectory reading (EE vs cube vs goal):**
1. **Approach (steps 1–~200):** EE moves from home toward cube; EE–cube dist drops from ~0.33 m → <0.01 m.
2. **Grasp + partial lift (~200–600):** gripper closes; cube_z rises to 0.23–0.29 m (well above table).
3. **Post-grasp wander (600–1250):** cube hovers ~10–36 cm from goal in xy; `dist_to_goal` never crosses 0.02 m threshold → timeout.

**Conclusion:** B1 (anchor mask only) learns approach + grasp from DAgger teacher demos but **cannot relocate cube to goal** without guidance field. This is exactly the hypothesis B7 tests.

**Artifacts:**
- `runs/s0_B1_restart/eval_diagnostics/eval_diagnostics.json` — full trajectories
- `runs/s0_B1_restart/eval_diagnostics/eval_analysis.md` — human summary

### Issues encountered
| Issue | Resolution |
|-------|------------|
| `mask_flow` supervision → align ~0.03 | Switched to **`state_goal`**: cube→goal direction in x-z from `student_state`; mask still input-only at inference |
| Weak signal from mask centroid Δ | Documented in PROGRESS; mask_flow kept as ablation option |
| Eval only reported 0% success, no failure reason | Added diagnostic eval + wired `collect_diagnostics` into training eval |
| B7 ckpt meta `visual_channels=8` but weights 10ch | Infer channels from conv weights; fixed checkpoint save for future runs |
| `TERM=dumb` breaks isaaclab.sh | Force `TERM=xterm-256color` in wrapper scripts |

### Reflections
- Step 1 target: predict demo relocation direction from **masks only**; proprio used only as training label (not leaked to student at test)
- B1 result validates research framing: mask anchors grasp, G needed for relocation
- Next: B7 should reduce `post_grasp_wander_m` and convert `lifted_not_at_goal` → `success`

---

## Day 1 — 2026-06-23 (S1 clean collect + disk cleanup)

### Done
- [x] **Clean S1 collect** (`run_s1_collect_clean.sh`): 100 ep, 100% success, 240px mask+depth, no teacher noise, topdown_close
- [x] **Disk cleanup** after Cursor crash: repo 520 GB → **2.2 GB**; archive at `/home/l/code/data/AutoResearch_Data` (~516 GB)
- [x] S1 DAgger B1/B7 both finished 300 iters (eval@300: 0% SR; 16/20 `never_reached_cube` — trained with **stale** G_θ)

### Archive policy
- **Moved:** `dagger_aggregate.pt`, `student_final.pt`, `student_iter_200.pt` (embedded 76 GB buffers) for S0/S1 runs
- **Kept in repo:** `student_best.pt`, `student_iter_300.pt`, logs, eval JSON, TensorBoard
- **Data:** clean240 + old noisy HDF5 → archive; symlinks in `demos/` and `demos_clean240/`
- **Deleted:** empty/partial failed HDF5 (`111054`, `111256`)
- Manifest: `/home/l/code/data/AutoResearch_Data/ARCHIVE_MANIFEST.md`

### Blocked
- [ ] **G_θ retrain on clean240** — `learn_guidance_field.py` OOM-killed (~84 GB RSS loading all samples). Checkpoint still old `guidance_field_s1_random_cube.pt` (0.723, noisy 84px). Need lazy HDF5 loading or RAM subsample.

### Next
```bash
# After fixing loader memory — retrain Step 1
DEMO_DIR=data/lift/s1_random_cube/demos_clean240 IMAGE_SIZE=240 \
  bash scripts/run_s1_step1.sh --batch-size 16

# Verify G on clean teacher
bash scripts/play_teacher_g_check.sh
```

---

## Metrics log

| Date | Step | field_alignment (val) | B1 eval SR | B7 eval SR | Notes |
|------|------|----------------------|------------|------------|-------|
| 2026-06-18 | Step 1 | **0.9845** | — | — | state_goal, v3+smoke, epoch 79 |
| 2026-06-19 | Step 2 B1 | — | **0%** | — | 100 iters; failure=lifted_not_at_goal (85% grasp) |
| 2026-06-22 | Step 2 B7 | — | 0% | **0%** | aux @5cm backfilled |
| 2026-06-22 | **S1** | — | — | — | subgoal field + dagger scripts ready |
| 2026-06-23 | S1 collect | — | — | — | clean240, 100% success, archived 17 GB |
| 2026-06-23 | S1 Step 2 | — | 0% | 0% | 300 iters each; stale G; buffers archived |

---

## Commands

### Diagnostic eval (any checkpoint)
```bash
CHECKPOINT=runs/s0_B1_restart/student_best.pt \
OUTPUT_DIR=runs/s0_B1_restart/eval_diagnostics \
bash scripts/run_diagnostic_eval.sh
```

### DAgger progress
```bash
tail -f runs/s0_B7_*/progress.txt
python3 scripts/dagger_progress.py runs/s0_B7_*/
```

### Launch B7
```bash
FIELD_CKPT=checkpoints/lift/guidance_field_s0_fixed.pt \
  bash scripts/run_s0_dagger_smoke.sh B7
```
