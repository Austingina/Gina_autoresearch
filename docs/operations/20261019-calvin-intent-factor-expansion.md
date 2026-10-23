# CALVIN 粗粒度 G 意图因子扩充操作记录

- 操作 ID：`20261019-calvin-intent-factor-expansion`
- 状态：`COMPLETED_PASS`
- 记录时间：`2026-10-19T13:11:52+08:00`
- 操作目录：`/home/ai-sz-a26394-u1/Gina_autoresearch`
- 只读基线：`/home/ai-sz-a26394-u1/AutoResearch`
- 基线提交：`3d4aeba1badaa3d6ce781033e0d9187729dfec54`

## 目标

将当前仅覆盖方块左右推动的 CALVIN 子集，扩展成用于研究粗粒度引导场
`G=[A(3), F(3)]` 普遍规律的“意图因子棋盘格”数据：`A` 只表示到可交互区域的
粗粒度吸引，`F` 只表示目标部件/物体的期望低维运动方向，不编码具体关节轨迹、
固定接触点或执行阶段。

## 当前基线

- 数据：`task_ABC_D_volume_g_subset_expanded_s12_h3`
- source：4 个方块任务 × 3 场景 × 12 段 = 144 段
- held-out：2 个颜色×方向重组 × 3 场景 × 3 段 = 18 段
- 总计：162 段、9,693 帧，约 2.6 GiB
- 现有全部 162 段必须保留，不能因新抽样而替换。

## 新规则

### source 训练任务

1. 方块推动：`push_red_block_left`、`push_blue_block_right`、
   `push_pink_block_left`、`push_pink_block_right`
2. 刚性机构：`open_drawer`、`move_slider_right`
3. 容器运输：`push_into_drawer`、`place_in_slider`

每个任务在 `calvin_scene_A/B/C` 各取 12 段；已有方块段作为强制保留集合，
新增任务按固定种子 `20260814` 在各任务×场景单元内确定性抽样。

### held-out 组合

- 方块重组：`push_red_block_right`、`push_blue_block_left`
- 机构反向：`close_drawer`、`move_slider_left`

每个 held-out 任务在每个场景取 3 段，只用于 audit，不写入 source-only 训练缓存。

### 排除项

- LED/灯泡开关：当前 translational `G` 为空，不能帮助识别几何 G。
- lift/rotate/stack/unstack：当前 `G=[A,F]` 尚无经验证的角流、抓取和对象对应定义；
  本操作不伪造标签，留待单独的表示扩展操作。
- 不混入 LIBERO/RoboTwin 数据；跨模拟器统一格式属于后续独立操作。

## 安全门

1. 新选择器必须先通过 RED→GREEN 测试。
2. source 与 held-out 任务集合必须不相交。
3. 所有 segment 帧区间全局不重叠。
4. 现有 162 段必须是新 manifest 的严格子集。
5. 每个 source 单元严格 12 段，每个 held-out 单元严格 3 段。
6. 下载前必须 dry-run 并输出任务、场景、机制、方向和帧分布。
7. 仅修改 `Gina_autoresearch`；原仓库状态和 HEAD 必须最终复核。

## 计划步骤

1. 从官方 ZIP 按 range 提取完整 annotation 元数据，不下载 555 GB 全包。
2. 新建意图 schema 与选择器测试，先确认缺少实现时测试失败。
3. 实现以现有 manifest 为强制保留集合的任务×场景平衡选择器。
4. dry-run 生成完整 selection manifest 与覆盖报告。
5. 新建 `task_ABC_D_g_intent_factorial_s12_h3` 数据目录；复用已验证 episode，
   只下载缺失成员，并对每个新成员执行 size/CRC 校验。
6. 核验本地 frame ID、manifest、source/held-out 隔离和字段。
7. 生成与旧子集的分布、机制覆盖和组合 held-out 对比报告。

## 预期变更

- 新增选择器、测试、报告生成脚本。
- 新增 `artifacts/data_distribution/intent_factorial_*` 证据。
- 新增 `data/calvin/task_ABC_D_g_intent_factorial_s12_h3/`。
- 不覆盖现有 `task_ABC_D_volume_g_subset_expanded_s12_h3/`。

## 失败与停止条件

- 任一任务×场景候选不足；
- 基线段未全部保留；
- source/held-out 或 segment 区间重叠；
- 官方成员 size/CRC 不符；
- 发现原仓库变化；
- 数据模型不足以给新任务定义现有粗粒度 `G`。

发生以上情况时保留日志和已验证文件，状态记为 `BLOCKED` 或 `PARTIAL`，不把不完整
目录声明为训练数据。

## 回滚

删除本操作新建的数据目录、脚本和 artifact 即可；现有 CALVIN 子集及原仓库不变。

## PASS 标准

- 新规则 manifest 满足全部安全门；
- 所有期望文件实际存在并通过完整性验证；
- 分布报告明确区分 source、held-out、机制族和方向支持；
- 原仓库仍为干净的基线提交。

## 实际结果

- 完成时间：`2026-10-19T14:14:14+08:00`
- 输出数据：`data/calvin/task_ABC_D_g_intent_factorial_s12_h3`
- 最终规模：324 段、19,354 帧，目录约 5.2 GiB。
- source：8 个任务、288 段、17,161 帧；每个任务×A/B/C 场景严格 12 段。
- held-out：4 个任务、36 段、2,193 帧；每个任务×A/B/C 场景严格 3 段。
- 家族覆盖：`block_push=162`、`articulation=90`、`container_transport=72`。
- 原 162 段全部保留，新增 162 段，删除 0 段；9,693 个旧 episode 以硬链接复用。
- 首次长下载在已完成大部分文件后遇到单个 600 秒阻塞请求，主动停止并以 60 秒超时、
  8 workers 断点恢复；恢复运行退出码为 0，manifest 的 `complete=true`。

### TDD 证据

- 初始 RED：选择器文件不存在，测试按预期失败。
- 元数据缓存 RED：`load_local_metadata` 不存在，测试按预期失败；实现后避免远端元数据
  短时断线导致 dry-run 失败。
- 反事实 RED：`build_counterfactual_audit_plan` 不存在，测试按预期失败。
- 最终 GREEN：6 个单元测试全部通过。
- 日志：`artifacts/evidence/20261019T131152-calvin-intent-factorial/`。

### 完整性核验

- episode：期望 19,354，实际 19,354，缺失 0，额外 0；
- archive file size 失败 0，ZIP CRC 失败 0，必要字段失败 0；
- 过滤后 annotation 共 324 条，与 manifest 逐条一致；
- segment 全局区间重叠为 false；source/held-out 任务交集为空；
- 选择脚本 SHA-256：`e39c7a506b1d984597be724941496e2af8467185f5ffffb1a2615d435cb0d5af`；
- 测试脚本 SHA-256：`0d2adb6f969f474b36fcf20ecc69e4e9ba15564685284a4a146c2f16818e61f0`；
- 实际 manifest SHA-256：`b7eeabef96bee1caada64b59ad53d862d887cfbd51939d3908238abff66874a5`。

### 反事实与失败样本边界

- manifest 加入 `zero_g`、`sign_flip_f`、inverse-intent、wrong-entity 审计条件；全部明确
  `usage=audit_only`、`expert_action_label=false`，不会伪装成专家训练样本。
- 官方 CALVIN 语言段是成功示范，本轮没有虚构失败动作；真实失败/恢复数据必须后续在
  模拟器中 rollout 后独立采集。

### 报告

- `artifacts/data_distribution/intent_factorial/INTENT_FACTORIAL_EXPANSION_REPORT.md`
- `artifacts/data_distribution/intent_factorial/distribution_by_task.csv`
- `artifacts/evidence/20261019T131152-calvin-intent-factorial/integrity_validation.json`

### 原仓库终态

`/home/ai-sz-a26394-u1/AutoResearch` 仍为
`master@3d4aeba1badaa3d6ce781033e0d9187729dfec54`，工作区干净，`git fsck --full`
退出码为 0。
