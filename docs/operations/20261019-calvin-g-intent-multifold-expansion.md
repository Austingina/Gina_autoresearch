# CALVIN 粗粒度 G 意图多折扩充操作记录

- 操作 ID：`20261019-calvin-g-intent-multifold-expansion`
- 开始时间：`2026-10-19T16:14:37+08:00`
- 目标仓库：`/home/ai-sz-a26394-u1/Gina_autoresearch`
- 原始只读仓库：`/home/ai-sz-a26394-u1/AutoResearch`
- 状态：`COMPLETED_PASS`
- 操作者：Hermes Agent

## 权威目标

继续按任务意图扩充 CALVIN 数据，唯一研究目标是识别可跨对象、方向、场景、机构并最终跨模拟器迁移的粗粒度引导量 `G`，而不是为了扩大文件数量或把颜色本身当作目标变量。

当前表示契约：

- `A`：当前状态到可交互区域的粗粒度位移或可达关系；
- `F`：任务坐标系中期望的实体运动或作用方向；
- task-level intent：`family / primitive / entity / target_relation / direction_or_polarity`；
- frame-level phase：后续由真实状态生成 `approach / align / contact / transport / release-or-stop`，本操作不伪造帧级阶段标签。

## 基线

- 实验仓库 HEAD：`3d4aeba1badaa3d6ce781033e0d9187729dfec54`，工作区已含前序实验文件；
- 单折意图数据：`task_ABC_D_g_intent_factorial_s12_h3`，324 段、19,354 帧；
- 现有任务家族：`block_push / articulation / container_transport`；
- 当前单折：Pink 为左右桥梁，held-out 为 red-right 与 blue-left；
- 磁盘：约 1.3 TiB 可用；
- 并发：PID `553555` 正由 `/home/ai-sz-a26394-u1/AST` 写入独立目录 `/home/ai-sz-a26394-u1/AST_data/...`，本操作不得终止、修改或复用其输出。

## 本操作范围

1. 构建意图超集：12 个已验证为平移型粗粒度 `G` 的 CALVIN 任务，每个任务×A/B/C 场景固定 12 段；
2. 方块推动构建 3 个平衡 connected-checkerboard：Pink、Red、Blue 分别作为双方向桥梁，每个精确对象×方向组合恰好在一个 block fold 中作为 held-out；
3. 机构平移构建 4 个 inverse-intent fold：drawer-open、drawer-close、slider-left、slider-right 各自恰好在一个 articulation fold 中作为 held-out，其余三个组合进入 source；
4. 将 3 个 block folds 与 4 个 articulation folds 做笛卡尔积，得到 12 个完整实验折；
5. 每折保持 container transport 两个任务为覆盖性 source，不宣称它们已实现 primitive×container 的正交识别；
6. 每折 source 任务×场景使用 12 段，held-out 任务×场景仅使用固定 3 段，同折 source/held-out 任务与帧区间必须完全隔离；
7. 复用现有已验证文件，只补齐当前四个低密度任务至 12 段/场景。

## 明确非目标

- 不把颜色作为 `G` 的语义；颜色仅作为实体身份代理和反捷径测试轴；
- 不把人工替换意图后的原动作标成专家动作；
- 不虚构失败、恢复或重规划轨迹；
- 不声称 A/B/C 均衡等于 ABC→D 场景泛化；
- 不加入尚未验证 `G` 表达的旋转、堆叠、抬升、灯光和开关任务；
- 不修改 `/home/ai-sz-a26394-u1/AutoResearch`；
- 不访问或干预 AST 并发任务的文件与进程。

## 计划产物

- `framework/scripts/data/build_calvin_g_intent_multifold.py`
- `framework/scripts/data/test_build_calvin_g_intent_multifold.py`
- `artifacts/data_distribution/intent_multifold/superset_selection_s12.json`
- `artifacts/data_distribution/intent_multifold/folds/*.json`
- `data/calvin/task_ABC_D_g_intent_multifold_s12/`
- `artifacts/data_distribution/intent_multifold/G_INTENT_MULTIFOLD_REPORT.md`
- `artifacts/evidence/20261019T161437-calvin-g-intent-multifold/`

## 执行步骤与预期

1. 保存当前 Git 状态、目标文件哈希、磁盘和并发快照；
2. 先写多折协议测试并运行，预期因实现不存在而 RED；
3. 最小实现意图超集选择、12 折生成、折内隔离和覆盖平衡，运行目标及回归测试至 GREEN；
4. 使用完整语言标注和 ZIP64 稀疏索引 dry-run，生成确定性 selection 与 fold manifests；
5. 等待 PID `553555` 结束后，以 CRC-before-write 和断点恢复方式补齐远端成员；
6. 核验 expected/local 文件集合、ZIP size/CRC、必要字段、标注顺序、区间、每折任务配额与泄漏；
7. 生成报告和 README，复核原仓库干净并关闭本记录。

## 安全门与中止条件

- 若 authoritative annotation 不足以满足任何任务×场景的 12 段配额，中止；
- 若任何折中 source 与 held-out 任务或帧重叠，中止；
- 若新选择不能保留当前 324 段，中止；
- 若发现目标目录已被其他进程写入，中止；
- 若 AST 并发任务仍在运行，不启动新的远端下载；
- 若 CRC、字段或过滤标注出现任一失败，不标记 PASS；
- 若需要扩展到未验证的 `G` 任务族，另开操作记录。

## 回滚

所有新文件写入独立脚本、artifact、evidence 和 dataset 目录；回滚时删除本操作新建目录并从证据备份恢复本操作触及的已有文件，不修改前序单折数据及原仓库。

## PASS 标准

- TDD RED 与 GREEN 证据齐全；
- 12 个 fold 的组合覆盖平衡且每折隔离通过；
- 超集选择保留当前 324 段并将 12 个任务×3 场景补齐至 12 段；
- 实际文件、CRC、字段、标注、区间和 manifest 全部通过；
- 报告明确区分已支持的 CALVIN 内组合泛化与仍未验证的 phase、场景、机器人和模拟器泛化；
- 原仓库仍为干净的 `master@3d4aeba1badaa3d6ce781033e0d9187729dfec54`。

## 实际结果

### 完成时间

`2026-10-19T19:50:05+08:00` 后完成最终验证和报告收尾。

### 实现与 TDD

- 新增多折构建器与协议测试，保存实现缺失 RED、固定 held-out pool RED 及后续 GREEN 证据；
- 最终多折测试 5 项全部通过，旧单折意图回归测试 6 项全部通过，相关脚本 `py_compile` 通过；
- 因独立 AST 下载暴露大 Range 停滞问题，为本提取器补充 2 MiB 物理 Range 上限并保存额外 RED/GREEN 证据；
- 最终下载使用 workers=2、timeout=60、retries=8、max-range-bytes=2,097,152。

### 数据与协议

- 协议：`calvin_g_intent_multifold_v1`，seed `20260814`；
- 超集：12 个任务、36 个任务×场景单元、432 段、25,852 帧；
- 保留旧单折数据全部 324 段，新增 108 段、6,498 帧；
- 旧数据硬链接复用 19,354 个文件，本轮远端下载 1,803,118,350 字节；
- 3 个 block bridge folds × 4 个 articulation folds = 12 个组合折；
- 每折 source 324 段、held-out 27 段，同一 held-out task 的评估 episode pool 跨组合折固定。

### 最终验证

- manifest `complete=true`；expected=actual=25,852，missing=0，extra=0；
- 25,852 个文件的 ZIP size、CRC 和必要 NPZ 字段全部通过，失败数均为 0；
- 本地语言标注顺序与 manifest 一致，432 段无全局区间重叠；
- 36 个任务×场景单元全部恰好 12 段；
- 12 个 fold 无 source/held-out 任务、split、配额或区间泄漏；
- 每个 block 组合 held-out 4 次，每个 articulation 组合 held-out 3 次；
- 每个 held-out task 的固定 episode pool 计数均为 1；
- manifest SHA-256：`c09c9f40886a54c3e35c3b5160093d575cb73c8eee42816b2cd807df2ee0a2a6`；
- 原始仓库保持 `master@3d4aeba1badaa3d6ce781033e0d9187729dfec54` 且干净，输出 `ORIGINAL_REPO_CLEAN_OK`。

### 科学边界

本操作完成的是 CALVIN 平移任务内部的对象×方向×机构逆意图组合协议和数据准备，并未产生训练性能结论；帧级 phase、真实失败/恢复、整场景 held-out、跨机器人和跨模拟器泛化仍未验证，不能据此宣称通用 `G` 已成立。
