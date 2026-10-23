# Gina_autoresearch 数据扩展与分布审计操作记录

- 计划时间：2026-10-19
- 只读基线仓库：`/home/ai-sz-a26394-u1/AutoResearch`
- 操作副本：`/home/ai-sz-a26394-u1/Gina_autoresearch`
- 基线提交：`3d4aeba1badaa3d6ce781033e0d9187729dfec54`

## 任务意图

保留原仓库不变，在独立副本中围绕当前活跃的 Phase-1 CALVIN interaction-volume G 任务扩展一部分训练数据，并提供扩展前后分布表。当前论文路径明确将 4 个 source 颜色×方向组合用于训练，将 `push_blue_block_left` 和 `push_red_block_right` 作为 held-out 重组任务，因此扩展不得将 held-out 样本写入训练缓存。

## 基线定义

仓库 `dataset.md` 记录的 interaction-volume 基线为：

- 固定选择种子：`20260814`；
- 3 个源场景：A、B、C；
- 4 个 source tasks，每个任务×场景 8 段，共 96 个训练 source segments；
- 2 个 held-out tasks，每个任务×场景 3 段，共 18 个仅审计 segments；
- 原始审计子集总计 114 段、6,810 帧；
- 训练缓存只包含 96 个 source segments、5,769 帧。

## 扩展方案

采用保守、平衡且可复现的 50% source-segment 扩展：

- 将 `source_per_task_scene` 从 8 提高到 12；
- 4 个 source tasks、3 个场景仍完全均衡；
- source 训练段从 96 增至 144；
- held-out 保持每任务×场景 3 段、总计 18 段，仅用于审计；
- 任务集合、场景集合、选择种子和验证模数保持不变；
- 不把 held-out 组合并入训练，不改变原仓库中的历史结果。

实际帧数、字节数和任务/场景交叉分布必须由官方 CALVIN ZIP 元数据和完成后的 manifest 实测，不用比例猜测。

## 计划步骤

1. 确认 `Gina_autoresearch` 不存在。
2. 从本机原仓库建立无硬链接的完整 Git 副本，并把 `origin` 恢复为 GitHub URL。
3. 记录原仓库和副本 HEAD、状态及关键文件哈希。
4. 在副本中创建隔离 Python venv，安装 `numpy`、`requests`、`remotezip`。
5. 使用仓库现有远程 ZIP 选择器分别审计基线参数 `8/3` 与扩展参数 `12/3`。
6. 将扩展数据下载到副本的项目自有目录：
   `data/calvin/task_ABC_D_volume_g_subset_expanded_s12_h3/`。
7. 生成机器可读 CSV/JSON 和 Markdown 分布报告，至少比较 task、scene、source/held-out、segments、frames 和占比。
8. 验证 episode 文件数量/区间、manifest、CRC、Git 状态、原仓库不变性与磁盘占用。

## 风险和安全边界

- 官方数据体积较大；先 dry-run 获取实际压缩字节估算，再决定是否执行下载。
- HTTP Range 服务若不可用或返回非 206，立即停止，不伪造扩展结果。
- 所有新增数据只写入 `Gina_autoresearch`；不写入 `AutoResearch`，不写外部 CALVIN 安装目录。
- held-out 数据只保留为审计集，不进入 source-only 训练统计与缓存。
- 不执行模型训练；本次边界是数据扩展和分布审计。

## 失败出口与回滚

- 目标目录已存在则停止，不覆盖。
- 下载中断时保留 manifest/日志并利用脚本的文件大小检查恢复；不报告完成。
- 若扩展破坏均衡性、出现区间重叠或缺文件，判定失败。
- 回滚仅删除本次新建的 `/home/ai-sz-a26394-u1/Gina_autoresearch`，原仓库不受影响。

## 实际执行记录

- 完成时间：`2026-10-19T12:24:34+08:00`。
- 原仓库 `/home/ai-sz-a26394-u1/AutoResearch` 保持在 `master@3d4aeba1badaa3d6ce781033e0d9187729dfec54`，工作区干净；`dataset.md` SHA-256 仍为 `b11e40c90b5f494793a842715138815224f5be6fb842b73417362cc2f331daef`。
- 使用 `git clone --no-hardlinks` 建立 `/home/ai-sz-a26394-u1/Gina_autoresearch`，副本初始 HEAD 与原仓库一致，origin 恢复为 GitHub URL。
- 官方 ZIP：`Content-Length=555309812705`、支持 byte ranges；远程 ZIP 有 1,894,126 个 entries，central directory 为 229,211,511 bytes。
- 原 `RemoteZip` 逐段索引耗时过长，因此在副本中新增稀疏 ZIP 索引构建工具，并为提取器增加 `--zip-index`、完整选择 manifest、并行下载和重试；central-directory tail SHA-256 为 `ff8c37006c8c0d0b16c36e1b63e8d7e118faea5eb9e1dc14ea6ff4e395301c7a`。
- 基线 dry-run 实测：114 段、6,810 帧、预计压缩读取 1,877,643,795 bytes；其中 source 96 段/5,769 帧，held-out 18 段/1,041 帧。
- 扩展 dry-run 实测：162 段、9,693 帧、预计压缩读取 2,663,383,906 bytes；其中 source 144 段/8,652 帧，held-out 保持 18 段/1,041 帧。
- 扩展保留全部 114 个基线段，删除 0 段，新增 48 个 source 段；每个 source task × scene 从 8 增至 12，任务和场景严格均衡。
- 实际数据已下载到 `data/calvin/task_ABC_D_volume_g_subset_expanded_s12_h3/`，占用约 2.6 GiB；manifest SHA-256 为 `49e63ca61684b3ebdaab57fead53623349c1bba31457feaf2bd0fd98849208d2`。
- 完整性验证：162 段的区间展开得到 9,693 个唯一 frame IDs，本地 episode 文件集合完全一致；提取时逐 member 校验未压缩大小和 CRC；另抽查 72 个任务×场景边界 NPZ，均含 `robot_obs`、`scene_obs`、`rel_actions`。
- 分布报告：`artifacts/data_distribution/CALVIN_DATA_DISTRIBUTION_COMPARISON.md`。
- 机器可读表：`distribution_by_task.csv`、`distribution_by_scene.csv`、`distribution_by_task_scene.csv`、`comparison_summary.json`。
- held-out `push_blue_block_left` 和 `push_red_block_right` 仅保留为 audit，不进入 source-only 训练统计，未发生目标组合泄漏。

## 执行中问题与修复

1. 原始 `RemoteZip` 直接读取 555 GB ZIP 的 218.6 MiB central directory 超时；改为一次性 range 下载 central-directory tail，并构建逻辑大小 555 GB、物理占用约 219 MiB 的稀疏索引。
2. 首次并行提取因官方 HTTP 服务读超时退出；所有已完成文件均经过 CRC 后才落盘，随后加入 bounded retry、降低为 4 workers 并从 660 个完整文件处恢复，最终退出码为 0。
3. `downloaded_bytes` 记录的是最后一次恢复进程实际传输量，不用作全量体积；完整选择的压缩字节口径来自 dry-run central-directory 审计。

## 最终结论

本次只修改和新增 `Gina_autoresearch` 内的代码、报告和数据，原仓库保持字节与 Git 状态不变；扩展数据已实际落盘并通过分布、ID、CRC 和字段验证，未执行模型训练。
