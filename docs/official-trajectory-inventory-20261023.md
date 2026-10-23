# 当前官方轨迹扩充与总库存对比（2026-10-23）

> **统计边界**：本文把两条库存分开记录：
>
> 1. **源仓库基线**：原有 CALVIN 数据，2,107 个 segments、48,069 个训练窗口；
> 2. **新增官方轨迹臂**：CALVIN、LIBERO、RoboTwin、ManiSkill、VIMA-Bench 的官方成功轨迹重组与标签增强，canonical 索引为 `data/semantic_indexes/official_reorg_20261023_v6/`。
>
> 文中的“当前仓库总库存 = 源仓库 + 新增官方臂”沿用此前报告的加总口径，用于库存规模和意图分布对比；它**不表示**已经完成跨来源去重、统一窗口化、统一动作向量或统一训练 manifest。

## 1. 总体对比

| 指标 | 源仓库 | 当前新增官方臂 | 当前仓库总库存 | 源→当前变化 |
|---|---:|---:|---:|---:|
| 模拟器/基准数 | 1 | 5 | 5 | +4 |
| simulator-qualified 任务数 | 14 | 117 | 121 | +107 |
| 任务记录数 | 2,107 segments | 26,193 parents | 28,300 task records | +26,193 |
| 意图家族数 | 4 | 20 | 20 | +16 |
| 原训练窗口 | 48,069 | 尚未统一窗口化 | 暂不可直接合计 | — |
| 新增官方臂原生帧/transitions | — | 2,507,756 | 不与 48,069 窗口直接相加 | — |
| 新增官方臂派生阶段视图 | — | 95,033 | 不计作独立成功轨迹 | — |

任务数口径说明：新增官方臂包含 CALVIN 12、LIBERO 40、RoboTwin 50、ManiSkill 9、VIMA 6，共 117 个 simulator-qualified 任务；源仓库 14 个任务中有 10 个与当前 CALVIN 官方臂重合，因此当前任务并集为 121。

## 2. 当前库存按模拟器分布

| 模拟器/基准 | 源仓库已有 | 当前新增官方臂 | 当前库存 | 当前占比 |
|---|---:|---:|---:|---:|
| CALVIN | 2,107 | +432 | 2,539 | 8.97% |
| LIBERO | 0 | +2,000 | 2,000 | 7.07% |
| RoboTwin | 0 | +2,500 | 2,500 | 8.83% |
| ManiSkill / SAPIEN | 0 | +15,262 | 15,262 | 53.93% |
| VIMA-Bench / PyBullet | 0 | +5,999 | 5,999 | 21.20% |
| **合计** | **2,107** | **+26,193** | **28,300** | **100.00%** |

新增官方臂内部占比为：CALVIN 1.65%、LIBERO 7.64%、RoboTwin 9.54%、ManiSkill 58.27%、VIMA-Bench 22.90%。

## 3. 任务意图分布变化

| 意图家族 | 源仓库数量 | 源仓库占比 | 当前新增官方臂 | 当前总量 | 当前占比 | 源→当前变化 |
|---|---:|---:|---:|---:|---:|---:|
| `tool_or_contact_operation` | 0 | 0.00% | +4,450 | 4,450 | 15.72% | +15.72 pp |
| `orientation_manipulation` | 0 | 0.00% | +4,008 | 4,008 | 14.16% | +14.16 pp |
| `insertion_assembly` | 0 | 0.00% | +2,975 | 2,975 | 10.51% | +10.51 pp |
| `planar_push` | 0 | 0.00% | +2,656 | 2,656 | 9.39% | +9.39 pp |
| `rolling_contact` | 0 | 0.00% | +2,373 | 2,373 | 8.39% | +8.39 pp |
| `pick_place` | 0 | 0.00% | +2,050 | 2,050 | 7.24% | +7.24 pp |
| `multi_object_arrangement` | 0 | 0.00% | +1,300 | 1,300 | 4.59% | +4.59 pp |
| `block_push` | 900 | 42.71% | +216 | 1,116 | 3.94% | −38.77 pp |
| `constraint_aware_tool_interaction` | 0 | 0.00% | +1,000 | 1,000 | 3.53% | +3.53 pp |
| `long_horizon_rearrangement` | 0 | 0.00% | +1,000 | 1,000 | 3.53% | +3.53 pp |
| `ordered_imitation` | 0 | 0.00% | +1,000 | 1,000 | 3.53% | +3.53 pp |
| `long_horizon_memory_manipulation` | 0 | 0.00% | +999 | 999 | 3.53% | +3.53 pp |
| `switch_activation` | 602 | 28.57% | +300 | 902 | 3.19% | −25.38 pp |
| `articulation` | 455 | 21.59% | +344 | 799 | 2.82% | −18.77 pp |
| `object_manipulation` | 0 | 0.00% | +450 | 450 | 1.59% | +1.59 pp |
| `long_horizon_composite` | 0 | 0.00% | +350 | 350 | 1.24% | +1.24 pp |
| `placement` | 0 | 0.00% | +350 | 350 | 1.24% | +1.24 pp |
| `container_transport` | 150 | 7.12% | +72 | 222 | 0.78% | −6.33 pp |
| `object_acquisition` | 0 | 0.00% | +200 | 200 | 0.71% | +0.71 pp |
| `handover` | 0 | 0.00% | +100 | 100 | 0.35% | +0.35 pp |

### 分布结论

- 原有 `block_push`、`switch_activation`、`articulation` 和 `container_transport` 数据没有减少，只是相对占比被更大规模的新任务家族稀释。
- `pick_place` 在三模拟器 v4 中曾占 41.57%；加入 ManiSkill 后在 v5 降至 10.15%，加入 VIMA 后在五来源官方臂 v6 中进一步降至 **7.83%**；按“源仓库＋官方臂”的总库存口径为 **7.24%**。
- 当前最大意图家族不再是 `pick_place`，而是 `tool_or_contact_operation`（15.72%）；随后为 `orientation_manipulation`（14.16%）和 `insertion_assembly`（10.51%）。
- 本轮扩充已经显著补齐旋转、插入装配、工具/持续接触、多对象重排、顺序执行和执行后恢复；柔性物体、真实力/触觉接触和跨模拟器在线 replay 仍未覆盖。

## 4. 当前五套官方数据主线

五套数据的主体均属于：

> **官方成功轨迹重组＋任务意图/阶段标签增强**

它们不是模拟器安装时自动附带的数据，也不是本机重新运行模拟器生成的新轨迹。原始图像、状态、动作、成功凭据和 provenance 保持各平台原生格式；统一的仅是任务语义 sidecar。candidate G 不作为接纳条件、分布定义或默认训练输入。

| 平台 | 固定官方来源 | 是否安装自带 | 当前扩充方式 | 当前状态 |
|---|---|---|---|---|
| CALVIN | Freiburg `task_ABC_D.zip` | 否 | 从官方成功标注中选 12 个任务，在 A/B/C 三场景各取 12 段；保留 NPZ 图像、状态和动作，增加任务意图及状态运动阶段 sidecar | 432 段、25,852 帧；全量回读通过 |
| LIBERO | `yifengzhu-hf/LIBERO-datasets`，revision `f13aa24a3da8c43c7225569f28c562979fa0e35a` | 否 | 纳入 `libero_10`、`libero_spatial`、`libero_object`、`libero_goal`；逐 HDF5 校验 SHA-256 和 terminal reward/done，增加语义 sidecar | 40 任务、2,000 parents、338,575 transitions；全量回读通过 |
| RoboTwin | `TianxingChen/RoboTwin2.0`，revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186` | 否 | 下载并核对 50 个 `demo_clean.zip`；回读 HDF5 状态、动作、seed、scene 和 4 路相机，增加语义 sidecar | 50/50 归档、2,500 parents、548,893 transitions、2,195,572 RGB 帧；全量回读通过 |
| ManiSkill | `haosulab/ManiSkill_Demonstrations`，revision `d674485bbffdd533914e52d272fdda34c0515608` | 否 | 选择 9 个非普通抓放任务，逐 ZIP/JSON/HDF5 检查原生 success、T/T+1、控制模式和环境参数 | 15,262 parents、1,581,768 transitions；9/9 ZIP 和全量回读通过 |
| VIMA-Bench | `VIMA/VIMA-Data`，revision `f35cee89382278b8d63d6d6f9c27cfeda3deb3f5`；代码 commit `97e0af11e126fd477c9d81eeabfb6c739021c680` | 否 | 对 21.5 GB 官方 ZIP 做 revision-pinned Range 稀疏抽取；保留 prompt、动作、观测、front/top RGB、分割和原生 success/failure | 5,999 parents、12,668 actions、18,667 observations；CRC/SHA 和全量回读通过 |

## 5. 各平台具体做了什么

### 5.1 CALVIN

- 数据来自官方 `task_ABC_D.zip`，不是本机专家新生成。
- 432 段 = 12 任务 × 3 场景 × 12 段。
- manifest 中的“324 retained＋108 new”表示 324 段沿用上一版选择、108 段从同一官方数据新增选择；不是重新执行 108 条轨迹。
- `rgb_static`、`rgb_gripper`、`robot_obs`、`scene_obs`、`rel_actions` 不改写。
- 只新增任务意图和基于原生状态运动的阶段视图。

证据：

```text
data/calvin/task_ABC_D_g_intent_multifold_s12/g_intent_multifold_manifest.json
```

### 5.2 LIBERO

- 官方 HDF5 来自固定 revision 的 Hugging Face 数据集。
- Gina 中的 payload 是从本机已有官方缓存按字节导入，但每个文件都重新与官方清单 SHA-256 核对；没有采用其他项目的实验输出。
- 40 个任务，每个任务 50 条 demo，共 2,000 条。
- 成功依据为原始 HDF5 terminal reward/done，不使用候选 G。

证据：

```text
data/libero/official_f13aa24a3da8c43c7225569f28c562979fa0e35a/official_inventory_manifest.json
```

### 5.3 RoboTwin

- 官方 `demo_clean.zip` 是作者运行原生规划专家后发布的数据，但不随代码仓库自动安装。
- 官方生成流程要求 `plan_success`、`check_success()`、`planned_joints_legal()`，并在保存后回放再次执行 `check_success()`。
- 本项目没有重新采集这 2,500 条，而是完成 50/50 归档下载、LFS SHA-256 核验、HDF5 全量回读和语义 sidecar。
- 原 HDF5 不改写，也不将 RoboTwin 动作向量与其他模拟器直接拼接。

证据：

```text
data/robotwin/official_3dc3b798668feb99ac61cc9086d84cbcc3d79186/remote_inventory.json
data/semantic_indexes/robotwin_official_20261022_v3/
```

### 5.4 ManiSkill

- 选择的 9 个任务为：`LiftPegUpright-v1`、`PegInsertionSide-v1`、`PlugCharger-v1`、`PokeCube-v1`、`PullCubeTool-v1`、`PushT-v1`、`RollBall-v1`、`DrawTriangle-v1`、`StackPyramid-v1`。
- 普通 `PickCube-v1` 被有意排除，避免继续堆普通抓放。
- 9/9 官方 ZIP 均与固定 revision 的 Hugging Face LFS SHA-256 一致。
- 25 个没有原生成功证据的 `PegInsertionSide-v1` 候选被排除，未放宽成功门槛。
- 官方压缩 payload 主要为 `none/state` observation mode；其中 5,975 个 parents 有 state observation group，不能宣称获得 RGB/RGB-D 轨迹。

证据：

```text
data/maniskill/official_d674485bbffdd533914e52d272fdda34c0515608/remote_inventory.json
data/semantic_indexes/maniskill_official_20261023_v2/
docs/operations/20261023-maniskill-official-trajectory-expansion.md
```

### 5.5 VIMA-Bench

- 选择六类语义缺口任务：`rotate`、`twist`、`sweep_without_exceeding`、`follow_order`、`pick_in_order_then_restore`、`rearrange_then_restore`。
- 每类确定性选择 1,000 个候选；最终接纳 5,999 条。
- `pick_in_order_then_restore/105017` 虽有原生 success，但其 `T=3`、结构化 observation=4、RGB=5，违反 T+1 对齐，被排除；没有静默裁帧或任务特例。
- 约 21.5 GB 的完整 ZIP 未整包复制；固定数据 revision 和公开 LFS SHA-256，读取完整 ZIP 中央目录，仅抽取选中成员并逐成员验证 compressed/uncompressed size、compression method、CRC32 和本地 SHA-256。
- 所有接纳轨迹保留 prompt、action、structured observation、front/top RGB、segmentation 和原生成功凭据。

证据：

```text
data/vima/official_f35cee89382278b8d63d6d6f9c27cfeda3deb3f5/selected_semantic_gap_v1/
data/semantic_indexes/vima_official_20261023_v2/
docs/operations/20261023-vima-official-trajectory-expansion.md
```

## 6. 唯一真正新生成的轨迹

单独封存的 5 条 RoboTwin `place_empty_cup` pilot 是本机实际运行 RoboTwin 原生规划/回放流程后新产生的：

- 5 条父轨迹；
- 866 transitions；
- native success 和 payload hash 已回读；
- provenance 为 `online_native_expert_collection`；
- 不并入上述 26,193 条官方数据统计。

对应索引：

```text
data/semantic_indexes/online_native_expert_20261022_v1/
```

## 7. Canonical 索引与核验证据

当前五来源官方 canonical 统一语义视图：

```text
data/semantic_indexes/official_reorg_20261023_v6/
```

关键结果：

- 26,193 个唯一官方成功父轨迹；
- 2,507,756 个原生 frames/transitions；
- 95,033 个派生阶段视图；
- `records.jsonl` SHA-256：`81023ae40c4a5e7680e026e782cf3ccaab64ef08836b0087dc00170817b26a2f`；
- 独立统一回读：`PASS`。

注意：统一视图只统一语义字段，不混合各模拟器原始动作/状态向量；阶段视图不构成新的独立成功轨迹。

## 8. 准确结论与在线边界

准确表述是：

> CALVIN、LIBERO、RoboTwin、ManiSkill 和 VIMA-Bench 当前主体扩充，都是从固定官方来源增加选取范围，并进行成功轨迹重组与任务意图/阶段标签增强；不是模拟器安装自带，也不是本机重新执行生成。只有隔离的 5 条 RoboTwin pilot 属于真正的新轨迹采集。

状态区分：

- **已实现并离线全量验证**：五来源官方 payload/index、原生成功证据、长度/阶段/唯一性、CRC/SHA 和 unified index。
- **已有在线原生执行证据**：仅隔离的 5 条 RoboTwin pilot。
- **未在线验证**：本轮 ManiSkill/SAPIEN 和 VIMA-Bench/PyBullet 官方数据没有在模拟器中逐条 replay。
- **未完成**：跨来源统一训练窗口、动作空间适配、训练、held-out 评估和可靠泛化验证。

## 9. 任务意图代表视频覆盖

按 20 个 canonical 任务意图逐类核对：

- 原有代表视频：9 类；
- 本次从完整官方成功父轨迹新增导出：9 类；
- 当前专属代表视频覆盖：**18/20**。

本次新增：

| 任务意图 | 来源 | 代表任务 | 视频 |
|---|---|---|---|
| `orientation_manipulation` | VIMA | `twist` | `10_orientation_vima.mp4` |
| `constraint_aware_tool_interaction` | VIMA | `sweep_without_exceeding` | `11_constraint_tool_vima.mp4` |
| `long_horizon_rearrangement` | VIMA | `rearrange_then_restore` | `12_long_rearrangement_vima.mp4` |
| `ordered_imitation` | VIMA | `follow_order` | `13_ordered_imitation_vima.mp4` |
| `long_horizon_memory_manipulation` | VIMA | `pick_in_order_then_restore` | `14_long_memory_vima.mp4` |
| `block_push` | CALVIN | `push_red_block_right` | `15_block_push_calvin.mp4` |
| `switch_activation` | LIBERO | `turn_on_the_stove` | `16_switch_activation_libero.mp4` |
| `articulation` | LIBERO | `open_the_middle_drawer_of_the_cabinet` | `17_articulation_libero.mp4` |
| `container_transport` | CALVIN | `push_into_drawer` | `18_container_transport_calvin.mp4` |

视频目录：

```text
artifacts/videos/missing_intents_20261023/
```

尚缺两类：

| 任务意图 | 当前数量 | 状态 | 原因 |
|---|---:|---|---|
| `insertion_assembly` | 2,975 | `BLOCKED_NO_OFFICIAL_VISUAL_PAYLOAD` | 当前选中的 ManiSkill 官方 payload 仅有 state/action；需要单独执行并持久化 SAPIEN replay |
| `rolling_contact` | 2,373 | `BLOCKED_NO_OFFICIAL_VISUAL_PAYLOAD` | 当前选中的 ManiSkill 官方 payload 仅有 state/action；需要单独执行并持久化 SAPIEN replay |

这些视频是对已有官方视觉观测的**导出**，不是新录制的物理轨迹，也不构成本机在线 simulator replay 证据。9/9 新视频均通过 ffprobe、哈希、早/中/晚帧解码和 contact-sheet 视觉检查。
