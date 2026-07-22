# Notion Papers Local Mirror

Source: Notion `Paper Tracking` / `Papers` database, with local-only public metadata enrichment.

Rules:

- This is a local, read-only project mirror.
- Do not edit cloud Notion from AutoResearch.
- `local_enrichment` fields are local verification/fill-ins, not cloud Notion fields.
- Use Notion hints as external evidence, then verify paper claims before
  turning them into local design decisions.

Rows mirrored: 102
arXiv API verified rows: 0
manual public-lookup rows: 18
deterministic arXiv-ID fill rows: 77

## High-Relevance Records

### GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors

- Rating: A
- Year: 2026
- URL: https://arxiv.org/abs/2606.05160
- Local verified: manual public lookup
- Topic: Humanoid, Loco-Manipulation, Data Generation, Sim-to-Real, Embodied AI
- Method Pattern: Synthetic 4D HOI Generation, Video Prior, Retargeting, Sim-to-Real
- Use For: Related Work, Method Pattern, Experiment, Dataset
- Idea Hook: 把GRAIL式生成轨迹作为高层memory/intent数据源，再训练低层spectral residual controller。
- Key Idea: 把视频生成先验放进完全可控的3D场景中，生成可重建、可retarget的人形loco-manipulation数据；用生成数据训练真实机器人策略。
- Limitation: 核心依赖可控3D资产和重建管线；对开放世界真实视频直接使用能力有限。
- Relevance to My Work: 与你的loco-manipulation、world model和数据扩展方向高度相关。可作为“用生成模型补机器人数据瓶颈”的强相关工作，也能支撑你的memory-conditioned hierarchical VLA/RL中数据生成和sim-to-real部分。

### FALCON: Actively Decoupled Visuomotor Policies for Loco-Manipulation with Foundation-Model-Based Coordination

- Rating: A
- Year: 2025
- URL: https://arxiv.org/abs/2512.04381
- Local verified: manual public lookup
- Topic: Loco-Manipulation, VLA, Diffusion Policy, Foundation Model, Hierarchical Policy
- Method Pattern: Decoupled Policies, Foundation Model Coordinator, Diffusion Policy, Phase Progress
- Use For: Related Work, Baseline, Method Pattern
- Idea Hook: 把FALCON的子系统解耦与CSP的频谱解耦对比，形成“结构解耦 vs 动作频谱解耦”的实验线。
- Key Idea: 主动解耦异质子系统，但用foundation model恢复全局协调，避免单一策略融合异构观测导致性能下降。
- Limitation: 主要是loco-manipulation场景，尚未覆盖个性化记忆和长期spatial memory。
- Relevance to My Work: 非常贴近你的“解耦策略 + foundation model协调”方向，可作为直接竞品/支撑论文。它给出与spectral hierarchy不同的模块化解耦路线。

### Bring My Cup! Personalizing Vision-Language-Action Models with Visual Attentive Prompting

- Rating: A
- Year: 2025
- URL: https://arxiv.org/abs/2512.20014
- Local verified: manual public lookup
- Topic: VLA, Personalization, Embodied AI, Visual Prompting, Robot Manipulation
- Method Pattern: Frozen VLA Adapter, Reference Memory, Visual Prompting, Open-vocabulary Grounding
- Use For: Related Work, Motivation, Method Pattern, Experiment
- Idea Hook: 能否把VAP式reference memory扩展成长期 embodied memory，并让低层RL使用视觉提示/目标mask作为持续奖励或goal signal？
- Key Idea: 把personal object grounding做成冻结VLA前端的visual attentive prompt：reference images -> object matching -> visual highlight + instruction rewrite。核心价值是不用训练VLA即可补足实例级个性化识别。
- Limitation: 主要解决实例级个性化grounding，动作学习本身仍依赖已有VLA；没有把个性化记忆转化为可学习的长期策略或RL奖励。
- Relevance to My Work: 对你的VLA/RL解耦路线很相关：它证明高层视觉 grounding 可以作为外部adapter服务于冻结VLA。可作为“冻结基础模型 + 可插拔感知/记忆模块”的相关工作，也能启发个性化目标、reference memory、视觉提示作为goal conditioning。

### VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training

- Rating: S
- Year: 2022
- URL: https://arxiv.org/abs/2210.00030
- Local filled: arXiv ID/URL from existing ID
- Topic: Visual Reward, Representation Learning, Offline RL, Robot Manipulation
- Method Pattern: Value-Implicit Pretraining, Visual Reward, Frozen Representation
- Use For: Duplicate Check, Related Work, Motivation
- Idea Hook: 可作为memory-conditioned control中的reward/goal embedding来源。
- Key Idea: 用value-implicit pretraining让视觉表示同时承担representation和reward function，为下游机器人任务提供稠密奖励。
- Limitation: 这是重复候选；应避免在digest和related work中重复计数。
- Relevance to My Work: 对你的RL reward生成、frozen perception + RL路线是关键基础论文，但当前条目应与已有VIP记录合并或去重。

### Goal2Pixel: Grounding Goals to Pixels for Vision-Language Navigation

- Rating: A
- Year: 2026
- URL: https://arxiv.org/abs/2606.01621
- Local verified: manual public lookup
- Topic: VLN, Navigation, Visual Grounding, Embodied AI, Spatial Memory
- Method Pattern: Pixel Grounding, Waypoint Interface, Keyframe Memory, VLM-to-Control Bridge
- Use For: Related Work, Method Pattern, Experiment
- Idea Hook: 把Goal2Pixel的pixel waypoint作为低频目标，把CSP/控制器作为高频执行层。
- Key Idea: 把语言导航决策从动作空间转成像素 grounding 接口：VLM负责可视目标点，几何模块负责导航执行。
- Limitation: 主要面向可见导航像素和VLN-CE，复杂交互和不可见目标的长期规划仍有限。
- Relevance to My Work: 非常契合“VLA输出低频/中间目标，低层控制执行”的框架。它是导航版的中间表示桥接，可以作为spectral hierarchical policy的VLN类比。

### Sumo: Dynamic and Generalizable Whole-Body Loco-Manipulation

- Rating: A
- Year: 2026
- URL: https://arxiv.org/abs/2604.08508
- Local verified: manual public lookup
- Topic: Loco-Manipulation, Whole-Body Control, Planning, Sim-to-Real, Legged Robot
- Method Pattern: Test-time Steering, Sample-based Planning, Whole-body Control
- Use For: Related Work, Baseline, Method Pattern
- Idea Hook: 用VLA/memory输出动态cost或目标，让Sumo式test-time steering成为低层执行器。
- Key Idea: 用测试时规划调制预训练whole-body policy，动态改变cost function实现新任务泛化。
- Limitation: 语言/视觉语义层较弱，更多是控制与规划层贡献。
- Relevance to My Work: 对低层控制/高层规划接口很有参考价值：高层不必重训策略，可以通过steering/cost调制已有控制器。

### FlowBot++: Learning Generalized Articulated Objects Manipulation via Articulation Projection

- Rating: A
- Year: 2023
- URL: https://arxiv.org/abs/2306.12893
- Local verified: manual public lookup
- Topic: Articulated Object, Manipulation, 3D Vision, Dense Flow, Robot Manipulation
- Method Pattern: Dense 3D Flow, Articulation Projection, Affordance-like Representation
- Use For: Related Work, Method Pattern, Baseline
- Idea Hook: 将FlowBot++的3D motion目标作为VLA低频意图，交给RL/优化控制器执行。
- Key Idea: 把可动关节物体的操作目标转成密集3D motion/articulation projection，再驱动下游操作。
- Limitation: 专注 articulated objects，语言和长期规划不是核心。
- Relevance to My Work: 对affordance/flow-based manipulation和3D中间表示很相关，可作为视觉目标/flow interface的基础论文。

### 3DFlowAction: Learning Cross-Embodiment Manipulation from 3D Flow World Model

- Rating: A
- Year: 2025
- URL: https://arxiv.org/abs/2506.06199
- Local verified: manual public lookup
- Topic: 3D Flow, World Model, Cross-Embodiment, Robot Manipulation, VLA
- Method Pattern: 3D Flow World Model, Cross-Embodiment Representation, Flow-guided Planning
- Use For: Related Work, Method, Idea Seed
- Idea Hook: 把3DFlowAction的3D flow当作低频语义动作，低层学习高频执行残差。
- Key Idea: 3D object flow是跨机器人形态的中间动作表示，能把人类/机器人数据统一到同一物体运动空间。
- Limitation: 依赖3D flow数据合成和优化策略；语言长时程规划不是核心。
- Relevance to My Work: 非常契合你的中间表示/低频目标路线，可作为memory-conditioned spectral controller的核心相关工作。

### RT-Affordance: Affordances are Versatile Intermediate Representations for Robot Manipulation

- Rating: A
- Year: 2024
- URL: https://arxiv.org/abs/2411.02704
- Local filled: arXiv ID/URL from existing ID
- Key Idea: [T1] Affordance = 末端位姿序列。分层：VLM生成affordance plan -> overlay手部轮廓 -> VLA执行。与用户框架关系：相似分层引导范式但用文本affordance+IL(非几何goal+GCRL)。Related Work必引直接竞品。

### AffordDP: Generalizable Diffusion Policy with Transferable Affordance

- Rating: A
- Year: 2024
- URL: https://arxiv.org/abs/2412.03142
- Local filled: arXiv ID/URL from existing ID
- Key Idea: [T1] 3D affordance (contact point + trajectory) 通过SD-DINOv2+ICP跨类别迁移。与用户框架关系：相同affordance as goal理念但用IL(非GCRL)。对GCRL的关键发现：trajectory对IL下游必要，对GCRL下游可能冗余(contact point作为goal即可)。

### Hierarchical Planning with Latent World Models

- Rating: A
- Year: 2026
- URL: https://arxiv.org/abs/2604.03208
- Local verified: manual public lookup
- Topic: World Model, Hierarchical Planning, MPC, Robot Manipulation, Navigation
- Method Pattern: Latent World Model, Temporal Hierarchy, MPC, Coarse-to-Fine Planning
- Use For: Related Work, Method, Baseline
- Idea Hook: 把多时间尺度world model和VLA/VLM的semantic intent结合，形成长期规划+低层执行闭环。
- Key Idea: 通过多时间尺度latent world model把长时程控制拆成高层粗规划和低层精细规划。
- Limitation: 偏planning/MPC范式，对VLA语言 grounding 和个性化记忆没有直接覆盖。
- Relevance to My Work: 直接支撑你的hierarchical world model和memory-conditioned planning方向，可作为Draft v0中的世界模型/规划核心相关工作。

### Hierarchical Policy Learning via Spectral Decomposition

- Rating: A
- Year: 2026
- URL: https://arxiv.org/abs/2606.29570
- Local verified: manual public lookup
- Topic: Hierarchical Policy, Robot Manipulation, VLA, Imitation Learning, Action Decomposition
- Method Pattern: Spectral Action Decomposition, Coarse-to-fine Policy, Hierarchical Control, DCT
- Use For: Method, Baseline, Idea Seed, Experiment
- Idea Hook: 把VLA输出或VLM grounding映射到低频动作/目标轨迹，再让RL学习高频修正，形成“semantic low-frequency plan + RL high-frequency controller”的论文主线。
- Key Idea: 把层次策略从语义子任务分解转为动作频域分解：低频=任务级运动意图，高频=执行级修正。CSP提供一种连续动作空间中的层次结构，而不是离散skill或subgoal。
- Limitation: 重点在动作序列结构和模仿学习/策略建模，对语言-视觉长期记忆、开放世界目标泛化和online RL探索覆盖有限。
- Relevance to My Work: 非常适合作为你“高层语义/目标 + 低层控制”框架的技术支撑。它提供了一个可验证的动作分解假设：高层模块不一定输出离散subtask，也可以输出低频轨迹/粗动作，低层RL负责高频接触修正。

### VIP: Vision Instructed Pre-training for Robotic Manipulation

- Rating: B
- Year: 2024
- URL: https://arxiv.org/abs/2410.07169
- Local verified: manual public lookup
- Topic: Robot Manipulation, Vision Instruction, Pretraining, Policy Learning
- Method Pattern: Vision Instruction, Point Flow, Goal-conditioned Policy
- Use For: Related Work, Method Pattern
- Idea Hook: 把视觉指令/point flow作为memory-conditioned VLA输出的低频目标。
- Key Idea: 用视觉目标/point flow替代或补充文本目标，让机器人策略通过视觉指令学习任务条件化。
- Limitation: 主要关注视觉目标表达，缺少长期记忆和开放世界语义目标。
- Relevance to My Work: 与3DFlowAction、Goal2Pixel都指向“中间视觉目标/flow作为低频控制接口”，可支撑你的视觉目标 grounding + 低层控制路线。

### HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control

- Rating: A
- Year: 2026
- URL: https://arxiv.org/abs/2607.03449
- Local verified: manual public lookup
- Topic: Embodied Memory, Long-Horizon Control, VLA, Hierarchical Policy, Embodied AI
- Method Pattern: Hierarchical Memory, Long-horizon VLA, Memory-conditioned Control
- Use For: Idea Seed, Related Work, Motivation
- Idea Hook: 把VAP的reference memory、HiMe的long-horizon memory、CSP的动作层次分解合并，形成“memory-conditioned hierarchical VLA/RL”的初版论文方向。
- Key Idea: 待核验：将embodied memory分层组织，用于支持长时程VLA控制，缓解单步VLA在长期任务中的遗忘、状态漂移和错误累积。
- Limitation: 公开元数据尚未核验；暂不能确认方法细节、实验平台和是否有真实机器人结果。
- Relevance to My Work: 与你的知识库核心方向高度相关：长期记忆、VLA控制、层次结构都直接支撑“从论文收集到idea孵化”的主线。需要优先补全论文链接和摘要后判断是否作为Key Paper。

### A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation

- Rating: S
- Year: 2025
- URL: https://arxiv.org/abs/2504.12636
- Local filled: arXiv ID/URL from existing ID
- Key Idea: 层次化可供性感知扩散模型。高层：与具身无关的可供性表示(Embodiment-Agnostic Affordance)，预测接触点+接触后轨迹。低层：动作执行。100万接触点预训练+轨迹微调，跨4种机器人泛化。ICCV 2025

### Open-World Drone Active Tracking with Goal-Centered Rewards

- Rating: A
- Year: 2024
- URL: https://arxiv.org/abs/2412.00744
- Local filled: arXiv ID/URL from existing ID
- Topic: Visual Tracking, Reinforcement Learning, Object Tracking
- Key Idea: 目标中心奖励塑形+课程训练策略提升无人机在开放世界复杂环境中的主动跟踪
- Relevance to My Work: 无人机视觉主动追踪，与 TrackVLA 方向直接相关

### TrackVLA++: Unleashing Reasoning and Memory Capabilities in VLA Models for Embodied Visual Tracking

- Rating: S
- Year: 2025
- URL: https://arxiv.org/abs/2509.12129
- Local filled: arXiv ID/URL from existing ID
- Topic: Embodied AI, Navigation, VLA, Foundation Model
- Key Idea: VLA模型融合极坐标思维链推理与门控记忆模块，实现鲁棒的长程具身视觉跟踪
- Relevance to My Work: 自己的核心工作，TrackVLA 系列

### World Action Models: The Next Frontier in Embodied AI

- Rating: A
- Year: 2026
- URL: https://arxiv.org/abs/2605.12090
- Local verified: manual public lookup
- Topic: World Action Model, VLA, World Model, Embodied AI, Survey
- Method Pattern: World Action Model, VLA + World Model, Survey Taxonomy
- Use For: Related Work, Motivation, Taxonomy
- Idea Hook: 把你的memory-conditioned spectral hierarchy定位为WAM的一种action decoding/conditioning机制。
- Key Idea: VLA的下一步不是单纯observation-to-action，而是联合建模未来状态和动作的World Action Model。
- Limitation: Survey本身不提供可直接复现实验，但适合定义问题空间和gap。
- Relevance to My Work: 可作为你论文intro和related work的上位框架，帮助把VLA、world model、memory和action hierarchy组织成统一叙事。

### VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-training

- Rating: A
- Year: 2022
- URL: https://arxiv.org/abs/2210.00030
- Local filled: arXiv ID/URL from existing ID
- Key Idea: 将视觉表示学习重构为离线goal-conditioned RL问题，自监督预训练得到通用视觉奖励和表示。冻结后可同时提供视觉表示和密集奖励信号。与R3M互补：VIP额外提供奖励信号。启发：VLM特征是否也能提供内在奖励？

### R3M: A Universal Visual Representation for Robot Manipulation

- Rating: S
- Year: 2022
- URL: https://arxiv.org/abs/2203.12601
- Local filled: arXiv ID/URL from existing ID
- Key Idea: 在Ego4D人类视频上用时间对比学习+视频语言对齐+稀疏正则化预训练视觉表示。冻结后作为downstream RL/IL的感知模块。frozen感知+RL路线的鼻祖，证明frozen视觉表示可高效迁移到manipulation，但缺少语言条件能力。

### TarGF: Learning Target Gradient Field to Rearrange Objects without Explicit Goal Specification

- Rating: S
- Year: 2022
- URL: https://arxiv.org/abs/2209.00853
- Local filled: arXiv ID/URL from existing ID
- Key Idea: 【方法类别】策略型 【核心技术】基于Score Matching学习目标梯度场(TarGF)，仅需目标分布样例无需显式目标规格。双应用范式：基于模型规划(梯度→参考控制+分布式路径规划器)和RL(似然变化奖励+残差策略学习)

### Realtime-VLA FLASH: Speculative Inference for Diffusion-based VLAs

- Rating: A
- Year: 2026
- URL: https://arxiv.org/abs/2605.13778
- Local filled: arXiv ID/URL from existing ID
- Topic: VLA, Diffusion Policy, Foundation Model, Embodied AI
- Key Idea: Realtime-VLA FLASH: 首个扩散VLA投机推理框架。轻量草稿模型(7.8ms)+主模型并行验证+相位感知回退，3.04x加速(58ms→19.1ms)几乎零性能损失。解决扩散VLA实时部署根本瓶颈
- Relevance to My Work: ⚠️ 部分参考: 救援场景实时控制需求高，3x加速方案有价值。但当前聚焦操作任务非导航/移动。相位感知回退思路可用于VLA-SAR任务阶段自适应。关联: VLA工程化/实时推理优化

### VL-Nav: A Neuro-Symbolic Approach for Reasoning-based Vision-Language Navigation

- Rating: A
- Year: 2025
- URL: https://arxiv.org/abs/2512.08186
- Local filled: arXiv ID/URL from existing ID
- Topic: Embodied AI, VLN, Navigation, Diffusion Policy, VLA
- Key Idea: Neuro-Symbolic 架构（神经感知 + 符号规划）；Curiosity-based Exploration；像素级 VL 特征对齐；零样本泛化。
- Relevance to My Work: 探索策略可迁移到 EVT 场景；Neuro-Symbolic 为具身 AI 推理提供新思路；开放世界语义理解可增强目标发现能力。

### Hierarchical Instruction-aware Embodied Visual Tracking

- Rating: S
- Year: 2025
- URL: https://arxiv.org/abs/2505.20710
- Local filled: arXiv ID/URL from existing ID
- Topic: Visual Tracking, Embodied AI, Vision-Language
- Key Idea: 分层指令感知框架：LLM语义-空间目标对齐器连接高层指令与RL动作策略
- Relevance to My Work: 自己的工作

### TrackVLA: Embodied Visual Tracking in the Wild

- Rating: S
- Year: 2025
- URL: https://arxiv.org/abs/2505.23189
- Local filled: arXiv ID/URL from existing ID
- Topic: Embodied AI, Visual Tracking, VLA
- Key Idea: 将 VLA 模型引入具身视觉追踪，实现识别+追踪的协同学习
- Relevance to My Work: 自己的核心工作，VLA+追踪的开创性尝试

### RoboDuet: Learning a Cooperative Policy for Whole-body Legged Loco-Manipulation

- Rating: B
- Year: 2024
- URL: https://arxiv.org/abs/2403.17367
- Local filled: arXiv ID/URL from existing ID
- Topic: Loco-Manipulation, Whole-Body Control, Legged Robot, Cooperative Policy
- Method Pattern: Two-policy Coordination, Whole-body Loco-Manipulation
- Use For: Related Work, Baseline
- Idea Hook: 比较RoboDuet的手工/学习协同与FALCON/VLA协调方式。
- Key Idea: 用two-policy framework解决四足移动底盘和机械臂的协调问题，而不是单一集中策略。
- Limitation: 缺少foundation model语义协调和长期任务规划。
- Relevance to My Work: 是FALCON之前/相邻的解耦loco-manipulation路线，可作为related work和baseline。

### Learning World Models for Unconstrained Goal Navigation

- Rating: B
- Year: 2024
- URL: https://arxiv.org/abs/2411.02446
- Local verified: manual public lookup
- Topic: World Model, Goal Navigation, Reinforcement Learning, Exploration
- Method Pattern: Goal-conditioned World Model, Replay Key States, Unconstrained Navigation
- Use For: Related Work, Method Pattern
- Idea Hook: 将memory检索出的关键状态作为MUN式subgoal图，服务长时程VLA导航。
- Key Idea: 把world model用于任意subgoal状态之间的可达性/转移建模，从而支持goal-directed exploration。
- Limitation: 与VLA/真实机器人接口较弱，主要是RL/world model范式。
- Relevance to My Work: 对你的world model + navigation + exploration方向有参考价值，特别是key states/subgoals和memory检索结合。

### Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects

- Rating: B
- Year: 2025
- URL: https://arxiv.org/abs/2502.07005
- Local verified: manual public lookup
- Topic: Reinforcement Learning, Deformable Object, Geometry, SE(3), Robot Manipulation
- Method Pattern: SE(3)-Equivariant Policy, Heterogeneous Graph, Geometry-aware RL
- Use For: Method Pattern, Baseline, Experiment
- Idea Hook: 让memory/VLA输出低频目标，低层采用geometry-aware RL执行复杂物体操作。
- Key Idea: 显式几何和异构关系建模能显著增强RL manipulation在不同形状/可变形物体上的泛化。
- Limitation: 主要解决几何泛化，不涉及语言和记忆。
- Relevance to My Work: 对低层RL控制器的几何归纳偏置很有价值，可作为high-frequency controller的policy architecture参考。

### LongSpace: Exploring Long-Horizon Spatial Memory from Perception to Recall in Video

- Rating: A
- Year: 2026
- URL: https://arxiv.org/abs/2606.05677
- Local verified: manual public lookup
- Topic: Spatial Memory, Long-Horizon Memory, Video Understanding, Embodied AI, Navigation
- Method Pattern: Long-video Spatial Memory, 3D Structural Cues, Layer-aware Memory
- Use For: Related Work, Method Pattern, Motivation
- Idea Hook: 把LongSpace式空间记忆作为VLA控制器的历史状态和目标检索模块。
- Key Idea: 空间记忆不是简单长上下文，而需要结构化3D cues和分层/层感知memory检索。
- Limitation: 主要是video MLLM reasoning，不直接输出控制动作。
- Relevance to My Work: 与HiMe、UniMapping和memory-conditioned VLA/RL强相关，可作为长期空间记忆层的关键参考。

### ActionX: Pre-training Action Experts with Reinforcement Learning for VLA Models

- Rating: A
- Year: 2026
- URL: https://app.notion.com/p/ActionX-Pre-training-Action-Experts-with-Reinforcement-Learning-for-VLA-Models-37e573cfca7c814a8c4ce79f0c0e8650
- Key Idea: 两阶段VLA训练框架：Stage1冻结VLM(InternVL3-1B)，用PPO预训练action expert(8层transformer+flow matching)；Stage2解冻全参数SFT。LIBERO 91.5%，Meta-World 81.3%。核心发现：RL预训练对空间推理任务至关重要(Spatial子集0%->95%)。

### SocialGFs: Learning Social Gradient Fields for Adaptive Multi-Agent Systems

- Rating: S
- Year: 2024
- URL: https://arxiv.org/abs/2405.01839
- Local filled: arXiv ID/URL from existing ID
- Key Idea: 【方法类别】策略型 【核心技术】Denoising Score Matching学习社会梯度场(SocialGFs)，将多智能体复杂交互编码为社会力(环境力/智能体间力/内在动机力)。离线数据学习，无需在线交互。集成MAPPO实现跨任务迁移+信用分配+可扩展性

### Tool-Aligned VLA: Towards Long-horizon Embodied Agents

- Rating: S
- Year: 2026
- URL: https://arxiv.org/abs/2605.13119
- Local filled: arXiv ID/URL from existing ID
- Topic: VLA, Foundation Model, Long-horizon Planning, Embodied AI, LLM Agent
- Key Idea: VLAs-as-Tools: VLA即工具族新范式。高层VLM(场景分析/全局规划/恢复)+专用VLA工具族(局部操作)，TAPT后训练残差适配器专化。LIBERO-Long +4.8pp, RoboTwin +23.1pp
- Relevance to My Work: ✅ 高参考价值: 长时序救援任务天然映射为工具链(搜→定→救→撤)。VLM高层规划+专用VLA工具分层设计与VLA-SAR架构高度吻合。事件触发重规划适合边缘部署。关联Idea: VLA-SAR系统架构/长时序具身任务分解

### NS-Mem: Advancing Multimodal Agent Reasoning with Long-Term Neuro-Symbolic Memory

- Rating: S
- Year: 2026
- URL: https://arxiv.org/abs/2603.15280
- Local filled: arXiv ID/URL from existing ID
- Topic: Embodied AI, Vision-Language
- Key Idea: 三层神经符号记忆（情景 + 语义 + 逻辑规则）；SK-Gen 自动知识整合；混合检索（相似性 + 符号查询）；长期记忆支持复杂推理。
- Relevance to My Work: 为 EVT 提供长期记忆方案：记录目标历史轨迹、环境布局、重获策略；神经符号混合查询支持复杂重获场景推理；可提升长时追踪的连贯性。

### VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation

- Rating: S
- Year: 2024
- URL: https://github.com/bdaiinstitute/vlfm
- Local filled: arXiv ID/URL from existing ID
- Topic: Navigation, Vision-Language, Benchmark
- Key Idea: Language-Grounded Value Map（语言锚定价值图）；前沿探索 + VLM 语义评估；Spot 真机部署；代码开源（MIT License）。
- Relevance to My Work: 前沿探索 + VLM 语义评估可直接用于目标重获；代码开源且已真机验证，工程实现可参考；与 UnrealZOO 仿真环境结合潜力大。

### VLM Can Be a Good Assistant: Enhancing Embodied Visual Tracking with Self-Improving VLMs

- Rating: S
- Year: 2025
- URL: https://scholar.google.com/citations?user=lzsXQ5AAAAAJ
- Local filled: arXiv ID/URL from existing ID
- Topic: Visual Tracking, Embodied AI, Vision-Language
- Key Idea: 混合策略：快速视觉跟踪+自改进VLM推理恢复，实现主动式目标重捕获
- Relevance to My Work: 自己的工作，VLM 辅助追踪方向

### UAV-ON: A Benchmark for Open-World Object Goal Navigation for UAVs

- Rating: A
- Year: 2025
- URL: https://arxiv.org/abs/2508.00288
- Local filled: arXiv ID/URL from existing ID
- Topic: Navigation, Benchmark, Embodied AI
- Key Idea: 无人机开放世界物体目标导航基准：14个UE环境+1270标注物体+实例级语义目标
- Relevance to My Work: 与 UnrealZOO 场景相关，UAV 追踪方向参考

### OpenVLA: An Open-Source Vision-Language-Action Model

- Rating: S
- Year: 2024
- URL: https://arxiv.org/abs/2406.09246
- Local filled: arXiv ID/URL from existing ID
- Topic: VLA, Embodied AI, Vision-Language
- Key Idea: 大规模开源 VLA baseline，为社区提供强力基础模型
- Relevance to My Work: VLA 领域最重要的开源 baseline，直接相关

### Embrace-3K: Embodied Reasoning and Action in Complex Environments

- Rating: S
- Year: 2025
- URL: https://arxiv.org/abs/2507.10548
- Local filled: arXiv ID/URL from existing ID
- Topic: Embodied AI, Benchmark, Vision-Language
- Key Idea: 基于UE+UnrealCV-Zoo构建3000+语言引导任务的具身推理与行动基准，评估VLMs具身能力
- Relevance to My Work: 自己的工作，具身推理方向

### Empowering Embodied Visual Tracking with Visual Foundation Models and Offline RL

- Rating: S
- Year: 2024
- URL: https://arxiv.org/abs/2409.09798
- Local filled: arXiv ID/URL from existing ID
- Topic: Visual Tracking, Reinforcement Learning, Embodied AI
- Key Idea: Visual Foundation Model + Offline RL 联合训练，高效具身追踪
- Relevance to My Work: 自己的代表作，Offline RL 方向

### UnrealZOO: Enriching Photo-realistic Virtual Worlds for Embodied AI

- Rating: S
- Year: 2025
- URL: https://arxiv.org/abs/2311.05597
- Local filled: arXiv ID/URL from existing ID
- Topic: Embodied AI, 3D Simulation, Benchmark
- Key Idea: 高质量虚拟仿真环境，提升 sim-to-real 迁移能力
- Relevance to My Work: 自己的核心基础设施工作

### SuperFlow: Training Flow Matching Models with RL on the Fly

- Rating: B
- Year: 2025
- URL: https://arxiv.org/abs/2512.17951
- Local verified: manual public lookup
- Topic: Flow Matching, Reinforcement Learning, Generative Model, Training
- Method Pattern: RL for Flow Matching, Step-level Advantage, Variance-aware Sampling
- Use For: Method Background, Idea Seed
- Idea Hook: 可借鉴其step-level advantage思想训练flow/diffusion action policy。
- Key Idea: 把RL credit assignment做进flow matching训练过程，提高采样和训练效率。
- Limitation: 主要验证在text-to-image生成，不是 embodied control。
- Relevance to My Work: 对diffusion/flow policy + RL的训练范式有方法启发，但不是机器人论文，需要作为方法背景而不是核心related work。

### UniMapping: Unified SLAM Framework for Map-Centric Embodied Perception

- Rating: B
- Year: 2026
- URL: https://app.notion.com/p/UniMapping-Unified-SLAM-Framework-for-Map-Centric-Embodied-Perception-397573cfca7c81528645f9d2669e868b
- Topic: SLAM, Embodied Perception, Mapping, Navigation, World Model
- Method Pattern: Map-centric Perception, Unified SLAM, Spatial Memory
- Use For: Related Work, Method Pattern, Background
- Idea Hook: 把map-centric SLAM作为长期空间记忆底座，结合VLA高层指令和RL低层控制，孵化面向导航/搜救的memory-grounded embodied agent。
- Key Idea: 待核验：把SLAM从几何定位模块提升为具身智能的map-centric perception substrate，为后续任务规划、目标搜索和长期记忆提供统一地图表示。
- Limitation: 公开元数据尚未核验；需要确认是否支持语义/开放词汇地图，以及是否能直接服务VLA或RL策略。
- Relevance to My Work: 对导航和长期具身记忆方向有价值，但与VLA操控/RL主线的直接性略低。可作为空间记忆和map-based world model方向的背景或工具论文。

### DexVLA: VLM with Plug-In Diffusion Expert for Generalist Robot Policy

- Rating: B
- Year: 2025
- URL: https://arxiv.org/abs/2502.05855
- Local filled: arXiv ID/URL from existing ID
- Key Idea: 可插拔扩散action expert：1B参数diffusion-based action expert，三阶段训练(对齐预训练->动作预训练->SFT)。action expert与VLM解耦设计但最终联合微调。启发：diffusion action expert比transformer更适合多模态动作分布。

### MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models

- Rating: Unrated
- Year: 2026
- URL: https://arxiv.org/abs/2606.13515
- Local verified: manual public lookup
- Key Idea: Masks are both precise visual prompts and future semantic prediction targets for WAMs.
- Limitation: World-action-model framing may be heavier than a lightweight policy interface; mask prediction alone does not encode all interaction geometry.
- Relevance to My Work: Directly supports the value of M as an object-centric anchor and clarifies a boundary for future mask/world-model ablations.

### What Can RL Bring to VLA Generalization? An Empirical Study

- Rating: A
- Year: 2025
- URL: https://arxiv.org/abs/2505.19789
- Local filled: arXiv ID/URL from existing ID
- Key Idea: 系统研究RL微调vs SFT对VLA泛化性的影响。RL微调在OOD场景显著优于SFT，但SFT在分布内更稳定。提供RL vs SFT实证基础，支撑'frozen VLM+RL'路线的合理性。

### SayCan: Grounding Language in Robotic Affordances

- Rating: A
- Year: 2022
- URL: https://arxiv.org/abs/2204.01691
- Local filled: arXiv ID/URL from existing ID
- Key Idea: LLM做高层任务规划，预训练skill的value function做affordance grounding。开创'冻结LLM+RL执行'的层次化路线。区别：SayCan是离散skill级解耦，用户是连续特征级解耦。

### Lossless Adaptation of Pretrained Vision Models for Robotic Manipulation

- Rating: A
- Year: 2024
- URL: https://arxiv.org/abs/2304.06600
- Local filled: arXiv ID/URL from existing ID
- Key Idea: 证明冻结预训练视觉模型在机器人操作中可能达不到最优性能，但直接微调会丢失预训练知识。提出Lossless Adaptation：冻结主干+轻量adapter。直接支撑'frozen VLM+轻量投影层'设计，但指出纯冻结可能不够，需轻量adapter辅助。

### DSRL: Steering Your Diffusion Policy with Latent Space Reinforcement Learning

- Rating: Unrated
- Year: 2025
- URL: https://arxiv.org/abs/2506.15799
- Local filled: arXiv ID/URL from existing ID
- Key Idea: [T0] 核心洞察：噪声空间RL=扩散策略的轻量RL微调，无需BPTT，黑箱访问。Noise Aliasing机制允许高效利用离线(s,a,r,s')数据。融合方向：DSRL的噪声空间RL可作为我的Affordance-GCRL框架的推理时增强层，两者不互斥而是互补。Sergey Levine+Abhishek Gupta团队。

### V-GPS: Steering Your Generalists via Value-Guided Policy Steering

- Rating: Unrated
- Year: 2024
- URL: https://arxiv.org/abs/2410.13816
- Local filled: arXiv ID/URL from existing ID
- Key Idea: [T0] 核心：离线RL价值函数作为测试时行为引导器，不修改策略本身（黑箱访问）。Cal-QL比IQL更稳健（大K不退化）。与Affordance-GCRL关系：V-GPS用语言条件化Q函数做重排序，我的框架用Affordance作为GCRL的goal——后者在精确性和跨任务复用上更强。Sergey Levine团队核心工作。

### ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation

- Rating: Unrated
- Year: 2026
- URL: https://arxiv.org/abs/2602.11598
- Local filled: arXiv ID/URL from existing ID
- Key Idea: 脑-动作分层架构（Hierarchical Brain-Action）：认知脑（LLM）负责语义推理，动作专家（Flow Matching）负责精确连续轨迹生成。数据引擎：1690万专家轨迹+500万推理样本，7802个3D场景（10.7km²）。任务统一：Point-Goal/Object-Goal/Instruction-Following/POI-Goal/Person-Following共5项具身导航任务统一建模。
