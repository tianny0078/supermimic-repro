# 按 GitHub star 排名前三的 SuperMimic 复现：源码审计

审计日期：2026-11-01，America/Los_Angeles。排名抓取时间为 2026-11-02 04:07 UTC。本项目对照提交为 `c00a98a1934fec59496486a152cd7a96b45f45be`，正式朝向实验的训练源码为 `5c0a006296f5db6bc91c82320c190d31263210af`。

结论：我们已具备 SuperMimic 的主要训练算法，但尚未实现一个可以与高 star 成功复现直接对照的完整配置。最值得补的是目标轨迹表示、critic 误差特征、运行统计归一化、世界模型预热、动作末尾处理，以及训练吞吐。另有一个已经用 CPU 隔离验证的接口缺口：压缩动作库训练后的检查点，不能通过现有 NPZ 评估入口直接评测。

这里把“缺实现”“已有但未启用/未验证”“不同设计、需要消融”分开。没有把别家的高分直接认定为原论文的正确实现，也没有在此次审计中修改训练代码、重训第三方模型或运行 GPU 仿真。

## 排名与范围

GitHub 官方 REST 搜索 `supermimic`，按 stars 降序，返回总数 119；读取第一页前 100 项已足够确定前三。另用 `supermimic fork:true stars:>200` 和 `super-mimic stars:>200` 复核，前三代码仓库不变。GitHub 连接器搜索超时后使用官方 REST 接口。`miao-org/awesome-supermimic` 为 420 stars，但它是资源索引，不是复现代码，因此不占前三。

| 名次 | 仓库 | Stars | 固定读取提交 | 主要后端 |
|---|---|---:|---|---|
| 1 | [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) | 770 | `0525a23a2dcfa61f8543f98b84c075fe66d9d286` | Isaac Lab 3 / Newton / MuJoCo Warp |
| 2 | [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) | 356 | `ba6105349069bf040f1fdad60873d13138d505a2` | mjlab / MuJoCo Warp |
| 3 | [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) | 220 | `faee48a875579c45b3a41e6399b05ad014641ebf` | UniLab；mjwarp / MuJoCo / Motrix |

下载保存了 148 份文本源码、配置和说明文件；重点阅读训练循环、网络、MDP、动作库、replay、终止、评估及发布记录，不把下载文件数当作逐文件完整审计数量。固定提交、证据行和 CPU 核查结果见 [机器可读审计索引](top3-code-audit-2026-11-01.json)。完整原始快照保存在 `/mnt/data/tianyuan/supermimic-repro/audits/top3-2026-11-01/`。

Uni 的网络和算法与 vibed 有明显继承关系，README 也保留了参考实现的归属。两者相同的设计不能算成两个独立的因果证据。Cuboi 是我们的上游；我对本地与其当前源码做了逐文件差异检查。

## 公开成绩应该怎样理解

以下均为作者自报，未在本机复跑第三方检查点。

| 仓库 / 配置 | 作者报告 | 本轮源码核对 |
|---|---|---|
| vibed，LAFAN1 | 1.24 h：0.824、1/40 失败；1.47 h：0.829、0/40，RTX PRO 6000 | 分母是整条 `N−1`；额外朝向观测、bf16、自碰撞关闭、500 轮 WM-only 预热。[记录][z-results] [评估][z-eval] |
| Cuboi，LAFAN1 +9D anchor | 3.3 h：0.904、3/40；纯参考约 0.56–0.59 | 当前评估仍用实际执行步数作分母；不能与我们的 full-reference-v1 排分。[说明][c-results] [代码][c-eval] |
| Cuboi，BONES-SEED | 4×5090、7.1 h，约 0.93、145/129491 失败 | 动作集合与论文不一致，分母仍是执行长度；作者报告 54 个 flip 全失败。README 的“filter now matches”不是 motion ID 已与论文逐一对齐的证明。[说明][c-results] |
| Uni，LAFAN1 | 12k：0.827227、0/40 | 整条 `N−1` 分母；99D 观测和 730D goal。[说明][u-results] [评估][u-eval] |
| Uni，BONES-SEED 优化配置 | 8k：0.9395、193/129487 失败 | EMA 0.995、no_sync、lr 4e−4；并非早期 2e−4 基线配置，也不是论文相同动作集合。[配置][u-config] [保存 EMA][u-checkpoint] |

我们的受控结果：32000 采集步 / 12784 更新时，纯参考 0.569920、3/40 失败；当前相对躯干朝向目标 0.799138、2/40。朝向实验最终 37450 采集步 / 14964 更新 / 3 h，为 0.807746、3/40。均为 seed0、完整 40 条训练动作、完整参考分母；尚无未见动作泛化结论。

## 优先补齐的实现和验证

### 1. 缺少与 vibed / Uni 等价的输入配置（P1，确定缺实现）

vibed / Uni 将当前参考相对**骨盆**朝向 6D 放进每帧观测，观测 99D，十帧历史 990D；goal 为 10×73D。每个未来参考帧包含相对当前参考航向的位移 3、旋转 6、线/角速度各 3、关节角 29、关节速度 29。[vibed 观测][z-obs] [轨迹目标][z-goal] [Uni 目标][u-goal]

我们使用 93×10=930D 本体历史，加 680D 参考目标和**当前躯干**相对朝向 6D，合计 goal 686D。缺未来参考位移和旋转序列，额外朝向没有十帧历史，也不在相同网络分支。这不是“都加了 6D，所以输入已经一样”。[本地目标][o-goal] [本地想象输入][o-imagine] [本地网络][o-policy]

建议：独立新增 `community_vibed` 输入模式，同时对齐真实仿真和想象中的 pelvis 定义、历史、噪声、缩放及未来帧时序；保留现有 680 / 686 / 689 模式。论文没有完整列出 goal，社区模式仍须明确标为复现假设。未来轨迹输入也不能自动补回机器人相对参考的全局 XY 位置反馈。

### 2. critic 跟踪误差特征未对齐（P1，部分开关已有，完整版本缺失）

vibed / Uni 的 critic 除 71D 不变量状态、goal 和 context 外，还有关节误差 29D、身体位置误差 42D、根位置误差 3D、相对旋转 6D；总输入为 914D。其 critic 还使用 RunningNorm。[vibed][z-critic] [Uni][u-critic]

我们的正式 run 用完整 71D 原始状态，686D goal，33D context，总 790D；`root_error_features=False`。代码虽然能追加 4D 根位置/偏航误差，但这与上面 80D 跟踪误差向量不等价，且未在现有实验启用。[本地配置/网络][o-policy] [实际输入][o-trainer]

原始世界坐标不等于显式参考误差。当前 goal 不含参考绝对 XY，不能仅凭 critic 接收了完整机器人状态，就认定它能唯一确定参考位置奖励。相对误差特征值得做单变量消融；这不是已证实的唯一性能瓶颈，也不能据此修改已报告 run 的含义。

现有最终模型执行期间的根位置奖励为 0.0010865 / 0.01，约 10.9%；根朝向为 0.0096254 / 0.01，约 96.3%。这说明朝向反馈后仍有明显路径漂移。这里是已执行步上的分项均值，不是另一个完整动作回报。

### 3. 缺运行统计归一化和 WM LayerNorm（P1，确定缺实现）

vibed / Uni 对 actor 观测、goal、critic 输入及 WM 输入/输出增量使用 RunningNorm，世界模型隐藏层含 LayerNorm；误差尺度与数据的增量统计相联系，并设下限。[vibed 网络][z-norm] [Uni 网络][u-norm] [误差][u-wmerr]

我们只有人工指定的固定输入、输出和误差缩放，没有 RunningNorm，也没有 WM LayerNorm。WM 还采用速度显式积分加位姿修正，而两家直接预测局部状态增量，函数参数化也不同。[固定尺度][o-state] [网络][o-dynamics]

建议先把运行统计、LayerNorm、状态增量参数化做成可独立配置项。统计量必须入检查点，仿真采集时更新、想象和评估时冻结。WM 误差的归一化不同，因此不能拿我们的 NSE 与论文或社区的 0.007–0.011 数值直接评判模型好坏。

### 4. 缺世界模型独立预热阶段（P1，确定缺实现）

两家先填 64 个采集步，再进行 500 个更新轮次的 WM-only 阶段，之后才更新 actor / critic。每轮有 10 个 WM minibatch，不能把 500 轮解释为 500 个 minibatch。[vibed 训练][z-warm] [Uni 循环][u-loop]

我们填 40 个采集步后就交替更新 WM 和 actor，没有 WM-only 开关。这可能让早期策略通过尚不准确的动力学和终止头优化；实际影响尚需消融验证。[本地循环][o-loop]

### 5. 动作尾部处理不完整（P1，确定行为差异）

vibed / Uni 采样想象起点时要求动作剩余长度能容纳 H=32，避免把片尾反复当作真实运动继续训练。vibed 最新提交还对超出片尾的**合成参考帧**保持最后姿态、把速度置零，同时保留真实最后一帧的原始速度；Uni 当前版本尚没有这项 padding 修正。[起点过滤][z-start] [vibed padding][z-padding] [Uni goal][u-goal]

我们的起点没有剩余长度限制，想象把参考索引截断到最后一帧，继续计算奖励、存活权重和 critic bootstrap；goal 也会重复最后一帧的非零参考速度。[想象][o-imagine] [goal_window][o-goal]

CPU 隔离调用现有 `goal_window` 已确认：当最后一帧关节速度为 2 时，查询的合成未来帧仍为 2。这个输出本身确定；它对最终分数的影响未测。建议明确比较“足够长度起点”“片尾 mask / bootstrap”“合成延伸零速度”三种约定，不能顺手把真实参考末帧速度或评估数据改掉。短 BONES 动作尤其需要处理。

### 6. 压缩动作库训练到评估的接口没有闭合（P1，CPU 已复核）

我们有 `build_library.py`、fp16 紧凑库、在线 FK、`DataCfg.library`，不是完全缺大库模块。问题是训练选用 `.pt` 压缩库时，`run_manifest` 只记录该 `.pt` 文件；`eval_benchmark.py` 仅发现 `.npz`，然后用 NPZ 文件名/哈希去匹配训练 manifest，直接拒绝。评估环境也未转发压缩库模式。[训练入口][o-loop] [评估入口][o-eval-cli] [压缩库][o-build]

此次把现有评估函数的文件读取、检查点和仿真依赖换成 CPU stub：NPZ→NPZ 场景通过数据校验并到达仿真 stub；packed→NPZ 场景在进入仿真前明确抛出 ValueError。未加载 GPU 检查点、未运行仿真。

Uni 已有压缩库到训练、自动沿用训练数据的完整评估流程。[Uni 动作库][u-goal] [评估][u-eval]

需补：压缩库评估入口、逐 motion ID / 来源哈希、训练与评估的量化数据一致性说明，以及匹配的全库验收。我们尚未跑过 BONES-SEED 全库训练/评估。上游后来恢复的杂技过滤关键词在本地仍未合入，但其筛选本身也未与论文精确对齐；不能为了接近数量直接删动作。[上游过滤][c-filter]

### 7. 训练吞吐仍有明确优化空间（P2，确定缺部分工程优化，收益未测）

我们的 replay 每个 WM minibatch 都在 4096×1000 条记录上构造窗口 mask，再对完整 mask 做 multinomial；并有 `.item()` 同步。vibed / Uni 从合法年龄区间直接抽取起点，再对跨回合的后缀应用 mask。[本地 replay][o-replay] [vibed replay][z-replay] [Uni replay][u-replay]

此外，本地训练在采集、WM、actor 后显式同步 GPU，actor 指标频繁转 Python float；优化器没有 fused Adam。Uni 已有 fused Adam、专用小矩阵运算和减少若干同步的路径。[本地循环][o-loop] [本地策略训练][o-trainer] [Uni learner][u-learner] [Uni 几何运算][u-geometry]

从已存在的朝向实验日志（采集步≥10000）重算，三个阶段每采集步耗时中位数约为：sim 0.028 s、WM 0.076 s、policy/critic 0.180 s。它们包含各阶段内部采样、更新等开销，不是单独算子的计时；尚不能把速度差归因于其中一个调用。应在 Slurm 下 profile 后再改采样器，尤其保留 terminal row、回合边界和环形缓冲区边界正确性。也不能把 Uni 的 `no_sync` 名称理解为全程没有同步，它仍有统计输出转 float 等操作。

## 需要作为独立实验的选项

| 差异 | 社区实现 | 我们当前情况 | 判断 |
|---|---|---|---|
| 解析软终止 | vibed / Uni 用阈值 sigmoid 软化 | 学习终止头 + BCE | 缺可切换解析终止模式；两种设计均可属于论文允许范围，不能预先判定学习头错误 |
| 精度 | vibed / Uni 默认 bf16 MLP，积分/FK 保持 fp32 | fp32/TF32 | 缺 bf16 可选加速；当前保留 TF32 是对齐论文的选择 |
| EMA | Uni 最新配置 0.995，保存/评估平均策略，同时另存 live 权重 | 无 | 可选稳定化；须同时评估 live / EMA，不能把平均策略成绩混成旧配置成绩 |
| 学习率 | Uni 最新默认 4e−4；早期基线 2e−4 | 2e−4 | 优化配置差异；不宜与 EMA、归一化同时改后归因 |
| 自碰撞/模型/锚点 | vibed 关闭自碰撞，pelvis 锚点，自有 URDF→MJCF；Uni 同系 pelvis MDP | mjlab G1 FULL_COLLISION，torso 相对身体奖励，pelvis 全局奖励/终止 | 应做资产/接触/锚点一致性核对，关闭自碰撞不是必须补的功能 |
| 训练 reset 噪声 | vibed / Uni 默认开启姿态、速度扰动 | 开关已有；正式 run 关闭 | 不是缺实现，是尚未验证的训练分布差异；评估仍须无噪声 |
| return-based sampler | vibed / Uni 主要更新回合起点 bin，用均值奖励近似 bootstrap | 更新轨迹经过的 bin；bootstrap_fn 未接入时屏蔽近截断步 | 三者均有 50/40/10 采样，但不是同一个实现；本地有开关/接口不等于已启用 critic bootstrap |
| iteration | vibed / Uni 一轮更新，平均采集 2.5 步 | 一次并行采集，平均 0.4 轮更新 | UTD 一致，日志/总预算/LR 衰减的单位不同；缺统一计数和可选 update-based 调度 |

例如 vibed 的 14000 iterations 包含 500 WM-only 轮，约 13500 次 actor 更新，约 35000 个并行采集步（另有初始填充）。我们最佳同进度对照是 32000 采集步 / 12784 更新；最终是 37450 / 14964。学习量是同一量级，不能只凭“14k 比 37k 少”推断样本效率。

解析终止、EMA、bf16、lr 翻倍、自碰撞关闭都不能整包加入后继续称作与原论文明确设置完全相同。论文的目标编码、状态尺度等未充分说明的部分，应继续记录为假设。

## 已有的能力与尚未完成的验收

已经具备，不能重复报为缺失：WM k=8 多步拟合、H=32 BPTT、critic bootstrap、GAE、5×4 critic 更新、动作执行噪声单独记录、十帧本体历史、50/40/10 重置采样、DR、torch.compile、数据并行梯度同步、紧凑动作库模块、逐动作失败时间、仿真视频。我们的完整参考 N−1 分母、失败后缀记零、数据/源码/机器人哈希、原子检查点，以及相关回归检查应继续保留。

Cuboi 最新源码仍有 actor objective 除 H、WM loss 除 k，以及执行步分母；这些我们已经修改，不能机械同步回去。上游新增的 `merge_benchmark_shards.py` 在本地缺失，但我们的 `--clip-list` 已支持分片；应补一个校验重复/漏 motion ID、使用 full-reference-v1 和一致 provenance 的合并器，而不是原样复制上游聚合公式。[上游合并][c-merge]

Uni 额外提供多后端 policy I/O 合约、自动恢复评估数据、Viser 在线播放及兼容性检查。我们目前只有 mjlab 后端和录制视频，缺交互式暂停/单步/切换环境页面及跨后端合约。[Uni 合约][u-contract] 这些是工程扩展，不是解释当前 0.8077 的首要因素。

尚缺实验验收：同配置多随机种子、BONES 全库、真实 PPO-Sim 对照、H×critic 消融、与论文匹配的抗推统计、真机。前三项目也没有全部完成这些：Cuboi README 明确列出未做 PPO / 速度接口 / 真机等，vibed 的主结果限于 LAFAN1。因此这些属于本项目完整复现范围的缺口，不应假称“前三都有、只有我们没有”。`eval_push.py` 和无 critic 开关在本地已存在，但有脚本不等于已有验收结果。

## 建议执行顺序

1. 先补公平比较的配置与计数：把输入、critic、WM 归一化、预热、终止、精度、资产、采集步/更新数/LR 单位显式记录；保留当前结果原样。
2. 低风险修补压缩库评估和片尾规则，并用 CPU / Slurm GPU 检查数据一致性、边界和梯度。片尾规则改变训练语义，需新实验名。
3. 同环境、同数据、同预算先做归一化 / WM-only / critic 误差 / 轨迹 goal 的受控比较；既报告匹配真实采集量，也报告时间。完整社区配置可作为参考线，但不能用整包改动建立单因素结论。
4. profile 后优化 replay、同步和优化器；再单独测试解析终止、EMA、bf16、reset 扰动。
5. 达到稳定 LAFAN1 指标后做多种子、PPO-Sim 和大库。BONES 必须先锁定 motion ID，不以库大小接近论文或失败数较小代替数据对齐。

本次产出为审计文档与可核查快照；没有宣称上述修改已完成或已提升训练成绩。CPU 核查仅验证压缩库校验路径和当前片尾输出，不代替端到端 GPU 训练与评估。

[z-results]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/MILESTONES.md#L21
[z-eval]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/evaluate.py#L84
[c-results]: https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/README.md#L9
[c-eval]: https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/src/supermimic/eval/benchmark.py#L97
[u-results]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/README.md#L254
[u-eval]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/tasks/motion_tracking/g1/supermimic/evaluate.py#L82
[u-config]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/conf/supermimic/config.yaml#L27
[u-checkpoint]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/learner.py#L571
[z-obs]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/mdp.py#L95
[z-goal]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/motion.py#L100
[u-goal]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/tasks/motion_tracking/g1/supermimic/motion.py#L296
[o-goal]: ../src/supermimic/envs/mdp/motion_library.py#L226
[o-imagine]: ../src/supermimic/imagination/rollout.py#L36
[o-policy]: ../src/supermimic/policy/nets.py#L20
[z-critic]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/mdp.py#L163
[u-critic]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/tasks/motion_tracking/g1/supermimic/mdp.py#L249
[o-trainer]: ../src/supermimic/imagination/policy_trainer.py#L56
[z-norm]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/networks.py#L24
[u-norm]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/networks.py#L34
[u-wmerr]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/learner.py#L279
[o-state]: ../src/supermimic/world_model/state_spec.py#L47
[o-dynamics]: ../src/supermimic/world_model/dynamics.py#L9
[z-warm]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/train.py#L37
[u-loop]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/learner.py#L632
[o-loop]: ../src/supermimic/training/loop.py#L35
[z-start]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/train.py#L159
[z-padding]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/motion.py#L104
[o-eval-cli]: ../scripts/eval_benchmark.py#L22
[o-build]: ../scripts/data/build_library.py#L18
[c-filter]: https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/scripts/data/filter_bones_seed.py#L29
[o-replay]: ../src/supermimic/data/replay_buffer.py#L97
[z-replay]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/buffer.py#L41
[u-replay]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/replay.py#L59
[u-learner]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/learner.py#L155
[u-geometry]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/geometry.py#L91
[c-merge]: https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/scripts/merge_benchmark_shards.py#L16
[u-contract]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/sim2sim.py#L142
