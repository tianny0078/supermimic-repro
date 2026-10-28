# SuperMimic 社区实现：结果与代码复核

复核日期：2026-10-25（America/Los_Angeles）。本次重新搜索到 53 个名称/描述匹配的公开仓库（含本项目），另补读 MorphMimic；取得 50 份 README，对其中 26 个项目读取了关键源码，共保存 169 个文本文件。所有来源固定到提交 SHA。完整列表见 [54 项索引](community-repositories-2026-10-25.md)，具体读取文件见 [机器可读索引](community-repositories-2026-10-25.json)。

本次执行的是源码阅读、公开结果文件检查和 CPU 数值重算；没有重训第三方模型、运行其仿真评估或验证其硬件耗时。读取 HEAD 与证明某个结果由该 HEAD 训练得到是两回事；有训练源码哈希时才可进一步绑定。公开成绩均按作者报告标注。

## 对当前复现的直接结论

最值得先验证的是 **goal 的坐标表示与相对朝向反馈**。这比单纯延长当前 reference-only 训练更有证据支持，但尚未在本项目完成因果消融。

需要修正之前的说法：**93 维本体观测不等于 goal 里禁止相对朝向。** 论文 Appendix A 分别定义 proprioception 与 goal，只给出未来 10 帧，没有穷举每帧目标特征。因此当前 reference-only 的 680 维目标只是一个解释，不能自动称为唯一“严格论文口径”。反过来，社区把某种相对朝向序列命名为 paper/beyondmimic_seq，也不能作为已确认原作者实现的证据。[论文](https://arxiv.org/abs/2610.28480)

本项目 Slurm 742/743 实测：单 H200，3 小时，最佳完整片段 macro return **0.569920**，3/40 失败；32000 是采集步计数，对应 **12784 次更新**。相对身体位置项达到其最大值约 97%，全局根位置/朝向仅约 6%/18%（后者均为执行步上的分项统计）。这种表现与缺少朝向纠偏信息相容；它不能单独排除世界模型、动作库、奖励和优化上的其他问题。

## 有训练成绩且值得优先参考的项目

表中 F = 分母为完整参考长度（具体 N 或 N−1 另核对），E = 分母为实际执行步数。失败后的后缀在 F 下会降低回报。F/E 不能混排成统一排行榜。

| 项目 | 作者公开结果 | 实际输入/关键实现 | 评估与适用范围 |
|---|---|---|---|
| [zita-ch/vibed](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/MILESTONES.md) | RTX PRO 6000，1.24 h 0.824、1失败；1.47 h 0.829、0/40 | 本体项外加 6D 参考相对朝向，观测共99维；bf16；关闭自碰撞；500次WM-only | F；40条LAFAN1；其多个下游重复引用同一组成绩 |
| [dhmbb2/vibe](https://github.com/dhmbb2/supermimic_vibe/blob/072f0910dadc9d95c720e625b546120e9ce917c9/notes/RESULTS.md) | A6000，1.34 h 0.833、1失败；12 h 0.946、0/40 | 本体93；每未来帧73维，包含参考根相对机器人根的位置3/旋转6；bf16；解析终止 | F；逐动作JSON重算均值0.9463、0失败；有PPO、H×critic等对照 |
| [Go-Youngster](https://github.com/Go-Youngster/supermimic-reproduce/blob/02db3d5ae75a54c76ddec969c8bd63cc8fefc661/README.md) | A100约3 h达0.82；最佳15.7 h 0.875、0/40 | 本体93；710维goal包含每未来帧相对根朝向；bf16；WM带FK特征 | F；单种子；部分计时来自无竞争迭代耗时换算，不能统一当实际墙钟 |
| [YanYuan98](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/README.md) | TF32单5090约3 h 0.850、0/40；纯参考对照0.564、5失败 | 目标增加6D相对躯干朝向；可选未来轨迹特征；保留自碰撞的正式结果 | F；对我们当前0.57最有参考价值的公开对照；并非本机独立复测 |
| [LaoYang1994](https://github.com/LaoYang1994/SuperMimic-Rep/blob/6f8db6b1e9e02062c44aee358d69c6c8852745ee/README.md) | LAFAN1 18k更新0.821、1失败；BONES44k更新0.934、210/129511失败 | 默认每未来帧关节29+速度29+相对朝向6；另有当前6D和完整9D位姿模式 | F，同时保留E诊断；“论文口径”是该项目解释；BONES失败数未对齐论文 |
| [CuboiLeo](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/docs/EXPERIMENTS.md) | +9位姿：1.66 h 0.824、5失败，3.3 h 0.904、3失败；reference-only约0.56 | 我们的代码基线；680维纯参考目标，可追加9维相对躯干位姿 | **E**；README虽称Appendix F，代码分母为实际步数；BONES0.935仅2000子集 |
| [panhainan2020](https://github.com/panhainan2020/SuperMimic-MuJoCo/blob/db045afa9c52458f213e2c66037387e9c109a03a/results/098000/summary.json) | 固定98k：LAFAN1 0.871322、40/40；100STYLE 0.804516、3569/3573 | vibed的MuJoCo/mjlab移植；legacy99含相对朝向 | F；根位置平均误差仍0.643 m，说明完成/高分不等于全局路径准确 |
| [Motphys/Uni](https://github.com/Motphys/Uni-SuperMimic/blob/faf7818e8abac3ec6abb8f00e6e904d3577f800b/README.md) | 12k：0.827227，0/40 | UniLab/vibed下游；相对朝向；解析奖励/终止 | F；README未给这次训练的完整硬件/耗时证据；不是新增独立算法 |
| [TrackerLab](https://github.com/Renforce-Dynamics/trackerlab-supermimic/blob/29d24524f2f731bf4df75b319c3e76544165849e/docs/results.md) | 32k：0.8694663，0/40；旧2k为0.5416、12失败 | IsaacLab/PhysX；旧ZouYan93仓库现迁移到组织 | 完整动作评估；后端不同，旧列表只列2k结果已过时 |
| [allmimic](https://github.com/Shimingyi/allmimic/blob/cb334ff674b2f57248f33f8c21b7fb526fbdf8cc/README.md) | 单4090 LAFAN1 2 h 0.94、0/40；双4090 6.6 h完成困难集306/306 | 基于dhmbb2；上身残差动作、latent128、25%无噪采集、均衡采样、值裁剪等 | F；强工程参考；算法/动作/数据均变化，不能称与论文原设置等价 |
| [SuperMimic_Dyn](https://github.com/Baojch/SuperMimic_Dyn/blob/28b15acb68ab3871f2ae1479964d53457ac066c6/reproducibility/e009/results.json) | E009验证3939/3966完成；完整片段macro0.864542；双4090约4.09 h | full689即+9位姿；自定义95323训练片段；均衡来源/类别；移除手部几何 | 正式套件报告F及N−1；继承benchmark.py仍为E，须看实际调用入口；验证集没有翻转 |
| [kangzejian1896](https://github.com/kangzejian1896/supermimic-g1-reproduction/blob/5daf8586f26da3ef3bd99bdd875c8f8ce1b4e18e/docs/benchmark_168k.json) | A800约25.86 h，168k；验证128条0.928972、0失败；训练84287条0.944235、9失败 | Cuboi下游；可选paper_protocol改变锚点/随机化/评估 | F在paper_protocol下；验证参与选模；数据非论文129514条 |
| [NJU Sim2Real](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/blob/99d310d5460333aabe853337b1d55ee27940a3fe/docs/METHOD.md) | 自报BONES129491条0.9305、46失败；有真机舞蹈Release | Cuboi下游；自定义舞蹈/部署；LAFAN e2/e3表沿袭上游 | 当前benchmark仍E；主结果表硬件/耗时空缺，不能据此比较训练速度 |
| [SuperMimicKit](https://github.com/OMEGA-i/SuperMimicKit/blob/6697ff8ca53adc9233ded12a7c8777610ca1154d/docs/preview-evaluation-g1-55500.json) | G1 55.5k：train40 0.798961、3失败；val37 0.3103、31失败 | 多本体工具链；保留较早检查点 | F；G1到100k出现40/40失败，值得参考退化诊断，不能只挑早期分数 |

## 代码层面确认的差异

### 1. 相对朝向与相对位置是两个问题

- vibed：`mdp.py:OBS_DIM` 是93+6；`obs()` 拼接 `R_robot^T R_reference` 的6D表示。[源码](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/mdp.py#L103)
- dhmbb2：`Task.goal()` 把未来参考根位置减去机器人根位置后旋到机体系，同时加入相对旋转。它需要位置估计，和只用相对朝向的方案不同。[源码](https://github.com/dhmbb2/supermimic_vibe/blob/072f0910dadc9d95c720e625b546120e9ce917c9/code/supermimic/task.py)
- Go-Youngster：`goal_obs()` 使用 `quat_conj(robot_q) * reference_q`；不含全局位置，但确实含机器人姿态反馈。README的“without global positions”不能解释为完全没有相对反馈。[源码](https://github.com/Go-Youngster/supermimic-reproduce/blob/02db3d5ae75a54c76ddec969c8bd63cc8fefc661/supermimic/task.py#L112)
- LaoYang：`beyondmimic_seq` 是10×64维，每帧含参考关节、速度和当前机器人锚点下的参考朝向；`anchor_ori_now` 则是680+6维。这两个接口内容和时间分布不同，不应合并。[源码](https://github.com/LaoYang1994/SuperMimic-Rep/blob/6f8db6b1e9e02062c44aee358d69c6c8852745ee/src/supermimic/envs/mdp/observations.py#L23)
- YanYuan：`anchor_obs` 明确分为 none / ori / pos_ori；目标轨迹也分68维旧表示与77维轨迹表示。[源码](https://github.com/YanYuan98/Supermimic_mod/blob/6932f1cd33532b4e84ff173df926d8195a58c45a/src/supermimic/goal_spec.py#L1)

因此本项目最合适的下一组实验是先保留93维本体和现有训练/评估设置，只追加当前参考相对躯干朝向6D；另把64×10的序列目标作为独立解释。完整9D位姿应是第三组，不能用来代替朝向消融。IMU提供的航向仍依赖初始参考对齐和漂移假设，不能把6D朝向描述成无条件准确可测。

### 2. 评估器的分母确实分叉

Cuboi的实现是：

```python
norm_return = ret_sum / (max_r * max(executed_steps, 1))
```

位于 [benchmark.py:97](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/src/supermimic/eval/benchmark.py#L97)；`aggregate()` 还按执行步加权。这会把提前失败后的未播放后缀排除掉。若每步均拿满分、却只执行整条动作的10%，E仍可为1，F最多为0.1。

vibed、dhmbb2、Go-Youngster和Uni的评估器按完整参考长度 N−1 归一化；YanYuan、LaoYang也已修改完整分母，但局部代码仍使用N或max(N,steps)，存在一帧级差异。我们的 `full-reference-v1` 使用N−1，并分别给出macro和参考长度加权结果。

这修正了旧调研的一处实质遗漏：Cuboi的0.904不能和我们的0.569920直接当作同口径指标。失败数少时差异可能有限，仍需要原始逐动作回报/长度才能精确转换，不能从总均值倒推。

### 3. 可从公开JSON直接验证的结果与不一致

- dhmbb2的40条 `supermimic_return`（四位小数）均值为 **0.9463**，失败字段合计0，与12小时表一致。它支持公开汇总内部一致性，不等于验证真实训练或墙钟。[逐动作记录](https://github.com/dhmbb2/supermimic_vibe/blob/072f0910dadc9d95c720e625b546120e9ce917c9/results/audit/per_clip.json)
- M5公开模型的E分数 **0.708095**，15失败。用逐片 `norm_return×steps` 重建归一化奖励，再除各自完整帧数N，macro为 **0.582600**；使用N−1约 **0.582653**。原评估25条完成动作执行了N次而非N−1次，因此只能作为分母影响的CPU重算，不能替代重新运行修正后的仿真。[原始逐片JSON](https://github.com/WXyuany/supermimic-m5/blob/42451591718e1a9a5ec601366997f47c0ad180a3/results/benchmark_retuned_21000.json)
- fan-ziqi的 `full_eval.json` 有13条分数超过其完成比例上界。例如 `fallAndGetUp1_subject1` 仅执行290/8409步，若每步最多0.1，完整回报最多 **0.03449**，文件却列 **0.80002**。当前 `evaluate.py` 已按完整长度计算；公开历史JSON与这段代码不能同时代表同一口径。原因可能是旧分母或证据未同步，目前无法确定，不应采信0.730896作为已核验的完整动作分数。[结果](https://github.com/fan-ziqi/supermimic-agy/blob/8513519b1b9b2d18ac61089748ddebbfe1bcf52b/runs/supermimic_lafan_1gpu/full_eval.json)、[当前代码](https://github.com/fan-ziqi/supermimic-agy/blob/8513519b1b9b2d18ac61089748ddebbfe1bcf52b/supermimic/evaluate.py#L47)、[奖励上界](https://github.com/fan-ziqi/supermimic-agy/blob/8513519b1b9b2d18ac61089748ddebbfe1bcf52b/supermimic/state.py#L22)

CPU重算结果保存为 [numeric-checks JSON](community-numeric-checks-2026-10-25.json)。没有下载第三方权重。

### 4. “iteration”至少有两种含义

| 代码路径 | 一次日志iteration | UTD=0.4的实际含义 |
|---|---|---|
| vibed、Uni、Go-Youngster、LaoYang、YanYuan | 一组WM/actor/critic更新 | 更新前平均采集2.5个并行仿真步 |
| dhmbb2、allmimic、Cuboi旧版、本项目已完成run | 一个并行仿真采集步 | 平均每5步执行2组更新 |

同为“32000迭代”会差约2.5倍更新数。论文Algorithm A.1明确每迭代更新WM与actor，但UTD的计数单位仍需要实现解释。我们的已完成run应标注采集步和更新数，而不能拿32000与vibed的14000直接比较。长期250k学习率衰减按哪个计数执行也需要单列为复现假设；本次不改旧run配置或追溯重命名其步数。

例如本项目最佳32000采集步/12784更新；vibed的14000包含约500个WM-only更新，其actor更新数量约13.5k。两者策略更新量接近，观测、模型表示、精度和评估仍不同，不能据此把差距归因于某一项。

### 5. 相同网络宽度不代表相同世界模型

- Cuboi系：71维完整仿真状态，规范化网络输入，预测速度增量/位姿修正，FK解析奖励，学习终止头。
- dhmbb2/Go-Youngster额外从FK得到身体位置/速度等输入；预测增量、运行统计归一化与名义误差尺度也不同。
- vibed/Uni使用解析软终止；500个WM-only预热、LayerNorm/运行归一化、bf16是额外选择。
- allmimic修正或增强梯度裁剪、值范围、采样与动作表示；MorphMimic还报告LayerNorm、值目标上限、避开片尾想象等做法。它们值得做消融，不能整包搬入后称原论文等价。
- actor目标按H相加还是再除H会影响固定阈值梯度裁剪：vibed/Go-Youngster使用 `-J.mean()`，dhmbb2为 `-J.mean()/H`；allmimic仍除H，但推荐preset把全局clip改到1/32。两者在Adam、epsilon及逐元素clip等细节下不能直接宣称严格等价。
- 本项目已经移除多余的 /H 和WM的 /k。当前未出现非有限值，不能仅因别家使用value clip就断言本项目必须加入。

## 哪些“复现成功”不能直接继承

- allmimic的306条困难动作包含54条flip和196条lying/resting/getup；原项目失败表列的是44条flip和99条lying/resting/getup，并另有dance。类别数量与动作ID未一一对齐，“306/306”不能代替完整129514条零失败。[原项目基准定义](https://supermimic.github.io/)
- SuperMimic_Dyn的验证集3966条没有flip/cartwheel；翻转表现来自训练集。继承的benchmark.py用E，但最终E009套件另有F统计，必须按实际入口区分。
- samindaa的0.829/1.47h标注为移植前IsaacLab结果；本地mjlab4k为0.732、4失败。HighTorque的README与vibed相同。CesarLiu继承Uni的12k表。它们不能增加“独立成功复现次数”。
- NJU的LAFAN e2/e3表与Cuboi同源；其新BONES及部署结果需单独核对，不能把那两行当作独立重复证据。
- BONES筛选库存在129943、129511、129491、95323和84287等不同集合。仅比较失败个数会误导；必须绑定motion ID清单。
- 不少展示把参考虚影逐步对齐机器人XY/航向，适合看关节姿态但会遮住全局漂移。评估应同时看未对齐根轨迹、完整动作回报和失败时间。
- 后空翻、太空步微调、E1/Oli/A3/H1或灵巧手项目各自有价值，但任务、训练先验和奖励不同，不纳入G1/LAFAN1的统一成绩表。

## 本项目建议顺序

1. 固定现有40条数据与 `full-reference-v1`，保留reference-only基线。
2. 优先做6D当前相对朝向单变量实验；使用相同critic、TF32、动作、奖励、碰撞和预算。训练与想象都从相应真实/预测机器人姿态重算goal，不能只改仿真输入。
3. 再单独比较10×64相对朝向序列解释，记录goal尺度、噪声、开始帧和片尾处理。
4. 明确区分采集步、WM更新组、actor更新和WM minibatch数；确认250k调度与320k总量的计数定义后再启动长训。
5. 增加未对齐根XY/航向轨迹和长片失败前后录像；完整9D位姿作为另一个标注清楚的参考组。
6. 若仍不收敛，再逐项验证WM输入归一化、reset噪声、片尾bootstrap、critic过估计和采样分布。保留每项独立实验，不同时更换整套实现。

本次交付是静态复核与实验设计；尚未产生6D/序列目标的本机训练结果。

